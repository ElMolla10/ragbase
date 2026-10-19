"""Authenticated retrieval, quote answers, cancellation and native-text previews."""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.answers.grounding import (
    REMOTE_BUDGET,
    input_bound,
    make_request,
    overview_question,
    validate_answer,
)
from backend.gateway.contracts import GatewayError
from backend.library import bounded_body, identity
from backend.policy import assemble
from backend.search.index import INDEX_VERSION
from backend.search.retrieve import document_context, retrieve
from backend.storage.local import StoredArtifact

router = APIRouter(prefix="/api/workspaces")


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=512)
    version_ids: list[UUID] = Field(min_length=1, max_length=8)
    client_request_id: UUID = Field(default_factory=uuid4)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Question is empty")
        return value.strip()

    @field_validator("version_ids")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Select each immutable version once")
        return value


async def parse_input(request):
    try:
        return SearchInput.model_validate_json(await bounded_body(request, 8192))
    except ValueError as exc:
        raise HTTPException(
            422, "A question and 1–8 distinct version IDs are required"
        ) from exc


async def evidence(request, actor, workspace, data, *, answer_mode=False):
    database = request.app.state.database
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        overview = answer_mode and overview_question(data.question)
        hits = (
            []
            if overview
            else await retrieve(conn, workspace, data.version_ids, data.question)
        )
        coverage = {"strategy": "lexical", "total_document_chunks": None}
        if answer_mode and (overview or not hits):
            hits, total = await document_context(conn, workspace, data.version_ids)
            coverage = {
                "strategy": "document_overview"
                if overview
                else "document_context_fallback",
                "total_document_chunks": total,
            }
        if hits:
            versions = list({str(hit["version_id"]) for hit in hits})
            originals = (
                (
                    await conn.execute(
                        text(
                            "SELECT storage_key,sha256,byte_size FROM enterprise.document_versions WHERE workspace_id=:w AND id=ANY(CAST(:versions AS uuid[]))"
                        ),
                        {"w": workspace, "versions": versions},
                    )
                )
                .mappings()
                .all()
            )
            for original in originals:
                await asyncio.to_thread(
                    request.app.state.storage.read,
                    workspace,
                    StoredArtifact(
                        original["storage_key"],
                        original["sha256"],
                        original["byte_size"],
                    ),
                )
        return (hits, coverage) if answer_mode else hits


@router.post("/{workspace}/search")
async def search(workspace: UUID, request: Request, principal=Depends(identity)):
    data = await parse_input(request)
    hits = await evidence(request, principal.user_id, workspace, data)
    return {
        "index_version": INDEX_VERSION,
        "retrieval": "lexical_baseline",
        "hits": jsonable_encoder(hits),
        "scope": [str(value) for value in data.version_ids],
        "exhaustive": False,
    }


@router.get("/{workspace}/sources/{source}")
async def source_preview(
    workspace: UUID, source: UUID, request: Request, principal=Depends(identity)
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace) as conn:
        await database.authorize(conn, workspace)
        row = (
            (
                await conn.execute(
                    text("""
            SELECT c.*,p.location,v.sha256,v.byte_size,v.storage_key,d.filename
            FROM enterprise.document_chunks c JOIN enterprise.extracted_pages p ON p.version_id=c.version_id AND p.ordinal=c.ordinal
            JOIN enterprise.document_versions v ON v.id=c.version_id JOIN enterprise.documents d ON d.id=v.document_id
            JOIN enterprise.ingestion_jobs j ON j.version_id=c.version_id
            WHERE c.id=:id AND c.workspace_id=:workspace AND c.index_version=:index AND j.status='extracted'
        """),
                    {"id": source, "workspace": workspace, "index": INDEX_VERSION},
                )
            )
            .mappings()
            .first()
        )
        if not row:
            raise HTTPException(404, "Source not available")
        await asyncio.to_thread(
            request.app.state.storage.read,
            workspace,
            StoredArtifact(row["storage_key"], row["sha256"], row["byte_size"]),
        )
    # Plain text avoids executing document Markdown/HTML or rendering instructions.
    return PlainTextResponse(
        f"Native-text source preview\nFile: {row['filename']}\nVersion: {row['version_id']}\nSHA-256: {row['sha256']}\nLocation: {json.dumps(row['location'], ensure_ascii=False)}\nCharacters: {row['char_start']}–{row['char_end']}\nIndex: {row['index_version']}\n\n{row['content']}\n\nNative text only; layout, tables, images and OCR are not validated.",
        headers={
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'"
        },
    )


async def run_status(database, actor, workspace, run_id):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT status,lease_until>clock_timestamp() AS current FROM enterprise.answer_runs WHERE id=:id"
                    ),
                    {"id": run_id},
                )
            )
            .mappings()
            .first()
        )
        if not row or row["status"] not in {"queued", "running"} or not row["current"]:
            raise GatewayError("Answer was cancelled, expired or revoked")
        return row["status"]


async def record(
    database, actor, workspace, run_id, status, *, metadata=None, error=None
):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        await conn.execute(
            text("""
            UPDATE enterprise.answer_runs SET status=:status,error_code=:error,metadata=CAST(:meta AS jsonb),updated_at=now()
            WHERE id=:id AND status IN ('queued','running') AND lease_until>clock_timestamp()
        """),
            {
                "id": run_id,
                "status": status,
                "error": error,
                "meta": json.dumps(metadata or {}),
            },
        )


@asynccontextmanager
async def slot(database, actor, workspace, run_id):
    deadline, connection = time.monotonic() + 90, None
    try:
        while time.monotonic() < deadline:
            await run_status(database, actor, workspace, run_id)
            candidate = await database.engine.connect()
            try:
                acquired = await candidate.scalar(
                    text("SELECT pg_try_advisory_lock(728003)")
                )
                if acquired and await candidate.scalar(
                    text("SELECT enterprise.heavy_lease_active(:id)"), {"id": run_id}
                ):
                    await candidate.execute(text("SELECT pg_advisory_unlock(728003)"))
                    acquired = False
                await candidate.commit()
            except BaseException:
                await candidate.invalidate()
                await candidate.close()
                raise
            if acquired:
                connection = candidate
                break
            await candidate.close()
            await asyncio.sleep(0.2)
        if connection is None:
            raise GatewayError("Generation queue deadline exceeded")
        await run_status(database, actor, workspace, run_id)
        async with database.scope(actor, workspace) as conn:
            await database.authorize(conn, workspace)
            await conn.execute(
                text(
                    "UPDATE enterprise.answer_runs SET status='running',lease_until=clock_timestamp()+interval '120 seconds' WHERE id=:id AND status='queued'"
                ),
                {"id": run_id},
            )
        yield
    finally:
        if connection is not None:
            try:
                await connection.execute(text("SELECT pg_advisory_unlock(728003)"))
                await connection.commit()
            except BaseException:
                await connection.invalidate()
                raise
            finally:
                await connection.close()


async def collect(gateway, model_request):
    parts, usage, first, started, finished = [], None, None, time.monotonic(), False
    provenance = None
    async for event in gateway.stream(model_request):
        if event.kind == "text":
            first = first if first is not None else time.monotonic() - started
            parts.append(event.text)
            if sum(len(part.encode()) for part in parts) > 16384:
                raise GatewayError("Answer exceeds size limit")
        elif event.kind == "done":
            if event.finish_reason != "stop":
                raise GatewayError("Incomplete answer")
            usage, finished = event.usage, True
            provenance = event.provenance
    if not finished:
        raise GatewayError("Incomplete answer")
    return "".join(parts), usage, first, provenance


def frame(kind, **value):
    return json.dumps({"event": kind, **value}, ensure_ascii=False, default=str) + "\n"


@router.post("/{workspace}/answers")
async def answer(workspace: UUID, request: Request, principal=Depends(identity)):
    data = await parse_input(request)
    actor, database = principal.user_id, request.app.state.database
    hits, coverage = await evidence(request, actor, workspace, data, answer_mode=True)
    gateway = request.app.state.gateway
    if gateway is None:
        raise HTTPException(
            503,
            "Generation is disabled; retrieval and source previews remain available",
        )
    instructions = assemble("retrieval")
    output_budget = (
        2048 if request.app.state.settings.generation_mode == "openrouter" else 256
    )
    budget = (
        REMOTE_BUDGET
        if request.app.state.settings.generation_mode == "openrouter"
        else 4096
    )
    selected = []
    for hit in hits:
        candidate = make_request(
            instructions,
            data.question,
            [*selected, hit],
            max_output_tokens=output_budget,
        )
        if input_bound(candidate) + candidate.max_output_tokens <= budget:
            selected.append(hit)
    model_request = make_request(
        instructions, data.question, selected, max_output_tokens=output_budget
    )
    if input_bound(model_request) + model_request.max_output_tokens > budget or (
        hits and not selected
    ):
        raise HTTPException(
            413,
            "Question and source evidence exceed the context budget; narrow the question",
        )
    run_id = uuid4()
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        if await conn.scalar(
            text(
                "SELECT id FROM enterprise.answer_runs WHERE client_request_id=:client"
            ),
            {"client": data.client_request_id},
        ):
            raise HTTPException(409, "This request has already been submitted")
        if not await conn.scalar(text("SELECT enterprise.admit_answer()")):
            raise HTTPException(
                429, "Answer queue is full or this user already has an active answer"
            )
        await conn.execute(
            text("""
            INSERT INTO enterprise.answer_runs(id,workspace_id,created_by,client_request_id,question_sha256,status,lease_until)
            VALUES(:id,:workspace,:actor,:client,:hash,'queued',clock_timestamp()+interval '180 seconds')
        """),
            {
                "id": run_id,
                "workspace": workspace,
                "actor": actor,
                "client": data.client_request_id,
                "hash": hashlib.sha256(data.question.encode()).hexdigest(),
            },
        )
        for hit in selected:
            await conn.execute(
                text(
                    "INSERT INTO enterprise.answer_sources(run_id,workspace_id,chunk_id) VALUES(:run,:workspace,:chunk)"
                ),
                {"run": run_id, "workspace": workspace, "chunk": hit["id"]},
            )

    async def generate():
        started, task, completed = time.monotonic(), None, False
        remote = request.app.state.settings.generation_mode == "openrouter"
        yield frame(
            "queued",
            run_id=run_id,
            external_processing=remote,
            privacy_policy=request.app.state.settings.remote_privacy
            if remote
            else "local",
        )
        try:
            async with slot(database, actor, workspace, run_id):
                queue_seconds = time.monotonic() - started
                yield frame("stage", stage="generating_and_validating")
                raw, usage, first = '{"insufficient":true,"claims":[]}', None, None
                provenance = None
                if selected:
                    task = asyncio.create_task(collect(gateway, model_request))
                    try:
                        while not task.done():
                            if await request.is_disconnected():
                                raise asyncio.CancelledError
                            await run_status(database, actor, workspace, run_id)
                            await asyncio.wait({task}, timeout=0.5)
                        raw, usage, first, provenance = await task
                    finally:
                        # Close the provider transport before releasing admission.
                        if not task.done():
                            task.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await task
                verified = validate_answer(raw, selected)
                if len(selected) < len(hits):
                    verified["scope_notice"] += (
                        f" {len(hits) - len(selected)} retrieved passages were not included because of the context budget."
                    )
                await run_status(database, actor, workspace, run_id)
                metadata = {
                    "constitution_version": instructions.constitution_version,
                    "instruction_sha256": model_request.instructions.sha256,
                    "index_version": INDEX_VERSION,
                    "model": {
                        **request.app.state.model_record,
                        "runtime_version": getattr(gateway, "runtime_version", None),
                        "details": getattr(gateway, "details", {}),
                        "resolved_route": provenance,
                    },
                    "external_processing": remote,
                    "privacy_policy": request.app.state.settings.remote_privacy
                    if remote
                    else "local",
                    "context_limit": None if remote else 4096,
                    "request_budget": budget,
                    "input_token_upper_bound": None
                    if remote
                    else input_bound(model_request),
                    "input_byte_budget_charge": input_bound(model_request)
                    if remote
                    else None,
                    "output_token_budget": output_budget,
                    "token_accounting": "development_byte_ceiling_provider_tokenizer_unvalidated"
                    if remote
                    else "utf8_bytes_plus_512_template_tokens",
                    "usage": usage,
                    "queue_seconds": queue_seconds,
                    "model_first_token_seconds": first,
                    "answer_release_seconds": time.monotonic() - started,
                    "retrieved_count": len(hits),
                    "included_count": len(selected),
                    "model_called": bool(selected),
                    "coverage": {
                        **coverage,
                        "included_chunks": len(selected),
                        "all_extracted_chunks_included": coverage[
                            "total_document_chunks"
                        ]
                        == len(selected)
                        if coverage["total_document_chunks"] is not None
                        else False,
                    },
                    "fixture": request.app.state.settings.generation_mode == "fixture",
                }
                await record(
                    database, actor, workspace, run_id, "completed", metadata=metadata
                )
                # Validate everything first. No raw model tokens or private
                # reasoning enter the application stream.
                if coverage["total_document_chunks"] is not None:
                    total = coverage["total_document_chunks"]
                    verified["scope_notice"] = (
                        f"Document context: {len(selected)} of {total} extracted passages supplied. "
                        + (
                            "All extracted text passages included; extraction gaps may remain. "
                            if len(selected) == total
                            else "Bounded overview context; some document text was not supplied. "
                        )
                        + "Not a calculation or a guaranteed exhaustive review."
                    )
                yield frame("answer", **verified, metadata=metadata)
                yield frame("done", run_id=run_id)
                completed = True
        except (
            GatewayError,
            HTTPException,
            SQLAlchemyError,
            OSError,
            TimeoutError,
        ) as exc:
            # Fixed categories only: never log exception text, prompts, provider
            # bodies or credentials. These make live adapter failures diagnosable.
            failure = (
                {
                    "Incomplete answer": "generation_incomplete",
                    "Remote generation rate limit reached": "provider_rate_limit",
                    "Answer schema validation failed": "invalid_answer_schema",
                    "Citation or quotation validation failed": "invalid_source_quote",
                    "Generation endpoint reported an error": "upstream_error",
                    "Routing metadata changed during generation": "routing_changed",
                    "Provider usage exceeds development budget": "provider_budget_exceeded",
                }.get(str(exc), "answer_not_released")
                if isinstance(exc, GatewayError)
                else "answer_not_released"
            )
            logging.getLogger(__name__).warning("Answer rejected: %s", failure)
            with suppress(Exception):
                await record(
                    database,
                    actor,
                    workspace,
                    run_id,
                    "failed",
                    error="answer_not_released",
                )
            yield frame(
                "error",
                code="answer_not_released",
                detail="The BYOK provider rate limit or quota was reached. No outside-provider fallback was used. Try again after the provider limit resets."
                if failure == "provider_rate_limit"
                else "The model returned an invalid source citation. No answer was released."
                if failure == "invalid_source_quote"
                else "The model returned an unexpected response format. No answer was released."
                if failure == "invalid_answer_schema"
                else "The answer was cancelled, unavailable, revoked or failed evidence validation. No generated answer was released.",
            )
        finally:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            if not completed:
                with suppress(Exception):
                    await record(
                        database,
                        actor,
                        workspace,
                        run_id,
                        "cancelled",
                        error="disconnected_or_cancelled",
                    )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/{workspace}/answers/{run_id}/cancel")
async def cancel_answer(
    workspace: UUID, run_id: UUID, request: Request, principal=Depends(identity)
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace) as conn:
        await database.authorize(conn, workspace)
        found = await conn.scalar(
            text(
                "UPDATE enterprise.answer_runs SET status='cancelled',updated_at=now() WHERE id=:id AND status IN ('queued','running') RETURNING id"
            ),
            {"id": run_id},
        )
        if not found:
            raise HTTPException(404, "Active answer not available")
    return {"status": "cancelled"}
