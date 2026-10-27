"""Durable exhaustive coverage. Interactive answers and extraction run between batches."""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.answers.grounding import REMOTE_BUDGET
from backend.coverage.planner import (
    MAX_MAP_CALLS,
    MAX_REDUCTION_CALLS,
    assign_aliases,
    build_sections,
    expand_sections,
    fail_section,
    gap_counts,
    public_section,
    scope_notice,
    split_cited,
    verdict,
)
from backend.coverage.prompt import (
    fits_request,
    make_coverage_request,
    parse_reduction,
    parse_section,
    plan_map,
    plan_reduction,
    release_answer,
)
from backend.gateway.contracts import GatewayError
from backend.policy import assemble
from backend.search.index import INDEX_VERSION
from backend.search.retrieve import check_versions
from backend.storage.local import StoredArtifact

logger = logging.getLogger(__name__)
ACTIVE = ("queued", "running", "yielding")


class StopCoverageError(Exception):
    def __init__(self, status, error):
        self.status = status
        self.error = error


def frame(kind, **value):
    return json.dumps({"event": kind, **value}, ensure_ascii=False, default=str) + "\n"


def budgets(settings):
    remote = settings.generation_mode == "openrouter"
    return (2048 if remote else 256), (REMOTE_BUDGET if remote else 4096)


def _json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def pages_from_rows(rows):
    grouped, order = {}, []
    for row in rows:
        key = (row["version_id"], row["ordinal"])
        if key not in grouped:
            grouped[key] = {
                "version_id": row["version_id"],
                "ordinal": row["ordinal"],
                "content": row["page_content"],
                "location": dict(_json(row["location"], {})),
                "warnings": list(_json(row["page_warnings"], [])),
                "job_warnings": list(_json(row["job_warnings"], [])),
                "filename": row["filename"],
                "document_id": row["document_id"],
                "sha256": row["sha256"],
                "chunks": [],
            }
            order.append(key)
        if row["chunk_id"] is not None:
            grouped[key]["chunks"].append(
                {
                    "id": row["chunk_id"],
                    "part": row["part"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "content": row["chunk_content"],
                    "index_version": row["index_version"],
                }
            )
    return [grouped[key] for key in order]


async def load_scope(request, actor, workspace, versions):
    database = request.app.state.database
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        await check_versions(conn, workspace, versions)
        originals = (
            (
                await conn.execute(
                    text(
                        "SELECT storage_key,sha256,byte_size FROM enterprise.document_versions WHERE workspace_id=:w AND id=ANY(CAST(:versions AS uuid[]))"
                    ),
                    {"w": workspace, "versions": [str(value) for value in versions]},
                )
            )
            .mappings()
            .all()
        )
        rows = (
            (
                await conn.execute(
                    text("""
                SELECT p.version_id, p.ordinal, p.content AS page_content, p.location,
                       p.warnings AS page_warnings, j.warnings AS job_warnings,
                       c.id AS chunk_id, c.part, c.char_start, c.char_end,
                       c.content AS chunk_content, c.index_version,
                       v.sha256, d.id AS document_id, d.filename
                FROM enterprise.extracted_pages p
                JOIN enterprise.document_versions v ON v.id=p.version_id
                JOIN enterprise.documents d ON d.id=v.document_id
                JOIN enterprise.ingestion_jobs j ON j.version_id=p.version_id
                LEFT JOIN enterprise.document_chunks c
                  ON c.version_id=p.version_id AND c.ordinal=p.ordinal
                 AND c.index_version=:index
                WHERE p.workspace_id=:w AND p.version_id=ANY(CAST(:versions AS uuid[]))
                  AND j.status='extracted'
                ORDER BY array_position(CAST(:versions AS uuid[]), p.version_id),
                         p.ordinal, c.part
                """),
                    {
                        "w": workspace,
                        "versions": [str(value) for value in versions],
                        "index": INDEX_VERSION,
                    },
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
                original["storage_key"], original["sha256"], original["byte_size"]
            ),
        )
    return pages_from_rows(rows)


async def _scalar(database, actor, workspace, statement, params):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        return await conn.scalar(text(statement), params)


async def ensure_active(database, actor, workspace, run_id, request):
    if await request.is_disconnected():
        raise StopCoverageError("cancelled", "disconnected_or_cancelled")
    try:
        row = await _mapping(
            database,
            actor,
            workspace,
            "SELECT status, lease_until>clock_timestamp() AS current FROM enterprise.coverage_runs WHERE id=:id",
            {"id": run_id},
        )
    except HTTPException as exc:
        raise StopCoverageError("failed", "authorization_revoked") from exc
    if not row or row["status"] not in ACTIVE or not row["current"]:
        if row and row["status"] == "cancelled":
            raise StopCoverageError("cancelled", "cancelled")
        raise StopCoverageError("failed", "interrupted_or_revoked")


async def _mapping(database, actor, workspace, statement, params):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        return (await conn.execute(text(statement), params)).mappings().first()


async def touch(database, actor, workspace, run_id, status):
    allowed = (
        ("queued", "yielding")
        if status == "running"
        else ("queued", "running", "yielding")
    )
    return await _scalar(
        database,
        actor,
        workspace,
        """
        UPDATE enterprise.coverage_runs
        SET status=:status, lease_until=clock_timestamp()+interval '180 seconds', updated_at=now()
        WHERE id=:id AND status=ANY(CAST(:allowed AS text[]))
        RETURNING id
        """,
        {"id": run_id, "status": status, "allowed": list(allowed)},
    )


async def flag(database, actor, workspace, statement):
    async with database.scope(actor, workspace) as conn:
        return await conn.scalar(text(statement))


@asynccontextmanager
async def coverage_slot(database, actor, workspace, run_id, request):
    deadline, connection = time.monotonic() + 90, None
    try:
        while time.monotonic() < deadline:
            await ensure_active(database, actor, workspace, run_id, request)
            if await flag(
                database, actor, workspace, "SELECT enterprise.coverage_must_yield()"
            ):
                await touch(database, actor, workspace, run_id, "yielding")
                await asyncio.sleep(0.2)
                continue
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
        await ensure_active(database, actor, workspace, run_id, request)
        if not await touch(database, actor, workspace, run_id, "running"):
            raise StopCoverageError("cancelled", "cancelled")
        yield
    finally:
        if connection is not None:
            with suppress(Exception):
                await touch(database, actor, workspace, run_id, "yielding")
            try:
                await connection.execute(text("SELECT pg_advisory_unlock(728003)"))
                await connection.commit()
            except BaseException:
                await connection.invalidate()
                raise
            finally:
                await connection.close()


async def read_model(gateway, request):
    parts, usage, provenance, finished = [], None, None, False
    async for event in gateway.stream(request):
        if event.kind == "text":
            parts.append(event.text)
            if sum(len(part.encode()) for part in parts) > 16384:
                raise GatewayError("Answer exceeds size limit")
        elif event.kind == "done":
            if event.finish_reason != "stop":
                raise GatewayError("Incomplete answer")
            usage, provenance, finished = event.usage, event.provenance, True
    if not finished:
        raise GatewayError("Incomplete answer")
    return "".join(parts), usage, provenance


async def call_model(
    gateway, model_request, database, actor, workspace, run_id, request
):
    task = asyncio.create_task(read_model(gateway, model_request))
    try:
        while not task.done():
            await ensure_active(database, actor, workspace, run_id, request)
            await asyncio.wait({task}, timeout=0.5)
        return await task
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


async def update_sections(database, actor, workspace, run_id, sections, status, code):
    if not sections:
        return
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        for section in sections:
            if status == "gap" and section["status"] == "covered":
                continue
            if status == "covered":
                section["status"] = "covered"
                section["gap_code"] = None
            elif section["status"] == "pending":
                fail_section(section, code)
            await conn.execute(
                text("""
                UPDATE enterprise.coverage_sections
                SET status=:status, gap_code=:code
                WHERE run_id=:run AND version_id=:version AND ordinal=:ordinal AND part=:part
                  AND (:status='covered' OR status='pending')
                """),
                {
                    "status": section["status"],
                    "code": section["gap_code"],
                    "run": run_id,
                    "version": section["version_id"],
                    "ordinal": section["ordinal"],
                    "part": section["part"],
                },
            )


async def insert_run(request, actor, workspace, data, sections):
    database = request.app.state.database
    run_id = uuid4()
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        if not await conn.scalar(text("SELECT enterprise.admit_coverage()")):
            raise HTTPException(
                429,
                "Coverage queue is full or this user already has an active coverage job",
            )
        if await conn.scalar(
            text(
                "SELECT id FROM enterprise.coverage_runs WHERE client_request_id=:client"
            ),
            {"client": data.client_request_id},
        ):
            raise HTTPException(409, "This request has already been submitted")
        await conn.execute(
            text("""
            INSERT INTO enterprise.coverage_runs(
                id,workspace_id,created_by,client_request_id,question_sha256,version_ids,
                status,coverage_complete,lease_until)
            VALUES(:id,:workspace,:actor,:client,:hash,CAST(:versions AS uuid[]),
                   'queued',false,clock_timestamp()+interval '180 seconds')
            """),
            {
                "id": run_id,
                "workspace": workspace,
                "actor": actor,
                "client": data.client_request_id,
                "hash": hashlib.sha256(data.question.encode()).hexdigest(),
                "versions": [str(value) for value in data.version_ids],
            },
        )
        for section in sections:
            await conn.execute(
                text("""
                INSERT INTO enterprise.coverage_sections(
                    run_id,workspace_id,version_id,ordinal,part,chunk_id,status,gap_code,location)
                VALUES(:run,:workspace,:version,:ordinal,:part,:chunk,:status,:code,CAST(:location AS jsonb))
                """),
                {
                    "run": run_id,
                    "workspace": workspace,
                    "version": section["version_id"],
                    "ordinal": section["ordinal"],
                    "part": section["part"],
                    "chunk": section["chunk_id"],
                    "status": section["status"],
                    "code": section["gap_code"],
                    "location": json.dumps(section["location"], ensure_ascii=False),
                },
            )
    return run_id


def _report(sections, *, complete, notes, versions, extra_gaps):
    return {
        "coverage_complete": complete,
        "label": "complete" if complete else "partial",
        "sections": [public_section(section) for section in sections],
        "gaps": [
            public_section(section)
            for section in sections
            if section["status"] != "covered"
        ]
        + [
            {
                "gap_code": code,
                "status": "gap",
                "location": None,
                "ordinal": None,
                "part": None,
            }
            for code in extra_gaps
        ],
        "scope_notice": scope_notice(
            sections,
            complete=complete,
            notes=notes,
            versions=versions,
            extra_gaps=extra_gaps,
        ),
    }


async def prepare(request, actor, workspace, data):
    if request.app.state.gateway is None:
        raise HTTPException(
            503,
            "Generation is disabled; retrieval and source previews remain available",
        )
    instructions = assemble("coverage")
    pages = await load_scope(request, actor, workspace, data.version_ids)
    sections, notes = build_sections(pages, workspace)
    output_tokens, budget = budgets(request.app.state.settings)

    def fits_text(content):
        return fits_request(
            instructions,
            data.question,
            [{"alias": "S1", "text": content}],
            kind="section",
            output_tokens=output_tokens,
            budget=budget,
        )

    sections = expand_sections(sections, fits_text)
    batches = plan_map(
        sections,
        instructions=instructions,
        question=data.question,
        output_tokens=output_tokens,
        budget=budget,
        max_calls=MAX_MAP_CALLS,
    )
    ordered = assign_aliases(batches)
    run_id = await insert_run(request, actor, workspace, data, sections)
    return {
        "instructions": instructions,
        "sections": sections,
        "notes": notes,
        "batches": batches,
        "ordered": ordered,
        "output_tokens": output_tokens,
        "budget": budget,
        "run_id": run_id,
        "question": data.question,
        "versions": data.version_ids,
    }


async def _persist_stop(
    database, actor, workspace, run_id, sections, stop, notes, versions
):
    code = "cancelled" if stop.error == "disconnected_or_cancelled" else stop.error
    for section in sections:
        if section["status"] == "pending":
            fail_section(section, code)
    metadata = {
        "coverage_complete": False,
        "label": "partial",
        "gap_counts": gap_counts(sections),
        "processing_notes": notes,
        "section_count": len(sections),
    }
    with suppress(HTTPException, SQLAlchemyError, OSError, TimeoutError):
        async with database.scope(actor, workspace) as conn:
            await database.authorize(conn, workspace)
            await conn.execute(
                text("""
                UPDATE enterprise.coverage_runs
                SET status=:status, error_code=:error, coverage_complete=false,
                    metadata=CAST(:meta AS jsonb), updated_at=now()
                WHERE id=:id AND status=ANY(CAST(:allowed AS text[]))
                """),
                {
                    "id": run_id,
                    "status": stop.status,
                    "error": stop.error,
                    "meta": json.dumps(metadata),
                    "allowed": list(ACTIVE),
                },
            )
            await conn.execute(
                text("""
                UPDATE enterprise.coverage_sections
                SET status='gap', gap_code=:code
                WHERE run_id=:id AND status='pending'
                """),
                {"id": run_id, "code": code},
            )
    if stop.error == "authorization_revoked":
        with suppress(Exception):
            async with database.scope(actor, workspace) as conn:
                await conn.execute(text("SELECT enterprise.reap_coverage()"))
    return _report(
        sections, complete=False, notes=notes, versions=versions, extra_gaps=()
    )


async def execute(context, request, actor, workspace):
    database = request.app.state.database
    gateway = request.app.state.gateway
    settings = request.app.state.settings
    sections = context["sections"]
    notes = context["notes"]
    run_id = context["run_id"]
    output_tokens = context["output_tokens"]
    budget = context["budget"]
    instructions = context["instructions"]
    question = context["question"]
    versions = context["versions"]
    remote = settings.generation_mode == "openrouter"
    summaries, usage, provenance = [], None, None
    calls = 0
    terminal = False
    started = time.monotonic()
    yield frame(
        "queued",
        run_id=run_id,
        external_processing=remote,
        privacy_policy=settings.remote_privacy if remote else "local",
    )
    try:
        yield frame("stage", stage="covering_sections")
        for batch in context["batches"]:
            items = [
                {"alias": f"S{index}", "text": section["content"]}
                for index, section in enumerate(batch, 1)
            ]
            model_request = make_coverage_request(
                instructions,
                question,
                items,
                kind="section",
                max_output_tokens=output_tokens,
            )
            try:
                async with coverage_slot(database, actor, workspace, run_id, request):
                    raw, usage, provenance = await call_model(
                        gateway,
                        model_request,
                        database,
                        actor,
                        workspace,
                        run_id,
                        request,
                    )
                text_value, aliases = parse_section(raw, batch)
            except GatewayError as exc:
                if str(exc) == "Generation queue deadline exceeded":
                    raise StopCoverageError(
                        "failed", "generation_queue_deadline"
                    ) from exc
                logger.warning("Coverage section rejected: section_processing_failed")
                await update_sections(
                    database,
                    actor,
                    workspace,
                    run_id,
                    batch,
                    "gap",
                    "section_processing_failed",
                )
                continue
            calls += 1
            cited, missed = split_cited(batch, aliases)
            if cited:
                await update_sections(
                    database, actor, workspace, run_id, cited, "covered", None
                )
                summaries.append({"text": text_value, "aliases": aliases})
            if missed:
                await update_sections(
                    database,
                    actor,
                    workspace,
                    run_id,
                    missed,
                    "gap",
                    "section_processing_failed",
                )
            yield frame(
                "section",
                covered=sum(section["status"] == "covered" for section in sections),
                total=len(sections),
            )
        extra = []
        root = None
        reduction_hash = None
        if summaries:
            yield frame("stage", stage="reducing_sections")
            root, reduction_calls, reduction_hash, usage, provenance = await _reduce(
                summaries,
                instructions,
                question,
                output_tokens,
                budget,
                gateway,
                database,
                actor,
                workspace,
                run_id,
                request,
            )
            calls += reduction_calls
            if root is None:
                extra.append("reduction_incomplete")
        section_complete = bool(sections) and all(
            section["status"] == "covered" for section in sections
        )
        reduction_complete = (not summaries and section_complete) or root is not None
        complete = verdict(
            sections, reduction_complete=reduction_complete and not extra
        )
        ordered = [section for section in sections if section.get("global_alias")]
        prose = None
        if root is not None:
            prose = root["text"]
        elif summaries and extra:
            prose = " ".join(summary["text"] for summary in summaries)
        elif section_complete and not summaries:
            prose = "The processed sections did not establish a summary."
        verified = None
        if prose is not None and ordered:
            try:
                verified = release_answer(prose, ordered)
            except GatewayError:
                logger.warning("Coverage release rejected: invalid_source_quote")
                extra.append("release_validation_failed")
                complete = False
                verified = None
        elif prose is not None and not ordered:
            verified = release_answer(prose, [])
        report = _report(
            sections,
            complete=complete,
            notes=notes,
            versions=versions,
            extra_gaps=tuple(extra),
        )
        if verified is None:
            verified = {
                "mode": "exhaustive_coverage",
                "insufficient": True,
                "answer": "No complete summary was released.",
                "claims": [],
                "limitation": "A missing or failed section is not proof the fact is absent.",
            }
            if complete:
                complete = False
                report = _report(
                    sections,
                    complete=False,
                    notes=notes,
                    versions=versions,
                    extra_gaps=(*extra, "release_validation_failed"),
                )
        verified.update(mode="exhaustive_coverage", **report)
        verified["limitation"] = (
            "Citations identify authorized source passages. Generated wording is a model "
            "interpretation, not a mechanical proof. Coverage completeness counts processed "
            "sections only."
        )
        sample = make_coverage_request(
            instructions,
            question,
            [{"alias": "S1", "text": "section"}],
            kind="section",
            max_output_tokens=output_tokens,
        )
        metadata = _metadata(
            request,
            instructions,
            model_request_hash=sample.instructions.sha256,
            reduction_hash=reduction_hash,
            report=report,
            notes=notes,
            calls=calls,
            usage=usage,
            provenance=provenance,
            budget=budget,
            output_tokens=output_tokens,
            started=started,
            remote=remote,
            versions=versions,
            extra=extra,
        )
        await _finish(database, actor, workspace, run_id, complete, metadata)
        terminal = True
        yield frame("answer", **verified, metadata=metadata)
        yield frame("done", run_id=run_id)
    except StopCoverageError as stop:
        report = await _persist_stop(
            database, actor, workspace, run_id, sections, stop, notes, versions
        )
        terminal = True
        yield frame("coverage", run_id=run_id, **report)
        yield frame("error", code=stop.error, detail=_detail(stop.error))
    except (GatewayError, HTTPException, SQLAlchemyError, OSError, TimeoutError):
        logger.warning("Coverage rejected: answer_not_released")
        stop = StopCoverageError("failed", "answer_not_released")
        report = await _persist_stop(
            database, actor, workspace, run_id, sections, stop, notes, versions
        )
        terminal = True
        yield frame("coverage", run_id=run_id, **report)
        yield frame("error", code=stop.error, detail=_detail(stop.error))
    finally:
        if not terminal:
            with suppress(Exception):
                await _persist_stop(
                    database,
                    actor,
                    workspace,
                    run_id,
                    sections,
                    StopCoverageError("cancelled", "disconnected_or_cancelled"),
                    notes,
                    versions,
                )


async def _reduce(
    summaries,
    instructions,
    question,
    output_tokens,
    budget,
    gateway,
    database,
    actor,
    workspace,
    run_id,
    request,
):
    level = summaries
    calls = 0
    usage = provenance = None
    reduction_hash = None
    while True:
        remaining = MAX_REDUCTION_CALLS - calls
        groups, overflow = plan_reduction(
            level,
            instructions=instructions,
            question=question,
            output_tokens=output_tokens,
            budget=budget,
            max_calls=remaining,
        )
        if (
            overflow
            or not groups
            or (len(level) > 1 and all(len(group) == 1 for group in groups))
        ):
            return None, calls, reduction_hash, usage, provenance
        reduced = []
        for group in groups:
            items = [
                {"alias": alias, "text": summary["text"]}
                for summary in group
                for alias in summary["aliases"]
            ]
            model_request = make_coverage_request(
                instructions,
                question,
                items,
                kind="reduction",
                max_output_tokens=output_tokens,
            )
            reduction_hash = model_request.instructions.sha256
            try:
                async with coverage_slot(database, actor, workspace, run_id, request):
                    raw, usage, provenance = await call_model(
                        gateway,
                        model_request,
                        database,
                        actor,
                        workspace,
                        run_id,
                        request,
                    )
                allowed = {alias for summary in group for alias in summary["aliases"]}
                text_value, aliases = parse_reduction(raw, allowed)
            except GatewayError as exc:
                if str(exc) == "Generation queue deadline exceeded":
                    raise StopCoverageError(
                        "failed", "generation_queue_deadline"
                    ) from exc
                logger.warning("Coverage reduction rejected: reduction_incomplete")
                return None, calls, reduction_hash, usage, provenance
            calls += 1
            reduced.append({"text": text_value, "aliases": aliases})
        level = reduced
        if len(level) == 1:
            return level[0], calls, reduction_hash, usage, provenance


def _metadata(
    request,
    instructions,
    *,
    model_request_hash,
    reduction_hash,
    report,
    notes,
    calls,
    usage,
    provenance,
    budget,
    output_tokens,
    started,
    remote,
    versions,
    extra,
):
    gateway = request.app.state.gateway
    return {
        "constitution_version": instructions.constitution_version,
        "instruction_sha256": model_request_hash or instructions.sha256,
        "reduction_instruction_sha256": reduction_hash,
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
        "output_token_budget": output_tokens,
        "token_accounting": "development_byte_ceiling_provider_tokenizer_unvalidated"
        if remote
        else "utf8_bytes_plus_512_template_tokens",
        "usage": usage,
        "model_calls": calls,
        "answer_release_seconds": time.monotonic() - started,
        "coverage_complete": report["coverage_complete"],
        "label": report["label"],
        "section_count": len(report["sections"]),
        "covered_count": sum(
            section["status"] == "covered" for section in report["sections"]
        ),
        "gap_counts": gap_counts(
            [
                {"status": section["status"], "gap_code": section["gap_code"]}
                for section in report["sections"]
            ],
            extra,
        ),
        "extra_gaps": list(extra),
        "processing_notes": notes,
        "selected_versions": [str(value) for value in versions],
        "fixture": request.app.state.settings.generation_mode == "fixture",
        "strategy": "explicit_section_coverage",
    }


async def _finish(database, actor, workspace, run_id, complete, metadata):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        await conn.execute(
            text("""
            UPDATE enterprise.coverage_runs
            SET status='finished', coverage_complete=:complete, error_code=NULL,
                metadata=CAST(:meta AS jsonb), updated_at=now()
            WHERE id=:id AND status=ANY(CAST(:allowed AS text[]))
            """),
            {
                "id": run_id,
                "complete": complete,
                "meta": json.dumps(metadata),
                "allowed": list(ACTIVE),
            },
        )


def _detail(code):
    return {
        "cancelled": "Coverage was cancelled. Processed sections were recorded; the result is not complete.",
        "disconnected_or_cancelled": "Coverage was cancelled. Processed sections were recorded; the result is not complete.",
        "authorization_revoked": "Access changed before coverage finished. No summary was released.",
        "generation_queue_deadline": "Coverage waited for interactive work and did not finish. The result is not complete.",
        "interrupted_or_revoked": "Coverage was interrupted or the lease expired. The result is not complete.",
        "answer_not_released": "Coverage failed. No complete summary was released.",
    }.get(code, "Coverage failed. No complete summary was released.")
