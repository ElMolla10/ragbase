"""Authorize a CSV version, then calculate outside PostgreSQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.analysis.engine import run
from backend.analysis.messages import clarification, describe
from backend.analysis.plan import (
    OperationPlan,
    bind_plan,
    interpretation_limited,
    margin_ambiguous,
)
from backend.analysis.planner import propose
from backend.analysis.profile import ClarifyError, DatasetError, build_dataset
from backend.gateway.contracts import GatewayError
from backend.search.index import INDEX_VERSION

logger = logging.getLogger(__name__)

DETAIL = {
    "not_csv": "Calculations run on one authorized CSV version that finished extraction.",
    "not_ready": "This CSV version is not ready for calculation. Finish extraction and review first.",
    "source_limit": "The CSV exceeds the calculation row or column limit. No result was calculated.",
    "invalid_plan": "The calculation plan is not an allowlisted operation. No result was calculated.",
    "unknown_column": "The plan names a column that is not in this CSV version.",
    "ambiguous_column": "The plan names a column that matches more than one field.",
    "column_not_numeric": "That column is text or a leading-zero identifier and cannot be used as a number.",
    "output_limit": "The calculation exceeded the output limit. No partial result was returned.",
    "runtime_limit": "The calculation exceeded its time limit. No result was returned.",
    "sandbox_rejected": "The calculation could not run inside the locked snapshot. No result was returned.",
    "no_numeric_values": "The aggregate column has no numeric values in the included rows. No zero was substituted.",
}


async def perform(request, actor, workspace: UUID, body) -> dict:
    database = request.app.state.database
    dataset = await _load(database, actor, workspace, body.version_id)
    question = body.question
    plan = body.plan
    model_called = False
    if interpretation_limited(question, plan):
        return await _clarification(
            database,
            actor,
            workspace,
            body,
            dataset,
            question,
            plan,
            "interpretation_limit",
            False,
        )
    if question is not None:
        gateway = request.app.state.gateway
        if gateway is None:
            raise HTTPException(
                503,
                "Generation is disabled. Submit an explicit calculation plan; retrieval remains available.",
            )
        model_called = True
        try:
            plan = await propose(gateway, question, dataset)
        except GatewayError as exc:
            logger.warning("Calculation rejected: planner_unavailable")
            raise HTTPException(
                503, "Calculation planning is unavailable. No result was calculated."
            ) from exc
        except DatasetError as exc:
            _fail(exc.code)
    if plan.action == "clarify":
        return await _clarification(
            database,
            actor,
            workspace,
            body,
            dataset,
            question,
            plan,
            "planner_clarification",
            model_called,
        )
    if plan.action != "calculate":
        return await _clarification(
            database,
            actor,
            workspace,
            body,
            dataset,
            question,
            plan,
            "not_a_calculation",
            model_called,
        )
    try:
        plan = bind_plan(plan, dataset)
    except DatasetError as exc:
        _fail(exc.code)
    if margin_ambiguous(question, plan, dataset):
        return await _clarification(
            database,
            actor,
            workspace,
            body,
            dataset,
            question,
            plan,
            "ambiguous_margin",
            model_called,
        )
    calculation_id = uuid4()
    await _start(database, actor, workspace, body, calculation_id, question, plan)
    try:
        outcome = await asyncio.to_thread(run, dataset, plan)
    except ClarifyError as exc:
        answer, record = _clarification_body(dataset, question, exc.code, exc.detail)
        await _finish(
            database,
            actor,
            workspace,
            calculation_id,
            "clarification",
            exc.code,
            None,
            plan,
            {},
            "",
            record,
        )
        return _payload(
            workspace, calculation_id, "clarification", answer, record, [], model_called
        )
    except DatasetError as exc:
        await _fail_running(database, actor, workspace, calculation_id, exc.code)
        _fail(exc.code)
    except Exception as exc:
        logger.warning("Calculation rejected: %s", type(exc).__name__)
        await _fail_running(
            database, actor, workspace, calculation_id, "sandbox_rejected"
        )
        _fail("sandbox_rejected")
    record = outcome["record"]
    answer = describe(record, question)
    record["answer"] = answer
    record["calculation_id"] = str(calculation_id)
    saved = await _finish(
        database,
        actor,
        workspace,
        calculation_id,
        "completed",
        None,
        outcome["dataset_sha256"],
        plan,
        outcome["record"]["result"],
        outcome["export_csv"],
        record,
    )
    if not saved:
        raise HTTPException(404, "Workspace or document not available")
    return _payload(
        workspace,
        calculation_id,
        "completed",
        answer,
        record,
        _with_preview(workspace, outcome["citations"]),
        model_called,
    )


async def read_calculation(
    request, actor, workspace: UUID, calculation_id: UUID
) -> dict:
    row = await _owned(request, actor, workspace, calculation_id)
    return {
        "calculation_id": str(calculation_id),
        "status": row["status"],
        "record": row["record"],
        "production_qualified": False,
    }


async def read_table(request, actor, workspace: UUID, calculation_id: UUID) -> str:
    row = await _owned(request, actor, workspace, calculation_id)
    if row["status"] != "completed" or not row["export_csv"]:
        raise HTTPException(409, "This calculation has no result table.")
    return row["export_csv"]


async def read_record(request, actor, workspace: UUID, calculation_id: UUID) -> dict:
    row = await _owned(request, actor, workspace, calculation_id)
    return row["record"]


def _fail(code: str) -> None:
    logger.warning("Calculation rejected: %s", code)
    raise HTTPException(422, DETAIL.get(code, DETAIL["invalid_plan"]))


async def _load(database, actor, workspace, version_id):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        row = (
            (
                await conn.execute(
                    text("""
                SELECT v.id AS version_id, v.document_id, v.sha256, d.filename,
                       j.status, j.warnings, j.extractor
                FROM enterprise.document_versions v
                JOIN enterprise.documents d ON d.id=v.document_id
                JOIN enterprise.ingestion_jobs j ON j.version_id=v.id
                WHERE v.id=:version AND v.workspace_id=:workspace
                """),
                    {"version": version_id, "workspace": workspace},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise HTTPException(404, "Workspace or document not available")
        if not str(row["filename"]).lower().endswith(".csv"):
            raise HTTPException(409, DETAIL["not_csv"])
        if row["status"] != "extracted":
            raise HTTPException(409, DETAIL["not_ready"])
        pages = (
            (
                await conn.execute(
                    text("""
                SELECT p.ordinal, p.content, p.location, p.warnings, c.id AS chunk_id
                FROM enterprise.extracted_pages p
                LEFT JOIN enterprise.document_chunks c
                  ON c.version_id=p.version_id AND c.ordinal=p.ordinal AND c.part=0
                 AND c.workspace_id=p.workspace_id AND c.index_version=:index
                WHERE p.version_id=:version AND p.workspace_id=:workspace
                ORDER BY p.ordinal
                """),
                    {
                        "version": version_id,
                        "workspace": workspace,
                        "index": INDEX_VERSION,
                    },
                )
            )
            .mappings()
            .all()
        )
    try:
        return build_dataset(
            version_id=row["version_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            extractor=row["extractor"],
            warnings=_jsonish(row["warnings"]),
            pages=[_page(page) for page in pages],
        )
    except DatasetError as exc:
        status = 409 if exc.code in {"not_csv", "not_ready"} else 422
        raise HTTPException(status, DETAIL.get(exc.code, DETAIL["not_csv"])) from exc


def _page(page) -> dict:
    return {
        "ordinal": page["ordinal"],
        "content": page["content"],
        "location": _jsonish(page["location"]),
        "warnings": _jsonish(page["warnings"]),
        "chunk_id": page["chunk_id"],
    }


def _jsonish(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


async def _clarification(
    database, actor, workspace, body, dataset, question, plan, code, model_called
):
    answer, record = _clarification_body(dataset, question, code, {})
    calculation_id = uuid4()
    await _insert(
        database,
        actor,
        workspace,
        body,
        calculation_id,
        question,
        plan,
        "clarification",
        code,
        record,
    )
    return _payload(
        workspace, calculation_id, "clarification", answer, record, [], model_called
    )


def _clarification_body(dataset, question, code, detail) -> tuple[str, dict]:
    columns = [column.name for column in dataset.columns]
    answer = clarification(code, question, columns)
    record = {
        "production_qualified": False,
        "status": "clarification",
        "code": code,
        "answer": answer,
        "dataset": {
            "filename": dataset.filename,
            "version_id": dataset.version_id,
            "document_id": dataset.document_id,
            "sha256": dataset.sha256,
            "columns": [
                {"name": column.name, "type": column.type} for column in dataset.columns
            ],
        },
        "filters": [],
        "operation": None,
        "units": "unspecified",
        "included_row_count": None,
        "excluded_row_count": None,
        "result": None,
        "currencies": list(detail.get("values") or []),
        "limitation": "No value was calculated. Not production qualification.",
    }
    return answer, record


async def _start(database, actor, workspace, body, calculation_id, question, plan):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        await _reserve(conn, body.client_request_id)
        if not await conn.scalar(text("SELECT enterprise.admit_calculation()")):
            raise HTTPException(
                429, "A calculation or another heavy job is already running."
            )
        await conn.execute(
            text("""
            INSERT INTO enterprise.calculations(
                id,workspace_id,created_by,client_request_id,version_id,question_sha256,
                status,lease_until,plan
            ) VALUES (
                :id,:workspace,:actor,:client,:version,:hash,'running',
                clock_timestamp()+interval '30 seconds',CAST(:plan AS jsonb)
            )
            """),
            _identity(workspace, actor, body, calculation_id, question, plan),
        )


async def _insert(
    database,
    actor,
    workspace,
    body,
    calculation_id,
    question,
    plan,
    status,
    code,
    record,
):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        await _reserve(conn, body.client_request_id)
        try:
            await conn.execute(
                text("""
                INSERT INTO enterprise.calculations(
                    id,workspace_id,created_by,client_request_id,version_id,question_sha256,
                    status,error_code,plan,record
                ) VALUES (
                    :id,:workspace,:actor,:client,:version,:hash,:status,:error,
                    CAST(:plan AS jsonb),CAST(:record AS jsonb)
                )
                """),
                _identity(workspace, actor, body, calculation_id, question, plan)
                | {
                    "status": status,
                    "error": code,
                    "record": json.dumps(record, ensure_ascii=False),
                },
            )
        except IntegrityError as exc:
            raise HTTPException(409, "This request has already been submitted") from exc


async def _reserve(conn, client_request_id):
    existing = await conn.scalar(
        text("SELECT id FROM enterprise.calculations WHERE client_request_id=:client"),
        {"client": client_request_id},
    )
    if existing:
        raise HTTPException(409, "This request has already been submitted")


def _identity(workspace, actor, body, calculation_id, question, plan) -> dict:
    return {
        "id": calculation_id,
        "workspace": workspace,
        "actor": actor,
        "client": body.client_request_id,
        "version": body.version_id,
        "hash": None
        if question is None
        else hashlib.sha256(question.encode()).hexdigest(),
        "plan": json.dumps(
            plan.model_dump() if isinstance(plan, OperationPlan) else {},
            ensure_ascii=False,
        ),
    }


async def _finish(
    database,
    actor,
    workspace,
    calculation_id,
    status,
    code,
    digest,
    plan,
    result,
    export_csv,
    record,
) -> bool:
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        saved = await conn.scalar(
            text("""
            UPDATE enterprise.calculations
            SET status=:status,error_code=:error,dataset_sha256=:digest,
                plan=CAST(:plan AS jsonb),result=CAST(:result AS jsonb),
                export_csv=:export,record=CAST(:record AS jsonb),
                lease_until=NULL,updated_at=now()
            WHERE id=:id AND status='running' AND lease_until>clock_timestamp()
            RETURNING id
            """),
            {
                "id": calculation_id,
                "status": status,
                "error": code,
                "digest": digest,
                "plan": json.dumps(plan.model_dump(), ensure_ascii=False),
                "result": json.dumps(result, ensure_ascii=False),
                "export": export_csv,
                "record": json.dumps(record, ensure_ascii=False),
            },
        )
        return bool(saved)


async def _fail_running(database, actor, workspace, calculation_id, code):
    with_context = True
    try:
        async with database.scope(actor, workspace) as conn:
            await database.authorize(conn, workspace)
            await conn.execute(
                text("""
                UPDATE enterprise.calculations
                SET status='failed',error_code=:error,lease_until=NULL,updated_at=now()
                WHERE id=:id AND status='running'
                """),
                {"id": calculation_id, "error": code},
            )
    except Exception:
        with_context = False
        logger.warning("Calculation rejected: failure_not_recorded")
    return with_context


async def _owned(request, actor, workspace, calculation_id):
    database = request.app.state.database
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace)
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT status,export_csv,record FROM enterprise.calculations WHERE id=:id"
                    ),
                    {"id": calculation_id},
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(404, "Calculation not available")
    return row


def _with_preview(workspace, citations: list[dict]) -> list[dict]:
    prepared = []
    for item in citations:
        citation = dict(item)
        if citation.get("chunk_id"):
            citation["preview_url"] = (
                f"/api/workspaces/{workspace}/sources/{citation['chunk_id']}"
            )
            citation["source_id"] = citation["chunk_id"]
        else:
            citation["source_id"] = f"row-{citation['row']}"
            citation["preview_url"] = None
        prepared.append(citation)
    return prepared


def _payload(
    workspace, calculation_id, status, answer, record, citations, model_called
):
    base = f"/api/workspaces/{workspace}/calculations/{calculation_id}"
    downloads = {"record": base + "/record.json"}
    if status == "completed":
        downloads["table"] = base + "/table.csv"
    return {
        "calculation_id": str(calculation_id),
        "status": status,
        "answer": answer,
        "record": record,
        "citations": citations,
        "downloads": downloads,
        "model_called": model_called,
        "production_qualified": False,
    }
