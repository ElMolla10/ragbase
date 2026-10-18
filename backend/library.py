"""Authenticated originals, durable native extraction and page evidence. No chat."""

import asyncio
import re
from urllib.parse import quote, unquote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from backend.auth.development import Principal, authenticate
from backend.storage.local import StoredArtifact

router = APIRouter(prefix="/api")


def identity(request: Request) -> Principal:
    return authenticate(
        request.headers.get("authorization"), request.app.state.settings.dev_tokens
    )


def filename_from_header(value: str | None) -> str:
    try:
        value = unquote(value, errors="strict") if value else value
    except UnicodeError as exc:
        raise HTTPException(400, "Invalid UTF-8 filename") from exc
    if (
        not value
        or len(value) > 240
        or value != value.strip()
        or re.search(r"[\x00-\x1f\x7f/\\]", value)
    ):
        raise HTTPException(
            400, "A plain filename of at most 240 characters is required"
        )
    if not value.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(
            415,
            "Supported originals are UTF-8 TXT, Markdown and native PDFs",
        )
    return value


async def bounded_body(request: Request, limit: int) -> bytes:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) < 0:
                raise ValueError
            if int(length) > limit:
                raise HTTPException(413, "Upload exceeds byte limit")
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length") from exc
    data = bytearray()
    async with asyncio.timeout(15):
        async for chunk in request.stream():
            if len(data) + len(chunk) > limit:
                raise HTTPException(413, "Upload exceeds byte limit")
            data.extend(chunk)
    if not data:
        raise HTTPException(400, "Empty originals are not accepted")
    return bytes(data)


@router.get("/workspaces")
async def workspaces(request: Request, principal: Principal = Depends(identity)):
    async with request.app.state.database.scope(principal.user_id) as conn:
        actor = await conn.scalar(
            text("SELECT id FROM enterprise.users WHERE id=enterprise.current_actor()")
        )
        if not actor:
            raise HTTPException(401, "Identity is not active")
        result = await conn.execute(
            text("SELECT id,name FROM enterprise.workspaces ORDER BY id LIMIT 100")
        )
        rows = [dict(row) for row in result.mappings()]
    return {"workspaces": rows, "limit": 100}


@router.get("/workspaces/{workspace_id}/documents")
async def documents(
    workspace_id: UUID, request: Request, principal: Principal = Depends(identity)
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id)
        result = await conn.execute(
            text("""
            SELECT d.id,d.filename,v.id AS version_id,v.version_number,v.byte_size,v.sha256,v.status,
                   coalesce(j.status,'not_queued') AS ingestion_status
            FROM enterprise.documents d JOIN enterprise.document_versions v
            ON v.document_id=d.id AND v.workspace_id=d.workspace_id
            LEFT JOIN enterprise.ingestion_jobs j ON j.version_id=v.id
            WHERE d.workspace_id=:workspace ORDER BY d.created_at DESC,d.id LIMIT 100
        """),
            {"workspace": workspace_id},
        )
        rows = [dict(row) for row in result.mappings()]
    return {"documents": rows, "limit": 100, "extraction": "native_text_only"}


@router.post("/workspaces/{workspace_id}/documents", status_code=201)
async def store_original(
    workspace_id: UUID, request: Request, principal: Principal = Depends(identity)
):
    filename = filename_from_header(request.headers.get("x-filename"))
    database = request.app.state.database
    store = request.app.state.storage
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id, write=True)
        if not await conn.scalar(text("SELECT enterprise.admit_ingestion()")):
            raise HTTPException(429, "Processing queue is full; wait for existing jobs")
        body = await bounded_body(request, request.app.state.settings.max_upload_bytes)
        if filename.lower().endswith((".txt", ".md")):
            try:
                body.decode("utf-8")
            except UnicodeError as exc:
                raise HTTPException(415, "Text originals must use UTF-8") from exc
        elif not body.startswith(b"%PDF-"):
            raise HTTPException(
                415,
                "A PDF header is required; full validation runs in the bounded worker",
            )
        document_id, version_id = uuid4(), uuid4()
        artifact = await asyncio.to_thread(store.put, workspace_id, version_id, body)
        await conn.execute(
            text("""
            INSERT INTO enterprise.documents(id,workspace_id,filename,created_by)
            VALUES(:id,:workspace,:filename,:actor)
        """),
            {
                "id": document_id,
                "workspace": workspace_id,
                "filename": filename,
                "actor": principal.user_id,
            },
        )
        await conn.execute(
            text("""
            INSERT INTO enterprise.document_versions
            (id,workspace_id,document_id,version_number,storage_key,sha256,byte_size,status)
            VALUES(:id,:workspace,:document,1,:key,:sha,:size,'stored')
        """),
            {
                "id": version_id,
                "workspace": workspace_id,
                "document": document_id,
                "key": artifact.key,
                "sha": artifact.sha256,
                "size": artifact.size,
            },
        )
        job_id = uuid4()
        await conn.execute(
            text("""
            INSERT INTO enterprise.ingestion_jobs(id,workspace_id,version_id,created_by)
            VALUES(:id,:workspace,:version,:actor)
        """),
            {
                "id": job_id,
                "workspace": workspace_id,
                "version": version_id,
                "actor": principal.user_id,
            },
        )
    # Commit precedes success. Reconciliation reports unreferenced files left by
    # failed commits; it never deletes user data automatically.
    return {
        "document_id": document_id,
        "version_id": version_id,
        "filename": filename,
        "sha256": artifact.sha256,
        "byte_size": artifact.size,
        "status": "stored",
        "extraction": "queued",
        "job_id": job_id,
    }


@router.get("/workspaces/{workspace_id}/versions/{version_id}/file")
async def original_file(
    workspace_id: UUID,
    version_id: UUID,
    request: Request,
    principal: Principal = Depends(identity),
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id)
        result = await conn.execute(
            text("""
            SELECT v.storage_key,v.sha256,v.byte_size,d.filename
            FROM enterprise.document_versions v JOIN enterprise.documents d ON d.id=v.document_id
            WHERE v.id=:version AND v.workspace_id=:workspace
        """),
            {"version": version_id, "workspace": workspace_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            raise HTTPException(404, "Workspace or document not available")
        artifact = StoredArtifact(row["storage_key"], row["sha256"], row["byte_size"])
        data = await asyncio.to_thread(
            request.app.state.storage.read, workspace_id, artifact
        )
    return Response(
        data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''"
            + quote(row["filename"], safe=""),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@router.get("/workspaces/{workspace_id}/versions/{version_id}/extraction")
async def extracted_pages(
    workspace_id: UUID,
    version_id: UUID,
    request: Request,
    principal: Principal = Depends(identity),
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id)
        row = (
            (
                await conn.execute(
                    text("""
            SELECT j.status,j.error_code,j.warnings,j.extractor,j.attempts,v.sha256,d.id AS document_id
            FROM enterprise.ingestion_jobs j JOIN enterprise.document_versions v ON v.id=j.version_id
            JOIN enterprise.documents d ON d.id=v.document_id
            WHERE j.workspace_id=:workspace AND j.version_id=:version
        """),
                    {"workspace": workspace_id, "version": version_id},
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise HTTPException(404, "Workspace or document not available")
        pages = []
        if row["status"] in {"extracted", "needs_review"}:
            result = await conn.execute(
                text("""
                SELECT ordinal,content,location,warnings FROM enterprise.extracted_pages
                WHERE workspace_id=:workspace AND version_id=:version ORDER BY ordinal
            """),
                {"workspace": workspace_id, "version": version_id},
            )
            pages = [dict(page) for page in result.mappings()]
    details = {
        "encrypted_pdf_unsupported": "Encrypted PDFs are unsupported; provide an unlocked copy.",
        "invalid_or_unsupported_pdf": "The PDF could not be validated or parsed.",
        "artifact_unavailable": "The original is missing or failed its integrity check; operator review is required.",
        "authorization_revoked": "The requesting identity no longer has permission to process this source.",
        "retry_limit": "Processing stopped after repeated interruptions; operator review is required.",
    }
    return {
        **dict(row),
        "error_detail": details.get(
            row["error_code"],
            "Processing failed within the configured limits."
            if row["error_code"]
            else None,
        ),
        "workspace_id": workspace_id,
        "version_id": version_id,
        "pages": pages,
        "retrieval_ready": False,
    }


@router.post("/workspaces/{workspace_id}/versions/{version_id}/cancel")
async def cancel_ingestion(
    workspace_id: UUID,
    version_id: UUID,
    request: Request,
    principal: Principal = Depends(identity),
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id, write=True)
        status = await conn.scalar(
            text("""
            UPDATE enterprise.ingestion_jobs SET status='cancelled',claim_token=NULL,
                lease_until=NULL,updated_at=now() WHERE workspace_id=:workspace
                AND version_id=:version AND status IN ('queued','extracting') RETURNING status
        """),
            {"workspace": workspace_id, "version": version_id},
        )
        if not status:
            raise HTTPException(404, "Active job not available")
    return {"status": status}


@router.post("/workspaces/{workspace_id}/versions/{version_id}/process")
async def queue_existing_original(
    workspace_id: UUID,
    version_id: UUID,
    request: Request,
    principal: Principal = Depends(identity),
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id, write=True)
        exists = await conn.scalar(
            text(
                "SELECT id FROM enterprise.document_versions WHERE id=:id AND workspace_id=:workspace"
            ),
            {"id": version_id, "workspace": workspace_id},
        )
        if not exists:
            raise HTTPException(404, "Workspace or document not available")
        admitted = await conn.scalar(text("SELECT enterprise.admit_ingestion()"))
        # Admission takes a transaction lock: concurrent repeats observe the same
        # committed job instead of creating duplicates or resetting its state.
        prior = (
            (
                await conn.execute(
                    text(
                        "SELECT id,status FROM enterprise.ingestion_jobs WHERE version_id=:id"
                    ),
                    {"id": version_id},
                )
            )
            .mappings()
            .first()
        )
        if prior:
            return {"job_id": prior["id"], "status": prior["status"]}
        if not admitted:
            raise HTTPException(429, "Processing queue is full; wait for existing jobs")
        job_id = uuid4()
        await conn.execute(
            text("""
            INSERT INTO enterprise.ingestion_jobs(id,workspace_id,version_id,created_by)
            VALUES(:id,:workspace,:version,:actor)
        """),
            {
                "id": job_id,
                "workspace": workspace_id,
                "version": version_id,
                "actor": principal.user_id,
            },
        )
    return {"job_id": job_id, "status": "queued"}
