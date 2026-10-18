"""Authenticated storage-only API. Extraction and generation are not enabled."""

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
    if not value.lower().endswith((".txt", ".md")):
        raise HTTPException(
            415,
            "This storage slice accepts TXT and Markdown; PDF validation follows in ingestion",
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
            SELECT d.id,d.filename,v.id AS version_id,v.version_number,v.byte_size,v.sha256,v.status
            FROM enterprise.documents d JOIN enterprise.document_versions v
            ON v.document_id=d.id AND v.workspace_id=d.workspace_id
            WHERE d.workspace_id=:workspace ORDER BY d.created_at DESC,d.id LIMIT 100
        """),
            {"workspace": workspace_id},
        )
        rows = [dict(row) for row in result.mappings()]
    return {"documents": rows, "limit": 100, "extraction": "not_implemented"}


@router.post("/workspaces/{workspace_id}/documents", status_code=201)
async def store_original(
    workspace_id: UUID, request: Request, principal: Principal = Depends(identity)
):
    filename = filename_from_header(request.headers.get("x-filename"))
    database = request.app.state.database
    store = request.app.state.storage
    async with database.scope(principal.user_id, workspace_id) as conn:
        await database.authorize(conn, workspace_id, write=True)
        body = await bounded_body(request, request.app.state.settings.max_upload_bytes)
        if filename.lower().endswith((".txt", ".md")):
            try:
                body.decode("utf-8")
            except UnicodeError as exc:
                raise HTTPException(415, "Text originals must use UTF-8") from exc
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
    # Commit precedes success. Reconciliation reports unreferenced files left by
    # failed commits; it never deletes user data automatically.
    return {
        "document_id": document_id,
        "version_id": version_id,
        "filename": filename,
        "sha256": artifact.sha256,
        "byte_size": artifact.size,
        "status": "stored",
        "extraction": "not_implemented",
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
