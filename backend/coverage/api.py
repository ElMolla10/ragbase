"""Authenticated exhaustive coverage. Ordinary overview answers stay bounded."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from backend.answers.api import parse_input
from backend.coverage.planner import gap_counts
from backend.coverage.workflow import execute, prepare
from backend.library import identity

router = APIRouter(prefix="/api/workspaces")


@router.post("/{workspace}/coverage")
async def start_coverage(
    workspace: UUID, request: Request, principal=Depends(identity)
):
    data = await parse_input(request)
    context = await prepare(request, principal.user_id, workspace, data)
    return StreamingResponse(
        execute(context, request, principal.user_id, workspace),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


def _visible_complete(row, sections):
    return bool(
        row["status"] == "finished"
        and row["coverage_complete"]
        and sections
        and all(section["status"] == "covered" for section in sections)
    )


@router.get("/{workspace}/coverage/{run_id}")
async def read_coverage(
    workspace: UUID, run_id: UUID, request: Request, principal=Depends(identity)
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace) as conn:
        await database.authorize(conn, workspace)
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT status,coverage_complete,error_code,metadata,version_ids FROM enterprise.coverage_runs WHERE id=:id"
                    ),
                    {"id": run_id},
                )
            )
            .mappings()
            .first()
        )
        if not row:
            raise HTTPException(404, "Coverage job not available")
        stored = (
            (
                await conn.execute(
                    text("""
                SELECT s.version_id,s.ordinal,s.part,s.status,s.gap_code,s.location,
                       v.sha256,d.filename
                FROM enterprise.coverage_sections s
                JOIN enterprise.coverage_runs r ON r.id=s.run_id
                JOIN enterprise.document_versions v ON v.id=s.version_id
                JOIN enterprise.documents d ON d.id=v.document_id
                WHERE s.run_id=:id
                ORDER BY array_position(r.version_ids, s.version_id), s.ordinal, s.part
                """),
                    {"id": run_id},
                )
            )
            .mappings()
            .all()
        )
    sections = [
        {
            "version_id": str(item["version_id"]),
            "ordinal": item["ordinal"],
            "part": item["part"],
            "status": item["status"],
            "gap_code": item["gap_code"],
            "location": item["location"],
            "filename": item["filename"],
            "sha256": item["sha256"],
        }
        for item in stored
    ]
    metadata = row["metadata"] or {}
    complete = _visible_complete(row, sections)
    return {
        "run_id": str(run_id),
        "status": row["status"],
        "coverage_complete": complete,
        "label": "complete" if complete else "partial",
        "error_code": row["error_code"],
        "sections": sections,
        "gap_counts": gap_counts(sections, metadata.get("extra_gaps") or ()),
        "processing_notes": list(metadata.get("processing_notes") or []),
        "selected_versions": [str(value) for value in row["version_ids"]],
        "strategy": "explicit_section_coverage",
    }


@router.post("/{workspace}/coverage/{run_id}/cancel")
async def cancel_coverage(
    workspace: UUID, run_id: UUID, request: Request, principal=Depends(identity)
):
    database = request.app.state.database
    async with database.scope(principal.user_id, workspace) as conn:
        await database.authorize(conn, workspace)
        found = await conn.scalar(
            text("""
            UPDATE enterprise.coverage_runs
            SET status='cancelled', updated_at=now()
            WHERE id=:id AND status IN ('queued','running','yielding')
            RETURNING id
            """),
            {"id": run_id},
        )
        if not found:
            raise HTTPException(404, "Active coverage job not available")
    return {"status": "cancelled"}
