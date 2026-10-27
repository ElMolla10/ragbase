"""Authenticated calculation routes. Plans are validated again in trusted code."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.analysis.plan import OperationPlan
from backend.analysis.service import perform, read_calculation, read_record, read_table
from backend.library import bounded_body, identity

router = APIRouter(prefix="/api/workspaces")


class CalculationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_id: UUID
    question: str | None = Field(default=None, min_length=1, max_length=512)
    plan: OperationPlan | None = None
    client_request_id: UUID

    @field_validator("question")
    @classmethod
    def stripped(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("empty question")
        return value

    @model_validator(mode="after")
    def one_input(self):
        if (self.question is None) == (self.plan is None):
            raise ValueError("Provide a question or a plan")
        return self


async def parse_calculation(request: Request) -> CalculationRequest:
    try:
        return CalculationRequest.model_validate_json(
            await bounded_body(request, 16384)
        )
    except ValueError as exc:
        raise HTTPException(
            422,
            "A calculation needs one CSV version and either a question or an allowlisted plan.",
        ) from exc


@router.post("/{workspace}/calculations")
async def create_calculation(
    workspace: UUID, request: Request, principal=Depends(identity)
):
    body = await parse_calculation(request)
    return await perform(request, principal.user_id, workspace, body)


@router.get("/{workspace}/calculations/{calculation_id}")
async def get_calculation(
    workspace: UUID,
    calculation_id: UUID,
    request: Request,
    principal=Depends(identity),
):
    return await read_calculation(request, principal.user_id, workspace, calculation_id)


@router.get("/{workspace}/calculations/{calculation_id}/table.csv")
async def download_table(
    workspace: UUID,
    calculation_id: UUID,
    request: Request,
    principal=Depends(identity),
):
    table = await read_table(request, principal.user_id, workspace, calculation_id)
    return Response(
        table.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="calculation.csv"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        },
    )


@router.get("/{workspace}/calculations/{calculation_id}/record.json")
async def download_record(
    workspace: UUID,
    calculation_id: UUID,
    request: Request,
    principal=Depends(identity),
):
    record = await read_record(request, principal.user_id, workspace, calculation_id)
    return Response(
        json_bytes(record),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="calculation-record.json"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        },
    )


def json_bytes(record) -> bytes:
    import json

    return json.dumps(record, ensure_ascii=False).encode("utf-8")
