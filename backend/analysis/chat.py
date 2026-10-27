"""Stream a spreadsheet calculation through the existing answer channel."""

import time
from uuid import uuid4

from fastapi import HTTPException

from backend.analysis.service import perform
from backend.answers.api import frame


def calculation_body(data, question: str):
    from backend.analysis.api import CalculationRequest

    return CalculationRequest(
        version_id=data.version_ids[0],
        question=question,
        client_request_id=data.client_request_id,
    )


async def stream_calculation(request, principal, workspace, data, question: str):
    started = time.monotonic()
    remote = request.app.state.settings.generation_mode == "openrouter"
    yield frame(
        "queued",
        run_id=str(uuid4()),
        external_processing=remote,
        privacy_policy=request.app.state.settings.remote_privacy if remote else "local",
    )
    yield frame("stage", stage="calculating")
    try:
        result = await perform(
            request, principal.user_id, workspace, calculation_body(data, question)
        )
    except HTTPException as exc:
        yield frame("error", code="calculation_rejected", detail=str(exc.detail))
        return
    claims = []
    for citation in result["citations"]:
        claims.append(
            {
                "source_id": citation["source_id"],
                "quote": citation["quote"],
                "explanation": None,
                "version_id": citation["version_id"],
                "document_id": citation["document_id"],
                "sha256": citation["sha256"],
                "filename": citation["filename"],
                "location": citation["location"],
                "preview_url": citation.get("preview_url"),
                "warnings": citation.get("warnings") or [],
            }
        )
    notice = (
        "No number was calculated. The request needs a clearer column, aggregate, filter, or definition."
        if result["status"] != "completed"
        else "Descriptive aggregate from the authorized CSV version. Not a forecast and not production qualification."
    )
    yield frame(
        "answer",
        mode="spreadsheet_calculation",
        insufficient=False,
        answer=result["answer"],
        claims=claims,
        scope_notice=notice,
        downloads=result["downloads"],
        metadata={
            "fixture": request.app.state.settings.generation_mode == "fixture",
            "model_called": result["model_called"],
            "answer_release_seconds": time.monotonic() - started,
            "calculation_id": result["calculation_id"],
            "calculation_status": result["status"],
            "production_qualified": False,
            "coverage": {},
            "external_processing": remote and result["model_called"],
        },
    )
    yield frame("done", run_id=result["calculation_id"])
