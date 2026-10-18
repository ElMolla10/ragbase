"""Lightweight foundation only; no document routes or production identity yet."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.policy import assemble


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RAGBASE_PROFILE", "dev-lite") != "dev-lite":
        raise RuntimeError("Only the development foundation is implemented")
    app.state.instructions = assemble("retrieval")
    yield


app = FastAPI(
    title="RAGBase Enterprise — development foundation",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health/live")
async def live() -> dict:
    return {"status": "alive", "profile": "dev-lite", "development_only": True}


@app.get("/health/ready")
async def ready() -> dict:
    # Revalidate assets so drift cannot be reported ready.
    try:
        instructions = assemble("retrieval")
    except ValueError as exc:
        raise HTTPException(503, "Policy assets need review") from exc
    return {
        "status": "foundation_ready",
        "generation": "fixture_only",
        "documents": "not_implemented",
        "constitution_version": instructions.constitution_version,
        "instruction_sha256": instructions.sha256,
    }
