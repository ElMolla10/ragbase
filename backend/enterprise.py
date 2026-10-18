"""Enterprise dev-lite entry point with optional scoped original-file storage."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from backend.enterprise_config import EnterpriseSettings
from backend.policy import assemble


def create_app(settings: EnterpriseSettings | None = None):
    settings = settings or EnterpriseSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.environ.get("RAGBASE_PROFILE", settings.profile) != "dev-lite":
            raise RuntimeError("Only the development enterprise profile is implemented")
        app.state.settings = settings
        app.state.instructions = assemble("retrieval")
        app.state.database = None
        if settings.database_url:
            from backend.enterprise_db import EnterpriseDatabase
            from backend.storage.local import LocalArtifactStore

            app.state.database = EnterpriseDatabase(settings.database_url)
            try:
                await app.state.database.check_ready()
                app.state.storage = LocalArtifactStore(
                    settings.storage_root, max_bytes=settings.max_upload_bytes
                )
                yield
            finally:
                await app.state.database.close()
        else:
            yield

    app = FastAPI(
        title="RAGBase Enterprise — development",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    if settings.database_url:
        from sqlalchemy.exc import SQLAlchemyError

        from backend.library import router
        from backend.storage.local import ArtifactError

        app.include_router(router)

        async def unavailable(request, exc):
            return JSONResponse(
                {"detail": "Storage service unavailable"}, status_code=503
            )

        for error in (SQLAlchemyError, ArtifactError, OSError, TimeoutError):
            app.add_exception_handler(error, unavailable)

    @app.get("/health/live")
    async def live():
        return {"status": "alive", "profile": "dev-lite", "development_only": True}

    @app.get("/health/ready")
    async def ready():
        try:
            instructions = assemble("retrieval")
            if app.state.database:
                await app.state.database.check_ready()
        except Exception as exc:
            raise HTTPException(503, "Enterprise dependencies need attention") from exc
        return {
            "status": "storage_ready" if app.state.database else "foundation_ready",
            "generation": "fixture_only",
            "documents": "original_storage_only"
            if app.state.database
            else "not_implemented",
            "constitution_version": instructions.constitution_version,
            "instruction_sha256": instructions.sha256,
        }

    return app


app = create_app()
