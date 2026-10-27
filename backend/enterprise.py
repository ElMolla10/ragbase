"""Development originals, native extraction, lexical search and verified quote answers."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.enterprise_config import EnterpriseSettings
from backend.openrouter_client import MODELS
from backend.policy import assemble
from backend.ui import install_browser


def create_app(settings: EnterpriseSettings | None = None, *, gateway=None):
    settings = settings or EnterpriseSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.environ.get("RAGBASE_PROFILE", settings.profile) != "dev-lite":
            raise RuntimeError("Only the development enterprise profile is implemented")
        app.state.settings = settings
        app.state.instructions = assemble("retrieval")
        app.state.database = None
        app.state.gateway = gateway
        app.state.model_record = {
            "alias": "company-assistant",
            "requested_model": settings.model_name,
            "checkpoint": None
            if settings.generation_mode == "openrouter"
            else settings.model_name,
            "digest": None
            if settings.generation_mode == "openrouter"
            else settings.model_digest,
            "adapter": settings.generation_mode,
            "adapter_version": "exhaustive-coverage-v1",
            "production_qualified": False,
            "external_processing": settings.generation_mode == "openrouter",
            "dynamic_routing": settings.generation_mode == "openrouter",
            "fallback_models": list(MODELS[1:])
            if settings.generation_mode == "openrouter"
            else [],
            "privacy_policy": settings.remote_privacy
            if settings.generation_mode == "openrouter"
            else "local",
            "byok_only": settings.generation_mode == "openrouter",
            "schema_enforcement": "application",
            "provider_json_schema": settings.generation_mode == "ollama",
            "enabled_capabilities": [
                "text",
                "structured_quote_output",
                "streaming",
                "exhaustive_text_coverage",
            ],
            "disabled_capabilities": ["vision", "tools", "analysis"],
            "exhaustive_coverage": "development_only_not_production_qualified",
        }
        if gateway is None and settings.generation_mode == "fixture":
            from backend.answers.fixtures import FixtureQuoteGateway

            app.state.gateway = FixtureQuoteGateway()
        elif gateway is None and settings.generation_mode == "ollama":
            from backend.gateway.ollama import OllamaGateway

            app.state.gateway = OllamaGateway(
                settings.model_endpoint, settings.model_name, settings.model_digest
            )
        elif gateway is None and settings.generation_mode == "openrouter":
            from backend.gateway.openrouter import OpenRouterGateway

            app.state.gateway = OpenRouterGateway(
                settings.model_endpoint, privacy=settings.remote_privacy
            )
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

        from backend.answers.api import router as answers_router
        from backend.coverage.api import router as coverage_router
        from backend.library import router
        from backend.storage.local import ArtifactError

        app.include_router(router)
        app.include_router(answers_router)
        app.include_router(coverage_router)

        async def unavailable(request, exc):
            return JSONResponse(
                {"detail": "Storage service unavailable"}, status_code=503
            )

        for error in (SQLAlchemyError, ArtifactError, OSError, TimeoutError):
            app.add_exception_handler(error, unavailable)

    if not install_browser(app, settings.ui_root, storage_root=settings.storage_root):

        @app.get("/", include_in_schema=False)
        async def browser_shell_unavailable():
            return PlainTextResponse(
                "Enterprise browser shell is not available.\n",
                status_code=503,
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )

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
            "generation": settings.generation_mode,
            "documents": "original_storage_and_native_extraction"
            if app.state.database
            else "not_implemented",
            "constitution_version": instructions.constitution_version,
            "instruction_sha256": instructions.sha256,
        }

    @app.get("/health/generation")
    async def generation_health():
        available = settings.generation_mode == "fixture"
        if settings.generation_mode == "openrouter":
            import httpx

            configured = False
            try:
                async with httpx.AsyncClient(
                    timeout=3, trust_env=False, follow_redirects=False
                ) as client:
                    response = await client.get(
                        settings.model_endpoint + "/health/configuration"
                    )
                configured = (
                    response.status_code == 200
                    and response.json().get("credential_configured") is True
                )
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                pass
            return JSONResponse(
                {
                    "mode": "openrouter",
                    "external_processing": True,
                    "model": settings.model_name,
                    "byok_only": True,
                    "fallback_models": list(MODELS[1:]),
                    "privacy_policy": settings.remote_privacy,
                    "credential_configured": configured,
                    "provider_verified": False,
                    "billable_probe_performed": False,
                    "production_qualified": False,
                },
                status_code=200 if configured else 503,
            )
        if settings.generation_mode == "ollama":
            import httpx

            try:
                async with httpx.AsyncClient(
                    timeout=3, trust_env=False, follow_redirects=False
                ) as client:
                    response = await client.get(settings.model_endpoint + "/api/tags")
                available = (
                    response.status_code == 200
                    and len(response.content) <= 65536
                    and any(
                        item.get("name") == settings.model_name
                        and item.get("digest") == settings.model_digest
                        for item in response.json().get("models", [])
                    )
                )
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                available = False
        return JSONResponse(
            {
                "mode": settings.generation_mode,
                "installed_checkpoint_available": available,
                "model_loaded_by_probe": False,
                "production_qualified": False,
            },
            status_code=200 if available else 503,
        )

    return app


app = create_app()
