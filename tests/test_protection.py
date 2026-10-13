"""Exercise rejection paths without database access or LLM calls."""

import httpx
import pytest

from backend.database import get_session
from backend.main import app
from backend.protection import limiter, settings


@pytest.fixture
def protected_client(monkeypatch):
    limiter.events.clear()
    monkeypatch.setattr(settings, "write_api_key", "")
    monkeypatch.setattr(settings, "query_rate_limit", 1)
    monkeypatch.setattr(settings, "ingest_rate_limit", 1)

    async def no_database():
        yield None

    app.dependency_overrides[get_session] = no_database
    yield httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    app.dependency_overrides.clear()
    limiter.events.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/query", "/ingest"])
async def test_rate_limit_returns_retry_after(protected_client, path):
    async with protected_client as client:
        assert (await client.post(path)).status_code == 422
        rejected = await client.post(path)
        assert rejected.status_code == 429
        assert int(rejected.headers["Retry-After"]) > 0
        assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_optional_key_protects_writes_only(protected_client, monkeypatch):
    monkeypatch.setattr(settings, "write_api_key", "secret")
    async with protected_client as client:
        assert (await client.post("/ingest")).status_code == 401
        assert (await client.delete("/documents/1")).status_code == 401
        assert (await client.post("/query")).status_code == 422
        assert (
            await client.post("/ingest", headers={"X-API-Key": "secret"})
        ).status_code == 422


@pytest.mark.asyncio
async def test_actual_file_size_limit(protected_client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    async with protected_client as client:
        response = await client.post(
            "/ingest", files={"file": ("paper.pdf", b"x" * 11)}
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_content_length_rejected_before_parsing(protected_client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    async with protected_client as client:
        response = await client.post("/ingest", headers={"Content-Length": "100000"})
    assert response.status_code == 413


def test_expired_limits_are_pruned(monkeypatch):
    monkeypatch.setattr("backend.protection.time.monotonic", lambda: 0)
    assert limiter.retry_after("a", "/query", 1) == 0
    assert limiter.retry_after("a", "/query", 1) > 0
    monkeypatch.setattr(
        "backend.protection.time.monotonic",
        lambda: settings.rate_limit_window_seconds + 1,
    )
    assert limiter.retry_after("b", "/query", 1) == 0
    assert ("a", "/query") not in limiter.events
    limiter.events.clear()
