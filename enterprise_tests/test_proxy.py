"""Development proxy only forwards fixed health routes to a fixed upstream."""

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import dev_proxy


@pytest.mark.parametrize("status", [200, 503])
def test_forward_fixed_upstream_and_health_status(monkeypatch, status):
    original = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"status": "fixture"})

    monkeypatch.setattr(
        dev_proxy.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond)),
    )
    with TestClient(dev_proxy.app) as client:
        response = client.get("/health/ready?url=https://example.com")
        assert response.status_code == status
        assert str(requests[0].url) == "http://api:8000/health/ready"
        assert client.get("/documents").status_code == 404
        assert client.post("/query").status_code == 404


def test_upstream_redirect_does_not_leave_network(monkeypatch):
    original = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://example.com"})

    monkeypatch.setattr(
        dev_proxy.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond)),
    )
    with TestClient(dev_proxy.app) as client:
        assert client.get("/health/live").status_code == 502
    assert len(requests) == 1
