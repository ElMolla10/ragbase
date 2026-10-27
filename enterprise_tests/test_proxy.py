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


def test_browser_and_coverage_stay_on_the_fixed_upstream(monkeypatch):
    original = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        path = request.url.path
        if path.startswith("/assets/"):
            return httpx.Response(
                200,
                content=b"console.log(1)",
                headers={"content-type": "text/javascript"},
            )
        if path.endswith("/coverage"):
            return httpx.Response(
                200,
                content=b'{"event":"queued"}\n',
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(
            200,
            content=b"<html></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(
        dev_proxy.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond)),
    )
    workspace = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    with TestClient(dev_proxy.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/index-abc.js").status_code == 200
        assert client.get("/assets/../secret").status_code == 404
        assert client.get("/api/workspaces/not-a-uuid/coverage").status_code == 404
        coverage = client.post(f"/api/workspaces/{workspace}/coverage", content=b"{}")
        assert coverage.status_code == 200
        assert b"queued" in coverage.content
    urls = {str(request.url) for request in requests}
    assert urls == {
        "http://api:8000/",
        "http://api:8000/assets/index-abc.js",
        f"http://api:8000/api/workspaces/{workspace}/coverage",
    }
    assert dev_proxy.api_path_allowed(f"/api/workspaces/{workspace}/coverage")
    assert dev_proxy.api_path_allowed(
        f"/api/workspaces/{workspace}/coverage/{workspace}/cancel"
    )
    assert not dev_proxy.api_path_allowed("/api/workspaces/not-a-uuid/coverage")
    assert not dev_proxy.browser_path_allowed("/assets/../../.env")
    assert not dev_proxy.browser_path_allowed("http://example.com/")


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
