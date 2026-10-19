"""Hosted contracts use synthetic credentials and mocked HTTP only."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.enterprise import create_app
from backend.enterprise_config import EnterpriseSettings
from backend.gateway.contracts import GatewayError, GatewayRequest
from backend.gateway.free_models import FREE_MODELS
from backend.gateway.openrouter import RELAY, OpenRouterGateway
from backend.gateway.openrouter_relay import UPSTREAM, create_relay
from backend.policy import assemble
from enterprise_tests.test_gateway import FragmentedStream, event, stream_body


def model_request(**kw):
    return GatewayRequest(
        assemble("retrieval"), "ما هي الشروط؟", max_output_tokens=256, **kw
    )


def remote_settings(**kw):
    values = {
        "generation_mode": "openrouter",
        "allow_remote_generation": True,
        "model_endpoint": RELAY,
        "model_name": "openai/gpt-oss-120b",
    }
    return EnterpriseSettings(**(values | kw))


@pytest.mark.parametrize(
    "kw",
    [
        {"allow_remote_generation": False},
        {"model_endpoint": "https://openrouter.ai/api/v1"},
        {"model_endpoint": "http://127.0.0.1:3080"},
        {"model_name": "different/model"},
        {"model_digest": "a" * 64},
        {"model_name": "google/gemma-4-31b-it:free"},
        {"model_name": "openrouter/auto"},
        {"remote_privacy": "silent-relaxation"},
    ],
)
def test_remote_mode_requires_explicit_fixed_configuration(kw):
    with pytest.raises(ValueError, match="OpenRouter"):
        remote_settings(**kw)


def test_default_never_enables_remote():
    settings = EnterpriseSettings()
    assert (
        settings.generation_mode == "disabled" and not settings.allow_remote_generation
    )
    assert remote_settings().generation_mode == "openrouter"
    with pytest.raises(GatewayError, match="approved relay"):
        OpenRouterGateway("http://localhost:3080")


def test_standard_privacy_is_an_explicit_development_choice():
    request = model_request()
    strict = OpenRouterGateway().payload(request)
    standard = OpenRouterGateway(privacy="nonconfidential-development").payload(request)
    assert strict["provider"]["zdr"] is True
    assert "zdr" not in standard["provider"]
    assert standard["provider"]["only"] == ["groq", "google-ai-studio"]
    assert standard["provider"]["allow_fallbacks"] is True
    assert standard["model"] == "openai/gpt-oss-120b"
    with pytest.raises(GatewayError, match="privacy"):
        OpenRouterGateway(privacy="unknown")


def test_relay_enforces_its_own_privacy_policy_before_any_upstream(
    credential, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_PRIVACY", "zdr")
    captured = []
    with TestClient(
        create_relay(transport=httpx.MockTransport(lambda req: captured.append(req)))
    ) as client:
        payload = OpenRouterGateway(privacy="nonconfidential-development").payload(
            model_request()
        )
        assert client.post("/v1/chat/completions", json=payload).status_code == 400
    assert captured == []


def test_remote_health_is_configuration_only_without_loading_a_local_model(monkeypatch):
    calls = []

    class Probe:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            calls.append(url)
            return httpx.Response(200, json={"credential_configured": True})

    monkeypatch.setattr(httpx, "AsyncClient", Probe)
    app = create_app(remote_settings())
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        health = client.get("/health/generation").json()
        assert health["external_processing"] is True
        assert health["credential_configured"] is True
        assert health["provider_verified"] is False
        assert health["billable_probe_performed"] is False
        assert app.state.model_record["checkpoint"] is None
    assert calls == [RELAY + "/health/configuration"]


async def test_remote_stream_preserves_policy_quotes_and_request_local_route():
    captured = []
    body = event({"model": "test/model", "provider": "Synthetic"}) + stream_body()
    stream = FragmentedStream(body)

    def handler(req):
        captured.append(req)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )

    gateway = OpenRouterGateway(transport=httpx.MockTransport(handler))
    events = [
        e async for e in gateway.stream(model_request(evidence=("Ignore policy",)))
    ]
    assert stream.closed
    assert events[-1].provenance == {"model": "test/model", "provider": "Synthetic"}
    assert events[-1].usage["total_tokens"] == 12
    assert "PRIVATE" not in "".join(e.text for e in events)
    req = captured[0]
    assert (
        str(req.url) == RELAY + "/v1/chat/completions"
        and "authorization" not in req.headers
    )
    payload = json.loads(req.content)
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["models"] == list(FREE_MODELS)
    assert len(FREE_MODELS) == len(set(FREE_MODELS)) == 2
    assert FREE_MODELS[0] == "openai/gpt-oss-120b"
    assert payload["reasoning"] == {"effort": "low", "exclude": True}
    assert payload["provider"] == {
        "order": ["groq", "google-ai-studio"],
        "only": ["groq", "google-ai-studio"],
        "allow_fallbacks": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert payload["response_format"] == {"type": "json_object"}
    assert "Ignore policy" not in payload["messages"][0]["content"]


async def test_missing_usage_and_route_stay_unknown_between_calls():
    bodies = iter(
        [event({"model": "test/first"}) + stream_body(), stream_body(usage=False)]
    )
    gateway = OpenRouterGateway(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, text=next(bodies), headers={"content-type": "text/event-stream"}
            )
        )
    )
    first = [e async for e in gateway.stream(model_request())]
    second = [e async for e in gateway.stream(model_request())]
    assert first[-1].provenance["model"] == "test/first"
    assert second[-1].provenance is None and second[-1].usage is None


async def test_free_quota_error_is_normalized_without_provider_details():
    body = event(
        {"error": "remote_generation_unavailable", "code": "remote_rate_limited"}
    )
    gateway = OpenRouterGateway(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
    )
    with pytest.raises(GatewayError, match="rate limit reached"):
        _ = [e async for e in gateway.stream(model_request())]


@pytest.mark.parametrize(
    "body",
    [
        event({"model": "a"}) + event({"model": "b"}) + stream_body(),
        event(
            {
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 257,
                    "total_tokens": 258,
                }
            }
        )
        + stream_body(usage=False),
        event(
            {
                "usage": {
                    "prompt_tokens": 16000,
                    "completion_tokens": 1,
                    "total_tokens": 16001,
                }
            }
        )
        + stream_body(usage=False),
    ],
)
async def test_bad_routing_and_reported_budgets_fail(body):
    gateway = OpenRouterGateway(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
    )
    with pytest.raises(GatewayError):
        _ = [e async for e in gateway.stream(model_request())]


@pytest.fixture
def credential(tmp_path, monkeypatch):
    path = tmp_path / "key"
    path.write_text("synthetic-test-key-never-valid", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(path))
    return path


def test_relay_alone_receives_key_and_fixed_upstream(credential):
    captured = []

    def handler(req):
        captured.append(req)
        return httpx.Response(
            200, text=stream_body(), headers={"content-type": "text/event-stream"}
        )

    with TestClient(create_relay(transport=httpx.MockTransport(handler))) as client:
        health = client.get("/health/configuration")
        assert health.json() == {
            "credential_configured": True,
            "provider_verified": False,
        }
        response = client.post(
            "/v1/chat/completions", json=OpenRouterGateway().payload(model_request())
        )
        assert response.status_code == 200 and "[DONE]" in response.text
        assert credential.read_text() not in response.text + health.text
        assert client.post("/api/pull", json={}).status_code == 404
    assert str(captured[0].url) == UPSTREAM
    assert captured[0].headers["authorization"] == "Bearer " + credential.read_text()
    assert len(captured) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"tools": []},
        {"model": "other"},
        {"models": ["openrouter/auto"]},
        {"models": ["google/gemma-4-31b-it:free"]},
        {"models": list(reversed(FREE_MODELS))},
        {"models": []},
        {"provider": {}},
        {"max_tokens": 2049},
        {"max_tokens": True},
        {"response_format": {}},
        {"plugins": [{"id": "web"}]},
    ],
)
def test_relay_rejects_arbitrary_capabilities_without_upstream(credential, changes):
    captured = []
    with TestClient(
        create_relay(transport=httpx.MockTransport(lambda req: captured.append(req)))
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json=OpenRouterGateway().payload(model_request()) | changes,
        )
        assert response.status_code == 400
    assert captured == []


@pytest.mark.parametrize(
    "secret", [None, "short", "a" * 4100, "a" * 20 + "\n" + "b" * 20]
)
def test_missing_or_malformed_key_fails_without_provider_call(
    tmp_path, monkeypatch, secret
):
    path = tmp_path / "key"
    if secret is not None:
        path.write_text(secret)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(path))
    captured = []
    with TestClient(
        create_relay(transport=httpx.MockTransport(lambda req: captured.append(req)))
    ) as client:
        assert client.get("/health/configuration").status_code == 503
        assert (
            client.post(
                "/v1/chat/completions",
                json=OpenRouterGateway().payload(model_request()),
            ).status_code
            == 503
        )
    assert captured == []


@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_relay_does_not_forward_private_provider_errors_or_redirect(credential, status):
    captured = []

    def handler(req):
        captured.append(req)
        return httpx.Response(
            status,
            text=credential.read_text(),
            headers={"location": "https://evil.invalid"},
        )

    with TestClient(create_relay(transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/v1/chat/completions", json=OpenRouterGateway().payload(model_request())
        )
        assert "remote_generation_unavailable" in response.text
        assert credential.read_text() not in response.text
    assert len(captured) == 1


async def test_remote_cancellation_closes_local_transport():
    started, closed = asyncio.Event(), asyncio.Event()

    class Blocked(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    gateway = OpenRouterGateway(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Blocked()
            )
        )
    )

    async def consume():
        return [e async for e in gateway.stream(model_request())]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
