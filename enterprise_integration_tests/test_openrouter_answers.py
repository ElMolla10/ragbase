"""RLS and quote validation stay active with mocked hosted generation."""

import json
from dataclasses import replace

import httpx
import pytest

from backend.enterprise import create_app
from backend.gateway.free_models import FREE_MODELS
from backend.gateway.openrouter import RELAY, OpenRouterGateway
from enterprise_integration_tests.test_pdf_answers import ask, frames, ready


@pytest.mark.parametrize(
    "forge", [False, True, "quota", "fallback", "invalid_fallback"]
)
async def test_hosted_mode_preserves_authorization_and_evidence_validation(
    world, forge
):
    record = await ready(world)
    captured = []

    def handler(req):
        captured.append(req)
        if forge == "quota":
            return httpx.Response(
                200,
                text='data: {"error":"remote_generation_unavailable","code":"remote_rate_limited"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        user = json.loads(json.loads(req.content)["messages"][1]["content"])
        evidence = json.loads(user["untrusted_evidence"][0])
        answer = {
            "insufficient": False,
            "claims": [
                {
                    "source_id": evidence["source_id"],
                    "quote": "Forged quote"
                    if forge is True or forge == "invalid_fallback"
                    else "Quarter 2 sales were 180.",
                }
            ],
        }
        chunk = {
            "model": FREE_MODELS[1].removesuffix(":free")
            if forge in {"fallback", "invalid_fallback"}
            else "synthetic/remote",
            "provider": "Synthetic",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": json.dumps(answer)},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    settings = replace(
        world["app"].state.settings,
        generation_mode="openrouter",
        allow_remote_generation=True,
        model_endpoint=RELAY,
        model_name="openai/gpt-oss-120b",
    )
    app = create_app(
        settings, gateway=OpenRouterGateway(transport=httpx.MockTransport(handler))
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        remote = world | {"app": app, "client": client}
        denied = await ask(remote, record, user="bob", workspace="b")
        assert denied.status_code == 404 and captured == []
        events = frames(await ask(remote, record))
        assert events[0]["external_processing"] is True
        answers = [e for e in events if e["event"] == "answer"]
        if forge is True or forge in {"quota", "invalid_fallback"}:
            assert not answers and events[-1]["event"] == "error"
            if forge == "quota":
                assert "BYOK provider rate limit" in events[-1]["detail"]
        else:
            metadata = answers[0]["metadata"]
            assert metadata["external_processing"] is True
            expected_model = (
                FREE_MODELS[1].removesuffix(":free")
                if forge == "fallback"
                else "synthetic/remote"
            )
            assert metadata["model"]["resolved_route"]["model"] == expected_model
            assert metadata["model"]["fallback_models"] == list(FREE_MODELS[1:])
            assert metadata["model"]["byok_only"] is True
            assert metadata["input_token_upper_bound"] is None
            assert answers[0]["claims"][0]["sha256"] == record["sha256"]
        assert len(captured) == 1
