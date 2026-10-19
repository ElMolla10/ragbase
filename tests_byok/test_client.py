"""Inspect actual SDK requests for every sync/async and stream combination."""

import json

import httpx
import pytest

from backend.openrouter_client import (
    BASE_URL,
    MODELS,
    PROVIDERS,
    async_client,
    async_completion,
    completion_args,
    routing_body,
    sync_client,
    sync_completion,
)


def response(request, captured):
    captured.append(request)
    data = json.loads(request.content)
    if data.get("stream"):
        chunk = {
            "id": "synthetic",
            "created": 0,
            "object": "chat.completion.chunk",
            "model": MODELS[1],
            "choices": [
                {"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}
            ],
        }
        return httpx.Response(
            200,
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            "id": "synthetic",
            "created": 0,
            "object": "chat.completion",
            "model": MODELS[1],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def check_request(request):
    assert str(request.url) == BASE_URL + "/chat/completions"
    body = json.loads(request.content)
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["model"] == MODELS[0]
    assert body["models"] == list(MODELS)
    assert body["provider"] == {
        "order": list(PROVIDERS),
        "only": list(PROVIDERS),
        "allow_fallbacks": True,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_sync_sdk_always_carries_provider_allowlist(stream):
    captured = []
    with sync_client(
        key="synthetic-not-a-real-key",
        transport=httpx.MockTransport(lambda req: response(req, captured)),
    ) as client:
        result = sync_completion(
            client, messages=[{"role": "user", "content": "synthetic"}], stream=stream
        )
        if stream:
            with result:
                assert list(result)[0].model == MODELS[1]
        else:
            assert result.model == MODELS[1]
    assert len(captured) == 1
    check_request(captured[0])


@pytest.mark.parametrize("stream", [False, True])
async def test_async_sdk_always_carries_provider_allowlist(stream):
    captured = []
    async with async_client(
        key="synthetic-not-a-real-key",
        transport=httpx.MockTransport(lambda req: response(req, captured)),
    ) as client:
        result = await async_completion(
            client, messages=[{"role": "user", "content": "synthetic"}], stream=stream
        )
        if stream:
            async with result:
                chunks = [chunk async for chunk in result]
                assert chunks[0].model == MODELS[1]
        else:
            assert result.model == MODELS[1]
    assert len(captured) == 1
    check_request(captured[0])


@pytest.mark.parametrize(
    "providers", [(), ("together",), ("fireworks",), ("groq", "google-ai-studio")]
)
def test_cannot_expand_or_reorder_provider_allowlist(providers):
    with pytest.raises(ValueError):
        routing_body(providers)


def test_cannot_supply_arbitrary_extra_body_or_model():
    with pytest.raises(ValueError):
        completion_args(extra_body={"provider": {}})
    with pytest.raises(ValueError):
        completion_args(model="openrouter/auto")


@pytest.mark.parametrize("status", [401, 429, 503])
async def test_sdk_does_not_add_a_second_retry_loop(status):
    from openai import APIStatusError

    captured = []

    def fail(req):
        captured.append(req)
        return httpx.Response(status, json={"error": {"message": "synthetic"}})

    async with async_client(
        key="synthetic-not-a-real-key", transport=httpx.MockTransport(fail)
    ) as client:
        with pytest.raises(APIStatusError):
            await async_completion(
                client, messages=[{"role": "user", "content": "synthetic"}]
            )
    assert len(captured) == 1
    check_request(captured[0])


def test_removed_mistral_cannot_be_selected():
    with pytest.raises(ValueError):
        routing_body(("mistral",))
    with pytest.raises(ValueError):
        completion_args(model="mistralai/mistral-large-4-0")
