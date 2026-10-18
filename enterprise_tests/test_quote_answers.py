"""Grounding validation and native transport contracts; no real model."""

import json
from uuid import uuid4

import httpx
import pytest

from backend.answers.grounding import input_bound, make_request, validate_answer
from backend.gateway.contracts import GatewayError
from backend.gateway.ollama import OllamaGateway
from backend.policy import assemble
from backend.search.index import chunks, normalize, query_terms


def hit(content="Revenue was 120."):
    return {
        "id": uuid4(),
        "source_id": str(uuid4()),
        "content": content,
        "version_id": uuid4(),
        "document_id": uuid4(),
        "sha256": "a" * 64,
        "filename": "synthetic.txt",
        "location": {"kind": "text", "line_start": 1},
        "index_version": "native-lexical-v1",
        "char_start": 0,
        "preview_url": "/source",
    }


def test_quote_must_match_a_supplied_source_exactly():
    source = hit()
    result = validate_answer(
        json.dumps(
            {
                "insufficient": False,
                "claims": [{"source_id": source["source_id"], "quote": "120"}],
            }
        ),
        [source],
    )
    assert result["claims"][0]["char_start"] == 12
    for source_id, quote in [
        (str(uuid4()), "120"),
        (source["source_id"], "Revenue was 999."),
    ]:
        with pytest.raises(GatewayError):
            validate_answer(
                json.dumps(
                    {
                        "insufficient": False,
                        "claims": [{"source_id": source_id, "quote": quote}],
                    }
                ),
                [source],
            )


@pytest.mark.parametrize(
    "raw",
    [
        "broken",
        '{"insufficient":false,"claims":[]}',
        '{"insufficient":true,"claims":[],"secret":"x"}',
    ],
)
def test_invalid_or_extra_model_fields_are_rejected(raw):
    with pytest.raises(GatewayError):
        validate_answer(raw, [])


def test_instruction_hash_covers_the_actual_runtime_contract():
    import hashlib

    req = make_request(assemble("retrieval"), "سؤال", [hit("الإيراد ١٢٠")])
    assert (
        req.instructions.sha256
        == hashlib.sha256(req.instructions.text.encode()).hexdigest()
    )
    assert (
        input_bound(req)
        >= len(req.instructions.text.encode()) + len(req.question.encode()) + 512
    )


def test_search_normalization_does_not_change_source_or_offsets():
    source = "إِجمالي المبيعات ١٢٠" + "x" * 1200
    version = uuid4()
    first = list(chunks(version, [{"ordinal": 1, "content": source}]))
    assert first == list(chunks(version, [{"ordinal": 1, "content": source}]))
    assert first[0]["content"] == source[:512]
    assert normalize("إِجمالي ١٢٠") == "اجمالي 120"
    assert query_terms("ما هي المبيعات؟") == ["المبيعات"]
    for chunk in first:
        assert source[chunk["char_start"] : chunk["char_end"]] == chunk["content"]


@pytest.mark.asyncio
async def test_ollama_uses_pinned_model_cpu_limits_and_ignores_reasoning():
    digest, captured = "a" * 64, []
    req = make_request(assemble("retrieval"), "Revenue?", [hit()])

    def respond(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "runtime_version": "test",
                    "models": [{"name": "local", "digest": digest}],
                },
            )
        captured.append(json.loads(request.content))
        body = (
            json.dumps(
                {
                    "message": {
                        "thinking": "private",
                        "content": '{"insufficient":true,"claims":[]}',
                    }
                }
            )
            + "\n"
        )
        body += (
            json.dumps(
                {
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 100,
                    "eval_count": 10,
                }
            )
            + "\n"
        )
        return httpx.Response(200, text=body)

    gateway = OllamaGateway(
        "http://local", "local", digest, transport=httpx.MockTransport(respond)
    )
    events = [event async for event in gateway.stream(req)]
    assert "private" not in "".join(event.text for event in events)
    assert events[-1].usage["prompt_tokens"] == 100
    assert (
        captured[0]["options"]["num_gpu"] == 0
        and captured[0]["options"]["num_ctx"] == 4096
    )
    assert captured[0]["think"] is False and captured[0]["keep_alive"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["digest", "truncated", "usage", "missing_done"])
async def test_ollama_fails_closed_on_checkpoint_and_completion_failures(failure):
    digest = "a" * 64

    def respond(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "local",
                            "digest": "b" * 64 if failure == "digest" else digest,
                        }
                    ]
                },
            )
        value = {
            "message": {"content": "{}"},
            "done": failure != "missing_done",
            "done_reason": "length" if failure == "truncated" else "stop",
            "prompt_eval_count": 9000 if failure == "usage" else 1,
            "eval_count": 1,
        }
        return httpx.Response(200, text=json.dumps(value) + "\n")

    gateway = OllamaGateway(
        "http://local", "local", digest, transport=httpx.MockTransport(respond)
    )
    with pytest.raises(GatewayError):
        _ = [
            event
            async for event in gateway.stream(
                make_request(assemble("retrieval"), "Question", [hit()])
            )
        ]
