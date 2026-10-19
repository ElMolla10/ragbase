"""Opt-in hosted text relay: fixed upstream, no storage, tools or local models.

Only this container receives the secret file. Provider errors never enter logs
or downstream responses. Production still requires verified host egress policy.
"""

import asyncio
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.gateway.contracts import GatewayRequest
from backend.gateway.openrouter import OpenRouterGateway
from backend.openrouter_client import api_key, async_client, async_completion, log_model
from backend.policy import Instructions

UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"


def read_key():
    try:
        return api_key()
    except ValueError:
        return None


def create_relay(*, transport=None):
    gateway = OpenRouterGateway(privacy=os.environ.get("OPENROUTER_PRIVACY", "zdr"))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health/configuration")
    async def health():
        configured = read_key() is not None
        return JSONResponse(
            {"credential_configured": configured, "provider_verified": False},
            status_code=200 if configured else 503,
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        data = bytearray()
        async for piece in request.stream():
            data.extend(piece)
            if len(data) > 24000:
                raise HTTPException(413, "Model request too large")
        try:
            payload = json.loads(data)
            messages = payload["messages"]
            if (
                len(messages) != 2
                or [m["role"] for m in messages] != ["system", "user"]
                or any(
                    set(m) != {"role", "content"} or not isinstance(m["content"], str)
                    for m in messages
                )
                or type(payload["max_tokens"]) is not int
                or not 1 <= payload["max_tokens"] <= 1024
            ):
                raise ValueError
            body = json.loads(messages[1]["content"])
            if (
                set(body) != {"question", "untrusted_evidence"}
                or not isinstance(body["question"], str)
                or not isinstance(body["untrusted_evidence"], list)
                or any(not isinstance(v, str) for v in body["untrusted_evidence"])
            ):
                raise ValueError
            normalized = GatewayRequest(
                Instructions(messages[0]["content"], "", "", ()),
                body["question"],
                tuple(body["untrusted_evidence"]),
                payload["max_tokens"],
            )
            normalized.validate()
            expected = gateway.payload(normalized)
            if payload != expected:
                raise ValueError
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise HTTPException(400, "Unsupported remote request") from exc
        key = read_key()
        if key is None:
            raise HTTPException(503, "Remote credential is not configured")

        async def generate():
            from openai import APIError, APIStatusError

            stream = None
            try:
                async with (
                    asyncio.timeout(60),
                    async_client(key=key, transport=transport) as client,
                ):
                    kwargs = {
                        k: v
                        for k, v in expected.items()
                        if k not in {"model", "models", "provider"}
                    }
                    stream = await async_completion(
                        client, privacy=gateway.privacy, **kwargs
                    )
                    size, first = 0, True
                    try:
                        async for chunk in stream:
                            if first:
                                log_model(chunk)
                                first = False
                            encoded = (
                                "data: "
                                + chunk.model_dump_json(exclude_none=True)
                                + "\n\n"
                            ).encode()
                            size += len(encoded)
                            if size > 262144:
                                yield b'data: {"error":"remote_stream_limit"}\n\n'
                                return
                            yield encoded
                    finally:
                        await stream.close()
                    yield b"data: [DONE]\n\n"
            except APIStatusError as exc:
                if exc.status_code == 429:
                    yield b'data: {"error":"remote_generation_unavailable","code":"remote_rate_limited"}\n\n'
                else:
                    yield b'data: {"error":"remote_generation_unavailable"}\n\n'
            except (APIError, TimeoutError, ValueError):
                yield b'data: {"error":"remote_generation_unavailable"}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


app = create_relay()
