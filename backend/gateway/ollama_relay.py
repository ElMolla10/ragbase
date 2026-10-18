"""Optional development relay. Only the preconfigured host model is reachable.

No model pull/create/delete endpoint, arbitrary URLs, storage or DB credentials.
This bridge is not an approved production network boundary.
"""

import json
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.answers.grounding import QuotedAnswer

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
MODEL = os.environ.get("DEV_OLLAMA_MODEL", "qwen3:1.7b")
HOST = "http://host.docker.internal:11434"


@app.get("/api/tags")
async def tags():
    try:
        async with httpx.AsyncClient(
            timeout=5, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.get(HOST + "/api/tags")
            version = await client.get(HOST + "/api/version")
        if response.status_code != 200 or len(response.content) > 65536:
            raise HTTPException(503, "Local inventory unavailable")
        return JSONResponse(
            {
                "runtime_version": version.json().get("version"),
                "models": [
                    {key: item[key] for key in ("name", "digest", "details")}
                    for item in response.json()["models"]
                    if item["name"] == MODEL
                ],
            }
        )
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(503, "Local inventory unavailable") from exc


@app.post("/api/chat")
async def chat(request: Request):
    data = bytearray()
    async for piece in request.stream():
        data.extend(piece)
        if len(data) > 24000:
            raise HTTPException(413, "Model request too large")
    try:
        payload = json.loads(data)
        if (
            payload.get("model") != MODEL
            or payload.get("tools")
            or len(payload.get("messages", [])) != 2
        ):
            raise ValueError
        if (
            [message.get("role") for message in payload["messages"]]
            != ["system", "user"]
            or any(
                set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                for message in payload["messages"]
            )
            or payload.get("format") != QuotedAnswer.model_json_schema()
        ):
            raise ValueError
        payload = {key: payload[key] for key in ("messages", "format")}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise HTTPException(400, "Unsupported model request") from exc
    payload.update(
        {
            "model": MODEL,
            "think": False,
            "stream": True,
            "keep_alive": 0,
            "options": {
                "num_ctx": 4096,
                "num_predict": 256,
                "num_gpu": 0,
                "num_thread": 2,
                "temperature": 0,
                "seed": 0,
            },
        }
    )

    async def generate():
        try:
            async with (
                httpx.AsyncClient(
                    timeout=90, trust_env=False, follow_redirects=False
                ) as client,
                client.stream("POST", HOST + "/api/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    yield b'{"error":"local_model_unavailable"}\n'
                    return
                size = 0
                async for piece in response.aiter_bytes():
                    size += len(piece)
                    if size > 262144:
                        yield b'{"error":"model_stream_limit"}\n'
                        return
                    yield piece
        except httpx.HTTPError:
            yield b'{"error":"local_model_unavailable"}\n'

    return StreamingResponse(generate(), media_type="application/x-ndjson")
