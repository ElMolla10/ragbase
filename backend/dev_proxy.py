"""Development reverse proxy for fixed health and authenticated storage routes."""

import asyncio
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


async def forward(path: str) -> JSONResponse:
    # Fixed upstream and route; never a user-controlled URL or general proxy.
    try:
        async with httpx.AsyncClient(
            timeout=5, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.get("http://api:8000" + path)
            if response.status_code not in {200, 503} or len(response.content) > 65536:
                raise HTTPException(
                    502, "Development API returned an invalid health response"
                )
            return JSONResponse(response.json(), status_code=response.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, "Development API unavailable") from exc


@app.get("/health/live")
async def live() -> JSONResponse:
    return await forward("/health/live")


@app.get("/health/ready")
async def ready() -> JSONResponse:
    return await forward("/health/ready")


@app.api_route("/api/{rest:path}", methods=["GET", "POST"])
async def storage_proxy(rest: str, request: Request):
    path = request.url.path
    if not re.fullmatch(
        r"/api/workspaces(?:/[0-9a-f-]{36}(?:/documents|/versions/[0-9a-f-]{36}/file))?",
        path,
    ):
        raise HTTPException(404, "Route not available")
    data = bytearray()
    try:
        async with asyncio.timeout(15):
            async for chunk in request.stream():
                if len(data) + len(chunk) > 5_000_000:
                    raise HTTPException(413, "Development upload exceeds byte limit")
                data.extend(chunk)
        forwarded = {
            key: request.headers[key]
            for key in ("authorization", "x-filename", "content-type")
            if key in request.headers
        }
        async with (
            httpx.AsyncClient(
                timeout=30, trust_env=False, follow_redirects=False
            ) as client,
            client.stream(
                request.method,
                "http://api:8000" + path,
                content=bytes(data),
                headers=forwarded,
            ) as upstream,
        ):
            if 300 <= upstream.status_code < 400:
                raise HTTPException(502, "Unexpected upstream redirect")
            body = bytearray()
            async for chunk in upstream.aiter_bytes():
                if len(body) + len(chunk) > 5_000_000:
                    raise HTTPException(502, "Development response exceeds byte limit")
                body.extend(chunk)
            headers = {
                key: upstream.headers[key]
                for key in (
                    "content-type",
                    "content-disposition",
                    "www-authenticate",
                )
                if key in upstream.headers
            }
            headers.update(
                {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
            )
            return Response(
                bytes(body), status_code=upstream.status_code, headers=headers
            )
    except (httpx.HTTPError, TimeoutError) as exc:
        raise HTTPException(502, "Development API unavailable") from exc
