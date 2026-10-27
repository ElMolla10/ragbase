"""Development reverse proxy for fixed health, browser shell and authenticated routes."""

import asyncio
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Fixed paths only. Coverage is an existing authenticated API, not a new data source.
_API_PATH = re.compile(
    r"/api/workspaces(?:/"
    r"[0-9a-f-]{36}"
    r"(?:/documents"
    r"|/versions/[0-9a-f-]{36}/(?:file|extraction|cancel|process)"
    r"|/sources/[0-9a-f-]{36}"
    r"|/search"
    r"|/answers(?:/[0-9a-f-]{36}/cancel)?"
    r"|/coverage(?:/[0-9a-f-]{36}(?:/cancel)?)?"
    r"))?"
)
_ASSET_NAME = re.compile(r"[A-Za-z0-9._-]{1,200}")
_BROWSER_ASSET_TYPES = {
    "text/javascript",
    "application/javascript",
    "text/css",
    "image/svg+xml",
    "image/png",
    "image/webp",
    "font/woff2",
}


def api_path_allowed(path: str) -> bool:
    return _API_PATH.fullmatch(path) is not None


def browser_path_allowed(path: str) -> bool:
    if path in {"/", "/favicon.svg"}:
        return True
    prefix = "/assets/"
    if not path.startswith(prefix):
        return False
    name = path[len(prefix) :]
    return bool(
        _ASSET_NAME.fullmatch(name) and name not in {".", ".."} and ".." not in name
    )


def streamed_generation(path: str, method: str) -> bool:
    return method == "POST" and (
        path.endswith("/answers") or path.endswith("/coverage")
    )


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


@app.get("/health/generation")
async def generation_health():
    return await forward("/health/generation")


@app.get("/")
async def browser_shell():
    return await forward_browser("/")


@app.get("/favicon.svg")
async def browser_icon():
    return await forward_browser("/favicon.svg")


@app.get("/assets/{name}")
async def browser_asset(name: str):
    return await forward_browser("/assets/" + name)


async def forward_browser(path: str) -> Response:
    if not browser_path_allowed(path):
        raise HTTPException(404, "Route not available")
    try:
        async with httpx.AsyncClient(
            timeout=10, trust_env=False, follow_redirects=False
        ) as client:
            upstream = await client.get("http://api:8000" + path)
        if 300 <= upstream.status_code < 400:
            raise HTTPException(502, "Unexpected upstream redirect")
        if (
            upstream.status_code not in {200, 404, 503}
            or len(upstream.content) > 2_000_000
        ):
            raise HTTPException(
                502, "Development API returned an invalid browser response"
            )
        media = upstream.headers.get("content-type", "").split(";")[0].strip().lower()
        if path == "/":
            allowed = {"text/html", "text/plain"}
        elif path == "/favicon.svg":
            allowed = {"image/svg+xml"}
        else:
            allowed = _BROWSER_ASSET_TYPES
        if upstream.status_code == 200 and media not in allowed:
            raise HTTPException(
                502, "Development API returned an invalid browser response"
            )
        headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        }
        if "content-type" in upstream.headers:
            headers["Content-Type"] = upstream.headers["content-type"]
        if "content-security-policy" in upstream.headers:
            headers["Content-Security-Policy"] = upstream.headers[
                "content-security-policy"
            ]
        return Response(
            upstream.content, status_code=upstream.status_code, headers=headers
        )
    except (httpx.HTTPError, TimeoutError) as exc:
        raise HTTPException(502, "Development API unavailable") from exc


@app.api_route("/api/{rest:path}", methods=["GET", "POST"])
async def storage_proxy(rest: str, request: Request):
    path = request.url.path
    if not api_path_allowed(path):
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
        if streamed_generation(path, request.method):
            client = httpx.AsyncClient(
                timeout=180, trust_env=False, follow_redirects=False
            )
            try:
                upstream = await client.send(
                    client.build_request(
                        "POST",
                        "http://api:8000" + path,
                        content=bytes(data),
                        headers=forwarded,
                    ),
                    stream=True,
                )
                if upstream.status_code != 200:
                    body = await upstream.aread()
                    await upstream.aclose()
                    await client.aclose()
                    return Response(
                        body[:65536],
                        status_code=upstream.status_code,
                        media_type="application/json",
                        headers={
                            "Cache-Control": "no-store",
                            "X-Content-Type-Options": "nosniff",
                        },
                    )
            except BaseException:
                await client.aclose()
                raise

            async def relay():
                size = 0
                try:
                    async for piece in upstream.aiter_bytes():
                        size += len(piece)
                        if size > 262144:
                            yield b'{"event":"error","code":"stream_limit"}\n'
                            return
                        yield piece
                finally:
                    await upstream.aclose()
                    await client.aclose()

            return StreamingResponse(
                relay(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "X-Accel-Buffering": "no",
                },
            )
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
