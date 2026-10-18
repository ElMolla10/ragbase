"""Development health-only reverse proxy; internal HTTPS is later server work."""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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
