"""Single-process demo limits; no external limiter service required."""

import hmac
import math
import time
from collections import deque

from fastapi import Request
from starlette.responses import JSONResponse

from backend.config import get_settings

settings = get_settings()


class RateLimiter:
    """Sliding window per client and endpoint, pruning expired client entries."""

    def __init__(self):
        self.events: dict[tuple[str, str], deque[float]] = {}

    def retry_after(self, client: str, path: str, limit: int) -> int:
        if limit == 0:
            return 0
        now = time.monotonic()
        cutoff = now - settings.rate_limit_window_seconds
        for key in list(self.events):
            events = self.events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                del self.events[key]
        events = self.events.setdefault((client, path), deque())
        if len(events) >= limit:
            return max(1, math.ceil(events[0] - cutoff))
        events.append(now)
        return 0


limiter = RateLimiter()


async def protect_request(request: Request, call_next):
    """Reject protected requests before multipart parsing or expensive work."""
    path = request.url.path
    is_ingest = request.method == "POST" and path == "/ingest"
    is_query = request.method == "POST" and path == "/query"
    is_delete = request.method == "DELETE" and path.startswith("/documents/")
    if settings.write_api_key and (is_ingest or is_delete):
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied.encode(), settings.write_api_key.encode()):
            return JSONResponse({"detail": "Valid X-API-Key required"}, status_code=401)
    if is_ingest:
        # Multipart overhead is bounded separately from the actual file limit.
        length = request.headers.get("content-length")
        if length:
            try:
                too_large = int(length) > settings.max_upload_bytes + 64 * 1024
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
            if too_large:
                return JSONResponse({"detail": "Upload too large"}, status_code=413)
    if is_ingest or is_query:
        client = request.client.host if request.client else "unknown"
        limit = settings.ingest_rate_limit if is_ingest else settings.query_rate_limit
        retry = limiter.retry_after(client, path, limit)
        if retry:
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry)},
            )
    return await call_next(request)
