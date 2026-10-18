"""Bounded local chat-completions adapter; text streaming only for now."""

import asyncio
import json
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx

from backend.gateway.contracts import (
    GatewayError,
    GatewayEvent,
    GatewayRequest,
    UnsupportedCapabilityError,
)


class LocalHTTPGateway:
    """Transport is injectable for contracts; deployment owns endpoint approval.

    The URL is server configuration, never a request field. Network policy is the
    security boundary for egress; syntax validation alone cannot approve an IP.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or timeout <= 0
        ):
            raise GatewayError("Invalid server-owned generation configuration")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.transport = transport

    async def stream(self, request: GatewayRequest) -> AsyncIterator[GatewayEvent]:
        request.validate()
        payload = {
            "model": "company-assistant",
            "messages": [
                {"role": "system", "content": request.instructions.text},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": request.question,
                            "untrusted_evidence": request.evidence,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "max_tokens": request.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        usage = None
        finish = None
        seen_done = False
        try:
            async with (
                asyncio.timeout(self.timeout),
                httpx.AsyncClient(
                    timeout=self.timeout,
                    transport=self.transport,
                    follow_redirects=False,
                    trust_env=False,
                ) as client,
                client.stream("POST", self.url, json=payload) as response,
            ):
                if response.status_code != 200:
                    raise GatewayError(
                        f"Generation endpoint returned HTTP {response.status_code}"
                    )
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise GatewayError(
                        "Generation endpoint did not return an event stream"
                    )
                async for line in response.aiter_lines():
                    if len(line) > 65536:
                        raise GatewayError("Generation event exceeds size limit")
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        seen_done = True
                        break
                    data = json.loads(raw)
                    if "error" in data:
                        raise GatewayError("Generation endpoint reported an error")
                    if data.get("usage") is not None:
                        candidate = data["usage"]
                        keys = {
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        }
                        if not isinstance(candidate, dict) or not keys.issubset(
                            candidate
                        ):
                            raise GatewayError("Invalid provider usage")
                        if any(
                            type(candidate[k]) is not int or candidate[k] < 0
                            for k in keys
                        ):
                            raise GatewayError("Invalid provider usage")
                        usage = {key: candidate[key] for key in sorted(keys)}
                    for choice in data.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise GatewayError(
                                "Multiple generation choices are unsupported"
                            )
                        delta = choice.get("delta") or {}
                        if delta.get("tool_calls") or delta.get("function_call"):
                            raise UnsupportedCapabilityError(
                                "Unexpected provider tool request"
                            )
                        # Reasoning fields intentionally never enter user text.
                        text = delta.get("content")
                        if text is not None and not isinstance(text, str):
                            raise GatewayError("Invalid text delta")
                        if text:
                            yield GatewayEvent("text", text)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                if not seen_done or not finish:
                    raise GatewayError("Generation stream ended without completion")
                if finish not in {"stop", "length", "content_filter"}:
                    raise GatewayError("Unsupported provider completion reason")
                yield GatewayEvent("done", finish_reason=finish, usage=usage)
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise GatewayError("Local generation timed out") from exc
        except httpx.HTTPError as exc:
            raise GatewayError("Local generation unavailable") from exc
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
            raise GatewayError("Malformed generation event") from exc
