"""Deterministic development output, not an answer-quality benchmark."""

from collections.abc import AsyncIterator

from backend.gateway.contracts import GatewayEvent, GatewayRequest


class FakeGateway:
    async def stream(self, request: GatewayRequest) -> AsyncIterator[GatewayEvent]:
        request.validate()
        yield GatewayEvent("text", "[DEVELOPMENT FIXTURE — not a model answer]")
        yield GatewayEvent("done", finish_reason="stop")
