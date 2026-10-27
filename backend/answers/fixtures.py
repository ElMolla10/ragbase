"""Explicit integration fixture. Never evidence of model answer quality."""

import json

from backend.analysis.planner import MARKER, fixture_plan
from backend.gateway.contracts import GatewayEvent


class FixtureQuoteGateway:
    async def stream(self, request):
        request.validate()
        # Only the calculation planner's trusted instructions select this shape.
        # A document that mentions the marker cannot change the quote path.
        if MARKER in request.instructions.text:
            profile = json.loads(request.evidence[0]).get("untrusted_profile") or {}
            yield GatewayEvent(
                "text", json.dumps(fixture_plan(request.question, profile))
            )
            yield GatewayEvent("done", finish_reason="stop")
            return
        # Only the coverage workflow's trusted instructions select this shape.
        # Document text that happens to contain the marker stays on the quote path.
        if "COVERAGE_WORKFLOW" in request.instructions.text:
            aliases = []
            for item in request.evidence:
                payload = json.loads(item)
                alias = payload.get("source_id")
                if isinstance(alias, str):
                    aliases.append(alias)
            if not aliases:
                result = {
                    "insufficient": True,
                    "answer": "No supplied sections established an answer.",
                }
            else:
                result = {
                    "insufficient": False,
                    "answer": "Reviewed only the supplied sections. "
                    + " ".join(f"[{alias}]" for alias in aliases),
                }
            yield GatewayEvent("text", json.dumps(result))
            yield GatewayEvent("done", finish_reason="stop")
            return
        if not request.evidence:
            result = {"insufficient": True, "claims": []}
        else:
            source = json.loads(request.evidence[0])
            result = {
                "insufficient": False,
                "claims": [
                    {"source_id": source["source_id"], "quote": source["text"][:200]}
                ],
            }
        yield GatewayEvent("text", json.dumps(result))
        yield GatewayEvent("done", finish_reason="stop")
