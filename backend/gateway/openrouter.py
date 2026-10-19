"""Explicit hosted generation through a fixed, credential-isolating relay."""

from backend.answers.grounding import REMOTE_BUDGET, input_bound
from backend.gateway.contracts import GatewayError
from backend.gateway.http import LocalHTTPGateway
from backend.openrouter_client import MODELS, routing_body

MODEL = MODELS[0]
RELAY = "http://openrouter-relay:8002"


class OpenRouterGateway(LocalHTTPGateway):
    def __init__(self, base_url=RELAY, *, privacy="zdr", **kwargs):
        if base_url != RELAY:
            raise GatewayError("OpenRouter requires the approved relay")
        if privacy not in {"zdr", "nonconfidential-development"}:
            raise GatewayError("Unknown remote privacy policy")
        self.privacy = privacy
        kwargs.setdefault("timeout", 65)
        super().__init__(base_url + "/v1", **kwargs)

    def payload(self, request):
        payload = super().payload(request)
        payload.update(
            model=MODEL,
            temperature=0,
            # OpenRouter tries these in order when the selected model fails
            # before a response starts. All retain identical request controls.
            models=list(MODELS),
            reasoning={"effort": "low", "exclude": True},
            provider=routing_body()["provider"],
            # JSON mode has no provider schema guarantee. Trusted application
            # validation still enforces schema, source identity and quotations.
            response_format={"type": "json_object"},
        )
        if self.privacy == "zdr":
            payload["provider"].update(zdr=True, data_collection="deny")
        if input_bound(request) + request.max_output_tokens > REMOTE_BUDGET:
            raise GatewayError("Remote request exceeds development byte budget")
        return payload

    def routing(self, data, provenance):
        for key in ("model", "provider"):
            value = data.get(key)
            if value is not None:
                if not isinstance(value, str) or not 1 <= len(value) <= 200:
                    raise GatewayError("Invalid routing metadata")
                if key in provenance and provenance[key] != value:
                    raise GatewayError("Routing metadata changed during generation")
                provenance[key] = value

    def validate_completion(self, request, usage):
        if usage and (
            usage["completion_tokens"] > request.max_output_tokens
            or usage["total_tokens"]
            != usage["prompt_tokens"] + usage["completion_tokens"]
            or usage["total_tokens"] > REMOTE_BUDGET
        ):
            raise GatewayError("Provider usage exceeds development budget")
