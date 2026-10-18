"""Native Ollama adapter behind a server-approved relay; never pulls models."""

import asyncio
import json

import httpx

from backend.answers.grounding import QuotedAnswer, input_bound
from backend.gateway.contracts import GatewayError, GatewayEvent


class OllamaGateway:
    def __init__(
        self, base_url, model, digest, *, transport=None, timeout=90, context=4096
    ):
        self.base_url, self.model, self.digest = base_url.rstrip("/"), model, digest
        self.transport, self.timeout, self.context = transport, timeout, context

    async def stream(self, request):
        request.validate()
        if input_bound(request) + request.max_output_tokens > self.context:
            raise GatewayError("Request exceeds conservative context budget")
        payload = {
            "model": self.model,
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
            "format": QuotedAnswer.model_json_schema(),
            "stream": True,
            "think": False,
            "keep_alive": 0,
            "options": {
                "num_ctx": self.context,
                "num_predict": request.max_output_tokens,
                "num_gpu": 0,
                "num_thread": 2,
                "temperature": 0,
                "seed": 0,
            },
        }
        try:
            async with (
                asyncio.timeout(self.timeout),
                httpx.AsyncClient(
                    timeout=self.timeout,
                    transport=self.transport,
                    trust_env=False,
                    follow_redirects=False,
                ) as client,
            ):
                tags = await client.get(self.base_url + "/api/tags")
                if tags.status_code != 200 or len(tags.content) > 65536:
                    raise GatewayError("Local model inventory unavailable")
                models = tags.json().get("models", [])
                if not any(
                    item.get("name") == self.model and item.get("digest") == self.digest
                    for item in models
                ):
                    raise GatewayError(
                        "Pinned local checkpoint is unavailable or changed"
                    )
                self.runtime_version = tags.json().get("runtime_version")
                self.details = next(
                    item.get("details", {})
                    for item in models
                    if item.get("name") == self.model
                )
                async with client.stream(
                    "POST", self.base_url + "/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        raise GatewayError("Local generation unavailable")
                    size, done = 0, False
                    async for line in response.aiter_lines():
                        if len(line) > 65536:
                            raise GatewayError("Model event exceeds size limit")
                        if not line:
                            continue
                        value = json.loads(line)
                        if value.get("error") or value.get("message", {}).get(
                            "tool_calls"
                        ):
                            raise GatewayError("Unsupported model event")
                        content = value.get("message", {}).get("content", "")
                        if not isinstance(content, str):
                            raise GatewayError("Invalid model content")
                        size += len(content.encode())
                        if size > 8192:
                            raise GatewayError("Model output exceeds size limit")
                        if content:
                            yield GatewayEvent("text", content)
                        if value.get("done"):
                            if value.get("done_reason") != "stop":
                                raise GatewayError(
                                    "Model answer did not finish normally"
                                )
                            prompt, output = (
                                value.get("prompt_eval_count"),
                                value.get("eval_count"),
                            )
                            if (
                                type(prompt) is not int
                                or type(output) is not int
                                or prompt < 0
                                or output < 0
                                or prompt > input_bound(request)
                                or prompt + output > self.context
                            ):
                                raise GatewayError(
                                    "Tokenizer usage violated the configured budget"
                                )
                            yield GatewayEvent(
                                "done",
                                finish_reason="stop",
                                usage={
                                    "prompt_tokens": prompt,
                                    "completion_tokens": output,
                                    "total_tokens": prompt + output,
                                },
                            )
                            done = True
                            break
                    if not done:
                        raise GatewayError("Model stream ended without completion")
        except (
            httpx.HTTPError,
            TimeoutError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            if isinstance(exc, GatewayError):
                raise
            raise GatewayError("Local generation failed") from exc
