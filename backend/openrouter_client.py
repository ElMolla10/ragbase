"""Single BYOK routing policy shared by sync/async SDK generation paths.

provider.only prevents other providers. Never-use-shared-capacity must also be
enabled for every model on each allowed provider in the OpenRouter workspace.
"""

import logging
import os
from pathlib import Path

BASE_URL = "https://openrouter.ai/api/v1"
MODELS = (
    "openai/gpt-oss-120b",
    "google/gemini-3.8-flash",
)
PROVIDERS = ("groq", "google-ai-studio")
logger = logging.getLogger(__name__)


def api_key():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    # Preserve the existing secret-file deployment; never copy keys into images.
    if not value and os.getenv("OPENROUTER_API_KEY_FILE"):
        try:
            with Path(os.environ["OPENROUTER_API_KEY_FILE"]).open(
                encoding="utf-8"
            ) as file:
                value = file.read(4097).strip()
        except (OSError, UnicodeError):
            value = ""
    if (
        not 20 <= len(value) <= 4096
        or not value.isascii()
        or any(c.isspace() for c in value)
    ):
        raise ValueError("OPENROUTER_API_KEY is not configured")
    return value


def routing_body(providers=PROVIDERS):
    providers = tuple(providers)
    # Verification may narrow the allowlist, never expand or reorder it.
    if not providers or providers not in (PROVIDERS, PROVIDERS[1:]):
        raise ValueError("Unapproved BYOK provider selection")
    return {
        "models": list(MODELS),
        "provider": {
            "order": list(providers),
            "only": list(providers),
            "allow_fallbacks": True,
        },
    }


def async_client(*, key=None, transport=None):
    import httpx
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=key or api_key(),
        max_retries=0,
        timeout=60,
        http_client=httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False
        ),
    )


def sync_client(*, key=None, transport=None):
    import httpx
    from openai import OpenAI

    return OpenAI(
        base_url=BASE_URL,
        api_key=key or api_key(),
        max_retries=0,
        timeout=60,
        http_client=httpx.Client(
            transport=transport, trust_env=False, follow_redirects=False
        ),
    )


def log_model(response):
    model = getattr(response, "model", None)
    if isinstance(model, str):
        logger.info("OpenRouter fulfilled model=%s", model[:200])


def completion_args(
    *,
    providers=PROVIDERS,
    model=None,
    privacy="nonconfidential-development",
    reasoning=None,
    **kwargs,
):
    route = routing_body(providers)
    primary = model or MODELS[0]
    if primary not in MODELS or "extra_body" in kwargs:
        raise ValueError("Unapproved BYOK routing override")
    if privacy == "zdr":
        route["provider"].update(zdr=True, data_collection="deny")
    elif privacy != "nonconfidential-development":
        raise ValueError("Unknown BYOK privacy policy")
    route["reasoning"] = (
        reasoning if reasoning is not None else {"effort": "low", "exclude": True}
    )
    return {"model": primary, "extra_body": route, **kwargs}


async def async_completion(client, **kwargs):
    result = await client.chat.completions.create(**completion_args(**kwargs))
    if not kwargs.get("stream"):
        log_model(result)
    return result


def sync_completion(client, **kwargs):
    result = client.chat.completions.create(**completion_args(**kwargs))
    if not kwargs.get("stream"):
        log_model(result)
    return result
