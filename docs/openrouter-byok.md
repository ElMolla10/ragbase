# OpenRouter BYOK generation

The public app now uses the shared OpenAI SDK adapter for both complete and
streamed answers. Primary model: `google/gemini-3.8-flash`; fallback:
`openai/gpt-oss-120b`. Every request restricts provider `only` and `order` to
`google-ai-studio`, `groq`, with fallback enabled. Mistral is excluded. Low
reasoning is enabled; Gemini requires it. Existing answer function signatures
and return types are preserved, and the selected model is logged without prompts.

In the same OpenRouter workspace as your application key, add valid Google AI
Studio and Groq provider keys as prioritized BYOK keys. Select **Never use shared
capacity for any model on this provider** on both providers. The request allowlist
alone cannot enforce that workspace setting. Restrict provider billing to the
intended free plan; quotas and OpenRouter BYOK fees/allowances are separate.
Successful HTTP responses do not establish zero cost or unlimited use.

For Docker Compose, initialize the private credential before starting services:

```powershell
python scripts/configure_openrouter.py
docker compose up -d --build
```

The key file stays in ignored `.local/` and is mounted only in the backend as a
Compose secret. Configure it independently in each checkout; never copy a secret
into Git. The optional `OPENROUTER_API_KEY` environment variable takes precedence.
For non-Compose deployments, set `OPENROUTER_API_KEY` in the backend service or
ignored `.env`, or set `OPENROUTER_API_KEY_FILE` to a private mounted file.
Legacy Groq/Ollama configuration fields remain parseable but no longer select
generation routes. The Compose key-file prerequisite still applies even when an
environment key is supplied.

Questions, conversation history and retrieved document text are sent to
OpenRouter and its selected provider. Use only nonconfidential development data
until privacy and company deployment are separately approved. This public app
does not acquire the enterprise branch's authorization/storage controls.

Run mocked adapter tests without model weights:

```powershell
python -m pytest tests_byok -q
```

For live synthetic checks, configure the credential and run:

```powershell
$env:OPENROUTER_API_KEY_FILE = ".local/openrouter-api-key"
python scripts/test_openrouter_fallbacks.py
```

Probes cover standard routing, Groq only, and Gemini-primary with only Groq
allowed to force same-request model fallback. This does not inject a real 429 or
qualify Arabic, image handling, latency or production reliability. No local
generation weights are required; existing local embedding setup is unchanged.
