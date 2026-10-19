> Historical free-pool configuration. The replacement BYOK implementation and
> its pending live/free-access gates are documented in [BYOK integration](openrouter-byok.md).

# Optional free Gemma generation

RAGBase can use `google/gemma-4-31b-it:free` without Ollama or local generation weights.
Choose this mode explicitly instead of the local Ollama override. Core services
still start with generation disabled when neither override is supplied.

Gemma is followed by five reviewed free candidates: Gemma 4 26B A4B, Nemotron 3
Super, Nemotron 3.5 Super VL, Dots3-Note Preview and Xing4.0 29B A4B. OpenRouter
tries them in order for model/provider failures under identical zero-price,
privacy and JSON controls. The actual chosen model is recorded. See the
[fallback order and verification record](governance/free-fallbacks.md).
These fallbacks do **not** bypass the account-wide daily free quota, retry
invalid answers or enable images. Current models remain unqualified candidates.

The owner approved this configuration for **nonconfidential development/testing only** under the provider standard data policy. It does not guarantee zero retention. Do not upload confidential company data for this mode.

Questions, runtime instructions and bounded retrieved text are sent through
OpenRouter to its selected provider. Originals, database, extraction, search,
authentication and application logs remain local. The queued event and answer
metadata disclose external processing. This is a development option, not a
qualified company deployment or an offline inference mode.

DeepSeek Harness (`dsh`, including `ollama launch dsh`) is a separate coding-agent
application. Its web UI is not the model API used here. RAGBase calls OpenRouter
through its own fixed relay, without harness filesystem, shell or web tools.
No existing harness or Ollama settings are changed by this feature.

## Setup on your laptop

From the repository directory, create an API key at
[OpenRouter keys](https://openrouter.ai/settings/keys). Set a spending limit there.
Then run:

```powershell
python scripts/configure_openrouter.py
```

Paste the key into the terminal's hidden prompt, then press Enter. The script
writes `.local/openrouter-api-key`; never paste the key into chat, a command line,
source code or a tracked `.env` file. `.local/` is excluded from Git and Docker
builds. The Compose secret is mounted only in the relay at
`/run/secrets/openrouter_api_key`. Avoid enabling prompt logging in your account.

If enterprise development has not been initialized, first run
`python scripts/init_enterprise_dev.py`. Existing development data is retained.

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml -f compose.enterprise-openrouter.yaml up -d --build --remove-orphans
```

Use **one generation override at a time**. `--remove-orphans` removes this
enterprise project's previous Ollama relay when switching; it does not remove
database volumes or the host's Ollama installation. No local model is loaded or
downloaded. The API remains at `http://127.0.0.1:8800`.

Check configuration with:

```powershell
Invoke-RestMethod http://127.0.0.1:8800/health/generation
```

`credential_configured: true` means the relay can read a plausibly formatted key.
It does **not** prove that the key is valid, has credits, or has an eligible model.
This health check never performs billable inference. The existing authenticated
answer endpoint/`scripts/query_enterprise.py` uses the selected mode; there is no
graphical settings page yet. Start live checks with synthetic documents only. The current free quota was exhausted during setup (HTTP 429); no automatic retries or paid fallbacks are used. Account quota metadata reported 54 used, limit 50, remaining 0 at the time of the check. No reset time was supplied. Wait for the provider reset rather than buying credits to use this development configuration.

## Switch back or rotate the key

To disable generation, use just the base file:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --build --remove-orphans
```

For local generation, use `compose.enterprise-ollama.yaml` instead of the remote
override. Switching generation does not require re-ingesting documents. To rotate
the key, revoke the old key in OpenRouter, rerun `configure_openrouter.py`, then
recreate the relay with the remote Compose files and `up -d --force-recreate openrouter-relay`.

## Current limits and acceptance

- Fixed HTTPS upstream: `https://openrouter.ai/api/v1/chat/completions`. Neither
  users nor documents can select URLs, providers, tools or request extensions.
- The relay receives no database credentials, document mounts or host port.
  The API/worker/database retain the internal network. Production requires host
  firewall/proxy enforcement of the relay's exact egress destination; its current
  Docker edge network alone is not a destination firewall.
- Requests use `response_format: {"type":"json_object"}`. Gemma does not enforce a JSON schema at the provider. Trusted code strictly validates JSON syntax, duplicate keys, types, extra fields, source IDs and exact quotes before releasing an answer. Truncated/prose/fenced responses fail rather than being rewritten or guessed.
- The fixed :free model and `provider.max_price` of zero for input/output exclude paid fallback. Free requests remain quota-limited.
- The explicit `nonconfidential-development` privacy choice allows standard provider data collection and does not request ZDR. Selecting `zdr` instead retains strict privacy routing and may reject all free Gemma endpoints. No automatic policy relaxation occurs.
- Existing exact-source quote/citation validation, RLS authorization, queues,
  revocation checks and local cancellation remain in force. There is no enabled
  remote vision, web search, tool execution, arithmetic or exhaustive summarizing.
- The model ID is fixed to Gemma 4 31B :free. The actual response model/provider identifiers are still recorded; absent identifiers and unexposed revisions are unknown. A hosted identifier does not pin unpublished weights or establish quality.
- Development retains a 4,096-unit byte-based prompt/output admission ceiling and
  1,024 remote output-token cap (local generation remains capped at 256). The byte ceiling is **not** a validated context capacity
  for all routed tokenizers. Reported token usage is retained and checked when
  present. Arabic, tokenization and route-quality acceptance remain pending.
- API and relay have a 30-second deadline; the relay bounds upstream bytes.
  Disconnect/cancel closes local transports. It cannot guarantee a provider stops
  processing already received data or cancels charges.
- Missing credentials, invalid keys, rate limits and provider outages cause an
  explicit generation failure; independent document/search features remain usable.
  There is no fallback to another processing mode.

Automated tests use fake HTTP and synthetic credentials. They make no OpenRouter requests or incur inference cost. The current live Gemma smoke was blocked by exhausted daily free quota (HTTP 429); no live answer/Arabic quality claim follows. See [the Gemma implementation record](governance/gemma-json-mode.md). The previous auto-router results are historical. Gemma reasoning is explicitly disabled for this bounded quote path.

Primary references: [OpenRouter API](https://openrouter.ai/docs/quickstart),
[Gemma free capabilities](https://openrouter.ai/google/gemma-4-31b-it:free),
[structured output](https://openrouter.ai/docs/guides/features/structured-outputs),
[ZDR routing](https://openrouter.ai/docs/guides/features/zdr),
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).
