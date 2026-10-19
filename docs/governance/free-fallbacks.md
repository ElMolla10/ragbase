# Ordered free model fallbacks

18 October 2026. Constitution 4.1.0. Owner requested multiple free fallbacks.

The server-owned text candidate order is:

1. `google/gemma-4-31b-it:free` — primary
2. `google/gemma-4-26b-a4b-it:free`
3. `nvidia/nemotron-3-super-120b-a12b:free`
4. `nvidia/nemotron-3.5-super-vl:free`
5. `dots-studio/dots-3-note-preview:free`
6. `xingchen-agi/xing4.0-29b-a4b:free`

OpenRouter's public catalog advertised zero input/output token prices, text
output, JSON mode and optional reasoning for these candidates when checked.
They are not ranked by measured Arabic accuracy. Small/mandatory-reasoning and
non-JSON candidates are excluded from this bounded quote path.

`backend/gateway/free_models.py` is the reviewed registry. The gateway supplies
Gemma as `model` and the other five as the ordered `models` array. OpenRouter
handles failover for upstream model/provider errors before streaming begins.
It is not an application retry loop. The existing total transport deadline and
limits remain, so attempting every candidate is not guaranteed.

All candidates share JSON-object mode, strict application schema/evidence checks,
zero input/output price ceiling and the selected privacy policy. The relay rejects
modified, reordered, paid or arbitrary lists supplied by callers. There is no
fallback after invalid JSON or forged evidence, no privacy relaxation, no paid
fallback and no new API keys/accounts. Actual response model/provider identifiers
remain request-local metadata; health also exposes the configured candidate list.

This remains nonconfidential development/testing only under the authorized
standard provider policies. NVIDIA's free endpoints explicitly prohibit
confidential/personal uploads. Stricter ZDR selection still filters out ineligible
endpoints. Image inputs are not enabled by this text fallback registry.

Fallbacks help individual endpoint/model failures; **they do not bypass
OpenRouter's account-wide daily free quota**. The last quota check reported zero
remaining. No live inference requests were made for this change. Live failover,
Arabic quality, latency and all production acceptance remain pending.

Validation: 149 unit/contract/parser tests and 58 actual PostgreSQL integration
tests pass. Checks cover the ordered free-only payload, relay rejection of
tampered lists, preserved zero-price/privacy controls, actual fallback-model
provenance, access denials, rejection of forged fallback quotations and quota
error handling. Mocked responses establish application behavior, not an upstream
failover result. Lint, format, policy assembly and Compose checks pass.

References: [OpenRouter model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks),
[account/free limits](https://openrouter.ai/docs/api_reference/limits),
[live catalog](https://openrouter.ai/api/v1/models).
