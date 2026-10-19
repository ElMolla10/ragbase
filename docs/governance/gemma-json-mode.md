# Free Gemma and application JSON validation

> This initial Gemma record remains historical evidence. The current deployment
> adds [ordered free fallbacks](free-fallbacks.md) under constitution 4.1.0.

18 October 2026. Constitution 4.0.0; replaces the previous auto-router selection.

The owner selected `google/gemma-4-31b-it:free` and explicitly authorized its
standard provider data policy for nonconfidential development/testing. This is
not company-confidential approval or a zero-retention guarantee. Privacy remains
an explicit server configuration; strict ZDR is still available and never relaxed
automatically. Credentials remain in the isolated relay.

The gateway now requests JSON-object mode instead of provider-enforced JSON
schema. The trusted application validates JSON syntax, duplicate keys, strict
types, extra keys, claim counts, source identifiers and exact source quotations
before releasing an answer. Raw tokens/private reasoning are never user output.
There is no permissive parser, invented field repair, second model or paid retry.
Schema validation cannot establish relevance or extraction correctness by itself.

The fixed :free identifier and zero input/output price ceiling prevent paid
fallback. Provider failures and free quota exhaustion remain explicit failures.
HTTP 429 produces a clear quota/rate-limit message without logging provider bodies
or credentials. The existing 4,096-unit development byte ceiling, 1,024 remote
output-token allowance, transport deadlines, auth/RLS and revocation checks remain.
Gemma reasoning is explicitly disabled for the current exact-quote path.

Validation: 145 enterprise unit/contract/parser tests and 56 PostgreSQL integration
tests pass. Cases include malformed/truncated/prose/fenced JSON, duplicate keys,
type coercion, extra fields, forged quotes, unauthorized documents, fixed free
configuration, explicit privacy choice, relay enforcement and a normalized quota
error through the answer endpoint. Ruff, policy and Compose checks also pass.

Live synthetic English, Arabic and mixed-question attempts were blocked by HTTP
429. The no-evidence path passed without calling the model. OpenRouter's key
metadata reported `used=54`, `limit=50`, `remaining=0`; no reset timestamp was
provided. These are blocked checks, not failed Arabic-quality scores or passing
live generation results. No paid fallback or confidential data was used. Stop live
requests until the provider quota resets; no purchase is required by this setup.

The configured endpoint is running, but Gemma live JSON compatibility and answer
quality remain pending because of the quota. Images, analysis, general summaries
and production qualification are not enabled by this change. Gemma's advertised
image support is not evidence that RAGBase's image path has been implemented.

References: [Gemma free capabilities](https://openrouter.ai/google/gemma-4-31b-it:free),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[free limits](https://openrouter.ai/docs/api_reference/limits).
