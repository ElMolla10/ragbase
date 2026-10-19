# OpenRouter BYOK integration

Updated 19 October 2026. This supersedes the public free-model pool described in
earlier development records. Hosted generation is authorized only for
nonconfidential development/testing. Default enterprise generation stays disabled.

All active generation adapters use `backend/openrouter_client.py`, the OpenAI
Python SDK, and the fixed `https://openrouter.ai/api/v1` base URL. Legacy answer
function signatures and enterprise evidence/schema checks are preserved.

The active candidate order is GPT-OSS 120B (`openai/gpt-oss-120b`) followed by Gemini 3.8 Flash
(`google/gemini-3.8-flash`). The owner removed Mistral on 19 October 2026; neither Mistral's model
nor its provider is permitted by the shared router. No replacement was added.

Every upstream completion includes `models` and provider `order`, `only`, and
`allow_fallbacks: true`. Production request allowlists are exactly
`groq`, `google-ai-studio`; verification probes may narrow that list.
Unknown models, arbitrary extra-body overrides and expanded provider lists are
rejected. OpenRouter performs fallback inside one request; the SDK adds no second
retry loop. This is request-level routing, not continuation of a partially
delivered answer. Model names are logged without prompt or answer contents.

`provider.only` blocks other providers, but cannot prove use of free keys or
prevent shared capacity on an allowed provider. The owner reports configuring
**Never use shared capacity for any model on this provider** for both
providers. That workspace setting and free-only provider account billing are
operational prerequisites. Provider and OpenRouter plan fees/allowances are
separate; there is no application guarantee of unlimited free inference.

Credentials come from `OPENROUTER_API_KEY` in the environment or ignored `.env`.
The existing Compose secret file remains supported via
`OPENROUTER_API_KEY_FILE`. Never commit credentials. Setup and the isolated relay
Compose command remain in [the earlier setup guide](openrouter.md).

Run the synthetic verification with:

```powershell
python scripts/test_openrouter_fallbacks.py --output .local/byok-verification.json
```

The three probes test the standard Groq/Google chain, Google only, and a GPT-OSS-primary request restricted to Google to force model fallback. They report
status/model and never print answers or keys. HTTP 200 plus an allowed model
establishes reachability, not zero cost or Arabic/company accuracy. No real
documents are used.

Current live evidence (19 October 2026): both updated models individually return
HTTP 200 with visible answers through their intended providers. The old Gemini
2.5 model is unavailable for this Google account; Groq credentials were corrected.
Gemini 3.8 requires reasoning enabled, so both routes use low reasoning and exclude
reasoning text from returned answers. Upstream generation has a bounded 60-second
timeout, with a 65-second API-to-relay envelope. Synthetic tests do not qualify
Arabic, production reliability, images or free-only billing. Workspace shared
capacity settings remain owner-reported. The verification script checks the standard chain, narrowed Groq-only path and forced Google-unavailable path; it does not manufacture a real 429. A live standard request also selected Groq successfully. This proves same-request model fallback, but does not identify why Google was skipped or validate a specific 429 recovery.

References: [OpenRouter BYOK](https://openrouter.ai/docs/guides/overview/auth/byok).

Deployment evidence: enterprise API, worker and relay rebuilt/recreated with the
new model configuration, retaining existing database/artifact volumes. Health
reports GPT-OSS primary and Gemini fallback. All 162 unit/contract and 58 isolated
PostgreSQL integration checks pass, as do lint and policy verification. A live
synthetic PDF question returned the exact quote `Quarter 1 sales: 120`, with the
correct page-1 source preview, in 24.37 seconds. This is one development smoke,
not a production latency or Arabic-quality benchmark. A broad `quarters` query
returned insufficient evidence because current retrieval is lexical; dense
retrieval remains pending.

Current default (constitution 5.3.0): Groq GPT-OSS 120B first on both master and
enterprise; Gemini is fallback. Enterprise services were recreated and health
confirms that order. The updated synthetic PDF quote/source-preview smoke passed
in 3.31 seconds. 163 unit/contract and 58 PostgreSQL integration checks pass.
The earlier fallback probes and timings above describe their original run, not
a freshly injected outage for the reversed chain.

Constitution v6.0.0 enables natural document chat with checked supporting citations. Overview requests and no-keyword-hit questions use authorized document context. Hosted requests have a 16,000-charge development byte envelope including 2,048 reserved output tokens; this is not a validated tokenizer capacity. Temperature is zero. Coverage reports included versus total passages; long documents may receive a partial overview, never a silent complete-review claim. Explanation semantics remain model-generated. Live synthetic overview through Groq passed in 4.53s; 190 unit/contract and 64 PostgreSQL tests pass.

V6.1 correction: natural generation returns cited explanations without copying PDF quotes. The app resolves short aliases to supplied authorized passages and attaches their original text. Explicit legacy quote responses remain strictly validated. The previously failing local document/question now returns through Groq in 3.93s with six valid passage citations. No document text or answer is recorded here. Suite: 192 unit/contract and 64 PostgreSQL checks pass. Unknown source IDs still fail closed.

V6.2 public-style document chat uses a complete answer string with short inline source aliases. Trusted code validates/resolves them into real page/line/row labels. Legacy claim responses remain compatible, but new generation is no longer forced into a fixed series of six statements. CLI defaults to clean prose; --verbose and --preview enable diagnostics/source inspection. 198 unit/contract and 64 PostgreSQL checks pass.
