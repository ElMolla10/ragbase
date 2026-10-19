# RAGBase Enterprise

Development branch for an access-controlled document question-answering and analysis
system, with Arabic and mixed-language evaluation planned throughout delivery.

**Updated: 19 October 2026.** Phase 3's document Q&A implementation is complete
in local development; Phase 4 has started with bounded CSV ingestion. The current working setup uses OpenRouter BYOK, with GPT-OSS
120B through Groq as primary and Gemini 3.8 Flash as fallback. Company-server
acceptance and production qualification remain open.

The [public app on master](https://github.com/ElMolla10/ragbase/tree/master) has a
separate deployment and data model. Its UI and Railway deployment are not the
enterprise application. The enterprise entry point is `backend.enterprise:app`.

## What works today

- Authenticated PDF, UTF-8 TXT, Markdown and comma-separated CSV uploads, with durable immutable
  originals and version/hash tracking.
- PostgreSQL workspace permissions and forced row-level security, including
  cross-workspace isolation, permission revocation and integrity checks.
- Durable extraction jobs, bounded native PDF/CSV text extraction, page/line/row lineage,
  extraction warnings and versioned lexical search over authorized evidence.
- Natural document answers and paper overviews through Groq, with generated
  explanations supported by exact quotations and source identifiers checked in
  trusted code. Overview questions do not require keyword overlap. When lexical
  retrieval finds nothing, bounded authorized document context is supplied to
  the model instead of immediately declaring insufficient evidence.
- Source citations and authenticated previews. The client displays actual
  response-reported model/provider and how much context was included. Long
  documents receive explicitly bounded overview context, not a complete-review
  claim. Interpretation accuracy remains a model-quality evaluation requirement.
- Bounded generation admission and queueing, cancellation, job recovery,
  idempotency and coordination between extraction and interactive answers.
- Separate core-service readiness and generation health. Search and source
  previews remain available during model outages.
- A terminal query client. A graphical enterprise UI is still pending.

Natural document chat is enabled. Source identities and supporting quotations
are validated; generated explanations are model interpretations and are not
mechanically proven true. Exhaustive summaries and spreadsheet calculations
remain separate, unfinished capabilities.

## Current model routing

| Priority | Model | OpenRouter model ID | Allowed provider |
|---|---|---|---|
| Primary | GPT-OSS 120B | `openai/gpt-oss-120b` | Groq |
| Fallback | Gemini 3.8 Flash | `google/gemini-3.8-flash` | Google AI Studio |

The shared OpenAI Python SDK client includes the ordered model list and explicit
provider `order`, `only` and `allow_fallbacks` on every upstream completion.
Only `groq` and `google-ai-studio` are permitted. Mistral is removed. Both models
use low reasoning; Gemini requires reasoning enabled. Private reasoning is not
released as answer text. The selected model is logged without prompts.

OpenRouter can select the fallback within the same request when the primary
cannot serve it. This does not resume a partially delivered answer. The relay
bounds upstream generation to 60 seconds; the API-to-relay envelope is 65 seconds.
No local generation weights are required for this mode.

**Hosted processing is authorized only for nonconfidential development/testing.**
Questions, instructions and bounded retrieved evidence leave through a separate
credential-isolating relay. Storage, parsing, search and authentication remain
local. The API, database and worker retain internal networking; only the relay
has external connectivity. Production egress enforcement still needs acceptance.

The owner reports enabling **Never use shared capacity for any model on this
provider** for both providers. That workspace setting is required in addition
to the request allowlist. Provider free-tier quotas and OpenRouter BYOK
fees/allowances are separate: there is no unlimited-free or guaranteed-zero-cost
claim. Standard provider privacy is not zero-data-retention approval.

See [BYOK implementation and evidence](docs/openrouter-byok.md).

## Verification reached

| Check | Recorded result |
|---|---|
| Unit and contract suite | 190 passed |
| Isolated PostgreSQL integration suite | 64 passed |
| Lint, formatting and policy consistency | Passed |
| Gemini through Google AI Studio | HTTP 200 with an answer |
| GPT-OSS 120B through Groq | HTTP 200 with an answer |
| Deployed routing probes | All 3 returned HTTP 200, including forced Google-unavailable fallback |
| Full document Q&A smoke | Exact `Quarter 1 sales: 120` quotation with correct page-1 citation and source preview |

The latest Groq-first Q&A smoke took **3.31 seconds** for one synthetic PDF
question; an earlier Gemini-first run took 24.37 seconds. These are individual
development observations with different serving conditions. This
is not a server latency or throughput guarantee. All live checks used synthetic,
nonconfidential data. The current forced fallback excludes Groq while keeping GPT-OSS
as the requested primary; it does not inject a real provider 429.

Arabic reliability, image understanding, representative business accuracy,
large-context behavior and company concurrency limits have not been qualified.
Current retrieval is lexical: a paraphrase can miss evidence that an exact-term
query finds. Cross-language semantic retrieval remains pending.

## Run the local development setup

Use Docker with Compose. On a fresh enterprise checkout, initialize development
credentials once:

```powershell
python scripts/init_enterprise_dev.py
```

Existing configuration is preserved; do not rerun initialization to rotate
credentials. The ignored `.env.enterprise` and `.local/dev-access.json` contain
local synthetic identities and credentials. They are not company SSO.

Start the document services with generation disabled:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --build
```

For the explicitly enabled BYOK mode, configure valid Google/Groq provider keys
in the same OpenRouter workspace as your application key, with shared capacity
disabled. Enter the application key through the hidden terminal prompt:

```powershell
python scripts/configure_openrouter.py
docker compose --env-file .env.enterprise -f compose.enterprise.yaml -f compose.enterprise-openrouter.yaml up -d --build
```

The credential is stored in ignored `.local/openrouter-api-key` and mounted only
in the relay. Never commit it or paste it into chat. Use one generation override
at a time. The previous Ollama configuration and phase-3 CPU smoke are historical;
the active generation configuration is the BYOK route described above.

The enterprise API is available at `http://127.0.0.1:8800`:

```powershell
Invoke-RestMethod http://127.0.0.1:8800/health/ready
Invoke-RestMethod http://127.0.0.1:8800/health/generation
```

Health reports configuration and readiness; it does not perform inference or
certify provider access. To ask about an authorized, extracted and indexed version:

```powershell
python scripts/query_enterprise.py --version YOUR_VERSION_UUID --question "What were Quarter 1 sales?" --preview
```

If the synthetic ingestion smoke was run previously, its version ID is available
in `.local/ingestion-smoke.json`. Retrieval is scoped to the selected versions;
a missing top-k result does not prove a fact is absent from the entire corpus.

## Try the enterprise version now

The local API is running at `http://127.0.0.1:8800`. There is no graphical
enterprise UI yet. From this repository folder, launch the interactive client:

```powershell
python scripts/try_enterprise.py
```

Paste a file path when prompted, then type questions. The client uploads the
original, waits for extraction/indexing, and displays explained answers with
citations. Add `--preview` to show full source previews. It uses the local synthetic Alice development identity without
printing its token. Use nonconfidential test files: retrieved evidence is sent
through the enabled OpenRouter route. Supported inputs are native PDF, UTF-8
TXT/Markdown and comma-separated UTF-8 CSV, up to 5 MB. Leave a question blank to
exit. Uploads remain in the development workspace after exiting.

You can also supply a path and question directly:

```powershell
python scripts/try_enterprise.py --file "C:\path\test.pdf" --question "What are the main terms?"
```

CSV is the first Phase 4 slice: up to 1,000 records and 100 columns, with original
strings, row/physical-line locations and quoted multiline cells preserved. No
header/type/date inference, formula execution or calculations are performed.
Inconsistent row widths require review and are blocked from Q&A. CSV answers
currently quote JSON arrays of source cells. OCR, DOCX, PPTX and XLSX remain
unsupported. See [Phase 4 progress](docs/governance/phase4-progress.md).

## Run checks

The disposable integration stack uses synthetic data and a separate database:

```powershell
docker compose -f compose.enterprise-tests.yaml build tests
docker compose -f compose.enterprise-tests.yaml run --rm tests
docker compose -f compose.enterprise-tests.yaml run --rm tests python -m pytest enterprise_tests -p no:cacheprovider -q
docker compose -f compose.enterprise-tests.yaml down
```

For synthetic live provider and fallback checks after enabling the relay:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml -f compose.enterprise-openrouter.yaml exec -T openrouter-relay python scripts/test_openrouter_fallbacks.py
```

Do not point tests at application/company databases or use company documents in
public CI. See [development operations](docs/operations/development.md).

## Seven-phase roadmap

| Phase | Scope | Current status |
|---|---|---|
| 1 / P0 | Hardware discovery, policy and acceptance cases | CPU plan and laptop inventory recorded; company-server inventory and reviewed cases pending |
| 2 / P1 | Foundation and scoped original storage | Development implementation tested; production identity and release gates pending |
| 3 / P2 | Native PDF ingestion, retrieval and cited answers | Local implementation complete; BYOK routing added and live-tested; server/company acceptance pending |
| 4 / P3 | Arabic/English OCR and Office formats | In progress: bounded CSV ingestion tested; OCR and DOCX/PPTX/XLSX pending |
| 5 / P4 | Trusted numerical analysis, charts and exports | Pending |
| 6 / P5 | Complete-document coverage, RTL UX and model qualification | Pending |
| 7 / P6 | Hardening, SSO, backup/restore and company pilot | Pending |

The company server has **64 GB system RAM and no GPU**. Its CPU, memory bandwidth
and serving envelope remain unmeasured. CPU-compatible local models remain
candidates for later evaluation; no local production model has been selected.
Hosted BYOK integration does not complete server/model qualification.

## Rules and next work

[CONSTITUTION.md](CONSTITUTION.md) is the canonical source of project rules.
Bounded runtime instructions are derived from it, and trusted code enforces
implemented access, scope and validation controls. A written rule is not proof
that every model follows it. [AGENTS.md](AGENTS.md) directs development agents to
that source; [the rule ledger](docs/governance/rule-ledger.json) records gaps.

Phase 4 is in progress: bounded CSV ingestion is implemented; next are OCR and Office fidelity. Company
hardware discovery and representative Arabic/mixed-language acceptance cases
remain parallel requirements. Dense retrieval, deterministic calculations,
complete-document workflows, graphical/RTL UX, company SSO, production networking,
restore drills and pilot approval are still open.

Further detail: [execution plan](docs/execution-plan.md),
[seven-phase roadmap](docs/governance/phase-roadmap.md),
[historical Phase 3 evidence](docs/governance/phase3-completion.md), and
[current BYOK evidence](docs/openrouter-byok.md). When older records describe a
superseded runtime or model, use the current BYOK record for generation behavior.

Document-chat correction: “whats the paper about?” now invokes Groq with document context and returns cited explanations. A synthetic live run passed in 4.53 seconds. Coverage is reported; this is not unrestricted full-document comprehension.
