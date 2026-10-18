# Phase 3 (P2) — local implementation completion record

Updated: 18 October 2026. Constitution 2.0.0; execution plan 2.0. Phase numbering remains seven phases, P0–P6.

**Result: Phase 3's local implementation is complete and tested. Target-server acceptance remains pending.** The company server is unavailable, and its CPU/ISA/operating system have not been inventoried. This record does not close production, Arabic-reliability or server-capacity gates.

## Working path

Authenticated PDF/TXT/Markdown upload -> immutable originals -> durable PostgreSQL extraction job -> bounded native text -> versioned lexical chunks -> authorized search -> local model selection of source quotations -> trusted quote/citation validation -> streamed validated answer -> permission-checked native-text source preview.

The API is available at http://127.0.0.1:8800. A terminal client is provided in scripts/query_enterprise.py. The graphical enterprise frontend and richer RTL experience remain later-phase work.

## Implemented controls

- Retrieval indexes are versioned independently from generation, using native-lexical-v1. Original text is retained; Arabic diacritic/alef/digit normalization affects only a separate search representation. PostgreSQL GIN lexical search runs inside the authenticated workspace and selected immutable-version scope. No unfiltered corpus is retrieved and filtered afterward. This is a lexical baseline; dense embeddings, reranking and cross-language semantic recall are unqualified.
- Native text is chunked with stable source IDs, original version/hash, page or line location, and character offsets. Background backfill indexes already-extracted originals without re-parsing files. New pages and chunks publish atomically. Extraction warnings remain visible. Sources requiring extraction/indexing/review cannot be queried as ready.
- The first answer mode is deliberately extractive: exact source quotations, not unrestricted generated analysis. The model returns a bounded schema with source IDs and quotes. Trusted code rejects unknown IDs, altered quotations, extra fields, unfinished output and context violations. No raw model text or private reasoning is streamed to the user. The stream exposes queued/progress events, then validated quotations with source metadata. Relevance and source-extraction accuracy still need review; an exact quote alone does not prove that it answers every aspect of a question.
- Source previews are authenticated plain text with no-store/nosniff and restrictive CSP. File integrity is checked before retrieved source content is used and before previews. Preview access and publication are rechecked after permission changes. Numerical operations and exhaustive summaries are not implemented by this lookup mode.
- One shared PostgreSQL advisory admission lock coordinates generation and native extraction. Interactive answers take priority over background work. A user can have one outstanding answer; the queue is bounded. Leases fence abandoned work, and stale/cancelled tasks cannot publish. Provider transports close before admission is released. Queue deadlines and a narrow global lease guard prevent over-admission after session loss. Actual inference-engine cancellation behavior still needs target-runtime qualification.
- Answer lifecycle, source dependencies, instruction/index/model identity, actual usage and timings are retained. Raw questions, prompts and answers are not persisted in answer-run records. Question hashes are retained. Routine metric metadata contains no document content. Answer-history replay and company retention policy remain later work.
- Generation is disabled in the default development profile. Fixture mode is explicit and labeled; no fixture is represented as model quality. An optional development Ollama relay reaches only the preconfigured local host model, exposes no pull/create/delete endpoint, rejects non-text messages/tools and arbitrary schemas, and has no document volume or database credentials. It is not a production egress boundary.
- Core readiness and generation availability are separate. Search and source previews remain usable during inference outages. Client requests cannot supply arbitrary model endpoints or checkpoint names.

## Verification and measurements

- **156 automated tests passed:** 103 unit/contract/parser checks plus 53 PostgreSQL integration checks. They passed against current source and the rebuilt packaged test image. Lint, formatting and deterministic constitution/policy checks passed. One existing TestClient dependency deprecation warning remains.
- Tests exercise scope filtering, all eleven tables' forced RLS, cross-workspace source access, revocation during generation, model outage, exact quote/citation rejection, request idempotency, context overflow, cancellation, generation priority, worker recovery and lease fencing after simulated session loss.
- **Four live synthetic checks passed:** English PDF lookup, Arabic text lookup, a mixed-language question over English PDF text, and absent evidence. The first three invoked the installed model; absent evidence returned without a model call. Expected value presence and citation/preview access were checked. These are functional smoke checks, not a representative business accuracy evaluation or proof of general cross-language retrieval.
- The existing laptop inventory now lists qwen3:1.7b and a separate harness tag; Gemma is absent from the current inventory. The smoke used the original qwen3:1.7b checkpoint, digest 8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7, Ollama 0.40.3, Q4_K_M metadata, CPU-only options, two threads, configured 4,096 total context and a 256-token output ceiling. The tag and its reported 2.0B parameter metadata are recorded as observed; no equivalence to a production candidate is inferred. It is a development integration checkpoint, not a new production recommendation.
- End-to-end times were approximately **20.75s English PDF**, **16.47s Arabic text**, **16.28s mixed question** and **0.15s absent evidence**. Keep-alive was zero; the model unloaded after requests. These observations are not throughput, warm-serving or server latency guarantees.
- Actual token counts came from the runtime. The English PDF call reported 519 prompt + 76 generated = 595 tokens; its preflight input bound was 2,923 tokens. The adapter uses UTF-8 bytes plus a reserved template allowance as a conservative admission policy for this tested byte-BPE configuration. It does not use English word estimates for Arabic. A 4K configured limit is not a filled-4K stress test, and does not validate larger contexts. Model/tokenizer changes require revalidation.
- Eighty periodic Windows memory snapshots observed a minimum of **1,630,232,576 available host bytes** (about 1.52 GiB). Ollama reported a 1.9 GB loaded size and 100% CPU. The process-name sampler did not comprehensively capture the runner, so its small process working-set value is not a model-memory estimate. Sampling can miss loading peaks; whole-server sizing remains unvalidated.
- With the development relay stopped, core readiness, search and previews returned 200; generation readiness returned 503. The answer stream returned a safe error and no answer content. The relay was restored afterward.
- Earlier extraction restart and original-file persistence smoke checks remain passing after the migrations. This is not a backup/restore drill.

All documents and questions in these checks were synthetic. No model weights were downloaded. No target server, company documents, GitHub publication or production deployment was used. Existing public-demo and unrelated services were left alone.

## Run

From the repository root, use the existing ignored development secrets:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --build
```

This starts storage/extraction/search with generation disabled. To enable the already-installed, pinned development Qwen model through the optional relay:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml -f compose.enterprise-ollama.yaml up -d --build
python scripts/smoke_enterprise_answers.py --output .local/phase3-live-evidence.json
```

A missing or changed checkpoint fails closed; no weights are pulled. The optional host relay requires Docker Desktop access to host.docker.internal and the existing host Ollama service. It is a laptop development convenience, not the final server runtime.

Ask a question over a selected version and inspect its source:

```powershell
$version = (Get-Content .local/ingestion-smoke.json -Raw | ConvertFrom-Json).version_id
python scripts/query_enterprise.py --version $version --question "What were Quarter 2 sales?" --preview
```

The example uses the earlier synthetic smoke PDF. Alternatively supply an authorized version ID; omit --question to enter it interactively. The client reads development tokens from ignored local configuration without printing them. More endpoint/startup details are in docs/operations/development.md.

## Remaining acceptance gates

Phase 3 local development acceptance is recorded separately from company-server acceptance. The latter still requires actual hardware discovery, approved runtime/artifact provisioning, resource and latency tests, and reviewed representative company evidence cases. Qwen3.5-9B/35B-A3B/27B, 4B fallback and Gemma remain unqualified candidates. Arabic business reliability requires its reviewed acceptance set. No production model selection has been made.

Later phases retain OCR/Office fidelity, trusted arithmetic, dense/hybrid retrieval tuning, exhaustive coverage, optional vision, model swaps, graphical/RTL UX, SSO, full network denial, release locking, backups/restoration and pilot approval. No later-phase feature is declared complete by this lookup milestone.

## Primary protocol references

[Ollama chat API](https://docs.ollama.com/api/chat) supplies streaming, structured format, thinking controls and actual usage counters. [PostgreSQL locking](https://www.postgresql.org/docs/16/explicit-locking.html) describes advisory locking. Project behavior and limits above are based on implementation/tests, not assumptions of provider compatibility.
