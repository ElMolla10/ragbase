# RAGBase Local Enterprise — Final Execution Plan

Version: 1.0  
Prepared: 18 October 2026  
Governing specification: [CONSTITUTION.md](../CONSTITUTION.md), version 1.0.0  
Project: [ElMolla10/ragbase](https://github.com/ElMolla10/ragbase)  
Status: execution baseline for implementation; application code, server configuration, and model performance have not been validated by this deliverable.

## 1. What we are building

We will turn RAGBase into a fully local company document and analysis platform. Users will upload documents, ask questions, summarize complete files, compare versions or files, extract structured information, calculate spreadsheet results, and generate charts and downloadable outputs. Arabic, English, and mixed-language work are first-class requirements.

The system has two evidence paths:

1. **Document path:** extract and preserve source structure, search authorized evidence, inspect relevant pages where necessary, and answer with precise citations.
2. **Analysis path:** normalize authorized tables, validate their meaning and quality, calculate results with trusted code, and explain those results with source ranges and reproducible operations.

The model coordinates bounded workflows and explains evidence. Application code controls permissions, validates tool requests, performs calculations, and enforces resource and network restrictions.

We will develop the application now without requiring access to the production server. Actual model ranking, Arabic reliability, supported context lengths, GPU capacity, production load, and offline server acceptance remain explicit later gates. A completed mock workflow will never be reported as a validated local-model deployment.

## 2. Agreed requirements and decision boundaries

| Area | Execution decision |
|---|---|
| Local operation | Company-controlled inference, OCR, parsing, storage, search, analysis, identity, logs, and browser assets; no hosted fallback |
| Development computer | 16 GB RAM laptop with lightweight services, synthetic fixtures, and mocked generation/embeddings by default |
| Production assumption | One server with one 64 GB GPU; vendor, usable VRAM, system RAM, CPU, OS, storage, and runtime remain unverified |
| Model candidates | Three candidates prepared behind one gateway; no winner selected before real testing |
| Context | 16K integration; 64K normal-use target; 128K priority validation; approximately 256K later where supported |
| Languages | Formal Arabic, Egyptian questions, English, mixed terminology, and cross-language retrieval |
| Formats | PDF, DOCX, XLSX, PPTX, CSV, TXT, Markdown, delivered incrementally |
| Identity | Company OIDC/SSO where available; local development identity only in a clearly separate development profile |
| Data isolation | Workspaces and roles; authorization applies to originals and all derived content |
| Deployment | Modular services under Docker Compose for the pilot; no Kubernetes requirement |
| Initial outputs | Cited answers, source previews, result tables, PNG charts, CSV/XLSX and Markdown exports |
| Deferred scope | Fine-tuning, arbitrary generated Python, unrestricted SQL, GraphRAG, general multi-agent frameworks, HA, legacy Office conversion, password handling, and polished DOCX/PDF exports |

The constitution owns policy. This plan owns sequence, deliverables, and verification. An implementation ledger will record whether each constitution rule is merely specified, implemented, or verified for a named release. No file or prompt alone guarantees model compliance.

## 3. Three model candidates

These are a practical comparison shortlist, not an exhaustive market ranking. No production suitability or hardware fit is claimed yet. Quantization and serving engine are part of the tested candidate configuration, not interchangeable implementation details.

| ID | Candidate | Role in comparison | Artifact and runtime decision |
|---|---|---|---|
| A | `Qwen/Qwen3.5-35B-A3B-FP8` | Initial leading candidate; mixture-of-experts comparison | Official FP8 checkpoint; verify GPU support and pin a compatible runtime |
| B | `Qwen/Qwen3.5-27B` | Dense-model comparison within the same family | Select and pin a compatible quantized artifact after license, lineage, kernel, and memory checks; the base model name does not identify a tested quantization |
| C | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Independent-family comparison for document work and instruction/tool behavior | Select and pin an appropriate quantization after compatibility and quality checks |

The Qwen model cards publish native 262,144-token contexts. That is a model specification, not a statement that our server can serve those contexts. [Candidate A model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8), [candidate B model card](https://huggingface.co/Qwen/Qwen3.5-27B).

Mistral's model card describes improvements to instruction following and function calling; its published configuration sets `max_position_embeddings` to 131,072. Candidate C therefore participates up to its supported limit and is not expected to pass a 256K native-context test. Its Arabic quality remains an open evaluation question. [Candidate C model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506), [published configuration](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/raw/main/config.json).

Published multilingual scores are screening evidence only. They do not establish reliable Arabic, Egyptian dialect understanding, numerical correctness, citation quality, or company-document performance. Recheck cards, licenses, immutable revisions, and serving support before acquiring artifacts. Do not copy unpinned `latest` or multi-GPU example commands into our single-card deployment.

All three candidates are evaluated sequentially; we do not need three resident model servers. The initial production design uses one active generation configuration. Automatic failover between unvalidated models is excluded.

## 4. What can be tested before the server is available

| Test | Can run now? | What it establishes |
|---|---|---|
| Gateway requests, streaming normalization, errors, timeouts, cancellation | Yes, against deterministic fake endpoints | Application and adapter contracts |
| Capability rejection, output schemas, malicious tool arguments | Yes | Trusted application validation |
| Workspace permissions, RLS, downloads, revocation | Yes, with local PostgreSQL and actual serving roles | Application/database isolation on the tested setup |
| Deterministic table operations, provenance, exports | Yes, with synthetic fixtures | Calculation and lineage correctness |
| Basic format parsing, job retries, source previews | Yes, with bounded fixtures and available local dependencies | Functional behavior on those fixtures |
| Tokenizer budgets for the actual candidates | Yes, if approved tokenizer artifacts can be obtained and supported locally | Token accounting; no answer-quality or GPU-capacity claims |
| Real embedding/retrieval evaluation on a small corpus | Potentially, using a separately bounded CPU profile | Retrieval quality for that embedding/corpus, independent of generation |
| Tiny 1–4B local model smoke test | Optional, only if resources permit | HTTP/streaming plumbing, not quality of A/B/C |
| Full A/B/C Arabic, vision, reasoning, or context benchmarks | Not on the assumed laptop as a credible acceptance setup | Requires suitable measured compute |
| Production memory, throughput, 64K/128K capacity, offline cold start | No | Requires target hardware and deployment |

CPU execution or heavy offloading of a large quantized candidate might be technically possible on some machines. We will not make that a dependency for this 16 GB laptop or treat it as a proxy for production serving. Check actual free RAM/storage before any optional model download; do not fetch large weights as part of ordinary development setup.

If suitable company-controlled alternative compute becomes available, the same harness can run there and record that hardware. Its results remain preliminary for target-server capacity. Hosted testing is not part of the default plan and company content must not be sent to an external service.

When compute is unavailable, benchmark results are **pending**, never passed, zero-scored, or replaced with invented mock scores. Development continues on independent work; release selection waits.

## 5. Constitution implementation

Keep one authoritative `CONSTITUTION.md` at the enterprise repository root. Repository entry instructions, including `AGENTS.md`, reference it instead of copying its rules. Development agents read it and relevant local instructions before changes.

Implement these supporting artifacts:

- `docs/governance/rule-ledger.yaml`: one row for each of the 55 rule IDs, including owner, enforcement location, verification reference, state, last validated release, and known gaps.
- `policy/manifest.yaml`: constitution version/hash, mandatory runtime sections, task-specific sections, and referenced rule IDs.
- `policy/templates/`: reviewed runtime instruction templates; derived assets are not independent policy sources.
- `scripts/build_policy.py`: deterministic assembly and validation of mandatory core plus applicable task rules.
- `tests/policy/`: missing-core, stale-hash, unknown-rule, malformed-manifest, evidence-separation, and adversarial instruction cases.

The build must detect constitution changes and require regeneration/review of affected instructions. Record the assembled instruction hash and constitution version with traces. A missing or invalid mandatory core stops the affected generation call.

Always include the mandatory runtime core. Select additional task sections through trusted application routing; the model does not decide which mandatory policies apply. Evidence and tool output remain in separate data fields. Do not send the entire constitution with every request or rely on prompting to enforce permissions, network isolation, or arithmetic.

The ledger distinguishes developer checks, application enforcement, deployment controls, and measured model behavior. A constitution version change triggers the relevant tests and evaluation slices before release.

## 6. Target architecture

```text
User browser
    |
Internal HTTPS reverse proxy — only user-facing service
    |
Frontend + FastAPI
    |-- Identity / workspace authorization / policy assembly
    |-- PostgreSQL + pgvector: metadata, ACLs, elements, jobs, history
    |-- Durable file store: originals, extraction, previews, datasets, outputs
    |-- DB outbox -> local broker -> bounded workers
    |                         |-- parsing / OCR / preview / indexing
    |                         |-- isolated deterministic analysis
    |-- Retrieval -> authorized evidence -> ModelGateway
    |                                      |
    |                              Internal model service
    |                              weights + tokenizer + engine
    |
Restricted local metrics, audit, backup, and administration
```

Retain FastAPI, PostgreSQL/pgvector, and useful existing streaming/retrieval foundations. Use Alembic for schema evolution. Proposed queue implementation is Celery with a pinned, tested local Redis-compatible broker such as Valkey; PostgreSQL remains durable job truth. Review compatibility before locking versions.

Use separate API, parser/OCR worker, analysis worker, and model-service images. Split heavyweight dependencies. No generation weights or model-engine imports in API startup. Embeddings, reranking, and OCR have their own replaceable interfaces; start CPU-first where practical and assign GPU use only after measuring the shared budget.

Proposed parsing stack: evaluate Docling for structured conversion, a dedicated openpyxl workbook adapter for XLSX fidelity, Arrow/Parquet for normalized tables, and DuckDB for trusted analytics. These are implementation choices to validate and pin, not claims that every format or feature will work automatically.

## 7. Environment profiles and configuration

| Profile | Purpose | Defaults |
|---|---|---|
| `dev-lite` | Daily laptop work | API, small PostgreSQL, minimal worker/broker, local assets, fake generation and embeddings; resource target approximately 6–8 GB total, verified rather than assumed |
| `dev-cpu-eval` | Optional limited extraction/retrieval checks | Small corpus, one heavy operation at a time, real bounded CPU models where supported |
| `dev-model` | Optional tiny-model integration | Short context, one request, synthetic content; never an A/B/C acceptance substitute |
| `server-staging` | Later integration and evaluation | Actual model service, representative internal corpus, isolated data/configuration |
| `server-local` | Pilot/production | Pinned offline bundle, SSO, network controls, production limits and backup |

Example application settings include `GENERATION_BASE_URL`, `GENERATION_MODEL_ALIAS`, `GENERATION_TIMEOUT_SECONDS`, `GENERATION_MAX_INPUT_TOKENS`, `GENERATION_MAX_OUTPUT_TOKENS`, and a server-owned model-profile identifier. Track the total configured context separately from input/output budgets. Credentials are server-side secrets.

The stable serving alias is `company-assistant`. Each candidate profile records checkpoint, immutable revision, quantization artifact/hash, engine/image version, tokenizer/chat template, parser settings, actual capabilities, sampling/reasoning limits, context budget, and validated operating envelope. Validate configuration at startup. Mock identities and fake models must be visibly marked and prohibited in production configuration.

## 8. Data and API contracts

Establish schemas before format-specific expansion:

- Users, groups, workspaces, memberships, roles, document permissions.
- Documents and immutable versions with hashes and lifecycle state.
- Stored artifacts, extraction runs, normalized elements, tables, and datasets.
- Embeddings/index versions, durable jobs, outbox events, attempts, and failures.
- Conversations/messages, source dependencies, tool runs, calculation records, generated artifacts, and audit events.

An evidence locator contains workspace, document/version/hash, element ID, format, location, and extraction-quality metadata. Locations are page/box, heading/paragraph/table cell, slide/shape, or sheet/table/range as appropriate. Derived rows and outputs retain source mappings.

Version gateway requests/results/events and ingestion/analysis schemas. Streaming supports text deltas, normalized completion/error status, usage or explicitly marked estimates, and cancellation/disconnect cleanup. Structured tool requests are validated against application schemas and scoped identity.

API contract groups: identity/workspaces; documents/versions; uploads/jobs/retry; authorized originals/previews/elements; streaming/nonstreaming query; datasets/analysis; history; artifacts/exports; and restricted health/admin endpoints. Final route names follow repository inspection. Do not break existing deployment assumptions unintentionally.

## 9. Delivery track A — work we can start now

Each package has a reviewable deliverable and an exit check. Tests run at the layer appropriate to the change; avoid broad retesting without a relevant reason. Record deferred checks instead of treating unavailable infrastructure as a pass.

### A0 — Repository baseline and project control

**Work:** Inspect the current upstream, repository instructions, dependency graph, tests, routes, deployment, and known risks. Compare with the originally reviewed commit `2f0a0204a14f6659b97a061168ccb0bdd8c061f7`; deliberately select and record the implementation base rather than assuming it is still current. Create the isolated `enterprise-local` branch and short feature branches. Preserve existing public deployment/data boundaries.

Install the constitution, agent entry instructions, rule ledger, execution backlog, decision log, and a baseline test report. Record existing failures separately from regressions. Separate synthetic public fixtures from internal acceptance data.

**Deliverables:** reviewed base SHA; enterprise branch; governance files; dependency/runtime inventory; baseline report; ordered backlog.

**Exit:** no company data added; every rule ID represented once in the ledger; branch/deployment isolation documented; current behavior and baseline checks recorded.

### A1 — Lightweight foundation, gateway, and policy assembly

**Work:** Add dependency injection for generation, embeddings, OCR, storage, and retrieval components. Remove heavy startup coupling. Split dependency groups and create reproducible dev-lite startup with fake endpoints. Introduce Alembic and migration discipline. Implement policy assembly and the ModelGateway contract plus HTTP adapter scaffold.

Create candidate profiles A/B/C with unvalidated capabilities disabled until proven. Test protocol variants with fixtures: fragmented streams, malformed structured requests, missing usage, unsupported images/tools, timeouts, disconnects, and endpoint outage. Test that changing the configured profile does not change routes, frontend, retrieval, or indexed documents.

**Deliverables:** dev-lite profile; lock strategy; migrations; interfaces/fakes; policy build; gateway schemas; three candidate manifests; contract tests.

**Exit:** clean laptop startup without large downloads or generation weights; API non-generation features work with inference unavailable; fake profile swaps pass; invalid mandatory policy fails explicitly. These checks establish contracts, not actual A/B/C compatibility.

### A2 — Identity, storage, authorization, and protected delivery

**Work:** Implement users/workspaces/roles and a production OIDC integration path. Use explicit development identity until company settings are available. Create durable storage with staged writes, hashes, immutable versions, safe paths, authorized delivery, and orphan reconciliation.

Implement application authorization plus RLS under the actual non-owner serving role. Scope workers and reset transaction identity safely. Add source dependencies and revocation handling for messages/artifacts. Protect all writes/admin/query paths; sanitize rendering, bundle browser dependencies locally, bound requests, and apply coordinated quotas.

**Deliverables:** scoped identity and permissions; storage abstraction; migrations/policies; authenticated download/preview paths; revocation and route-isolation tests.

**Exit:** one user/workspace cannot read another's originals, evidence, jobs, datasets, history, or artifacts by changing identifiers; pooled connections do not leak identity; revoked sources cannot reappear through history/export. Production SSO validation remains pending if internal identity is unavailable.

### A3 — Durable PDF ingestion and source previews

**Work:** Build upload validation, immutable original storage, DB job/outbox creation, queue delivery, idempotent extraction/indexing, and reconciliation. Start with text PDFs and basic TXT/Markdown; add scan routing and bounded Arabic/English OCR through replaceable interfaces. Preserve pages, structure, tables, locations, quality flags, and previews.

Test parser limits, worker termination, duplicate delivery, retries, queue outage, partially failed extraction, and file/database cleanup. Only publish a consistent ready version; show review/failure states and gaps.

**Deliverables:** durable job lifecycle; PDF adapter; source locator schemas; preview delivery; recovery fixtures; ingestion UI state.

**Exit:** authorized upload survives restart; duplicate jobs do not duplicate elements; failures remain visible; citations resolve to the correct immutable source. Mock embeddings are sufficient for wiring tests; extraction quality is assessed independently.

### A4 — Retrieval and the first end-to-end development slice

**Work:** Implement authorized dense plus indexed lexical candidates, fusion/deduplication, bounded reranking, structure-aware chunking, and parent-section expansion. Compare current E5 baseline with BGE-M3 and a multilingual reranker when bounded local resources allow; otherwise prepare the runner and leave scores pending. Keep index dimensions/revisions explicit.

Initial tuning ranges: approximately 400–800 tokens per child chunk, 30–50 candidates per search path, and roughly 6–12 retained evidence items; these are experiments, not fixed quality guarantees. Preserve headers when splitting tables. Evaluate Arabic normalization separately from displayed text and identifier handling. Do not label ordinary PostgreSQL full-text ranking as BM25.

Resolve conversational references into a scoped search question; assemble budgeted evidence; validate source IDs; disclose searched scope and missing evidence. Implement chat streaming, source selection, and preview links. Test Arabic RTL and mixed identifiers from this stage.

**Deliverables:** retrieval pipeline; gold evidence fixtures; citation validation; scoped chat/source UI; real-versus-mocked evaluation labels.

**Exit:** authorized PDF upload -> durable job -> indexed elements -> fake streaming answer -> valid source preview works end to end. Retrieval metrics are meaningful only for real embeddings/reranking; fake output does not pass answer-quality gates. The later server slice replaces the fake provider and validates actual local answers.

### A5 — Office formats and dataset normalization

**Work:** Add DOCX, PPTX, XLSX, and CSV adapters and strengthen PDF/OCR fixtures. For Word, define final-view versus revision handling and inclusion of headers, footers, notes, and comments. For slides, preserve notes, text/tables, slide previews, and inspectable chart/image regions.

For workbooks, preserve formulas separately from cached values, cell types, named/table ranges, identifiers, hidden-region policy, dates, units, and source mappings. Detect table regions and ambiguous headers; allow mapping confirmation. Do not imply formula recalculation. Reject unsupported encrypted/macro-enabled inputs explicitly. Disable or block external references and active content.

**Deliverables:** per-format adapters and gold fixtures; normalized element/dataset schema; range/slide/paragraph locators; extraction warnings; library filtering and reprocessing controls.

**Exit:** required text, values, relationships, and source locations match gold fixtures; unsupported content is disclosed; uncertain/stale values are flagged; memory/expansion limits work. Internal representative Arabic layouts remain a later acceptance requirement.

### A6 — Deterministic analysis, charts, and result exports

**Work:** Implement typed schemas for listing/describing datasets, filtering, aggregation, comparison, approved joins, and chart creation. Validate currencies, units, dates, duplicate keys, row counts, nulls, subtotals, and join multiplication. Ask for material mapping/metric ambiguity rather than guess.

Run trusted operations against authorized job-local datasets in isolated workers. Lock DuckDB external access/extensions/settings and enforce OS resource/network/filesystem boundaries. Record calculations, source ranges, data versions, exclusions, and outputs. Generate CSV/XLSX results, PNG charts, and Markdown reports, with formula-injection protection and authorized delivery.

**Deliverables:** analysis tool registry; data validation; isolated execution; provenance/calculation panel; charts/exports; critical numerical gold fixtures.

**Exit:** exact expected numerical results within defined tolerances; correct handling of stale formula caches, mixed currencies, leading zeros, duplicate joins, and inaccessible data; bounded execution and attempted unauthorized file/network access fail safely. Model tool-planning quality remains a separate later test.

### A7 — Coverage workflows, bounded vision, history, and UX

**Work:** Add section enumeration and coverage records for complete summaries, exhaustive extraction, and comparisons across selected files/versions. Reduce bounded section results while preserving citations and gaps. Never claim complete coverage for skipped or failed regions. Surface source conflicts.

Add application-controlled page inspection, image budgets, explicit capability failures, history with source dependencies, cancellation, progress, retry controls, and duplicate-submission prevention. Support Arabic RTL and mixed-language tables/citations. Vision requests send only authorized local inputs; do not fetch document-supplied URLs.

**Deliverables:** summary/comparison/extraction workflow engines; coverage report; visual tool scaffold; polished core chat/library/evidence/results flows; history/revocation behavior.

**Exit:** fixtures demonstrate every required section/version is visited or marked missing; unsupported vision is explicit; history respects revoked access; canceled work releases application resources. Actual visual interpretation and model synthesis quality remain pending until real inference tests.

### A8 — Evaluation harness and deployment preparation

**Work:** Build one reusable runner for gateway-only, retrieval-only, fixed-evidence generation, tool planning, complete end-to-end, context, vision, authorization, and load evaluations. Use the same cases and scoring definitions across candidate profiles. Preserve immutable results and configuration metadata. Build offline packaging scripts, release manifest templates, network policy, health checks, backup/restore tooling, and operator runbooks.

**Deliverables:** runnable benchmark suite; reviewed synthetic baseline; internal-data import format; blank A/B/C scorecards marked pending; reproducible deployment bundle specification; release checklist and runbooks.

**Exit:** the runner works against fakes and rejects invalid results without manufacturing quality scores; repeat runs record comparable configurations; laptop restore works on synthetic data; production bundle validation is explicitly deferred. Continue package preparation without asserting offline server acceptance.

## 10. Delivery track B — when suitable company compute is available

### B0 — Hardware and company integration inventory

Record GPU vendor/model, usable/free VRAM, driver/runtime/kernel compatibility, CPU architecture/cores, RAM, storage, Linux environment, virtualization/passthrough, and competing workloads. Collect expected users, simultaneous requests, corpus volume, upload complexity, Arabic mix, permission groups, identity-provider details, internal dependencies, retention, and availability needs.

Begin with 20–30 representative sanitized documents and 30 reviewed questions/calculations for initial integration. Review confidential samples only internally. Company IT supplies internal TLS/network/identity and backup decisions. Complete the hardware-specific model and artifact plan after inventory.

**Exit:** an approved staging environment, measured resource inventory, internal identity/network path, and representative acceptance slice exist. If hardware is unsuitable, change candidate/runtime or deployment expectations explicitly; do not silently weaken the requirements.

### B1 — Candidate serving and controlled benchmarks

Acquire verified pinned artifacts on approved staging infrastructure and transfer them through the approved process. Test candidates sequentially. First verify 16K startup, tokenizer, text/streaming/tools/vision capabilities, cancellation, and no external fetches. Then test actual supported contexts and shared resource limits.

Start with one active generation and increase only after measurement; test two and four where safe. Begin embeddings/reranking/OCR on bounded CPU resources to avoid unmeasured GPU contention. Do not assume a fixed percentage of VRAM or copy engine defaults blindly.

Run the A/B/C test matrix in Section 12. Exclude unsupported context/capability cells explicitly. Record failed startups and OOMs as failures, not missing observations. A candidate's different quantization is a new tested configuration.

**Exit:** real compatibility records, Arabic and evidence scores, 32K/64K/128K results where supported, and memory/latency/concurrency measurements. All required 128K tests must be attempted and documented; a failed priority goal cannot be called achieved. Decide whether a measured 64K deployment is acceptable or the priority goal needs a different candidate/configuration. Approximately 256K remains optional and is not required of candidate C.

### B2 — Local integration, identity, OCR, and model swap

Replace fakes with the validated local components. Validate company SSO, actual serving-role RLS, representative Arabic OCR, preview rendering, durable jobs, and the full authorized upload-to-answer path. Run complete analysis tasks where the model proposes plans and trusted tools calculate results.

Demonstrate replacing one validated model with a second through model-service/profile changes only: drain requests, replace the configuration, warm and check readiness, run smoke tests, and resume. Application routes, frontend, retrieval logic, documents, and embedding index remain unchanged. Test rollback to the prior configuration. A new engine may require adapter work, kept within that boundary.

**Exit:** two real local models satisfy the gateway swap contract; enabled features pass company-data tests; unsupported features fail explicitly; generation outage leaves authorized non-generation functions available.

### B3 — Offline hardening, load, recovery, and pilot

Build the final pinned bundle; remove public egress from both browser and services; cold-start from prepared artifacts on a clean runtime. Exercise all enabled formats, Arabic OCR, text and visual questions, workbook analysis, citations, exports, restart/retry, and cancellation. Inspect network-denial logs. Run mixed interactive/ingestion/report load and agreed user/concurrency budgets.

Back up consistent database/files and required secrets/configuration to a separate approved destination. Restore into a clean staging deployment; verify permissions, originals, citations, datasets, retrieval, and calculations. Validate release rollback/forward recovery and document single-server downtime expectations.

**Exit:** Section 13 gates pass for a named release; operator and user guides exist; known limitations are explicit; project owner and company IT accept the pilot scope and measured operating envelope. Public production rollout is a separate release action, not implied by drafting or merging code.

## 11. Model swapping as an acceptance requirement

The application must not import a candidate name into routes, retrieval logic, frontend behavior, or calculation logic. Candidate-specific templates, reasoning parsers, tool parsers, sampling settings, and response quirks stay in deployment profiles/adapters.

Swap procedure:

1. Select a prepared model profile and verify its artifact/license/runtime manifest.
2. Run adapter and capability checks, then required quality evaluations.
3. Record the new tokenizer/context limits, constitution/prompt version, and enabled capabilities.
4. Drain pending generation; keep other authorized features available.
5. Stop the old model process, start the replacement, warm it, and verify generation readiness.
6. Run text, Arabic, citation, tools, and enabled-vision smoke cases.
7. Resume within the measured operating envelope or revert the prior profile.

No re-ingestion or re-embedding is required for a generation-only change. Embedding-model changes use a separate versioned index migration with validation and controlled cutover. Reranker changes require retrieval evaluation. Never assume two full generation models fit concurrently on the single card.

## 12. Evaluation design

### Corpus and scoring

Build 200–300 reviewed cases with tuning and held-out splits. At least 100 cases are Arabic or mixed language and may be part of that total. Include matched English/Arabic questions; label Egyptian, formal Arabic, cross-language retrieval, scans, tables, exceptions/negation, missing answers, numerical analysis, conflicts/versions, and exhaustive tasks. Case tags may overlap; report unique-case counts and slice denominators.

Each case records authorized identity/scope, immutable input versions, question, expected supporting elements, required facts or exact result, acceptable refusal, coverage requirement, language/task tags, and scoring rubric. Numerical fixtures define precision, rounding, units, and expected excluded rows. Human reviewers adjudicate ambiguous facts/citations and approve gold labels.

Begin with a small reviewed slice, then grow the corpus throughout development. Never tune on held-out cases or use a model's own fluent answer as the gold result.

### Candidate comparison matrix

| Dimension | Procedure | Record |
|---|---|---|
| Arabic and English grounding | Same fixed evidence and reviewed questions for A/B/C | Required facts, unsupported claims, refusal, dialect/negation handling |
| Retrieval | Hold generation out; compare embedding/index/reranker configurations | Recall@k, MRR/nDCG, exact element hits, per-language and ACL slices |
| End-to-end answers | Same authorized corpus, question set, and pipeline | Answer correctness, citations, scope disclosure, extraction/retrieval/generation error attribution |
| Tool planning | Same typed schemas and analytical tasks | Valid schema, correct dataset/filters/operations, ambiguity detection, bounded recovery |
| Numerical outcomes | Trusted execution against gold tables | Exact expected values/tolerances, lineage, row counts, units and exclusions |
| Long context | 32K/64K/128K total budgets; facts at different positions; realistic distractors | Quality, actual token count, latency, peak memory, failures, output reserve |
| Vision | Same authorized page images and task limits | Correct supported interpretation, citations, uncertainty, memory and latency |
| Injection and access | Adversarial files, notes, tool outputs, user scopes, revocation | Unauthorized content, blocked operations, safe explanations |
| Serving | Fixed context/output budgets at 1/2/4 active requests where safe | Time to first token, generation rate, p50/p95 latency, queue wait, VRAM/RAM and cancellation |

Use identical case content and defined budgets; tokenize separately for each model. For position-controlled context tests, construct inputs per tokenizer and report the differences. Include fixed-evidence tests to separate generation quality from retrieval quality. Keep tuning effort and inference budgets documented; record repeat runs for variable generation instead of assuming bit-for-bit determinism.

Do not mix a published model benchmark, a laptop fixture result, and a server measurement into one undifferentiated score. Select a winner by required gates first, then compare quality, latency, resource needs, and operational complexity among candidates that pass. No aggregate score can compensate for an access leak or an unacceptable required Arabic slice.

## 13. Acceptance gates and definition of done

| Gate | Requirement | When assessed |
|---|---|---|
| G0 — Governance | Constitution installed; all 55 rule IDs tracked; instruction assembly/versioning checks pass | A0–A1, then every relevant change |
| G1 — Development | Repeatable dev-lite startup; inference outage tolerated; no heavy generation dependency | A1 |
| G2 — Isolation | Zero unauthorized content in dedicated access tests, including direct IDs, workers, history and exports | A2 onward; rerun on server |
| G3 — Ingestion | Enabled formats pass gold extraction/location cases, durable retries, consistent publication, and limits | A3–A5; internal corpus later |
| G4 — Retrieval | Proposed pilot target: at least 90% supporting-evidence recall@10 on answerable gold cases | Real embeddings/corpus only |
| G5 — Grounding | Proposed pilot target: at least 95% manually reviewed citation precision, plus agreed factual correctness/refusal/coverage thresholds per required slice | Real model evaluation |
| G6 — Analysis | 100% correct fixed critical numerical cases within stated tolerances; source lineage and sandbox checks pass | A6, then actual model-planned tasks |
| G7 — Arabic | At least 100 reviewed Arabic/mixed cases; matched-language gap and required business slices meet frozen thresholds | B1–B2 |
| G8 — Context and load | 64K normal target measured; 128K priority tests completed with explicit outcome; selected operating envelope meets agreed latency/concurrency limits | B1–B3 |
| G9 — Replacement | Two validated real local models swap without application/retrieval changes or re-ingestion/re-embedding | B2 |
| G10 — Offline and recovery | Clean offline boot, enabled workflows, restricted-network checks, backup restore and recovery runbook validated | B3 |
| G11 — Release | Every mandatory applicable rule verified for the named release or explicitly resolved through authorized policy change; known limitations and pilot scope accepted | B3 |

Before held-out evaluation, project owner/reviewers freeze: Arabic task correctness thresholds, permitted unsupported-claim rate, calibrated refusal criteria, extraction/OCR tolerances, whole-document coverage requirements, matched-language gap, and intended latency/concurrency. These are decision records, not values invented after seeing favorable results. Preliminary recall/citation targets remain proposed until the baseline review confirms definitions and thresholds.

Passing the minimum reviewed-case count alone does not pass the Arabic gate. A documented failure of 128K is not a pass for 128K capability; it requires an explicit scope/candidate decision. If 64K cannot meet requirements, revisit the model/runtime/hardware rather than silently declaring 16K sufficient. Optional 256K failure does not block the agreed pilot unless later made a requirement.

Application-complete, model-validated, and production-accepted are separate statuses. The first does not imply the other two.

## 14. Proposed repository layout and review packages

Adapt this layout to existing repository conventions after A0; avoid gratuitous rewrites.

```text
CONSTITUTION.md
AGENTS.md
docs/
  execution-plan.md
  governance/rule-ledger.yaml
  decisions/
  operations/
policy/
  manifest.yaml
  templates/
backend/
  gateway/              # interfaces, normalized events, HTTP adapters
  auth/                 # identity, workspace scope, permissions
  storage/              # durable originals/artifacts, reconciliation
  ingestion/            # format adapters, normalized extraction
  jobs/                 # durable state, outbox, retries
  analysis/             # typed operations, validation, execution, lineage
  retrieval.py          # extend or split as repository warrants
  migrations/
frontend/                # internally bundled assets and source/result UI
deployment/
  profiles/
  model-profiles/       # A/B/C manifests; no weights in Git
  offline/
scripts/
tests/
eval/
  synthetic/
  schemas/
  runners/
```

Suggested PR sequence into `enterprise-local`: `local-00-governance`, `local-01-foundation`, `local-02-access-storage`, `local-03-durable-ingest`, `local-04-retrieval-chat`, `local-05-formats`, `local-06-analysis`, `local-07-coverage-ux`, `local-08-eval-packaging`, then server-specific integration/hardening PRs. Split packages further where review size warrants it.

Each change records the concrete behavior, affected rule IDs, relevant verification, and deferred server checks. Avoid coupling large model downloads or server-only tests to ordinary public CI. Sensitive internal runners must never execute untrusted external PR code with production access.

## 15. Constitution coverage ownership

Every ID below gets an individual ledger row; ranges here describe ownership only.

| Rules | Primary work packages |
|---|---|
| LOC-001, LOC-002, LOC-003, LOC-004 | A0–A2, A8, B0–B3 |
| SEC-001, SEC-002, SEC-003, SEC-004, SEC-005, SEC-006, SEC-007 | A1–A2; exercised by all feature packages and B2–B3 |
| MOD-001, MOD-002, MOD-003, MOD-004, MOD-005, MOD-006, MOD-007, MOD-008 | A1, A8, B0–B2 |
| ING-001, ING-002, ING-003, ING-004, ING-005, ING-006, ING-007 | A2–A5, B2–B3 |
| EVD-001, EVD-002, EVD-003, EVD-004, EVD-005 | A4, A7, A8, B1–B2 |
| ANA-001, ANA-002, ANA-003, ANA-004, ANA-005, ANA-006, ANA-007 | A5–A6, B1–B3 |
| UX-001, UX-002, UX-003 | A3–A7, B2 |
| DEV-001, DEV-002, DEV-003 | A0–A1, continuous review |
| OPS-001, OPS-002, OPS-003, OPS-004 | A1–A3, A6–A8, B0–B3 |
| VAL-001, VAL-002, VAL-003, VAL-004, VAL-005, VAL-006, VAL-007 | A0, A8, B0–B3, release review |

## 16. Risks and planned responses

| Risk or dependency | Response |
|---|---|
| No server access now | Complete track A; keep performance/model acceptance pending; retain a runnable later test bundle |
| Candidate/quantization does not fit or kernels fail | Benchmark another pinned configuration/candidate; preserve gateway contract; record failed configuration |
| Long context harms latency/concurrency | Publish measured profiles, queue and limit concurrency, use bounded coverage workflows, and explicitly resolve unmet targets |
| Arabic OCR loses numbers or reading order | Preserve previews and uncertain values; compare local OCR alternatives; require review where quality is inadequate |
| Fluent but unsupported answers | Evaluate required facts/citations/refusal; validate source IDs and scope; revise retrieval/prompts or candidate based on attributed errors |
| Spreadsheet layout or formulas are ambiguous | Confirm mappings, preserve cached-value uncertainty, require refreshed/confirmed data before dependent calculations |
| Queue/worker crashes or repeated delivery | Durable DB state, outbox/reconciliation, idempotent steps, staged publication and recovery tests |
| Permission change exposes historical content | Track source dependencies, recheck display/history/export, invalidate scoped caches |
| Parser/model tries external access | No public egress, controlled local image paths, sandbox limits, offline acceptance checks |
| Company SSO/data/retention decisions unavailable | Use explicit development substitutes; continue independent work; block production acceptance of affected integration |
| Single-server failure | Separate backups, tested restoration, agreed downtime and RPO/RTO; HA remains a separate infrastructure decision |

## 17. Effort, responsibilities, and progress reporting

The original 32–51 engineering-day estimate remains a rough planning envelope for one developer with AI assistance, subject to re-estimation after A0–A1. It is not a promised calendar schedule or a validated estimate for all added governance and three-model evaluation work. Benchmark runs, corpus review, hardware access, SSO, and company approvals can add elapsed time. Do not count unavailable-server waiting as completed engineering work.

Engineering owns implementation, tests, model adapters, packaging, and evidence records. The project owner owns business priorities, representative questions, acceptance definitions, and policy decisions. Company IT owns hardware/access, internal identity/TLS/network, approved transfer, retention, and recovery requirements. Domain reviewers approve factual/citation/numerical gold answers and Arabic task judgments. One person may hold multiple roles; name the responsible people at kickoff.

Track each package as not started, in progress, implemented awaiting verification, verified, or blocked on a named dependency. Progress reports state what works, what was measured, unresolved failures, and the next independent task. Record server-dependent items separately so they do not stop useful laptop work or disappear from release requirements.

## 18. Immediate execution batch

1. Inspect and record the upstream base and repository instructions; initialize the enterprise branch without modifying the public deployment.
2. Install the constitution and this plan, add agent entry references, and create the 55-rule ledger and decision log.
3. Run existing checks, record baseline failures, and prepare small synthetic PDF/workbook/Arabic fixtures.
4. Implement dev-lite, injection interfaces, policy assembly, fake model endpoint, and versioned gateway contracts.
5. Add A/B/C profile manifests and fake profile-swap tests; prepare the real benchmark runner with pending scorecards.
6. Implement migrations, scoped identity, durable storage, RLS, and protected source delivery before expanding the corpus.
7. Build the first durable PDF upload-to-citation workflow with clearly labeled fake generation.
8. Continue formats, deterministic analysis, coverage workflows, UX, and packaging through track A while server access is unavailable.

The first useful development milestone is an authorized, durable, cited workflow on the laptop. The first validated company milestone is the same workflow using tested local components on approved infrastructure. Production acceptance follows the full gates above.

## 19. Baseline documents and future amendments

This plan consolidates the supplied `RAGBase_Local_Enterprise_Implementation_Plan.md`, constitution version 1.0.0, revised context/Arabic requirements, and the decision to prepare three candidates while server testing is unavailable. It supersedes the earlier execution sequence where they differ; it does not silently amend mandatory constitutional rules.

Architecture, candidate shortlist, tuning ranges, and effort are engineering decisions or estimates. Published model specifications are linked in Section 3 and must be pinned/rechecked before implementation. Runtime behavior, model quality, server capacity, and operational readiness require the verification described here.

Record amendments with date, rationale, affected package/gate/rule IDs, and required revalidation. Keep one canonical repository copy of this plan after implementation begins; the delivered Markdown file is the handoff baseline.
