# RAGBase execution roadmap — seven phases

Updated: 18 October 2026. Canonical plan: execution-plan.md v2.0; constitution v2.0.0. These are seven main phases, P0 through P6. Subtasks and release checks are not additional phases.

| Phase | Specific work | Completion evidence | Current state |
|---|---|---|---|
| 1 — P0: Discovery and policy | Inventory server CPU/ISA/RAM/storage/OS; confirm users and corpus; reconcile CPU assumptions; pin candidate artifacts; prepare reviewed Arabic/English cases | Server inventory, reviewed requirements and acceptance dataset, runtime/budget decisions | CPU plan, constitution and laptop inventory done; actual server inventory and reviewed company cases pending |
| 2 — P1: Foundation and access-controlled storage | Separate enterprise configuration; migrations; durable original files; development identities; workspace permissions and RLS; revocation and integrity checks | Actual serving-role isolation tests, durable-file checks and working lightweight deployment | Development storage milestone implemented and tested; production identity/provisioning and full release gates remain open |
| 3 — P2: PDF ingestion, retrieval and cited answers | Persist jobs; bounded native PDF/TXT/Markdown extraction; page lineage; versioned chunks and authorized retrieval; replaceable generation adapter; streamed answers and source preview; small-context baseline | Authorized PDF upload to evidence-backed local answer, verified citations and resource/quality record | Local implementation complete: jobs, extraction, lexical search, local quote answers and previews tested; target-server/company-case acceptance pending |
| 4 — P3: OCR and Office formats | Bounded Arabic/English OCR; DOCX/PPTX/XLSX/CSV adapters; preserve tables, formulas/caches, identifiers and source locations; show uncertain extraction | Reviewed format fixtures and warnings, bounded resource use and correct lineage | Pending |
| 5 — P4: Trusted numerical analysis | Typed filtering/aggregation/comparison/join tools; dataset validation; decimal arithmetic; isolated DuckDB jobs and spill limits; charts and exports with lineage | Correct numerical gold cases, sandbox checks and inspectable calculation records | Pending |
| 6 — P5: Complete coverage and model qualification | Whole-document summaries and comparisons; coverage records; RTL/mixed-language UX and history; optional image understanding; compare 9B/35B-A3B/27B, 4B fallback and existing Gemma; prove swapping | Arabic/coverage/citation tests and measured enabled capabilities; validated model swaps without re-ingestion | Pending; candidates registered but unbenchmarked |
| 7 — P6: Hardening and company pilot | Integrate company SSO; enforce offline networking/assets; exercise quotas, cancellation and load; pin release bundle; restore backups; document operating procedures | Offline cold start, access tests, restore drill, measured service envelope and explicit pilot acceptance | Pending |

## Execution rules

P0 server discovery can remain open while independent development continues with fixtures. Real-server capacity and production selection cannot be accepted from laptop tests. Development evidence is not production qualification. The constitution applies throughout every phase, with rule-specific enforcement and verification records.

Current milestone evidence: 156 passing checks, a four-case synthetic live CPU smoke (three model calls), source-preview/isolation checks and an inference-outage check. See [Phase 3 completion](phase3-completion.md). The default profile starts without generation; server and production qualification remain pending.

The original phase allowances totalled 33–56 engineering days, excluding access/review delays; they are historical planning estimates, not a fresh remaining-time forecast. Re-estimate against current progress and server discovery.
