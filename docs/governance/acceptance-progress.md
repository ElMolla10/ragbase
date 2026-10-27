# Acceptance groundwork

Updated 27 October 2026. Constitution 6.2.0. This record covers VAL-001, VAL-002, VAL-003, VAL-004, VAL-006, and VAL-007 only as far as code and a fixture run exist.

Nothing here qualifies production, company readiness, Arabic reliability, context length, latency, or offline recovery.

## Corpus 1.0.0

Path: `eval/acceptance/v1`. Review status: `machine_drafted_pending_human_review`. Reviewed cases: 0. Synthetic only. No company documents.

| Count | Value |
|---|---:|
| Cases | 246 |
| Tuning | 138 |
| Held out | 108 |
| Arabic or mixed | 125 |
| English | 121 |
| Arabic | 107 |
| Mixed language | 18 |
| Matched English/Arabic pairs | 20 |
| OCR pending | 14 |
| Chart pending | 6 |

Families: matched pairs 40, formal Arabic 24, English evidence 24, mixed terminology 20, Egyptian 18, Arabic over English 18, calculation 18, absent 16, OCR pending 14, long document 12, version conflict 12, coverage 12, denied access 12, chart pending 6.

Arabic registers on the cases: formal 41, matched pair 40, Egyptian 27, Arabic over English 19, mixed terminology 18. The remaining 101 cases are English-only and have no Arabic register.

Tasks: evidence 156, calculation 18, absent 16, OCR pending 14, version conflict 12, coverage 12, denied access 12, chart pending 6.

Held-out assignment is deterministic inside each family: indexes 0 and 1 modulo 5 are held out. Do not tune on that split until the owner confirms it.

OCR and chart cases are marked pending and are excluded from scored denominators. They are not transcribed gold.

## Fixture baseline

Command: `python scripts/run_acceptance.py`.

Mode: `fixture_gateway_in_process`. Gateway: `FixtureQuoteGateway`. Retrieval: local token overlap using `backend.search.index` normalization. That retrieval is not PostgreSQL `tsvector`, dense retrieval, or a context-window position test. Calculation uses the trusted DuckDB engine. Coverage uses the product planner and verdict. Live provider called: false. Weights downloaded: false.

Report: `eval/acceptance/reports/v1-fixture-baseline.json` and the matching Markdown file. `production_qualified` is false. There is no overall pass rate. 122 scored slice rows, 70 of them with at least one failure, and 16 pending slice rows.

| Gate | Decision |
|---|---|
| VAL-001 | not_qualified |
| VAL-002 | not_qualified |
| VAL-003 | pending_hardware |
| VAL-004 | pending_owner_thresholds |
| VAL-005 | not_qualified |
| VAL-006 | pending_checklist |
| VAL-007 | not_qualified |

Selected fixture results, summed across tuning and held-out. A passing row does not cancel a failing row.

| Slice | Metric | Passed / denominator |
|---|---|---:|
| calculation | numerical_exact | 18 / 18 |
| coverage | coverage_honesty | 12 / 12 |
| denied_access | denial_correct | 12 / 12 |
| long_document | evidence_recall_at_10 | 12 / 12 |
| arabic_over_english_unshared | evidence_recall_at_10 | 0 / 8 |
| egyptian_paraphrase | evidence_recall_at_10 | 0 / 6 |
| absent_with_distractor | refusal_correct | 0 / 6 |
| distractor_ambiguity | citation_precision_proxy | 0 / 24 |

Matched pairs: 20. Pairs whose fixture recall differs: 2 (`pair-17` and `pair-18`, English recalled and Arabic not). That is a lexical gap in this runner, not a reviewed Arabic quality gap.

The denial slice is a fixture scope check. It does not close VAL-005.

HTTP mode exists and reads the token from `ACCEPTANCE_DEV_TOKEN`. There is no `--token` argument. `--live-provider` is refused when `CI`, `GITHUB_ACTIONS`, `GITLAB_CI`, or `BUILDKITE` is set, and it is refused unless health generation is `openrouter`.

## VAL-003 and VAL-006

`python scripts/val003_context_harness.py` writes `eval/acceptance/reports/val003-pending.json`. Stages 4K and 8K are required. 16K and 32K stay conditional. Every measurement is null. No model starts and no weights are downloaded.

`python scripts/val006_recovery_checklist.py` writes a 17-item checklist. Every item is pending, including offline boot, Arabic OCR, network denial, and backup restore. The script refuses to mark the checklist complete.

## Open decisions for the owner

- Confirm VAL-004 definitions, denominators, and per-slice thresholds. Proposed figures are at least 90% supporting-evidence recall@10, at least 95% manually assessed citation precision, and 100% exact critical numerical results. They are not confirmed.
- Confirm VAL-002 business slices and Arabic thresholds, including the matched-pair gap. Do not average a failing slice into a passing total.
- Citation figures in the fixture report are proxies. They are not the manual citation assessment.
- Name a human reviewer. Reviewed status needs a new corpus version. Company samples stay on internal infrastructure.
- Keep scanned Arabic and charts pending until OCR and visual interpretation exist.
- Keep VAL-003 pending until the company server is measured.
- Keep VAL-006 pending until offline boot, network denial, and restore are run.
- Confirm that the held-out split stays unused for tuning.
