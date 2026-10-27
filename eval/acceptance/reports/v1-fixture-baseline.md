# Acceptance report 1.0.0

This is a fixture-mode measurement. **Production is not qualified.**

Corpus `1.0.0` (343d9f046527), constitution 6.2.0, generated 2026-10-27T13:20:37Z.

Cases: 246. Arabic or mixed: 125. Tuning: 138. Held out: 108. Reviewed: 0.

OCR pending: 14. Chart pending: 6. Those cases stay out of scored denominators.

## Gates

| Gate | Decision |
|---|---|
| VAL-001 | not_qualified |
| VAL-002 | not_qualified |
| VAL-003 | pending_hardware |
| VAL-004 | pending_owner_thresholds |
| VAL-005 | not_qualified |
| VAL-006 | pending_checklist |
| VAL-007 | not_qualified |

## Slices

Each row is one slice, one split, and one metric. A higher row does not cancel a lower one.

| Slice | Split | Metric | Passed | Denominator | Failures |
|---|---|---|---:|---:|---:|
| absent | held_out | refusal_correct | 4 | 7 | 3 |
| absent | tuning | refusal_correct | 6 | 9 | 3 |
| absent_with_distractor | held_out | refusal_correct | 0 | 3 | 3 |
| absent_with_distractor | tuning | refusal_correct | 0 | 3 | 3 |
| arabic_arabic_over_english | held_out | citation_precision_proxy | 4 | 8 | 4 |
| arabic_arabic_over_english | held_out | evidence_recall_at_10 | 4 | 8 | 4 |
| arabic_arabic_over_english | tuning | citation_precision_proxy | 6 | 10 | 4 |
| arabic_arabic_over_english | tuning | evidence_recall_at_10 | 6 | 10 | 4 |
| arabic_egyptian | held_out | citation_precision_proxy | 6 | 8 | 2 |
| arabic_egyptian | held_out | denial_correct | 1 | 1 | 0 |
| arabic_egyptian | held_out | evidence_recall_at_10 | 6 | 8 | 2 |
| arabic_egyptian | held_out | numerical_exact | 1 | 1 | 0 |
| arabic_egyptian | held_out | refusal_correct | 0 | 1 | 1 |
| arabic_egyptian | tuning | citation_precision_proxy | 6 | 10 | 4 |
| arabic_egyptian | tuning | coverage_honesty | 1 | 1 | 0 |
| arabic_egyptian | tuning | evidence_recall_at_10 | 6 | 10 | 4 |
| arabic_egyptian | tuning | refusal_correct | 2 | 2 | 0 |
| arabic_formal | held_out | citation_precision_proxy | 5 | 10 | 5 |
| arabic_formal | held_out | coverage_honesty | 1 | 1 | 0 |
| arabic_formal | held_out | evidence_recall_at_10 | 10 | 10 | 0 |
| arabic_formal | held_out | numerical_exact | 4 | 4 | 0 |
| arabic_formal | held_out | refusal_correct | 1 | 1 | 0 |
| arabic_formal | tuning | citation_precision_proxy | 7 | 14 | 7 |
| arabic_formal | tuning | coverage_honesty | 2 | 2 | 0 |
| arabic_formal | tuning | denial_correct | 2 | 2 | 0 |
| arabic_formal | tuning | evidence_recall_at_10 | 14 | 14 | 0 |
| arabic_formal | tuning | numerical_exact | 1 | 1 | 0 |
| arabic_formal | tuning | refusal_correct | 1 | 2 | 1 |
| arabic_matched_pair | held_out | citation_precision_proxy | 15 | 16 | 1 |
| arabic_matched_pair | held_out | evidence_recall_at_10 | 15 | 16 | 1 |
| arabic_matched_pair | tuning | citation_precision_proxy | 23 | 24 | 1 |
| arabic_matched_pair | tuning | evidence_recall_at_10 | 23 | 24 | 1 |
| arabic_mixed_terminology | held_out | citation_precision_proxy | 4 | 4 | 0 |
| arabic_mixed_terminology | held_out | denial_correct | 1 | 1 | 0 |
| arabic_mixed_terminology | held_out | evidence_recall_at_10 | 4 | 4 | 0 |
| arabic_mixed_terminology | held_out | numerical_exact | 1 | 1 | 0 |
| arabic_mixed_terminology | tuning | citation_precision_proxy | 5 | 8 | 3 |
| arabic_mixed_terminology | tuning | coverage_honesty | 1 | 1 | 0 |
| arabic_mixed_terminology | tuning | evidence_recall_at_10 | 6 | 8 | 2 |
| arabic_mixed_terminology | tuning | numerical_exact | 1 | 1 | 0 |
| arabic_or_mixed | held_out | citation_precision_proxy | 26 | 38 | 12 |
| arabic_or_mixed | held_out | coverage_honesty | 1 | 1 | 0 |
| arabic_or_mixed | held_out | denial_correct | 2 | 2 | 0 |
| arabic_or_mixed | held_out | evidence_recall_at_10 | 31 | 38 | 7 |
| arabic_or_mixed | held_out | numerical_exact | 6 | 6 | 0 |
| arabic_or_mixed | held_out | refusal_correct | 1 | 2 | 1 |
| arabic_or_mixed | tuning | citation_precision_proxy | 35 | 54 | 19 |
| arabic_or_mixed | tuning | coverage_honesty | 4 | 4 | 0 |
| arabic_or_mixed | tuning | denial_correct | 2 | 2 | 0 |
| arabic_or_mixed | tuning | evidence_recall_at_10 | 43 | 54 | 11 |
| arabic_or_mixed | tuning | numerical_exact | 2 | 2 | 0 |
| arabic_or_mixed | tuning | refusal_correct | 3 | 4 | 1 |
| arabic_over_english | held_out | citation_precision_proxy | 4 | 8 | 4 |
| arabic_over_english | held_out | evidence_recall_at_10 | 4 | 8 | 4 |
| arabic_over_english | tuning | citation_precision_proxy | 6 | 10 | 4 |
| arabic_over_english | tuning | evidence_recall_at_10 | 6 | 10 | 4 |
| arabic_over_english_shared_token | held_out | citation_precision_proxy | 4 | 4 | 0 |
| arabic_over_english_shared_token | held_out | evidence_recall_at_10 | 4 | 4 | 0 |
| arabic_over_english_shared_token | tuning | citation_precision_proxy | 6 | 6 | 0 |
| arabic_over_english_shared_token | tuning | evidence_recall_at_10 | 6 | 6 | 0 |
| arabic_over_english_unshared | held_out | citation_precision_proxy | 0 | 4 | 4 |
| arabic_over_english_unshared | held_out | evidence_recall_at_10 | 0 | 4 | 4 |
| arabic_over_english_unshared | tuning | citation_precision_proxy | 0 | 4 | 4 |
| arabic_over_english_unshared | tuning | evidence_recall_at_10 | 0 | 4 | 4 |
| calculation | held_out | numerical_exact | 8 | 8 | 0 |
| calculation | tuning | numerical_exact | 10 | 10 | 0 |
| coverage | held_out | coverage_honesty | 6 | 6 | 0 |
| coverage | tuning | coverage_honesty | 6 | 6 | 0 |
| denied_access | held_out | denial_correct | 6 | 6 | 0 |
| denied_access | tuning | denial_correct | 6 | 6 | 0 |
| distractor_ambiguity | held_out | citation_precision_proxy | 0 | 10 | 10 |
| distractor_ambiguity | held_out | evidence_recall_at_10 | 10 | 10 | 0 |
| distractor_ambiguity | tuning | citation_precision_proxy | 0 | 14 | 14 |
| distractor_ambiguity | tuning | evidence_recall_at_10 | 14 | 14 | 0 |
| egyptian | held_out | citation_precision_proxy | 6 | 8 | 2 |
| egyptian | held_out | evidence_recall_at_10 | 6 | 8 | 2 |
| egyptian | tuning | citation_precision_proxy | 6 | 10 | 4 |
| egyptian | tuning | evidence_recall_at_10 | 6 | 10 | 4 |
| egyptian_paraphrase | held_out | citation_precision_proxy | 0 | 2 | 2 |
| egyptian_paraphrase | held_out | evidence_recall_at_10 | 0 | 2 | 2 |
| egyptian_paraphrase | tuning | citation_precision_proxy | 0 | 4 | 4 |
| egyptian_paraphrase | tuning | evidence_recall_at_10 | 0 | 4 | 4 |
| english_evidence | held_out | citation_precision_proxy | 5 | 10 | 5 |
| english_evidence | held_out | evidence_recall_at_10 | 10 | 10 | 0 |
| english_evidence | tuning | citation_precision_proxy | 7 | 14 | 7 |
| english_evidence | tuning | evidence_recall_at_10 | 14 | 14 | 0 |
| evidence | held_out | citation_precision_proxy | 49 | 66 | 17 |
| evidence | held_out | evidence_recall_at_10 | 59 | 66 | 7 |
| evidence | tuning | citation_precision_proxy | 64 | 90 | 26 |
| evidence | tuning | evidence_recall_at_10 | 79 | 90 | 11 |
| formal_ar | held_out | citation_precision_proxy | 5 | 10 | 5 |
| formal_ar | held_out | evidence_recall_at_10 | 10 | 10 | 0 |
| formal_ar | tuning | citation_precision_proxy | 7 | 14 | 7 |
| formal_ar | tuning | evidence_recall_at_10 | 14 | 14 | 0 |
| long_document | held_out | citation_precision_proxy | 6 | 6 | 0 |
| long_document | held_out | evidence_recall_at_10 | 6 | 6 | 0 |
| long_document | tuning | citation_precision_proxy | 6 | 6 | 0 |
| long_document | tuning | evidence_recall_at_10 | 6 | 6 | 0 |
| matched_pair | held_out | citation_precision_proxy | 15 | 16 | 1 |
| matched_pair | held_out | evidence_recall_at_10 | 15 | 16 | 1 |
| matched_pair | tuning | citation_precision_proxy | 23 | 24 | 1 |
| matched_pair | tuning | evidence_recall_at_10 | 23 | 24 | 1 |
| matched_pair_paraphrase | held_out | citation_precision_proxy | 0 | 1 | 1 |
| matched_pair_paraphrase | held_out | evidence_recall_at_10 | 0 | 1 | 1 |
| matched_pair_paraphrase | tuning | citation_precision_proxy | 0 | 1 | 1 |
| matched_pair_paraphrase | tuning | evidence_recall_at_10 | 0 | 1 | 1 |
| mixed_paraphrase | held_out | citation_precision_proxy | 1 | 1 | 0 |
| mixed_paraphrase | held_out | evidence_recall_at_10 | 1 | 1 | 0 |
| mixed_paraphrase | tuning | citation_precision_proxy | 0 | 3 | 3 |
| mixed_paraphrase | tuning | evidence_recall_at_10 | 1 | 3 | 2 |
| mixed_terminology | held_out | citation_precision_proxy | 8 | 8 | 0 |
| mixed_terminology | held_out | evidence_recall_at_10 | 8 | 8 | 0 |
| mixed_terminology | tuning | citation_precision_proxy | 9 | 12 | 3 |
| mixed_terminology | tuning | evidence_recall_at_10 | 10 | 12 | 2 |
| version_ambiguous | held_out | citation_precision_proxy | 0 | 3 | 3 |
| version_ambiguous | held_out | evidence_recall_at_10 | 3 | 3 | 0 |
| version_ambiguous | tuning | citation_precision_proxy | 0 | 3 | 3 |
| version_ambiguous | tuning | evidence_recall_at_10 | 3 | 3 | 0 |
| version_conflict | held_out | citation_precision_proxy | 3 | 6 | 3 |
| version_conflict | held_out | evidence_recall_at_10 | 6 | 6 | 0 |
| version_conflict | tuning | citation_precision_proxy | 3 | 6 | 3 |
| version_conflict | tuning | evidence_recall_at_10 | 6 | 6 | 0 |

## Pending slices

- arabic_arabic_over_english / held_out: 1 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_egyptian / held_out: 1 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- arabic_egyptian / tuning: 2 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_formal / held_out: 2 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_formal / tuning: 1 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- arabic_formal / tuning: 1 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_mixed_terminology / held_out: 1 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_mixed_terminology / tuning: 1 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_or_mixed / held_out: 1 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- arabic_or_mixed / held_out: 4 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- arabic_or_mixed / tuning: 1 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- arabic_or_mixed / tuning: 4 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- chart_pending / held_out: 3 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- chart_pending / tuning: 3 cases, pending_not_scored. Visual interpretation is not implemented. Chart cases are not scored.
- ocr_pending / held_out: 6 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.
- ocr_pending / tuning: 8 cases, pending_not_scored. OCR is not implemented. Scanned cases are not scored and are not reviewed.

## Matched-pair lexical gap

Pairs: 20. Pairs whose recall proxy differs: 2.
This is a fixture-lexical gap between matched questions, not a reviewed Arabic quality gap and not a release threshold.

## Open decisions

- Confirm VAL-004 definitions, denominators, and per-slice thresholds before any pilot. Proposed figures are at least 90% supporting-evidence recall@10, at least 95% manually assessed citation precision, and 100% exact critical numerical results. They are not confirmed and are not measured results.
- Confirm VAL-002 business slices and acceptable Arabic thresholds, including the matched English/Arabic gap. Do not average a failing slice into a passing total.
- Citation figures in this report are proxies from fixture citations. They are not the manual citation-precision assessment VAL-004 requires.
- Name the reviewer for this machine-drafted corpus. Reviewed status requires a new corpus version. Synthetic fixtures do not establish company readiness; representative company samples stay on internal infrastructure.
- Scanned Arabic and chart cases stay pending until OCR and visual interpretation exist. Do not transcribe them into verified gold.
- VAL-003 stays pending until the company server is measured. No latency, context, or concurrency number should be inferred from this laptop or fixture run.
- VAL-006 stays pending until the offline boot, network-denial record, and backup restore are actually run.
- Held-out cases are separated so they can stay unused for tuning. Confirm that split before anyone tunes prompts or retrieval on this corpus.

## Execution

- mode: fixture_gateway_in_process
- gateway: backend.answers.fixtures.FixtureQuoteGateway
- retrieval: local_token_overlap_using_backend.search.index.normalize_and_query_terms
- retrieval_limitation: Fixture retrieval is not PostgreSQL tsvector, dense retrieval, or a context-window position test. VAL-003 remains pending.
- calculation: trusted_duckdb_after_fixture_planner_or_gold_plan
- coverage: backend.coverage_planner_and_verdict_with_fixture_gateway
- live_provider_called: False
- weights_downloaded: False
- production_qualified: False
- qualifies_company_readiness: False
- qualifies_arabic_reliability: False
- qualifies_context_or_latency: False
