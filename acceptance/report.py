"""Versioned acceptance report. Gate decisions stay explicit."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from acceptance.metrics import aggregate, matched_pair_gaps, pending_slices

OPEN_DECISIONS = (
    "Confirm VAL-004 definitions, denominators, and per-slice thresholds before any pilot. Proposed figures are at least 90% supporting-evidence recall@10, at least 95% manually assessed citation precision, and 100% exact critical numerical results. They are not confirmed and are not measured results.",
    "Confirm VAL-002 business slices and acceptable Arabic thresholds, including the matched English/Arabic gap. Do not average a failing slice into a passing total.",
    "Citation figures in this report are proxies from fixture citations. They are not the manual citation-precision assessment VAL-004 requires.",
    "Name the reviewer for this machine-drafted corpus. Reviewed status requires a new corpus version. Synthetic fixtures do not establish company readiness; representative company samples stay on internal infrastructure.",
    "Scanned Arabic and chart cases stay pending until OCR and visual interpretation exist. Do not transcribe them into verified gold.",
    "VAL-003 stays pending until the company server is measured. No latency, context, or concurrency number should be inferred from this laptop or fixture run.",
    "VAL-006 stays pending until the offline boot, network-denial record, and backup restore are actually run.",
    "Held-out cases are separated so they can stay unused for tuning. Confirm that split before anyone tunes prompts or retrieval on this corpus.",
)


def build_report(
    manifest: dict, cases: list[dict], outcomes: list[dict], execution: dict
) -> dict:
    slices = aggregate(outcomes)
    below = [
        row
        for row in slices
        if row["denominator"] and row["numerator"] != row["denominator"]
    ]
    return {
        "report_version": "1.0.0",
        "corpus_version": manifest["corpus_version"],
        "constitution_version": manifest["constitution_version"],
        "cases_sha256": manifest["cases_sha256"],
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "execution": execution,
        "corpus": {
            "cases": manifest["case_count"],
            "arabic_or_mixed": manifest["arabic_or_mixed_count"],
            "tuning": manifest["tuning_count"],
            "held_out": manifest["held_out_count"],
            "matched_pairs": manifest["matched_pair_count"],
            "ocr_pending": manifest["ocr_pending_count"],
            "chart_pending": manifest["chart_pending_count"],
            "reviewed": 0,
            "review_status": manifest["review_status"],
            "synthetic_only": True,
            "contains_company_documents": False,
        },
        "production_qualified": False,
        "gates": _gates(),
        "slices": slices,
        "slices_with_failures": below,
        "pending_slices": pending_slices(outcomes),
        "matched_pair_lexical_gap": matched_pair_gaps(cases, outcomes),
        "proposed_objectives_not_confirmed": {
            "evidence_recall_at_10": 0.90,
            "manual_citation_precision": 0.95,
            "critical_numerical_exact": 1.0,
            "status": "proposed_in_VAL-004_not_confirmed",
        },
        "cases": [
            {
                "id": outcome["id"],
                "split": outcome["split"],
                "family": outcome["family"],
                "task": outcome["task"],
                "status": outcome["status"],
                "failures": outcome["failures"],
            }
            for outcome in outcomes
            if outcome["failures"] or outcome["status"] != "scored"
        ],
        "open_decisions": list(OPEN_DECISIONS),
    }


def _gates() -> list[dict]:
    return [
        {
            "id": "VAL-001",
            "decision": "not_qualified",
            "production_qualified": False,
            "reasons": [
                "The corpus is machine-drafted and pending human review.",
                "Fixtures are synthetic. Company samples are not included.",
                "Scanned and chart cases are pending, not passed.",
                "A fixture-gateway run is not representative acceptance.",
            ],
        },
        {
            "id": "VAL-002",
            "decision": "not_qualified",
            "production_qualified": False,
            "reasons": [
                "Arabic and mixed-language cases are machine-drafted, not reviewed.",
                "Required business slices and thresholds are not agreed.",
                "Scanned Arabic is pending because OCR is not implemented.",
                "Fixture lexical proxies are not an Arabic quality measurement.",
                "Failing Arabic slices stay visible with their own denominators.",
            ],
        },
        {
            "id": "VAL-003",
            "decision": "pending_hardware",
            "production_qualified": False,
            "reasons": [
                "No company-server context or serving measurement is available.",
                "The harness records null measurements and does not download weights.",
            ],
        },
        {
            "id": "VAL-004",
            "decision": "pending_owner_thresholds",
            "production_qualified": False,
            "reasons": [
                "Proposed recall, citation, and numerical objectives are not confirmed.",
                "Citation precision here is a proxy, not a manual assessment.",
                "Per-slice rates are reported and are not averaged into a pass.",
            ],
        },
        {
            "id": "VAL-005",
            "decision": "not_qualified",
            "production_qualified": False,
            "reasons": [
                "Denied-access cases exercise the fixture scope check only.",
                "They do not replace serving-role, revocation, injection, or network tests.",
            ],
        },
        {
            "id": "VAL-006",
            "decision": "pending_checklist",
            "production_qualified": False,
            "reasons": [
                "The recovery and offline checklist is pending.",
                "No offline boot or backup restore was run for this report.",
            ],
        },
        {
            "id": "VAL-007",
            "decision": "not_qualified",
            "production_qualified": False,
            "reasons": [
                "This report publishes fixture-mode denominators, failures, and open gates.",
                "It does not label the system enterprise-ready, Arabic-reliable, or offline-validated.",
            ],
        },
    ]


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = path.with_suffix(".md")
    markdown.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    corpus = payload["corpus"]
    lines = [
        "# Acceptance report " + payload["report_version"],
        "",
        "This is a fixture-mode measurement. **Production is not qualified.**",
        "",
        f"Corpus `{payload['corpus_version']}` ({payload['cases_sha256'][:12]}), "
        f"constitution {payload['constitution_version']}, generated {payload['generated_at']}.",
        "",
        f"Cases: {corpus['cases']}. Arabic or mixed: {corpus['arabic_or_mixed']}. "
        f"Tuning: {corpus['tuning']}. Held out: {corpus['held_out']}. "
        f"Reviewed: {corpus['reviewed']}.",
        "",
        "OCR pending: "
        f"{corpus['ocr_pending']}. Chart pending: {corpus['chart_pending']}. "
        "Those cases stay out of scored denominators.",
        "",
        "## Gates",
        "",
        "| Gate | Decision |",
        "|---|---|",
    ]
    for gate in payload["gates"]:
        lines.append(f"| {gate['id']} | {gate['decision']} |")
    lines.extend(
        [
            "",
            "## Slices",
            "",
            "Each row is one slice, one split, and one metric. "
            "A higher row does not cancel a lower one.",
            "",
            "| Slice | Split | Metric | Passed | Denominator | Failures |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["slices"]:
        lines.append(
            f"| {row['slice']} | {row['split']} | {row['metric']} | "
            f"{row['numerator']} | {row['denominator']} | {len(row['failures'])} |"
        )
    lines.extend(["", "## Pending slices", ""])
    for row in payload["pending_slices"]:
        lines.append(
            f"- {row['slice']} / {row['split']}: {row['cases']} cases, {row['decision']}. {row['reason']}"
        )
    gap = payload["matched_pair_lexical_gap"]
    lines.extend(
        [
            "",
            "## Matched-pair lexical gap",
            "",
            f"Pairs: {gap['denominator']}. Pairs whose recall proxy differs: {gap['differing_pairs']}.",
            gap["note"],
            "",
            "## Open decisions",
            "",
        ]
    )
    for item in payload["open_decisions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Execution", ""])
    for key, value in payload["execution"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)
