"""Per-slice acceptance metrics.

Rates stay with their own numerators and denominators. No combined score is a
release decision.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


def score_case(case: dict, observation: dict, elements: dict) -> dict:
    status = "pending_not_scored" if _pending(case, observation) else "scored"
    metrics: dict[str, dict] = {}
    failures = []
    if status == "scored" and observation.get("error"):
        for name in _applicable(case):
            metrics[name] = _result(0, 1, False)
            failures.append({"metric": name, "reason": observation["error"]})
    elif status == "scored":
        for name in _applicable(case):
            metric, reason = _METRIC[name](case, observation, elements)
            metrics[name] = metric
            if not metric["pass"]:
                failures.append({"metric": name, "reason": reason})
    return {
        "id": case["id"],
        "split": case["split"],
        "family": case["family"],
        "slices": list(case["slices"]),
        "language": case["language"],
        "task": case["task"],
        "status": status,
        "question": case["question"],
        "metrics": metrics,
        "failures": failures,
    }


def aggregate(outcomes: list[dict]) -> list[dict]:
    """One row per slice, split, and metric. Rows are not averaged together."""
    grouped: dict[tuple, list] = {}
    for outcome in outcomes:
        if outcome["status"] != "scored":
            continue
        for sliver in outcome["slices"]:
            for name, metric in outcome["metrics"].items():
                grouped.setdefault((sliver, outcome["split"], name), []).append(
                    (outcome, metric)
                )
    rows = []
    for (sliver, split, name), items in sorted(grouped.items()):
        passed = [item for item in items if item[1]["pass"]]
        element_numerator = sum(item[1]["numerator"] for item in items)
        element_denominator = sum(item[1]["denominator"] for item in items)
        failures = [item[0]["id"] for item in items if not item[1]["pass"]]
        rows.append(
            {
                "slice": sliver,
                "split": split,
                "metric": name,
                "numerator": len(passed),
                "denominator": len(items),
                "element_numerator": element_numerator,
                "element_denominator": element_denominator,
                "failures": failures,
                "threshold": None,
                "threshold_status": "unconfirmed",
                "decision": "not_a_release_decision",
            }
        )
    return rows


def pending_slices(outcomes: list[dict]) -> list[dict]:
    grouped: dict[tuple, int] = {}
    for outcome in outcomes:
        if outcome["status"] != "pending_not_scored":
            continue
        for sliver in outcome["slices"]:
            key = (sliver, outcome["split"], outcome["task"])
            grouped[key] = grouped.get(key, 0) + 1
    return [
        {
            "slice": sliver,
            "split": split,
            "task": task,
            "cases": count,
            "scored_denominator": 0,
            "decision": "pending_not_scored",
            "reason": _pending_reason(task),
        }
        for (sliver, split, task), count in sorted(grouped.items())
    ]


def matched_pair_gaps(cases: list[dict], outcomes: list[dict]) -> dict:
    by_id = {outcome["id"]: outcome for outcome in outcomes}
    pairs: dict[str, dict] = {}
    for case in cases:
        pair = case["matched_pair_id"]
        if pair:
            pairs.setdefault(pair, {})[case["language"]] = case["id"]
    differing = []
    for pair, sides in sorted(pairs.items()):
        english = _recall_pass(by_id.get(sides.get("en")))
        arabic = _recall_pass(by_id.get(sides.get("ar")))
        if english != arabic:
            differing.append(
                {"pair": pair, "english_recalled": english, "arabic_recalled": arabic}
            )
    return {
        "pairs": len(pairs),
        "denominator": len(pairs),
        "differing_pairs": len(differing),
        "pairs_with_a_gap": differing,
        "note": (
            "This is a fixture-lexical gap between matched questions, "
            "not a reviewed Arabic quality gap and not a release threshold."
        ),
    }


def _pending(case: dict, observation: dict) -> bool:
    return case["task"] in {"ocr_pending", "chart_pending"} or bool(
        observation.get("pending")
    )


def _applicable(case: dict) -> list[str]:
    task = case["task"]
    gold = case["gold"]
    names = []
    if task in {"evidence", "version_conflict"}:
        names.extend(["evidence_recall_at_10", "citation_precision_proxy"])
    if task == "absent" or gold["expect_refusal"]:
        names.append("refusal_correct")
    if task == "denied_access" or gold["expect_denial"]:
        names.append("denial_correct")
    if task == "calculation":
        names.append("numerical_exact")
    if task == "coverage":
        names.append("coverage_honesty")
    return names


def _result(numerator: int, denominator: int, passed: bool) -> dict:
    return {"numerator": numerator, "denominator": denominator, "pass": passed}


def _supports(citation: dict, gold_ids: list[str], elements: dict) -> bool:
    if citation.get("id") in gold_ids:
        return True
    snippet = citation.get("text") or ""
    if len(snippet) < 12:
        return False
    for gold_id in gold_ids:
        body = elements.get(gold_id, {}).get("text") or ""
        if body and (snippet in body or body in snippet):
            return True
    return False


def _retrieved(gold_id: str, observation: dict, elements: dict) -> bool:
    if gold_id in observation.get("retrieved_ids", []):
        return True
    body = elements.get(gold_id, {}).get("text") or ""
    return bool(body) and any(
        body in text for text in observation.get("retrieved_texts", [])
    )


def _recall(case, observation, elements):
    gold_ids = case["gold"]["supporting_element_ids"]
    hits = sum(1 for gold_id in gold_ids if _retrieved(gold_id, observation, elements))
    passed = hits == len(gold_ids) and bool(gold_ids)
    reason = "supporting_element_missing_from_top_10"
    return _result(hits, len(gold_ids), passed), reason


def _citation(case, observation, elements):
    gold_ids = case["gold"]["supporting_element_ids"]
    citations = observation.get("citations") or []
    if not citations:
        return _result(0, 1, False), "no_citation"
    precise = sum(
        1 for citation in citations if _supports(citation, gold_ids, elements)
    )
    passed = precise == len(citations)
    return _result(precise, len(citations), passed), "cited_unsupporting"


def _refusal(case, observation, elements):
    del case, elements
    leaked = bool(observation.get("leaked"))
    refused = bool(observation.get("insufficient")) and not observation.get("citations")
    passed = refused and not leaked and not observation.get("denied")
    reason = "answered_without_support" if not refused else "response_leaked_text"
    return _result(int(passed), 1, passed), reason


def _denial(case, observation, elements):
    del elements
    body = observation.get("denial_body") or ""
    leaked = bool(observation.get("leaked")) or any(
        snippet and snippet in body for snippet in case["gold"]["forbidden_snippets"]
    )
    passed = bool(observation.get("denied")) and not leaked
    return _result(int(passed), 1, passed), "denial_missing_or_content_leaked"


def _numerical(case, observation, elements):
    del elements
    gold = case["gold"]
    if gold["expect_clarification"]:
        passed = (
            bool(observation.get("clarification"))
            and observation.get("numeric_value") is None
        )
        return _result(
            int(passed), 1, passed
        ), "clarification_missing_or_number_returned"
    try:
        actual = Decimal(str(observation.get("numeric_value")))
        expected = Decimal(str(gold["numerical"]["value"]))
    except (InvalidOperation, TypeError):
        return _result(0, 1, False), "numeric_result_missing"
    passed = actual == expected
    return _result(int(passed), 1, passed), "numeric_result_differs"


def _coverage(case, observation, elements):
    del elements
    sections = observation.get("coverage_sections") or []
    gaps = [section for section in sections if section.get("status") != "covered"]
    label = observation.get("coverage_label")
    complete = observation.get("coverage_complete") is True
    if label == "complete" and (gaps or not complete):
        return _result(0, 1, False), "complete_label_with_gap"
    if label == "complete" and complete != (not gaps):
        return _result(0, 1, False), "complete_label_disagrees_with_sections"
    expected = bool(case["gold"]["coverage"]["expect_complete"])
    if expected:
        passed = label == "complete" and complete and not gaps
        return _result(int(passed), 1, passed), "expected_complete_coverage_was_partial"
    disclosed = {section.get("id") for section in gaps}
    required = set(case["gold"]["coverage"]["gap_section_ids"])
    passed = label == "partial" and not complete and required <= disclosed
    return _result(int(passed), 1, passed), "gap_missing_or_labeled_complete"


def _recall_pass(outcome: dict | None) -> bool | None:
    if not outcome:
        return None
    metric = outcome["metrics"].get("evidence_recall_at_10")
    if not metric:
        return None
    return bool(metric["pass"])


def _pending_reason(task: str) -> str:
    if task == "ocr_pending":
        return (
            "OCR is not implemented. Scanned cases are not scored and are not reviewed."
        )
    return "Visual interpretation is not implemented. Chart cases are not scored."


_METRIC = {
    "evidence_recall_at_10": _recall,
    "citation_precision_proxy": _citation,
    "refusal_correct": _refusal,
    "denial_correct": _denial,
    "numerical_exact": _numerical,
    "coverage_honesty": _coverage,
}
