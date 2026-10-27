"""Case schema for the synthetic acceptance corpus.

This version is machine-drafted. A reviewed flag in the file is rejected.
Human review requires a new corpus version.
"""

from __future__ import annotations

import re

from acceptance import CONSTITUTION_VERSION, CORPUS_VERSION, REVIEW_STATUS

CASE_ID = re.compile(r"^acc-v1-\d{4}$")
TASKS = frozenset(
    {
        "evidence",
        "absent",
        "calculation",
        "coverage",
        "denied_access",
        "version_conflict",
        "ocr_pending",
        "chart_pending",
    }
)
SPLITS = frozenset({"tuning", "held_out"})
LANGUAGES = frozenset({"ar", "en", "mixed"})
REGISTERS = frozenset(
    {
        "formal",
        "egyptian",
        "mixed_terminology",
        "arabic_over_english",
        "matched_pair",
    }
)
ACTORS = frozenset({"member", "outsider", "revoked"})


class CorpusError(ValueError):
    """The acceptance corpus or a case is not usable."""


def _gold():
    return {
        "supporting_element_ids": [],
        "expect_refusal": False,
        "expect_denial": False,
        "expect_clarification": False,
        "numerical": None,
        "plan": None,
        "ocr_status": "not_applicable",
        "chart_status": "not_applicable",
        "coverage": None,
        "forbidden_snippets": [],
    }


def make_gold(**overrides):
    gold = _gold()
    unknown = set(overrides) - set(gold)
    if unknown:
        raise CorpusError(f"Unknown gold fields: {sorted(unknown)}")
    gold.update(overrides)
    return gold


def validate_case(case: dict) -> None:
    required = {
        "id",
        "corpus_version",
        "constitution_version",
        "split",
        "review_status",
        "reviewed",
        "synthetic",
        "confidential",
        "language",
        "arabic_register",
        "matched_pair_id",
        "task",
        "family",
        "slices",
        "question",
        "actor",
        "documents",
        "gold",
        "notes",
    }
    missing = required - set(case)
    if missing:
        raise CorpusError(f"Case is missing {sorted(missing)}")
    if not CASE_ID.fullmatch(case["id"]):
        raise CorpusError(f"Bad case id {case['id']}")
    if case["corpus_version"] != CORPUS_VERSION:
        raise CorpusError("Unexpected corpus version")
    if case["constitution_version"] != CONSTITUTION_VERSION:
        raise CorpusError("Unexpected constitution version")
    if case["review_status"] != REVIEW_STATUS or case["reviewed"] is not False:
        raise CorpusError("Cases in this corpus version are pending human review")
    if case["synthetic"] is not True or case["confidential"] is not False:
        raise CorpusError("Only nonconfidential synthetic cases are allowed")
    if case["split"] not in SPLITS or case["language"] not in LANGUAGES:
        raise CorpusError(f"{case['id']} has an unknown split or language")
    if case["task"] not in TASKS or case["actor"] not in ACTORS:
        raise CorpusError(f"{case['id']} has an unknown task or actor")
    register = case["arabic_register"]
    if register is not None and register not in REGISTERS:
        raise CorpusError(f"{case['id']} has an unknown Arabic register")
    if case["language"] == "en" and register not in {None, "matched_pair"}:
        raise CorpusError(f"{case['id']} pairs an English case with an Arabic register")
    if case["language"] in {"ar", "mixed"} and register is None:
        raise CorpusError(f"{case['id']} needs an Arabic register")
    question = case["question"]
    if not isinstance(question, str) or not question.strip() or len(question) > 512:
        raise CorpusError(f"{case['id']} has an unusable question")
    if not isinstance(case["slices"], list) or not case["slices"]:
        raise CorpusError(f"{case['id']} needs slices")
    if not isinstance(case["documents"], list):
        raise CorpusError(f"{case['id']} documents must be a list")
    for document in case["documents"]:
        for key in (
            "fixture",
            "document_id",
            "version_label",
            "role",
            "authorized_for",
        ):
            if key not in document:
                raise CorpusError(f"{case['id']} document is missing {key}")
        if ".." in document["fixture"] or document["fixture"].startswith("/"):
            raise CorpusError(f"{case['id']} fixture path is not relative")
    gold = case["gold"]
    if set(gold) != set(_gold()):
        raise CorpusError(f"{case['id']} gold fields do not match the schema")
    _validate_task(case)


def _validate_task(case: dict) -> None:
    gold = case["gold"]
    task = case["task"]
    if task in {"ocr_pending", "chart_pending"}:
        if case["documents"]:
            raise CorpusError(
                f"{case['id']} pending case must not ship a source document"
            )
        if task == "ocr_pending" and gold["ocr_status"] != "pending_not_implemented":
            raise CorpusError(f"{case['id']} OCR case must stay pending")
        if (
            task == "chart_pending"
            and gold["chart_status"] != "pending_not_implemented"
        ):
            raise CorpusError(f"{case['id']} chart case must stay pending")
        return
    if (
        gold["ocr_status"] != "not_applicable"
        or gold["chart_status"] != "not_applicable"
    ):
        raise CorpusError(f"{case['id']} scored case cannot claim OCR or chart review")
    if task in {"evidence", "version_conflict"} and not gold["supporting_element_ids"]:
        raise CorpusError(f"{case['id']} needs supporting elements")
    if task == "absent" and not gold["expect_refusal"]:
        raise CorpusError(f"{case['id']} absent case must expect refusal")
    if task == "denied_access":
        if case["actor"] == "member" or not gold["expect_denial"]:
            raise CorpusError(f"{case['id']} denial case is not a denial")
        if not gold["forbidden_snippets"]:
            raise CorpusError(f"{case['id']} denial case needs a leak check")
    if task == "calculation":
        if len(case["documents"]) != 1 or not case["documents"][0]["fixture"].endswith(
            ".csv"
        ):
            raise CorpusError(f"{case['id']} calculation needs one CSV fixture")
        if gold["expect_clarification"]:
            if gold["numerical"] is not None:
                raise CorpusError(
                    f"{case['id']} clarification cannot also expect a number"
                )
        elif not gold["numerical"] or gold["numerical"].get("tolerance") != "exact":
            raise CorpusError(f"{case['id']} numerical gold must be exact")
    if task == "coverage":
        coverage = gold["coverage"] or {}
        if "expect_complete" not in coverage or "gap_section_ids" not in coverage:
            raise CorpusError(f"{case['id']} coverage gold is incomplete")


def validate_corpus(cases: list[dict]) -> dict:
    if not 200 <= len(cases) <= 300:
        raise CorpusError(f"Corpus size {len(cases)} is outside 200–300")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise CorpusError("Duplicate case ids")
    arabic = [case for case in cases if case["language"] in {"ar", "mixed"}]
    if len(arabic) < 100:
        raise CorpusError("Fewer than 100 Arabic or mixed-language cases")
    splits = {case["split"] for case in cases}
    if splits != SPLITS:
        raise CorpusError("Tuning and held-out splits are both required")
    registers = {case["arabic_register"] for case in arabic}
    for register in ("formal", "egyptian", "mixed_terminology", "arabic_over_english"):
        if register not in registers:
            raise CorpusError(f"Missing Arabic register {register}")
    pairs: dict[str, set[str]] = {}
    for case in cases:
        if case["matched_pair_id"]:
            pairs.setdefault(case["matched_pair_id"], set()).add(case["language"])
    if len(pairs) < 8 or any(sides != {"en", "ar"} for sides in pairs.values()):
        raise CorpusError("Matched pairs need one English and one Arabic case")
    tasks = {case["task"] for case in cases}
    for task in TASKS:
        if task not in tasks:
            raise CorpusError(f"Missing task {task}")
    pending_ocr = [case for case in cases if case["task"] == "ocr_pending"]
    if len(pending_ocr) < 10 or any(case["reviewed"] for case in pending_ocr):
        raise CorpusError("Scanned cases must be present and pending")
    if any(case["confidential"] or not case["synthetic"] for case in cases):
        raise CorpusError("Confidential or non-synthetic case")
    families = {case["family"] for case in cases}
    for family in families:
        family_splits = {case["split"] for case in cases if case["family"] == family}
        if family_splits != SPLITS:
            raise CorpusError(f"{family} is missing a split")
    return {
        "cases": len(cases),
        "arabic_or_mixed": len(arabic),
        "tuning": sum(case["split"] == "tuning" for case in cases),
        "held_out": sum(case["split"] == "held_out" for case in cases),
        "matched_pairs": len(pairs),
        "ocr_pending": len(pending_ocr),
        "chart_pending": sum(case["task"] == "chart_pending" for case in cases),
        "reviewed": 0,
    }
