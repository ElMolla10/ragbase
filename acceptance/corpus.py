"""Load the versioned acceptance corpus and reject a mismatched manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from acceptance.schema import CorpusError, validate_case, validate_corpus


def load_corpus(path: Path) -> tuple[dict, list[dict]]:
    manifest_path = path / "manifest.json"
    cases_path = path / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise CorpusError("Acceptance corpus files are missing")
    raw = cases_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cases_sha256") != digest:
        raise CorpusError("cases.jsonl does not match the manifest hash")
    cases = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    for case in cases:
        validate_case(case)
    summary = validate_corpus(cases)
    if manifest.get("case_count") != summary["cases"]:
        raise CorpusError("Manifest case count does not match the cases")
    if manifest.get("arabic_or_mixed_count") != summary["arabic_or_mixed"]:
        raise CorpusError("Manifest Arabic count does not match the cases")
    if manifest.get("reviewed") is not False:
        raise CorpusError("Manifest must keep this corpus pending human review")
    if manifest.get("contains_company_documents") is not False:
        raise CorpusError("Manifest must say that company documents are absent")
    return manifest, cases
