"""Acceptance corpus and fixture-gateway runner. No live provider and no weights."""

import ast
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from acceptance.build import write_corpus
from acceptance.corpus import load_corpus
from acceptance.execute import (
    HttpEnterpriseClient,
    LiveProviderError,
    load_elements,
    retrieve,
)
from acceptance.metrics import aggregate, score_case
from acceptance.runner import main, run

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval/acceptance/v1"
REPORT = ROOT / "eval/acceptance/reports" / "v1-fixture-baseline.json"


def test_corpus_is_synthetic_pending_review_and_split():
    manifest, cases = load_corpus(CORPUS)
    assert manifest["reviewed"] is False
    assert manifest["contains_company_documents"] is False
    assert 200 <= len(cases) <= 300
    assert manifest["arabic_or_mixed_count"] >= 100
    assert manifest["tuning_count"] and manifest["held_out_count"]
    assert manifest["ocr_pending_count"] >= 10
    assert all(not case["reviewed"] and case["synthetic"] for case in cases)
    assert all(case["confidential"] is False for case in cases)
    registers = {case["arabic_register"] for case in cases}
    for register in (
        "formal",
        "egyptian",
        "mixed_terminology",
        "arabic_over_english",
        "matched_pair",
    ):
        assert register in registers
    tasks = {case["task"] for case in cases}
    for task in (
        "calculation",
        "coverage",
        "denied_access",
        "absent",
        "ocr_pending",
        "chart_pending",
    ):
        assert task in tasks
    pairs = {}
    for case in cases:
        if case["matched_pair_id"]:
            pairs.setdefault(case["matched_pair_id"], set()).add(case["language"])
    assert all(sides == {"en", "ar"} for sides in pairs.values())
    assert len(pairs) >= 8


def test_rebuilding_the_corpus_is_deterministic(tmp_path):
    first = write_corpus(tmp_path / "a")
    second = write_corpus(tmp_path / "b")
    assert first["cases_sha256"] == second["cases_sha256"]
    committed, _cases = load_corpus(CORPUS)
    assert committed["cases_sha256"] == first["cases_sha256"]


def test_fixture_baseline_does_not_qualify_and_keeps_failing_slices():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["production_qualified"] is False
    assert payload["execution"]["live_provider_called"] is False
    assert payload["execution"]["weights_downloaded"] is False
    assert {gate["decision"] for gate in payload["gates"]} <= {
        "not_qualified",
        "pending_hardware",
        "pending_owner_thresholds",
        "pending_checklist",
    }
    assert all(gate["production_qualified"] is False for gate in payload["gates"])
    assert "overall_pass_rate" not in payload
    pending = [row for row in payload["pending_slices"] if row["task"] == "ocr_pending"]
    assert pending
    assert all(row["scored_denominator"] == 0 for row in pending)
    failures = {
        (row["slice"], row["metric"]): row for row in payload["slices_with_failures"]
    }
    assert (
        failures[("arabic_over_english_unshared", "evidence_recall_at_10")][
            "denominator"
        ]
        > 0
    )
    assert failures[("distractor_ambiguity", "citation_precision_proxy")]["failures"]
    assert failures[("absent_with_distractor", "refusal_correct")]["numerator"] == 0
    numerical = [
        row
        for row in payload["slices"]
        if row["metric"] == "numerical_exact" and row["slice"] == "calculation"
    ]
    assert numerical
    assert all(row["numerator"] == row["denominator"] for row in numerical)
    coverage = [row for row in payload["slices"] if row["metric"] == "coverage_honesty"]
    assert coverage
    assert all(row["numerator"] == row["denominator"] for row in coverage)
    denial = [row for row in payload["slices"] if row["metric"] == "denial_correct"]
    assert denial
    assert all(row["numerator"] == row["denominator"] for row in denial)


def test_runner_refuses_live_provider_in_ci(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(LiveProviderError):
        run(
            CORPUS,
            tmp_path / "out.json",
            api_url="http://127.0.0.1:9",
            live_provider=True,
        )
    assert main(["--live-provider", "--api-url", "http://127.0.0.1:9"]) == 2


def test_http_fixture_gateway_is_opt_in(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"generation": "openrouter"})
        return httpx.Response(500)

    with pytest.raises(LiveProviderError):
        client = HttpEnterpriseClient(
            "http://enterprise.internal",
            "synthetic-token-value-0123456789",
            live_provider=False,
            version_map={
                "fixtures/harbor-en-v1.md": "00000000-0000-4000-8000-000000000001"
            },
            workspace="00000000-0000-4000-8000-000000000002",
            transport=httpx.MockTransport(handler),
        )
        import asyncio

        asyncio.run(client.check_gateway())
    monkeypatch.setenv("CI", "true")
    with pytest.raises(LiveProviderError):
        HttpEnterpriseClient(
            "http://enterprise.internal",
            "synthetic-token-value-0123456789",
            live_provider=True,
            version_map={
                "fixtures/harbor-en-v1.md": "00000000-0000-4000-8000-000000000001"
            },
            workspace="00000000-0000-4000-8000-000000000002",
            transport=httpx.MockTransport(handler),
        )


def test_http_errors_redact_the_token():
    client = HttpEnterpriseClient(
        "http://enterprise.internal",
        "synthetic-token-value-0123456789",
        live_provider=False,
        version_map={"fixtures/a.md": "00000000-0000-4000-8000-000000000001"},
        workspace="00000000-0000-4000-8000-000000000002",
    )
    assert "synthetic-token-value-0123456789" not in client.redact(
        "failed synthetic-token-value-0123456789"
    )


def test_execution_error_stays_in_the_denominator():
    case = {
        "id": "acc-v1-0001",
        "split": "held_out",
        "family": "calculation",
        "slices": ["calculation"],
        "language": "en",
        "task": "calculation",
        "question": "sum",
        "gold": {
            "supporting_element_ids": [],
            "expect_refusal": False,
            "expect_denial": False,
            "expect_clarification": False,
            "numerical": {"value": "1.00", "tolerance": "exact"},
            "plan": None,
            "ocr_status": "not_applicable",
            "chart_status": "not_applicable",
            "coverage": None,
            "forbidden_snippets": [],
        },
    }
    outcome = score_case(case, {"error": "execution_failed", "pending": False}, {})
    rows = aggregate([outcome])
    assert rows[0]["denominator"] == 1
    assert rows[0]["numerator"] == 0
    assert rows[0]["failures"] == ["acc-v1-0001"]


def test_exact_decimal_comparison_and_local_retrieval_do_not_download():
    assert Decimal("10.200000") == Decimal("10.20")
    elements = load_elements(CORPUS)
    harbor = [item for item in elements.values() if item["id"] == "en-harbor-berth-fee"]
    hits = retrieve(
        harbor, "What is the Synthetic Harbor Cooperative daily fee for Berth A?"
    )
    assert hits[0]["id"] == "en-harbor-berth-fee"
    tree = ast.parse((ROOT / "acceptance" / "val003.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "torch" not in names


def test_runner_has_no_token_argument():
    source = (ROOT / "acceptance" / "runner.py").read_text(encoding="utf-8")
    assert 'add_argument("--token"' not in source
    assert 'add_argument(\n        "--token"' not in source
    assert "--token-env" in source
    assert "ACCEPTANCE_DEV_TOKEN" in source
