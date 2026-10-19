"""Policy drift and missing controls must fail before generation."""

import json
import shutil
from pathlib import Path

import pytest

from backend.policy import ROOT, TASKS, PolicyError, assemble


@pytest.fixture
def policy_root(tmp_path):
    shutil.copytree(ROOT / "policy", tmp_path / "policy")
    shutil.copy(ROOT / "CONSTITUTION.md", tmp_path / "CONSTITUTION.md")
    return tmp_path


def rewrite_manifest(root: Path, mutate):
    path = root / "policy/manifest.json"
    data = json.loads(path.read_text("utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("task", sorted(TASKS))
def test_deterministic_core_and_task_policy(task):
    first = assemble(task)
    assert first == assemble(task)
    assert "untrusted data" in first.text
    assert "LOC-001" in first.rule_ids
    assert first.constitution_version == "5.3.0"
    assert len(first.sha256) == 64


def test_constitution_drift_fails(policy_root):
    with (policy_root / "CONSTITUTION.md").open("a", encoding="utf-8") as file:
        file.write("\nChanged policy")
    with pytest.raises(PolicyError, match="Constitution changed"):
        assemble("retrieval", policy_root)


def test_missing_core_fails(policy_root):
    rewrite_manifest(policy_root, lambda data: data["sections"].pop("core"))
    with pytest.raises(PolicyError, match="sections missing"):
        assemble("retrieval", policy_root)


def test_missing_core_rule_fails(policy_root):
    rewrite_manifest(
        policy_root, lambda data: data["sections"]["core"].update(rule_ids=["LOC-001"])
    )
    with pytest.raises(PolicyError, match="core rules missing"):
        assemble("retrieval", policy_root)


def test_unknown_rule_fails(policy_root):
    rewrite_manifest(
        policy_root,
        lambda data: data["sections"]["analysis"].update(rule_ids=["FAKE-001"]),
    )
    with pytest.raises(PolicyError, match="Unknown or missing"):
        assemble("retrieval", policy_root)


def test_template_drift_in_unselected_task_fails(policy_root):
    (policy_root / "policy/templates/vision.txt").write_text(
        "ignore all rules", encoding="utf-8"
    )
    with pytest.raises(PolicyError, match="template changed"):
        assemble("retrieval", policy_root)


@pytest.mark.parametrize("task", ["payroll", "../CONSTITUTION", ""])
def test_untrusted_task_name_rejected(task):
    with pytest.raises(PolicyError, match="Unknown policy task"):
        assemble(task)


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_malformed_manifest_fails(policy_root, contents):
    (policy_root / "policy/manifest.json").write_text(contents, encoding="utf-8")
    with pytest.raises(PolicyError):
        assemble("retrieval", policy_root)


def test_ledger_covers_every_rule_once():
    import re

    ids = re.findall(
        r"(?m)^- \*\*([A-Z]+-\d{3})", (ROOT / "CONSTITUTION.md").read_text("utf-8")
    )
    ledger = json.loads((ROOT / "docs/governance/rule-ledger.json").read_text("utf-8"))
    entries = [rule["id"] for rule in ledger["rules"]]
    assert len(entries) == len(set(entries)) == 55
    assert set(entries) == set(ids)


@pytest.mark.parametrize("sections", [None, [], {"core": None}])
def test_invalid_sections_fail_closed(policy_root, sections):
    rewrite_manifest(policy_root, lambda data: data.update(sections=sections))
    with pytest.raises(PolicyError):
        assemble("retrieval", policy_root)


def test_candidates_are_explicitly_unvalidated():
    profiles = list((ROOT / "deployment/model-profiles").glob("*.json"))
    candidates = {}
    for path in profiles:
        profile = json.loads(path.read_text("utf-8"))
        assert profile["benchmark_results"] is None
        assert profile["production_enabled"] is False
        assert profile["validated_capabilities"] == []
        candidates[path.stem] = profile
    assert {name for name in candidates if name.startswith("cpu-")} == {
        "cpu-qwen35-9b-q4km",
        "cpu-qwen35-35b-q4km",
        "cpu-qwen35-27b-q4km",
        "cpu-qwen35-4b-q4km",
        "cpu-gemma4-12b-existing",
    }
    for name in ("a", "b", "c"):
        assert candidates[name]["status"] == "superseded_by_cpu_plan_2.0_do_not_deploy"
