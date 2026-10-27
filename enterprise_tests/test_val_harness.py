"""VAL-003 and VAL-006 stay pending without hardware or a recovery drill."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from acceptance.val003 import pending_report, validate_measurement_file, write_report
from acceptance.val006 import ChecklistError, checklist, claim_complete, write_checklist


def test_val003_pending_report_has_null_measurements(tmp_path):
    payload = write_report(tmp_path / "val003.json")
    assert payload["decision"] == "pending_hardware"
    assert payload["production_qualified"] is False
    assert payload["weights_downloaded"] is False
    assert payload["model_started"] is False
    for stage in payload["stages"]:
        assert set(stage["positions"]) == {"beginning", "middle", "end"}
        assert all(value is None for value in stage["measurements"].values())
        assert stage["decision"] == "pending_hardware"


def test_val003_measurement_file_cannot_qualify_production(tmp_path):
    pending = pending_report()
    incomplete = tmp_path / "partial.json"
    incomplete.write_text(json.dumps(pending), encoding="utf-8")
    checked = validate_measurement_file(incomplete)
    assert checked["decision"] == "pending_hardware"
    assert checked["production_qualified"] is False
    assert checked["missing"]
    for stage in pending["stages"]:
        if stage["required_before_promotion"]:
            stage["measurements"] = dict.fromkeys(stage["measurements"], 1)
            stage["evidence_positions_tested"] = True
    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps(pending), encoding="utf-8")
    measured = validate_measurement_file(complete)
    assert measured["decision"] == "measured_pending_owner_review"
    assert measured["production_qualified"] is False
    assert measured["qualifies_latency_or_context"] is False


def test_val006_checklist_cannot_be_closed(tmp_path):
    payload = write_checklist(tmp_path / "val006.json")
    assert payload["decision"] == "pending_checklist"
    assert payload["offline_qualified"] is False
    assert payload["production_qualified"] is False
    assert payload["items"]
    assert all(
        item["status"] == "pending" and item["evidence"] is None
        for item in payload["items"]
    )
    assert (tmp_path / "val006.md").is_file()
    with pytest.raises(ChecklistError):
        claim_complete()
    assert checklist()["decision"] == "pending_checklist"


def test_harness_scripts_write_pending_records(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys, "argv", ["val003", "--output", str(tmp_path / "context.json")]
    )
    with pytest.raises(SystemExit) as success:
        runpy.run_path(
            str(root / "scripts/val003_context_harness.py"), run_name="__main__"
        )
    assert success.value.code == 0
    assert json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))[
        "decision"
    ] == ("pending_hardware")
    monkeypatch.setattr(
        sys,
        "argv",
        ["val006", "--output", str(tmp_path / "check.json"), "--claim-complete"],
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(
            str(root / "scripts/val006_recovery_checklist.py"), run_name="__main__"
        )
    assert caught.value.code == 2
    assert "cannot be marked complete" in capsys.readouterr().out
    monkeypatch.setattr(
        sys, "argv", ["val006", "--output", str(tmp_path / "check.json")]
    )
    with pytest.raises(SystemExit) as success:
        runpy.run_path(
            str(root / "scripts/val006_recovery_checklist.py"), run_name="__main__"
        )
    assert success.value.code == 0
    written = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert written["decision"] == "pending_checklist"
