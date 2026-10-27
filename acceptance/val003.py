"""VAL-003 context and serving harness.

The measurement record stays pending until the company server is available.
This module does not download weights, start a model, or invent latency.
"""

from __future__ import annotations

import json
from pathlib import Path

STAGES = (
    {
        "name": "4k",
        "total_context_tokens": 4096,
        "required_before_promotion": True,
        "positions": ("beginning", "middle", "end"),
    },
    {
        "name": "8k",
        "total_context_tokens": 8192,
        "required_before_promotion": True,
        "positions": ("beginning", "middle", "end"),
    },
    {
        "name": "16k",
        "total_context_tokens": 16384,
        "required_before_promotion": False,
        "promotion": "Only after 4K and 8K meet agreed quality, memory, and responsiveness.",
        "positions": ("beginning", "middle", "end"),
    },
    {
        "name": "32k",
        "total_context_tokens": 32768,
        "required_before_promotion": False,
        "promotion": "Only after 16K meets the same gates. Not a pilot requirement by itself.",
        "positions": ("beginning", "middle", "end"),
    },
)

MEASUREMENTS = (
    "cold_load_seconds",
    "prompt_processing_seconds",
    "time_to_first_user_visible_token_seconds",
    "generation_seconds",
    "whole_task_seconds",
    "queue_delay_seconds",
    "peak_process_ram_bytes",
    "peak_host_ram_bytes",
    "major_faults",
    "swap_activity",
    "cpu_contention",
)


def pending_report() -> dict:
    stages = []
    for stage in STAGES:
        stages.append(
            {
                **stage,
                "positions": list(stage["positions"]),
                "measurements": dict.fromkeys(MEASUREMENTS),
                "evidence_positions_tested": False,
                "decision": "pending_hardware",
            }
        )
    return {
        "gate": "VAL-003",
        "decision": "pending_hardware",
        "production_qualified": False,
        "qualifies_latency_or_context": False,
        "hardware_available": False,
        "weights_downloaded": False,
        "model_started": False,
        "stages": stages,
        "reason": (
            "Company-server context and serving measurements are pending. "
            "Passing startup or fitting weights would not close this gate. "
            "No latency, rate, or concurrency figure is claimed."
        ),
    }


def validate_measurement_file(path: Path) -> dict:
    """Check a supplied file. Present numbers still do not close production."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    for stage in payload.get("stages") or []:
        required = stage.get("required_before_promotion")
        measurements = stage.get("measurements") or {}
        for name in MEASUREMENTS:
            if required and measurements.get(name) is None:
                missing.append(f"{stage.get('name')}:{name}")
        if required and not stage.get("evidence_positions_tested"):
            missing.append(f"{stage.get('name')}:positions")
    decision = "pending_hardware" if missing else "measured_pending_owner_review"
    return {
        "gate": "VAL-003",
        "decision": decision,
        "production_qualified": False,
        "qualifies_latency_or_context": False,
        "missing": missing,
        "reason": (
            "Measurements can be recorded only from the target hardware. "
            "This harness does not turn those numbers into a production qualification."
        ),
    }


def write_report(path: Path, payload: dict | None = None) -> dict:
    payload = pending_report() if payload is None else payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
