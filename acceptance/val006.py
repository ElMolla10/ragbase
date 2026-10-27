"""VAL-006 offline and recovery checklist.

The checklist is the deliverable. This script cannot close the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

ITEMS = (
    (
        "local-boot-offline",
        "local",
        "Boot a prepared local-mode release without public internet.",
    ),
    (
        "local-enabled-formats",
        "local",
        "Exercise each enabled document format on the prepared release.",
    ),
    (
        "local-arabic-ocr",
        "local",
        "Run Arabic OCR on the prepared release. OCR is not implemented yet.",
    ),
    (
        "local-evidence-question",
        "local",
        "Ask an evidence-backed question and keep the citation.",
    ),
    (
        "local-workbook-calculation",
        "local",
        "Perform a workbook calculation through the trusted calculator.",
    ),
    ("local-open-citation", "local", "Open a citation from the answer."),
    ("local-export", "local", "Export a result table."),
    (
        "local-restart",
        "local",
        "Restart services and confirm authorized data is still available.",
    ),
    (
        "local-restore",
        "local",
        "Restore a backup onto a separate check, including database and files.",
    ),
    ("local-network-denial", "local", "Record network-denial evidence for local mode."),
    (
        "local-no-public-dependency",
        "local",
        "Show that local mode did not require a public connection.",
    ),
    (
        "remote-relay-only",
        "remote",
        "Show that only the generation relay reaches the approved provider.",
    ),
    ("remote-missing-key", "remote", "Missing or invalid provider keys fail safely."),
    ("remote-upstream-outage", "remote", "An upstream outage fails safely."),
    (
        "remote-local-services",
        "remote",
        "Local document services stay usable when generation is down.",
    ),
    (
        "remote-cancellation",
        "remote",
        "Cancellation closes local transports. Remote billing may already have started.",
    ),
    (
        "remote-not-offline",
        "remote",
        "Hosted inference is not described as offline-qualified.",
    ),
)


class ChecklistError(RuntimeError):
    """An attempt to close VAL-006 without a completed drill."""


def checklist() -> dict:
    return {
        "gate": "VAL-006",
        "decision": "pending_checklist",
        "production_qualified": False,
        "offline_qualified": False,
        "recovery_qualified": False,
        "items": [
            {
                "id": item_id,
                "mode": mode,
                "description": description,
                "status": "pending",
                "evidence": None,
            }
            for item_id, mode, description in ITEMS
        ],
        "reason": (
            "The recovery and offline drill has not been run. "
            "A checklist is not evidence that the drill passed."
        ),
    }


def claim_complete() -> None:
    raise ChecklistError(
        "VAL-006 cannot be marked complete from this script. "
        "The offline boot, restore drill, and network-denial records are still pending."
    )


def write_checklist(path: Path) -> dict:
    payload = checklist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = path.with_suffix(".md")
    lines = [
        "# VAL-006 recovery and offline checklist",
        "",
        "Decision: **pending**. This file does not qualify offline operation or backup recovery.",
        "",
        "| Item | Mode | Status | Check |",
        "|---|---|---|---|",
    ]
    for item in payload["items"]:
        lines.append(
            f"| `{item['id']}` | {item['mode']} | pending | {item['description']} |"
        )
    lines.append("")
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return payload
