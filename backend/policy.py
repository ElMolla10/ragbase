"""Fail-closed, deterministic runtime instructions derived from reviewed policy."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = frozenset({"retrieval", "coverage", "analysis", "vision"})
CORE_RULES = frozenset(
    {
        "LOC-001",
        "LOC-002",
        "SEC-002",
        "SEC-005",
        "SEC-007",
        "EVD-002",
        "EVD-003",
        "ANA-001",
        "MOD-003",
        "MOD-004",
        "UX-001",
    }
)


class PolicyError(ValueError):
    """Required policy is missing, stale, or invalid."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Instructions:
    text: str
    sha256: str
    constitution_version: str
    rule_ids: tuple[str, ...]


def assemble(task: str, root: Path = ROOT) -> Instructions:
    """Validate the complete manifest before assembling a trusted task profile."""
    if task not in TASKS:
        raise PolicyError(f"Unknown policy task: {task}")
    try:
        manifest = json.loads((root / "policy/manifest.json").read_text("utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise PolicyError("Unsupported policy manifest schema")
        constitution = (root / "CONSTITUTION.md").read_bytes()
        if manifest["constitution_sha256"] != digest(constitution):
            raise PolicyError(
                "Constitution changed: review and rebuild policy manifest"
            )
        version = re.search(r"(?m)^Version: (\S+)", constitution.decode("utf-8"))
        if not version or manifest["constitution_version"] != version[1]:
            raise PolicyError("Constitution version mismatch")
        known = set(
            re.findall(r"(?m)^- \*\*([A-Z]+-\d{3})", constitution.decode("utf-8"))
        )
        sections = manifest["sections"]
        if not isinstance(sections, dict) or set(sections) != {"core", *TASKS}:
            raise PolicyError("Mandatory policy sections missing or unknown")
        for section in sections.values():
            if not isinstance(section, dict) or not isinstance(
                section.get("rule_ids"), list
            ):
                raise PolicyError("Invalid policy section")
            if any(not isinstance(rule, str) for rule in section["rule_ids"]):
                raise PolicyError("Invalid policy rule IDs")
        if not CORE_RULES.issubset(sections["core"]["rule_ids"]):
            raise PolicyError("Mandatory core rules missing")
        contents = {}
        for name, section in sections.items():
            # Fixed template paths: manifests cannot read arbitrary files.
            data = (root / "policy/templates" / f"{name}.txt").read_bytes()
            if digest(data) != section["sha256"] or not data.strip():
                raise PolicyError(f"Policy template changed or empty: {name}")
            rules = section["rule_ids"]
            if not rules or not set(rules).issubset(known):
                raise PolicyError(f"Unknown or missing policy rule IDs: {name}")
            contents[name] = data.decode("utf-8").strip()
        text = "\n\n".join([contents["core"], contents[task]])
        rule_ids = tuple(
            sorted(set(sections["core"]["rule_ids"] + sections[task]["rule_ids"]))
        )
        return Instructions(text, digest(text.encode("utf-8")), version[1], rule_ids)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PolicyError):
            raise
        raise PolicyError("Missing or malformed policy assets") from exc
