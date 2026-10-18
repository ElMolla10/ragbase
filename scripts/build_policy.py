"""Verify policy assets or refresh hashes after an explicit policy review."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.policy import ROOT, TASKS, assemble, digest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reviewed-hashes", action="store_true")
    args = parser.parse_args()
    path = ROOT / "policy/manifest.json"
    if args.refresh_reviewed_hashes:
        manifest = json.loads(path.read_text("utf-8"))
        manifest["constitution_sha256"] = digest(
            (ROOT / "CONSTITUTION.md").read_bytes()
        )
        for name, section in manifest["sections"].items():
            section["sha256"] = digest(
                (ROOT / "policy/templates" / f"{name}.txt").read_bytes()
            )
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for task in sorted(TASKS):
        instructions = assemble(task)
        print(f"{task}: {instructions.sha256} ({len(instructions.rule_ids)} rules)")


if __name__ == "__main__":
    main()
