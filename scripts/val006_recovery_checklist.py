"""Write the pending VAL-006 checklist. Cannot mark the drill complete."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance.val006 import ChecklistError, claim_complete, write_checklist  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VAL-006 recovery and offline checklist"
    )
    parser.add_argument(
        "--output",
        default="eval/acceptance/reports/val006-checklist.json",
    )
    parser.add_argument("--claim-complete", action="store_true")
    args = parser.parse_args()
    if args.claim_complete:
        try:
            claim_complete()
        except ChecklistError as exc:
            print(str(exc))
            return 2
    payload = write_checklist(Path(args.output))
    print(f"VAL-006={payload['decision']} items={len(payload['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
