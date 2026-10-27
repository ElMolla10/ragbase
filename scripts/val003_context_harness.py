"""Write the pending VAL-003 context and serving record. Does not start a model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance.val003 import pending_report, validate_measurement_file, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="VAL-003 measurement harness")
    parser.add_argument(
        "--output",
        default="eval/acceptance/reports/val003-pending.json",
    )
    parser.add_argument("--measurement-file", default=None)
    args = parser.parse_args()
    if args.measurement_file:
        payload = validate_measurement_file(Path(args.measurement_file))
    else:
        payload = pending_report()
    write_report(Path(args.output), payload)
    print(
        f"VAL-003={payload['decision']} production_qualified={payload['production_qualified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
