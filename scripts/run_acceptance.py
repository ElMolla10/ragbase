"""Run the acceptance corpus. Fixture gateway unless an API is configured."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
