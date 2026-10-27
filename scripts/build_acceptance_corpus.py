"""Regenerate the synthetic acceptance corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceptance.build import write_corpus  # noqa: E402

if __name__ == "__main__":
    manifest = write_corpus(Path(__file__).resolve().parents[1] / "eval/acceptance/v1")
    print(
        f"cases={manifest['case_count']} arabic_or_mixed={manifest['arabic_or_mixed_count']} "
        f"reviewed={manifest['reviewed']}"
    )
