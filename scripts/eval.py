"""Run the frozen golden set and write baseline metrics."""

import argparse
import asyncio
import json
from pathlib import Path

from backend.database import get_session_context, init_db
from backend.evalset import run_eval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "eval" / "results" / "baseline.json"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also score citations and refusals. Needs a working LLM.",
    )
    args = parser.parse_args()

    await init_db()
    async with get_session_context() as session:
        report = await run_eval(session, generate=args.generate)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
