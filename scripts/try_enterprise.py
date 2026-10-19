"""Interactive local enterprise upload and Q&A; credentials are never printed."""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.query_enterprise import main as query  # noqa: E402

BASE = "http://127.0.0.1:8800"
MAX_BYTES = 5_000_000


def read_upload(path):
    if path.suffix.lower() not in {".pdf", ".txt", ".md", ".csv"}:
        raise ValueError("Choose a PDF, UTF-8 TXT, Markdown or CSV file.")
    with path.open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a nonempty file of at most 5 MB.")
    return data


def upload_and_wait(path, *, deadline=120):
    access = json.loads((ROOT / ".local/dev-access.json").read_text("utf-8"))
    opener = build_opener(ProxyHandler({}))
    base = BASE + f"/api/workspaces/{access['workspace_a']}"
    headers = {"Authorization": "Bearer " + access["tokens"]["alice"]}
    data = read_upload(path)
    request = Request(
        base + "/documents",
        data=data,
        headers=headers | {"X-Filename": quote(path.name, safe="")},
    )
    with opener.open(request, timeout=30) as response:
        record = json.load(response)
    version = UUID(record["version_id"])
    print(
        f"Uploaded {path.name}; version {version}. Waiting for extraction...",
        flush=True,
    )
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with opener.open(
            Request(base + f"/versions/{version}/extraction", headers=headers),
            timeout=10,
        ) as response:
            status = json.load(response)
        if status["status"] in {"failed", "cancelled", "needs_review"}:
            print("Extraction warnings:", ", ".join(status.get("warnings", [])))
            raise ValueError(
                f"Extraction {status['status']}: {status.get('error_code') or 'review required'}. "
                "The uploaded original is retained."
            )
        if status.get("retrieval_ready"):
            print("Ready. Extractor:", status.get("extractor"))
            for warning in status.get("warnings", []):
                print("Extraction note:", warning)
            return version
        time.sleep(2)
    raise ValueError(
        f"Still processing version {version}; the durable job continues. "
        "Use --version with that ID once extraction is ready."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--file", type=Path)
    selection.add_argument("--version", type=UUID)
    parser.add_argument("--question")
    parser.add_argument(
        "--verbose", action="store_true", help="Show processing/model diagnostics"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show full citation previews"
    )
    args = parser.parse_args()
    print("Enterprise development test — local API at " + BASE)
    print("Use nonconfidential test documents only. Questions and retrieved text")
    print("are sent through OpenRouter BYOK; standard provider privacy applies.")
    try:
        version = args.version
        if version is None:
            path = args.file or Path(input("File path: ").strip().strip('"'))
            version = upload_and_wait(path)
        if args.question:
            query(version, args.question, args.preview, verbose=args.verbose)
            return
        print(
            "Ask a question or an overview such as 'What is this paper about?'. Leave blank to exit."
        )
        while question := input("Question: ").strip():
            query(version, question, args.preview, verbose=args.verbose)
    except HTTPError as exc:
        raise SystemExit(
            f"Request failed (HTTP {exc.code}). Check readiness, permissions and supported input."
        ) from exc
    except URLError as exc:
        raise SystemExit(
            "Enterprise API unavailable. Start its Compose services first."
        ) from exc
    except (ValueError, OSError, KeyError) as exc:
        # Never print credential/configuration contents from exceptions.
        if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
            raise SystemExit(str(exc)) from exc
        raise SystemExit(
            "Check the selected file and local development configuration."
        ) from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nTest client closed. Uploaded originals and durable jobs are retained.")
