"""Query a selected development document; keep bearer tokens out of shell output."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]


def main(version, question, preview):
    access = json.loads((ROOT / ".local/dev-access.json").read_text("utf-8"))
    opener = build_opener(ProxyHandler({}))
    base = "http://127.0.0.1:8800"
    headers = {
        "Authorization": "Bearer " + access["tokens"]["alice"],
        "Content-Type": "application/json",
    }
    payload = {
        "version_ids": [str(version)],
        "question": question,
        "client_request_id": str(uuid4()),
    }
    request = Request(
        base + f"/api/workspaces/{access['workspace_a']}/answers",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with opener.open(request, timeout=180) as response:
            for line in response:
                event = json.loads(line)
                if event["event"] in {"queued", "stage"}:
                    print(event.get("stage", "queued"), flush=True)
                elif event["event"] == "error":
                    raise SystemExit(event["detail"])
                elif event["event"] == "answer":
                    if event["insufficient"]:
                        print("Insufficient evidence in the selected versions.")
                    for claim in event["claims"]:
                        print(claim["quote"])
                        print(
                            f"Source: {claim['filename']} / {claim['location']} / {claim['preview_url']}"
                        )
                        if preview:
                            with opener.open(
                                Request(base + claim["preview_url"], headers=headers),
                                timeout=10,
                            ) as source:
                                print(source.read().decode())
                    print(event["scope_notice"])
                    print(
                        f"Validated answer released after {event['metadata']['answer_release_seconds']:.2f}s; fixture={event['metadata']['fixture']}"
                    )
    except HTTPError as error:
        raise SystemExit(
            f"Request failed (HTTP {error.code}); verify permissions, indexing and generation availability."
        ) from error


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=UUID, required=True)
    parser.add_argument(
        "--question", help="If omitted, read the question interactively"
    )
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    main(args.version, args.question or input("Question: "), args.preview)
