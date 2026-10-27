"""Query a selected development document; keep bearer tokens out of shell output."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]


def main(version, question, preview, *, verbose=True, coverage=False):
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
    route = "coverage" if coverage else "answers"
    request = Request(
        base + f"/api/workspaces/{access['workspace_a']}/{route}",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with opener.open(request, timeout=180) as response:
            for line in response:
                event = json.loads(line)
                if event["event"] in {"queued", "stage"}:
                    if not verbose:
                        continue
                    if event["event"] == "queued" and event.get("external_processing"):
                        if event.get("privacy_policy") == "nonconfidential-development":
                            print(
                                "Nonconfidential development only: standard provider privacy policy; zero retention is not guaranteed.",
                                flush=True,
                            )
                        print(
                            "Remote generation: questions and retrieved text are sent through OpenRouter to its selected provider.",
                            flush=True,
                        )
                    print(event.get("stage", "queued"), flush=True)
                elif event["event"] == "error":
                    raise SystemExit(event["detail"])
                elif event["event"] == "coverage":
                    print(event.get("scope_notice", "Coverage stopped."))
                    for gap in event.get("gaps") or []:
                        if gap.get("gap_code"):
                            print("Gap:", gap["gap_code"], gap.get("location"))
                elif event["event"] == "answer":
                    metadata = event["metadata"]
                    if verbose and metadata.get("model_called"):
                        route = metadata.get("model", {}).get("resolved_route") or {}
                        print(
                            "Model used:",
                            route.get("model") or "not reported",
                            "/ provider:",
                            route.get("provider") or "not reported",
                            flush=True,
                        )
                    elif verbose:
                        print(
                            "No model call: no usable extracted document context was available."
                        )
                    if event["insufficient"] and not event.get("answer"):
                        print(
                            "The supplied document context did not establish an answer."
                        )
                    sources = {}
                    conversational = event.get("mode") in {
                        "grounded_document_chat",
                        "exhaustive_coverage",
                        "spreadsheet_calculation",
                    }
                    if conversational:
                        print("\n" + event["answer"])
                    if event.get("mode") == "exhaustive_coverage":
                        print("\n" + event["scope_notice"])
                        if not event["coverage_complete"]:
                            print("Not complete.")
                        for gap in event.get("gaps") or []:
                            if gap.get("gap_code"):
                                print("Gap:", gap["gap_code"], gap.get("location"))
                    for claim in event["claims"]:
                        key = claim["source_id"]
                        if key not in sources:
                            sources[key] = (len(sources) + 1, claim)
                        number = sources[key][0]
                        if not conversational:
                            print(
                                f"\n{claim.get('explanation') or claim['quote']} [{number}]"
                            )
                    if sources and preview:
                        print("\nSources:")
                    for number, claim in sources.values() if preview else ():
                        location = claim["location"]
                        label = (
                            f"page {location['page']}"
                            if "page" in location
                            else f"row {location['row']}"
                            if "row" in location
                            else f"lines {location.get('line_start', '?')}–{location.get('line_end', '?')}"
                        )
                        print(f"[{number}] {claim['filename']} — {label}")
                        if preview:
                            print("Supporting quote:", claim["quote"])
                            with opener.open(
                                Request(base + claim["preview_url"], headers=headers),
                                timeout=10,
                            ) as source:
                                print(source.read().decode())
                    coverage = metadata.get("coverage", {})
                    if verbose:
                        print(event["scope_notice"])
                        print(
                            f"Answer released after {metadata['answer_release_seconds']:.2f}s; fixture={metadata['fixture']}"
                        )
                    elif coverage.get("total_document_chunks") and not coverage.get(
                        "all_extracted_chunks_included"
                    ):
                        print(
                            f"\nContext: {coverage['included_chunks']} of {coverage['total_document_chunks']} passages supplied; partial overview."
                        )
                    if not verbose and metadata.get("fixture"):
                        print("Development fixture output; no real model was called.")
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
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run exhaustive section coverage instead of a bounded answer",
    )
    args = parser.parse_args()
    main(
        args.version,
        args.question or input("Question: "),
        args.preview,
        coverage=args.coverage,
    )
