"""Synthetic Phase 3 local-model smoke; no downloads or production qualification."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from smoke_enterprise_ingestion import synthetic_pdf

ROOT = Path(__file__).resolve().parents[1]


def main(output):
    access = json.loads((ROOT / ".local/dev-access.json").read_text("utf-8"))
    opener = build_opener(ProxyHandler({}))
    base = "http://127.0.0.1:8800"
    scope = f"/api/workspaces/{access['workspace_a']}"

    def call(path, *, user="alice", body=None, filename=None):
        headers = {"Authorization": "Bearer " + access["tokens"][user]}
        if isinstance(body, dict):
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode()
        if filename:
            headers["X-Filename"] = filename
        try:
            with opener.open(
                Request(base + path, headers=headers, data=body), timeout=180
            ) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers

    def upload_and_wait(content, filename):
        status, body, _ = call(scope + "/documents", body=content, filename=filename)
        assert status == 201, "Synthetic upload failed"
        record = json.loads(body)
        deadline = time.monotonic() + 30
        while True:
            status, body, _ = call(
                scope + f"/versions/{record['version_id']}/extraction"
            )
            state = json.loads(body)
            if status == 200 and state["retrieval_ready"]:
                return record
            assert time.monotonic() < deadline, "Extraction/indexing did not complete"
            time.sleep(0.5)

    records = [
        upload_and_wait(synthetic_pdf(), "phase3-quarters.pdf"),
        upload_and_wait("إجمالي المبيعات ١٢٠ جنيه.".encode(), "phase3-arabic.txt"),
    ]
    (ROOT / ".local/phase3-demo-sources.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    cases = [
        ("english_pdf", records[0], "What were Quarter 2 sales?", "180"),
        ("arabic_text", records[1], "ما هي المبيعات؟", "١٢٠"),
        ("mixed_question", records[0], "Quarter 2 المبيعات؟", "180"),
        ("missing_evidence", records[0], "Neptune telemetry?", None),
    ]
    results = []
    for name, record, question, expected in cases:
        started = time.monotonic()
        request = {
            "question": question,
            "version_ids": [record["version_id"]],
            "client_request_id": str(uuid4()),
        }
        status, body, _ = call(scope + "/answers", body=request)
        events = (
            [json.loads(line) for line in body.splitlines()] if status == 200 else []
        )
        answer = next(
            (value for value in events if value.get("event") == "answer"), None
        )
        success = answer is not None and (
            expected in " ".join(c["quote"] for c in answer["claims"])
            if expected
            else answer["insufficient"] and not answer["claims"]
        )
        preview_ok, denial_ok = True, True
        if answer and answer["claims"]:
            preview = answer["claims"][0]["preview_url"]
            preview_status, preview_body, headers = call(preview)
            preview_ok = (
                preview_status == 200
                and answer["claims"][0]["quote"] in preview_body.decode()
                and headers["Content-Type"].startswith("text/plain")
            )
            denial_ok = call(preview, user="bob")[0] == 404
        results.append(
            {
                "case": name,
                "passed": bool(success and preview_ok and denial_ok),
                "http_status": status,
                "elapsed_seconds": time.monotonic() - started,
                "source_preview_checked": preview_ok,
                "cross_workspace_denied": denial_ok,
                "metadata": answer.get("metadata") if answer else None,
                "events": [value.get("event") for value in events],
            }
        )
        print(
            json.dumps(
                {
                    "case": name,
                    "passed": results[-1]["passed"],
                    "elapsed_seconds": results[-1]["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    status, _, _ = call(
        f"/api/workspaces/{access['workspace_b']}/search",
        user="bob",
        body={"question": "sales", "version_ids": [records[0]["version_id"]]},
    )
    report = {
        "synthetic_only": True,
        "production_qualified": False,
        "target_server_tested": False,
        "retrieval": "lexical_baseline",
        "answer_mode": "exact_source_quotations",
        "cases": results,
        "cross_workspace_search_denied": status == 404,
        "passed": all(row["passed"] for row in results) and status == 404,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "Synthetic Phase 3 evidence saved; no prompts, answers or tokens retained in the report."
    )
    if not report["passed"]:
        raise SystemExit(
            "One or more local-model checks failed; inspect the evidence record"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".local/phase3-live-evidence.json"
    )
    main(parser.parse_args().output)
