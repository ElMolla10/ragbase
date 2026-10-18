"""Synthetic native-PDF queue/restart smoke check through the loopback proxy."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


def synthetic_pdf():
    # Independent small PDF fixture, generated only in memory.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for content in (b"Quarter 1 sales: 120", b"Quarter 2 sales: 180"):
        stream = b"BT /F1 12 Tf 20 350 Td (" + content + b") Tj ET"
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


def main(queue_only, verify_existing):
    access = json.loads((ROOT / ".local/dev-access.json").read_text("utf-8"))
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:8800/api/workspaces/{access['workspace_a']}"
    payload = synthetic_pdf()

    def call(path, user="alice", body=None):
        headers = {
            "Authorization": "Bearer " + access["tokens"][user],
            "X-Filename": "synthetic-quarters.pdf",
        }
        request = Request(base + path, data=body, headers=headers)
        try:
            with opener.open(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
        except (URLError, OSError):
            return 0, b""

    record_path = ROOT / ".local/ingestion-smoke.json"
    if verify_existing:
        record = json.loads(record_path.read_text("utf-8"))
    else:
        status, body = call("/documents", body=payload)
        assert status == 201, "Synthetic PDF upload failed"
        record = json.loads(body)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    path = f"/versions/{record['version_id']}/extraction"
    if queue_only:
        status, body = call(path)
        assert status == 200 and json.loads(body)["status"] == "queued"
        print(json.dumps({"status": "queued", "synthetic_data_only": True}))
        return
    deadline = time.monotonic() + 30
    while True:
        status, body = call(path)
        if status in {0, 502, 503} and time.monotonic() < deadline:
            time.sleep(0.5)
            continue
        assert status == 200
        result = json.loads(body)
        if result["status"] in {"extracted", "needs_review", "failed", "cancelled"}:
            break
        assert time.monotonic() < deadline, (
            "Worker did not finish within smoke deadline"
        )
        time.sleep(0.5)
    assert result["status"] == "extracted", "Native extraction did not complete"
    assert [page["location"]["page"] for page in result["pages"]] == [1, 2]
    assert (
        "120" in result["pages"][0]["content"]
        and "180" in result["pages"][1]["content"]
    )
    assert result["sha256"] == record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["retrieval_ready"] is True
    assert call(path, user="bob")[0] == 404
    assert call(path, user="viewer")[0] == 200
    assert call(f"/versions/{record['version_id']}/file")[1] == payload
    print(
        json.dumps(
            {
                "status": "passed",
                "synthetic_data_only": True,
                "pages": 2,
                "native_text_and_lineage": True,
                "cross_workspace_denied": True,
                "resumed_existing_job": verify_existing,
                "retrieval_ready": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--queue-only", action="store_true")
    options.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    main(args.queue_only, args.verify_existing)
