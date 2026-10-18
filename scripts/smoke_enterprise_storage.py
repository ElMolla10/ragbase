"""Exercise loopback development storage with synthetic data; never print tokens.

Run once to upload, then --verify-existing after recreating services to check
durability. The ignored record retains only synthetic IDs and a content hash.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8800"
PAYLOAD = "Synthetic durability fixture / بيانات تجريبية لاختبار الاستمرارية\n".encode()


def main(verify_existing):
    access = json.loads((ROOT / ".local/dev-access.json").read_text("utf-8"))
    opener = build_opener(ProxyHandler({}))

    def call(path, user=None, data=None):
        headers = {"X-Filename": "synthetic-durability.txt"}
        if user:
            headers["Authorization"] = "Bearer " + access["tokens"][user]
        request = Request(BASE + path, data=data, headers=headers)
        try:
            with opener.open(request, timeout=15) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    status, body = call("/health/ready")
    assert (
        status == 200
        and json.loads(body)["documents"] == "original_storage_and_native_extraction"
    )
    path = f"/api/workspaces/{access['workspace_a']}"
    record_path = ROOT / ".local/storage-smoke.json"
    if verify_existing:
        record = json.loads(record_path.read_text("utf-8"))
    else:
        status, body = call(path + "/documents", "alice", PAYLOAD)
        assert status == 201, "Synthetic upload failed"
        record = json.loads(body)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    file_path = path + f"/versions/{record['version_id']}/file"
    for user in ("alice", "viewer"):
        status, body = call(file_path, user)
        assert status == 200 and hashlib.sha256(body).hexdigest() == record["sha256"]
        assert body == PAYLOAD
    assert call(file_path, "bob")[0] == 404
    assert call(file_path)[0] == 401
    assert call(path + "/documents", "viewer", PAYLOAD)[0] == 404
    print(
        json.dumps(
            {
                "status": "passed",
                "synthetic_data_only": True,
                "mode": "verify_after_recreation"
                if verify_existing
                else "upload_and_verify",
                "authorized_readers": 2,
                "cross_workspace_denied": True,
                "anonymous_denied": True,
                "viewer_write_denied": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    main(parser.parse_args().verify_existing)
