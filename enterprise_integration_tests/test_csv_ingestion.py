"""CSV originals pass through the real queue, scoped index and source preview."""

import json

import pytest

from enterprise_integration_tests.conftest import headers
from enterprise_integration_tests.test_ingestion_jobs import extraction_path, process


async def test_csv_lineage_search_preview_and_original_are_scoped(world):
    body = "id,label,value\n001,مبيعات,120\n002,Returns,10\n".encode()
    base = f"/api/workspaces/{world['ids']['a']}"
    response = await world["client"].post(
        base + "/documents", headers=headers(world, filename="sales.csv"), content=body
    )
    assert response.status_code == 201
    record = response.json()
    assert await process(world)
    extraction = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world)
        )
    ).json()
    assert extraction["retrieval_ready"] and extraction["status"] == "extracted"
    assert json.loads(extraction["pages"][1]["content"]) == ["001", "مبيعات", "120"]
    original = await world["client"].get(
        base + f"/versions/{record['version_id']}/file", headers=headers(world)
    )
    assert original.content == body
    search = await world["client"].post(
        base + "/search",
        headers=headers(world),
        json={"version_ids": [record["version_id"]], "question": "مبيعات"},
    )
    assert search.status_code == 200
    hit = search.json()["hits"][0]
    assert hit["location"]["kind"] == "csv_row" and hit["location"]["row"] == 2
    preview = await world["client"].get(hit["preview_url"], headers=headers(world))
    assert preview.status_code == 200 and '"001"' in preview.text
    assert (
        await world["client"].get(hit["preview_url"], headers=headers(world, "bob"))
    ).status_code == 404


@pytest.mark.parametrize(
    "body,status", [(b'a,"bad', "failed"), (b"a,b\n1,2,3\n", "needs_review")]
)
async def test_failed_or_review_csv_cannot_be_used_as_answer_evidence(
    world, body, status
):
    base = f"/api/workspaces/{world['ids']['a']}"
    response = await world["client"].post(
        base + "/documents", headers=headers(world, filename="review.csv"), content=body
    )
    assert response.status_code == 201 and await process(world)
    record = response.json()
    result = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world)
        )
    ).json()
    assert result["status"] == status and not result["retrieval_ready"]
    search = await world["client"].post(
        base + "/search",
        headers=headers(world),
        json={"version_ids": [record["version_id"]], "question": "1"},
    )
    assert search.status_code == 409


async def test_csv_encoding_is_checked_before_durable_publication(world):
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename="invalid.csv"),
        content=b"\xff,123",
    )
    assert response.status_code == 415
