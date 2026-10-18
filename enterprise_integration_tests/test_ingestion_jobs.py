"""Durable queue, permission-scoped workers, lease fencing and page provenance."""

import asyncio
from uuid import UUID, uuid4

from sqlalchemy import text

from backend.ingestion.worker import publish, run_once
from enterprise_integration_tests.conftest import headers, upload
from enterprise_tests.pdf_fixtures import pdf_bytes


def extraction_path(world, record, workspace="a"):
    return f"/api/workspaces/{world['ids'][workspace]}/versions/{record['version_id']}/extraction"


async def process(world, actor="alice"):
    return await run_once(
        world["app"].state.database, world["app"].state.storage, [world["ids"][actor]]
    )


async def claim(world, token):
    database = world["app"].state.database
    async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        return await conn.scalar(
            text("SELECT enterprise.claim_ingestion(:token)"), {"token": token}
        )


async def test_upload_commits_job_and_text_result_with_lineage(world):
    record = (
        await upload(world, content="Ù…ØµØ¯Ø± ØªØ¬Ø±ÙŠØ¨ÙŠ\nSales 120".encode())
    ).json()
    path = extraction_path(world, record)
    queued = (await world["client"].get(path, headers=headers(world))).json()
    assert queued["status"] == "queued" and queued["pages"] == []
    assert await process(world)
    result = (await world["client"].get(path, headers=headers(world))).json()
    assert result["status"] == "extracted"
    assert result["version_id"] == record["version_id"]
    assert result["sha256"] == record["sha256"]
    assert result["pages"][0]["content"].startswith("Ù…ØµØ¯Ø± ØªØ¬Ø±ÙŠØ¨ÙŠ")
    assert result["pages"][0]["location"]["line_end"] == 2
    assert result["retrieval_ready"] is True
    assert not await process(world)


async def test_pdf_page_evidence_and_revocation(world):
    body = pdf_bytes(("Quarter 1: 120", "Quarter 2: 180"))
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename="report.pdf"),
        content=body,
    )
    assert response.status_code == 201, response.text
    path = extraction_path(world, response.json())
    assert await process(world)
    result = (await world["client"].get(path, headers=headers(world, "viewer"))).json()
    assert [p["location"]["page"] for p in result["pages"]] == [1, 2]
    assert "180" in result["pages"][1]["content"]
    assert (
        await world["client"].get(path, headers=headers(world, "bob"))
    ).status_code == 404
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
            {"id": world["ids"]["viewer"]},
        )
    assert (
        await world["client"].get(path, headers=headers(world, "viewer"))
    ).status_code == 404


async def test_encrypted_and_blank_pdf_report_honest_states(world):
    for data, expected in [
        (pdf_bytes(encrypted=True), "failed"),
        (pdf_bytes(("",)), "needs_review"),
    ]:
        response = await world["client"].post(
            f"/api/workspaces/{world['ids']['a']}/documents",
            headers=headers(world, filename="input.pdf"),
            content=data,
        )
        assert response.status_code == 201
        assert await process(world)
        result = (
            await world["client"].get(
                extraction_path(world, response.json()), headers=headers(world)
            )
        ).json()
        assert result["status"] == expected
        if expected == "failed":
            assert (
                result["error_code"] == "encrypted_pdf_unsupported"
                and result["pages"] == []
            )
        else:
            assert result["pages"][0]["warnings"]


async def test_queue_quota_rejects_before_creating_more_files(world):
    record = (await upload(world)).json()
    assert (await upload(world)).status_code == 429
    assert len(list(world["root"].glob("*/*/*.bin"))) == 1
    path = extraction_path(world, record).replace("/extraction", "/cancel")
    assert (
        await world["client"].post(path, headers=headers(world, "viewer"))
    ).status_code == 404
    assert (await world["client"].post(path, headers=headers(world))).json()[
        "status"
    ] == "cancelled"
    assert not await process(world)
    assert (await upload(world)).status_code == 201


async def test_restarted_worker_recovers_expired_lease_without_duplicate_pages(world):
    record = (await upload(world)).json()
    assert await claim(world, uuid4())
    async with world["owner"].begin() as conn:
        await conn.execute(
            text(
                "UPDATE enterprise.ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"
            ),
            {"id": UUID(record["job_id"])},
        )
    assert await process(world)
    result = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world)
        )
    ).json()
    assert result["attempts"] == 2 and len(result["pages"]) == 1
    assert not await process(world)


async def test_expired_claim_cannot_publish_over_new_worker(world):
    record = (await upload(world)).json()
    token = uuid4()
    await claim(world, token)
    async with world["owner"].begin() as conn:
        await conn.execute(
            text(
                "UPDATE enterprise.ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"
            ),
            {"id": UUID(record["job_id"])},
        )
    assert await process(world)
    assert not await publish(
        world["app"].state.database,
        world["ids"]["alice"],
        world["ids"]["a"],
        UUID(record["job_id"]),
        token,
        {"error_code": "parser_failed"},
    )
    result = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world)
        )
    ).json()
    assert result["status"] == "extracted"


async def test_cancelled_claim_cannot_publish(world):
    record = (await upload(world)).json()
    token = uuid4()
    await claim(world, token)
    path = extraction_path(world, record).replace("/extraction", "/cancel")
    assert (await world["client"].post(path, headers=headers(world))).status_code == 200
    assert not await publish(
        world["app"].state.database,
        world["ids"]["alice"],
        world["ids"]["a"],
        UUID(record["job_id"]),
        token,
        {"error_code": "parser_failed"},
    )


async def test_two_workers_do_not_run_two_extractions(world):
    assert (await upload(world)).status_code == 201
    assert (await upload(world, user="bob", workspace="b")).status_code == 201
    results = await asyncio.gather(process(world), process(world, "bob"))
    assert results.count(True) == 1
    assert await process(world, "bob" if results[0] else "alice")


async def test_worker_cannot_process_after_membership_is_revoked(world):
    record = (await upload(world)).json()
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
            {"id": world["ids"]["alice"]},
        )
    assert not await process(world)
    async with world["owner"].connect() as conn:
        assert (
            await conn.scalar(
                text("SELECT status FROM enterprise.ingestion_jobs WHERE id=:id"),
                {"id": UUID(record["job_id"])},
            )
            == "cancelled"
        )
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.extracted_pages"))
            == 0
        )


async def test_extracted_page_rls_does_not_depend_on_route_filters(world):
    await upload(world)
    assert await process(world)
    async with world["app"].state.database.scope(
        world["ids"]["bob"], world["ids"]["b"]
    ) as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.extracted_pages"))
            == 0
        )
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.ingestion_jobs"))
            == 0
        )


async def test_process_request_is_idempotent_and_preserves_completed_evidence(world):
    record = (await upload(world)).json()
    path = extraction_path(world, record).replace("/extraction", "/process")
    assert (await world["client"].post(path, headers=headers(world))).json()[
        "job_id"
    ] == record["job_id"]
    assert await process(world)
    response = await world["client"].post(path, headers=headers(world))
    assert response.json()["job_id"] == record["job_id"]
    assert response.json()["status"] == "extracted"
    assert not await process(world)


async def test_existing_original_without_a_job_can_be_queued(world):
    record = (await upload(world)).json()
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.ingestion_jobs WHERE id=:id"),
            {"id": UUID(record["job_id"])},
        )
    path = extraction_path(world, record).replace("/extraction", "/process")
    first, second = await asyncio.gather(
        world["client"].post(path, headers=headers(world)),
        world["client"].post(path, headers=headers(world)),
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    assert await process(world)


async def test_global_queue_bound_is_checked_across_user_scopes(world):
    record = (await upload(world)).json()
    async with world["owner"].begin() as conn:
        for number in range(2, 9):
            version = uuid4()
            key = f"{world['ids']['a']}/{version}/{record['sha256']}.bin"
            await conn.execute(
                text("""
                INSERT INTO enterprise.document_versions(id,workspace_id,document_id,version_number,storage_key,sha256,byte_size)
                VALUES(:id,:workspace,:document,:number,:key,:sha,:size)
            """),
                {
                    "id": version,
                    "workspace": world["ids"]["a"],
                    "document": UUID(record["document_id"]),
                    "number": number,
                    "key": key,
                    "sha": record["sha256"],
                    "size": record["byte_size"],
                },
            )
            await conn.execute(
                text("""
                INSERT INTO enterprise.ingestion_jobs(id,workspace_id,version_id,created_by)
                VALUES(:id,:workspace,:version,:actor)
            """),
                {
                    "id": uuid4(),
                    "workspace": world["ids"]["a"],
                    "version": version,
                    "actor": world["ids"]["alice"],
                },
            )
    assert (await upload(world, user="bob", workspace="b")).status_code == 429
    assert len(list(world["root"].glob("*/*/*.bin"))) == 1


async def test_retry_limit_stops_repeated_worker_crashes(world):
    record = (await upload(world)).json()
    for _ in range(3):
        assert await claim(world, uuid4())
        async with world["owner"].begin() as conn:
            await conn.execute(
                text(
                    "UPDATE enterprise.ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"
                ),
                {"id": UUID(record["job_id"])},
            )
    assert not await process(world)
    result = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world)
        )
    ).json()
    assert result["status"] == "failed" and result["error_code"] == "retry_limit"
    assert result["attempts"] == 3 and result["pages"] == []


async def test_worker_does_not_publish_after_revocation(world):
    import pytest
    from fastapi import HTTPException

    record = (await upload(world)).json()
    token = uuid4()
    await claim(world, token)
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
            {"id": world["ids"]["alice"]},
        )
    with pytest.raises(HTTPException):
        await publish(
            world["app"].state.database,
            world["ids"]["alice"],
            world["ids"]["a"],
            UUID(record["job_id"]),
            token,
            {"error_code": "parser_failed"},
        )
    async with world["owner"].connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.extracted_pages"))
            == 0
        )


async def test_corrupt_artifact_fails_job_without_exposing_content(world):
    record = (await upload(world)).json()
    next(world["root"].glob("*/*/*.bin")).write_bytes(b"sensitive-looking-invalid-data")
    assert await process(world)
    response = await world["client"].get(
        extraction_path(world, record), headers=headers(world)
    )
    assert response.json()["error_code"] == "artifact_unavailable"
    assert response.json()["pages"] == []
    assert "sensitive-looking" not in response.text
