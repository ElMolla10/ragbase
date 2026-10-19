"""End-to-end evidence lookup using actual RLS and explicit fixture generation."""

import asyncio
import json
from uuid import UUID, uuid4

from sqlalchemy import text

from backend.gateway.contracts import GatewayEvent
from backend.ingestion.worker import run_once
from enterprise_integration_tests.conftest import headers, upload


async def ready(
    world, content=b"Quarter 2 sales were 180.", user="alice", workspace="a"
):
    response = await upload(world, content=content, user=user, workspace=workspace)
    assert response.status_code == 201
    record = response.json()
    assert await run_once(
        world["app"].state.database, world["app"].state.storage, [world["ids"][user]]
    )
    return record


def payload(record, question="Quarter 2 sales?"):
    return {
        "question": question,
        "version_ids": [record["version_id"]],
        "client_request_id": str(uuid4()),
    }


async def ask(
    world, record, question="Quarter 2 sales?", user="alice", workspace="a", body=None
):
    return await world["client"].post(
        f"/api/workspaces/{world['ids'][workspace]}/answers",
        headers=headers(world, user),
        json=body or payload(record, question),
    )


def frames(response):
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines()]


async def test_lookup_streams_verified_quotes_with_private_run_metadata(world):
    record = await ready(world)
    response = await ask(world, record)
    events = frames(response)
    answer = next(event for event in events if event["event"] == "answer")
    assert answer["metadata"]["fixture"] is True
    assert "180" in answer["claims"][0]["quote"]
    assert answer["claims"][0]["sha256"] == record["sha256"]
    preview = await world["client"].get(
        answer["claims"][0]["preview_url"], headers=headers(world)
    )
    assert preview.status_code == 200 and "180" in preview.text
    assert preview.headers["content-type"].startswith("text/plain")
    async with world["owner"].connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT * FROM enterprise.answer_runs WHERE id=:id"),
                    {"id": UUID(events[0]["run_id"])},
                )
            )
            .mappings()
            .one()
        )
        assert row["status"] == "completed" and "Quarter" not in str(dict(row))


async def test_missing_keyword_hits_use_document_context_and_can_return_insufficient(
    world,
):
    record = await ready(world)

    class Insufficient:
        async def stream(self, request):
            assert request.evidence
            yield GatewayEvent("text", '{"insufficient":true,"claims":[]}')
            yield GatewayEvent("done", finish_reason="stop")

    world["app"].state.gateway = Insufficient()
    events = frames(await ask(world, record, "Neptune telemetry?"))
    answer = next(event for event in events if event["event"] == "answer")
    assert answer["insufficient"] and answer["claims"] == []
    assert answer["metadata"]["model_called"] is True
    assert answer["metadata"]["coverage"]["strategy"] == "document_context_fallback"


async def test_arabic_search_preserves_source_and_citation(world):
    record = await ready(world, "إِجمالي المبيعات ١٢٠ جنيه".encode())
    events = frames(await ask(world, record, "ما هي المبيعات؟"))
    answer = next(event for event in events if event["event"] == "answer")
    assert "١٢٠" in answer["claims"][0]["quote"]


async def test_denied_versions_and_sources_never_reach_generation(world):
    record = await ready(world)
    assert (await ask(world, record, user="bob", workspace="b")).status_code == 404
    search = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/search",
        headers=headers(world),
        json=payload(record),
    )
    source = search.json()["hits"][0]["id"]
    assert (
        await world["client"].get(
            f"/api/workspaces/{world['ids']['b']}/sources/{source}",
            headers=headers(world, "bob"),
        )
    ).status_code == 404
    async with world["app"].state.database.scope(
        world["ids"]["bob"], world["ids"]["b"]
    ) as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.document_chunks"))
            == 0
        )
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.answer_runs")) == 0
        )


async def test_forged_model_citation_is_never_released(world):
    record = await ready(world)

    class Forged:
        async def stream(self, request):
            yield GatewayEvent(
                "text",
                json.dumps(
                    {
                        "insufficient": False,
                        "claims": [
                            {
                                "source_id": str(uuid4()),
                                "quote": "secret forged content",
                            }
                        ],
                    }
                ),
            )
            yield GatewayEvent("done", finish_reason="stop")

    world["app"].state.gateway = Forged()
    response = await ask(world, record)
    assert frames(response)[-1]["event"] == "error"
    assert "secret forged content" not in response.text


async def test_revocation_during_generation_prevents_answer_and_preview(world):
    record = await ready(world)

    class Revoke:
        async def stream(self, request):
            async with world["owner"].begin() as conn:
                await conn.execute(
                    text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
                    {"id": world["ids"]["alice"]},
                )
            source = json.loads(request.evidence[0])
            yield GatewayEvent(
                "text",
                json.dumps(
                    {
                        "insufficient": False,
                        "claims": [
                            {"source_id": source["source_id"], "quote": source["text"]}
                        ],
                    }
                ),
            )
            yield GatewayEvent("done", finish_reason="stop")

    world["app"].state.gateway = Revoke()
    response = await ask(world, record)
    assert frames(response)[-1]["event"] == "error" and "180" not in response.text


async def test_generation_priority_quota_and_cancel_release_heavy_slot(world):
    record = await ready(world)
    bob = (await upload(world, user="bob", workspace="b")).json()
    entered, closed = asyncio.Event(), asyncio.Event()

    class Paused:
        async def stream(self, request):
            entered.set()
            try:
                await asyncio.sleep(20)
                yield GatewayEvent("text", "unreachable")
            finally:
                closed.set()

    world["app"].state.gateway = Paused()
    task = asyncio.create_task(ask(world, record))
    await asyncio.wait_for(entered.wait(), 5)
    assert not await run_once(
        world["app"].state.database, world["app"].state.storage, [world["ids"]["bob"]]
    )
    assert (await ask(world, record)).status_code == 429
    async with world["owner"].connect() as conn:
        run_id = await conn.scalar(
            text("SELECT id FROM enterprise.answer_runs WHERE status='running'")
        )
    path = f"/api/workspaces/{world['ids']['a']}/answers/{run_id}/cancel"
    assert (
        await world["client"].post(path, headers=headers(world, "viewer"))
    ).status_code == 404
    assert (await world["client"].post(path, headers=headers(world))).status_code == 200
    assert frames(await asyncio.wait_for(task, 5))[-1]["event"] == "error"
    assert closed.is_set()
    assert await run_once(
        world["app"].state.database, world["app"].state.storage, [world["ids"]["bob"]]
    )
    assert bob["job_id"]


async def test_repeated_request_id_does_not_generate_again(world):
    record = await ready(world)
    body = payload(record)
    assert (await ask(world, record, body=body)).status_code == 200
    assert (await ask(world, record, body=body)).status_code == 409


async def test_retrieval_works_when_model_is_disabled(world):
    record = await ready(world)
    world["app"].state.gateway = None
    assert (await ask(world, record)).status_code == 503
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/search",
        headers=headers(world),
        json=payload(record),
    )
    assert response.status_code == 200 and response.json()["hits"]


async def test_source_integrity_is_checked_before_retrieval_and_answer(world):
    record = await ready(world)
    next(world["root"].glob("*/*/*.bin")).write_bytes(b"altered original")
    assert (await ask(world, record)).status_code == 503
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/search",
        headers=headers(world),
        json=payload(record),
    )
    assert response.status_code == 503
    assert "altered original" not in response.text


async def test_unapproved_endpoint_field_and_unprocessed_sources_are_rejected(world):
    record = (await upload(world)).json()
    assert (await ask(world, record)).status_code == 409
    body = {**payload(record), "endpoint": "https://unapproved.invalid"}
    assert (await ask(world, record, body=body)).status_code == 422


async def test_active_ingestion_lease_fences_generation_after_session_loss(world):
    record = await ready(world)
    bob = (await upload(world, user="bob", workspace="b")).json()
    async with world["app"].state.database.scope(
        world["ids"]["bob"], world["ids"]["b"]
    ) as conn:
        assert await conn.scalar(
            text("SELECT enterprise.claim_ingestion(:token)"), {"token": uuid4()}
        )
    task = asyncio.create_task(ask(world, record))
    await asyncio.sleep(0.4)
    assert not task.done()
    async with world["owner"].begin() as conn:
        await conn.execute(
            text(
                "UPDATE enterprise.ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"
            ),
            {"id": UUID(bob["job_id"])},
        )
    events = frames(await asyncio.wait_for(task, 5))
    assert events[-1]["event"] == "done"


async def test_context_overflow_rejects_before_model_or_run_creation(world):
    record = await ready(world)
    assert (await ask(world, record, "sales " + "😀" * 506)).status_code == 413
    async with world["owner"].connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.answer_runs")) == 0
        )
