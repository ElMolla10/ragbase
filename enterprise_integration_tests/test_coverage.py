"""Exhaustive coverage against real RLS. The gateway is local and never a provider."""

import asyncio
import json
from uuid import UUID, uuid4

from sqlalchemy import text

from backend.answers.fixtures import FixtureQuoteGateway
from backend.gateway.contracts import GatewayEvent
from enterprise_integration_tests.conftest import headers
from enterprise_integration_tests.test_ingestion_jobs import process
from enterprise_integration_tests.test_pdf_answers import ask, frames, payload, ready
from enterprise_tests.pdf_fixtures import pdf_bytes


def cover_body(record, question, versions=None):
    return {
        "question": question,
        "version_ids": versions or [record["version_id"]],
        "client_request_id": str(uuid4()),
    }


async def cover(
    world, record, question, user="alice", workspace="a", versions=None, body=None
):
    return await world["client"].post(
        f"/api/workspaces/{world['ids'][workspace]}/coverage",
        headers=headers(world, user),
        json=body or cover_body(record, question, versions),
    )


def answer_of(response):
    events = frames(response)
    return next(event for event in events if event["event"] == "answer"), events


async def upload_named(world, filename, content, user="alice", workspace="a"):
    response = await world["client"].post(
        f"/api/workspaces/{world['ids'][workspace]}/documents",
        headers=headers(world, user, filename),
        content=content if isinstance(content, bytes) else content.encode(),
    )
    assert response.status_code == 201, response.text
    assert await process(world, user)
    return response.json()


async def test_multisection_pdf_is_complete_only_after_every_page_is_processed(world):
    record = await upload_named(
        world,
        "sections.pdf",
        pdf_bytes(tuple(f"Section {n} rainfall total {n}." for n in range(1, 9))),
    )
    calls = []

    class Recorded(FixtureQuoteGateway):
        async def stream(self, request):
            calls.append(request)
            async for event in super().stream(request):
                if event.kind == "done":
                    yield GatewayEvent(
                        "done",
                        finish_reason="stop",
                        provenance={"model": "openai/gpt-oss-120b", "provider": "Groq"},
                    )
                else:
                    yield event

    world["app"].state.gateway = Recorded()
    answer, events = answer_of(
        await cover(world, record, "Summarize the selected documents completely.")
    )
    assert answer["coverage_complete"] is True
    assert answer["label"] == "complete"
    assert answer["mode"] == "exhaustive_coverage"
    assert "not complete" not in answer["scope_notice"]
    assert {section["ordinal"] for section in answer["sections"]} == set(range(1, 9))
    assert all(section["status"] == "covered" for section in answer["sections"])
    assert {claim["location"]["page"] for claim in answer["claims"]} >= {1, 8}
    assert answer["metadata"]["strategy"] == "explicit_section_coverage"
    assert answer["metadata"]["model"]["resolved_route"]["provider"] == "Groq"
    assert calls and all(
        "COVERAGE_WORKFLOW" in call.instructions.text for call in calls
    )
    assert all(call.images == () and call.tools == () for call in calls)
    overview = frames(await ask(world, record, "overview"))
    bounded = next(event for event in overview if event["event"] == "answer")
    assert bounded.get("coverage_complete") is not True
    assert "exhaustive review" in bounded["scope_notice"]
    search = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/search",
        headers=headers(world),
        json=payload(record, "rainfall"),
    )
    assert search.json()["exhaustive"] is False
    stored = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/coverage/{events[0]['run_id']}",
        headers=headers(world),
    )
    assert stored.status_code == 200
    body = stored.json()
    assert body["coverage_complete"] is True
    assert "Section 8 rainfall" not in stored.text
    async with world["owner"].connect() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "SELECT question_sha256,metadata::text AS metadata FROM enterprise.coverage_runs WHERE id=:id"
                    ),
                    {"id": UUID(events[0]["run_id"])},
                )
            )
            .mappings()
            .one()
        )
    assert "Summarize" not in row["metadata"]
    assert "rainfall" not in row["metadata"]
    assert len(row["question_sha256"]) == 64


async def test_arabic_and_mixed_documents_keep_original_passages(world):
    arabic = await upload_named(world, "arabic.txt", "إِجمالي المبيعات ١٢٠ جنيه")
    mixed = await upload_named(world, "mixed.md", "Policy 12: لا يجوز تجاوز الحد 40.")
    arabic_answer, _ = answer_of(await cover(world, arabic, "لخص الوثيقة بالكامل"))
    assert arabic_answer["coverage_complete"] is True
    assert "إِجمالي" in arabic_answer["claims"][0]["quote"]
    assert "١٢٠" in arabic_answer["claims"][0]["quote"]
    mixed_answer, _ = answer_of(
        await cover(world, mixed, "Summarize the English and Arabic conditions.")
    )
    assert mixed_answer["coverage_complete"] is True
    assert "لا يجوز" in mixed_answer["claims"][0]["quote"]
    assert "40" in mixed_answer["claims"][0]["quote"]
    long = await upload_named(
        world,
        "long-ar.txt",
        "القسم يذكر هطول الأمطار والحد الأقصى ١٢ ملم. " * 40,
    )
    long_answer, _ = answer_of(await cover(world, long, "لخص كل الأقسام"))
    assert len(long_answer["sections"]) > 1
    assert len(long_answer["sections"]) == long_answer["metadata"]["section_count"]
    accounted = {section["status"] for section in long_answer["sections"]} <= {
        "covered",
        "gap",
    }
    assert accounted
    if not long_answer["coverage_complete"]:
        assert "not complete" in long_answer["scope_notice"]
        assert long_answer["gaps"]
    else:
        assert all(
            section["status"] == "covered" for section in long_answer["sections"]
        )
        assert "١٢" in "".join(claim["quote"] for claim in long_answer["claims"])


async def test_csv_rows_are_covered_and_formula_text_is_not_executed(world):
    record = await upload_named(
        world,
        "synthetic.csv",
        "id,note\n001,Quarter 1 المبيعات\n002,=SUM(A1)\n",
    )
    answer, _ = answer_of(await cover(world, record, "Summarize every row."))
    assert answer["coverage_complete"] is True
    assert {claim["location"]["row"] for claim in answer["claims"]} == {1, 2, 3}
    assert any("001" in claim["quote"] for claim in answer["claims"])
    assert "not executed" in answer["scope_notice"]
    assert "=SUM(A1)" in "".join(claim["quote"] for claim in answer["claims"])


async def test_failed_section_and_unreadable_page_stay_partial(world):
    record = await upload_named(
        world,
        "partial.pdf",
        pdf_bytes(("Visible rainfall total.", "UNIQUEGAPTOKEN later page")),
    )

    class FailToken(FixtureQuoteGateway):
        async def stream(self, request):
            if any("UNIQUEGAPTOKEN" in item for item in request.evidence):
                aliases = []
                for item in request.evidence:
                    payload = json.loads(item)
                    if "UNIQUEGAPTOKEN" not in str(payload.get("text", "")):
                        aliases.append(payload["source_id"])
                if aliases:
                    result = {
                        "insufficient": False,
                        "answer": "Reviewed only the supplied sections. "
                        + " ".join(f"[{alias}]" for alias in aliases),
                    }
                else:
                    result = {
                        "insufficient": True,
                        "answer": "The supplied section could not be summarized.",
                    }
                yield GatewayEvent("text", json.dumps(result))
                yield GatewayEvent("done", finish_reason="stop")
                return
            async for event in super().stream(request):
                yield event

    world["app"].state.gateway = FailToken()
    answer, _ = answer_of(await cover(world, record, "Summarize all pages."))
    assert answer["coverage_complete"] is False
    assert answer["label"] == "partial"
    assert "not complete" in answer["scope_notice"]
    assert "section_processing_failed" in {gap["gap_code"] for gap in answer["gaps"]}
    assert "UNIQUEGAPTOKEN" not in json.dumps(answer)
    assert any(section["status"] == "covered" for section in answer["sections"])
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO enterprise.extracted_pages(workspace_id,version_id,ordinal,content,location,warnings)
            VALUES(:workspace,:version,3,'',CAST(:location AS jsonb),CAST(:warnings AS jsonb))
            """),
            {
                "workspace": world["ids"]["a"],
                "version": record["version_id"],
                "location": json.dumps({"kind": "pdf_page", "page": 3}),
                "warnings": json.dumps(
                    ["no_native_text_ocr_or_visual_review_required"]
                ),
            },
        )
    world["app"].state.gateway = FixtureQuoteGateway()
    unread, _ = answer_of(await cover(world, record, "Summarize all pages again."))
    assert unread["coverage_complete"] is False
    assert "unreadable_section" in {gap["gap_code"] for gap in unread["gaps"]}
    assert "not complete" in unread["scope_notice"]


async def test_denied_access_never_starts_coverage_or_releases_text(world):
    record = await ready(world, b"Quarter 9 secret total 999.")
    called = []

    class Trap(FixtureQuoteGateway):
        async def stream(self, request):
            called.append(request)
            async for event in super().stream(request):
                yield event

    world["app"].state.gateway = Trap()
    assert (
        await cover(world, record, "Summarize", user="bob", workspace="b")
    ).status_code == 404
    assert (
        await world["client"].post(
            f"/api/workspaces/{world['ids']['a']}/coverage",
            headers=headers(world, "bob"),
            json=cover_body(record, "Summarize"),
        )
    ).status_code == 404
    foreign = cover_body(
        record, "Summarize", versions=[record["version_id"], str(uuid4())]
    )
    assert (await cover(world, record, "Summarize", body=foreign)).status_code == 404
    assert (
        await world["client"].post(
            f"/api/workspaces/{world['ids']['a']}/coverage",
            json=cover_body(record, "Summarize"),
        )
    ).status_code == 401
    assert called == []
    viewer, events = answer_of(await cover(world, record, "Summarize", user="viewer"))
    assert viewer["coverage_complete"] is True
    assert "999" in viewer["claims"][0]["quote"]
    hidden = await world["client"].get(
        f"/api/workspaces/{world['ids']['b']}/coverage/{events[0]['run_id']}",
        headers=headers(world, "bob"),
    )
    assert hidden.status_code == 404
    assert "999" not in hidden.text
    cancel = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/coverage/{events[0]['run_id']}/cancel",
        headers=headers(world, "viewer"),
    )
    assert cancel.status_code == 404


async def test_duplicate_request_and_disabled_generation_do_not_run(world):
    record = await ready(world)
    body = cover_body(record, "Summarize the document.")
    assert (await cover(world, record, "Summarize", body=body)).status_code == 200
    assert (await cover(world, record, "Summarize", body=body)).status_code == 409
    world["app"].state.gateway = None
    assert (await cover(world, record, "Summarize again")).status_code == 503
    async with world["owner"].connect() as conn:
        count = await conn.scalar(
            text(
                "SELECT count(*) FROM enterprise.coverage_runs WHERE question_sha256=:hash"
            ),
            {"hash": __import__("hashlib").sha256(b"Summarize again").hexdigest()},
        )
    assert count == 0


async def test_coverage_yields_to_interactive_work_and_can_be_cancelled(world):
    record = await upload_named(
        world,
        "cancel.pdf",
        pdf_bytes(("First visible page.", "UNIQUEGAPTOKEN second page")),
    )
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO enterprise.answer_runs(
                id,workspace_id,created_by,client_request_id,question_sha256,status,lease_until)
            VALUES(:id,:workspace,:actor,:client,:hash,'queued',clock_timestamp()+interval '10 minutes')
            """),
            {
                "id": uuid4(),
                "workspace": world["ids"]["a"],
                "actor": world["ids"]["alice"],
                "client": uuid4(),
                "hash": "b" * 64,
            },
        )
    calls = []

    class Counting(FixtureQuoteGateway):
        async def stream(self, request):
            calls.append(1)
            async for event in super().stream(request):
                yield event

    world["app"].state.gateway = Counting()
    task = asyncio.create_task(cover(world, record, "Summarize every page."))
    deadline = asyncio.get_running_loop().time() + 3
    status = None
    while asyncio.get_running_loop().time() < deadline:
        async with world["owner"].connect() as conn:
            status = await conn.scalar(
                text(
                    "SELECT status FROM enterprise.coverage_runs WHERE status IN ('queued','yielding','running')"
                )
            )
        if status == "yielding":
            break
        await asyncio.sleep(0.05)
    assert status == "yielding"
    assert calls == []
    async with world["owner"].begin() as conn:
        await conn.execute(
            text(
                "UPDATE enterprise.answer_runs SET status='cancelled' WHERE status='queued'"
            )
        )
    waited = await asyncio.wait_for(task, 8)
    finished, _ = answer_of(waited)
    assert finished["coverage_complete"] is True
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
    running = asyncio.create_task(cover(world, record, "Summarize once more."))
    await asyncio.wait_for(entered.wait(), 5)
    blocked = await cover(world, record, "Another coverage")
    assert blocked.status_code == 429
    async with world["owner"].connect() as conn:
        run_id = await conn.scalar(
            text("SELECT id FROM enterprise.coverage_runs WHERE status='running'")
        )
    path = f"/api/workspaces/{world['ids']['a']}/coverage/{run_id}/cancel"
    assert (
        await world["client"].post(path, headers=headers(world, "viewer"))
    ).status_code == 404
    assert (await world["client"].post(path, headers=headers(world))).status_code == 200
    cancelled = frames(await asyncio.wait_for(running, 5))
    assert cancelled[-1]["event"] == "error"
    assert closed.is_set()
    assert "UNIQUEGAPTOKEN" not in json.dumps(cancelled)
    coverage = next(event for event in cancelled if event["event"] == "coverage")
    assert coverage["coverage_complete"] is False
    assert "not complete" in coverage["scope_notice"]


async def test_active_extraction_lease_delays_coverage(world):
    record = await ready(world, b"Rainfall total 12.")
    bob = await upload_named(
        world, "other.txt", "bob document", user="bob", workspace="b"
    )
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("""
            UPDATE enterprise.ingestion_jobs
            SET status='extracting', lease_until=clock_timestamp()+interval '10 minutes'
            WHERE id=:id
            """),
            {"id": UUID(bob["job_id"])},
        )
    calls = []

    class Counting(FixtureQuoteGateway):
        async def stream(self, request):
            calls.append(1)
            async for event in super().stream(request):
                yield event

    world["app"].state.gateway = Counting()
    task = asyncio.create_task(cover(world, record, "Summarize the rainfall."))
    deadline = asyncio.get_running_loop().time() + 3
    status = None
    while asyncio.get_running_loop().time() < deadline:
        async with world["owner"].connect() as conn:
            status = await conn.scalar(
                text(
                    "SELECT status FROM enterprise.coverage_runs WHERE status IN ('queued','yielding','running')"
                )
            )
        if status == "yielding":
            break
        await asyncio.sleep(0.05)
    assert status == "yielding"
    assert calls == []
    async with world["owner"].begin() as conn:
        await conn.execute(
            text(
                "UPDATE enterprise.ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"
            ),
            {"id": UUID(bob["job_id"])},
        )
    answer, _ = answer_of(await asyncio.wait_for(task, 5))
    assert answer["coverage_complete"] is True


async def test_revoked_membership_releases_no_summary(world):
    record = await ready(world, b"UNIQUEGAPTOKEN confidential rainfall")

    class Revoke(FixtureQuoteGateway):
        async def stream(self, request):
            async with world["owner"].begin() as conn:
                await conn.execute(
                    text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
                    {"id": world["ids"]["alice"]},
                )
            async for event in super().stream(request):
                yield event

    world["app"].state.gateway = Revoke()
    response = await cover(world, record, "Summarize the document.")
    body = response.text
    assert frames(response)[-1]["event"] == "error"
    assert "UNIQUEGAPTOKEN" not in body
    assert "confidential" not in body


async def test_two_selected_versions_are_both_required(world):
    first = await upload_named(world, "alpha.txt", "Alpha rainfall is 3.")
    second = await upload_named(world, "beta.txt", "Beta runoff is 4.")
    answer, _ = answer_of(
        await cover(
            world,
            first,
            "Compare the selected documents.",
            versions=[first["version_id"], second["version_id"]],
        )
    )
    assert answer["coverage_complete"] is True
    assert {section["filename"] for section in answer["sections"]} == {
        "alpha.txt",
        "beta.txt",
    }
    assert "alpha.txt" in answer["answer"] and "beta.txt" in answer["answer"]
    quotes = " ".join(claim["quote"] for claim in answer["claims"])
    assert "Alpha rainfall" in quotes and "Beta runoff" in quotes
