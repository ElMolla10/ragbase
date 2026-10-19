"""Generic paper questions invoke generation on scoped document context."""

import json

from backend.gateway.contracts import GatewayEvent
from backend.search.retrieve import document_context
from enterprise_integration_tests.conftest import headers
from enterprise_integration_tests.test_ingestion_jobs import process
from enterprise_integration_tests.test_pdf_answers import ask, frames, ready
from enterprise_tests.pdf_fixtures import pdf_bytes


async def test_paper_overview_uses_groq_gateway_contract_without_keyword_overlap(world):
    record = await ready(world, b"Rainfall predictions use sensor measurements.")
    calls = []

    class Explain:
        async def stream(self, request):
            calls.append(request)
            source = json.loads(request.evidence[0])
            yield GatewayEvent(
                "text",
                json.dumps(
                    {
                        "insufficient": False,
                        "answer": "The document concerns predicting rainfall from sensor data. ["
                        + source["source_id"]
                        + "]",
                    }
                ),
            )
            yield GatewayEvent(
                "done",
                finish_reason="stop",
                provenance={
                    "model": "openai/gpt-oss-120b",
                    "provider": "Groq",
                },
            )

    world["app"].state.gateway = Explain()
    result = frames(await ask(world, record, "whats the paper about?"))
    answer = next(event for event in result if event["event"] == "answer")
    assert len(calls) == 1 and not answer["insufficient"]
    assert (
        answer["claims"][0]["quote"] == "Rainfall predictions use sensor measurements."
    )
    assert answer["claims"][0]["support_kind"] == "source_passage"
    assert answer["answer"].startswith("The document concerns")
    assert answer["metadata"]["model_called"]
    assert answer["metadata"]["coverage"]["strategy"] == "document_overview"
    assert answer["metadata"]["coverage"]["all_extracted_chunks_included"]
    assert answer["metadata"]["model"]["resolved_route"]["provider"] == "Groq"
    assert (
        await ask(world, record, "whats the paper about?", user="bob", workspace="b")
    ).status_code == 404


async def test_large_document_candidates_span_first_and_last_pages_without_false_full_coverage(
    world,
):
    base = f"/api/workspaces/{world['ids']['a']}"
    response = await world["client"].post(
        base + "/documents",
        headers=headers(world, filename="synthetic-paper.pdf"),
        content=pdf_bytes(
            tuple(f"Section {n}: Rainfall findings. " * 2 for n in range(1, 21))
        ),
    )
    assert response.status_code == 201 and await process(world)
    record = response.json()
    database = world["app"].state.database
    async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        from uuid import UUID

        hits, total = await document_context(
            conn, world["ids"]["a"], [UUID(record["version_id"])]
        )
    assert total == 20
    assert {1, 20} <= {hit["ordinal"] for hit in hits}
    result = frames(await ask(world, record, "overview"))
    answer = next(event for event in result if event["event"] == "answer")
    assert answer["metadata"]["coverage"]["included_chunks"] < total
    assert not answer["metadata"]["coverage"]["all_extracted_chunks_included"]
    assert "some document text was not supplied" in answer["scope_notice"]
