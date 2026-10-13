"""Tiny retrieval regression harness: real embeddings, pgvector, and RRF.

Run against a disposable test database: pytest tests/test_eval.py -q
No Groq key or external PDF downloads are needed.
"""

import pytest

from backend.ingest import PageText, chunk_text, embed_chunks
from backend.models import Chunk, Document
from backend.retrieval import search_chunks, settings


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
async def test_retrieval_keyword_regressions(db_session, monkeypatch, hybrid):
    monkeypatch.setattr(settings, "use_hybrid_search", hybrid)
    pages = [
        PageText(
            1,
            "Solar panels convert sunlight into electricity. Photovoltaic cells use silicon semiconductors.",
        ),
        PageText(
            2,
            "Vaccines train the immune system to recognize pathogens. Immunization helps prevent infectious disease.",
        ),
    ]
    chunks = chunk_text(pages)
    document = Document(
        filename="eval-texts", status="completed", num_chunks=len(chunks)
    )
    db_session.add(document)
    await db_session.flush()
    for chunk, vector in zip(chunks, embed_chunks(chunks), strict=True):
        db_session.add(
            Chunk(
                document_id=document.id,
                content=chunk.content,
                embedding=vector,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
            )
        )
    await db_session.commit()
    cases = [
        ("How do solar panels generate power?", "photovoltaic"),
        ("How does immunization protect against disease?", "vaccines"),
        ("كيف تحول الألواح الشمسية ضوء الشمس إلى كهرباء؟", "photovoltaic"),
    ]
    for query, keyword in cases:
        results = await search_chunks(
            db_session, query, document_id=document.id, top_k=1
        )
        assert results and keyword in results[0].content.lower(), (
            hybrid,
            query,
            results,
        )
