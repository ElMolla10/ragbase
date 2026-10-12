"""Tests for vector retrieval."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ingest import TextChunk, embed_chunks
from backend.models import Chunk, Document
from backend.retrieval import embed_query, search_chunks
from backend.retrieval import settings as retrieval_settings


class TestEmbedQuery:
    """Tests for query embedding."""

    def test_embed_query_returns_correct_dimension(self) -> None:
        """Test that query embedding has 384 dimensions."""
        embedding = embed_query("What is machine learning?")
        assert len(embedding) == 384

    def test_embed_query_returns_floats(self) -> None:
        """Test that embedding contains floats."""
        embedding = embed_query("Test query")
        assert all(isinstance(v, float) for v in embedding)


@pytest.mark.asyncio
class TestSearchChunks:
    """Tests for similarity search."""

    @pytest_asyncio.fixture
    async def populated_db(self, db_session: AsyncSession) -> Document:
        """Create a document with chunks for testing."""
        doc = Document(filename="test.pdf", status="completed", num_chunks=5)
        db_session.add(doc)
        await db_session.flush()

        # Create chunks with embeddings
        test_chunks = [
            TextChunk(
                content="Machine learning is a subset of AI",
                page_number=1,
                chunk_index=0,
            ),
            TextChunk(
                content="Deep learning uses neural networks",
                page_number=1,
                chunk_index=1,
            ),
            TextChunk(
                content="Natural language processing for text",
                page_number=2,
                chunk_index=2,
            ),
            TextChunk(
                content="Computer vision for image analysis",
                page_number=2,
                chunk_index=3,
            ),
            TextChunk(
                content="Reinforcement learning for agents",
                page_number=3,
                chunk_index=4,
            ),
        ]
        embeddings = embed_chunks(test_chunks)

        for chunk, emb in zip(test_chunks, embeddings, strict=True):
            db_chunk = Chunk(
                document_id=doc.id,
                content=chunk.content,
                embedding=emb,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
            )
            db_session.add(db_chunk)

        await db_session.commit()
        return doc

    async def test_search_returns_top_k_results(
        self, db_session: AsyncSession, populated_db: Document
    ) -> None:
        """Test that search returns requested number of results."""
        results = await search_chunks(db_session, "machine learning", top_k=3)
        assert len(results) == 3

    async def test_search_results_sorted_by_similarity(
        self, db_session: AsyncSession, populated_db: Document
    ) -> None:
        """Test that results are sorted by similarity score (descending)."""
        results = await search_chunks(
            db_session, "neural networks deep learning", top_k=5
        )
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_with_document_filter(
        self, db_session: AsyncSession, populated_db: Document
    ) -> None:
        """Test that document filter restricts results."""
        results = await search_chunks(
            db_session,
            "machine learning",
            document_id=populated_db.id,
            top_k=3,
        )
        assert all(r.document_id == populated_db.id for r in results)

    async def test_search_empty_database(self, db_session: AsyncSession) -> None:
        """Test search on empty database returns empty list."""
        results = await search_chunks(db_session, "test query")
        assert results == []

    async def test_hybrid_promotes_keyword_result_outside_vector_candidates(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = Document(filename="hybrid.pdf", status="completed", num_chunks=10)
        other = Document(filename="other.pdf", status="completed", num_chunks=1)
        db_session.add_all([doc, other])
        await db_session.flush()
        chunks = []
        for i in range(10):
            chunk = Chunk(
                document_id=doc.id,
                content=(
                    "rareterm " * 8 if i == 9 else "rareterm" if i == 0 else "filler"
                ),
                embedding=[1.0 - i / 5, i / 5] + [0.0] * 382,
                chunk_index=i,
                page_number=1,
            )
            db_session.add(chunk)
            chunks.append(chunk)
        db_session.add(
            Chunk(
                document_id=other.id,
                content="rareterm " * 20,
                embedding=[1.0] + [0.0] * 383,
                chunk_index=0,
                page_number=2,
            )
        )
        await db_session.commit()
        monkeypatch.setattr(
            "backend.retrieval.embed_query", lambda _: [1.0] + [0.0] * 383
        )
        monkeypatch.setattr(retrieval_settings, "use_hybrid_search", False)
        vector = await search_chunks(
            db_session, "rareterm", document_id=doc.id, top_k=2
        )
        assert chunks[-1].id not in [r.id for r in vector]

        monkeypatch.setattr(retrieval_settings, "use_hybrid_search", True)
        hybrid = await search_chunks(
            db_session, "rareterm", document_id=doc.id, top_k=2
        )
        assert [r.id for r in hybrid] == [chunks[0].id, chunks[-1].id]
        assert len({r.id for r in hybrid}) == 2
        assert all(r.document_id == doc.id for r in hybrid)

    async def test_hybrid_without_keyword_matches_returns_vector_order(
        self,
        db_session: AsyncSession,
        populated_db: Document,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(retrieval_settings, "use_hybrid_search", False)
        vector = await search_chunks(db_session, "xyznonexistent", top_k=3)
        monkeypatch.setattr(retrieval_settings, "use_hybrid_search", True)
        hybrid = await search_chunks(db_session, "xyznonexistent", top_k=3)
        assert [r.id for r in hybrid] == [r.id for r in vector]
        assert [r.similarity_score for r in hybrid] == sorted(
            [r.similarity_score for r in hybrid], reverse=True
        )
