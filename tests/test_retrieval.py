"""Tests for vector retrieval."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ingest import TextChunk, embed_chunks
from backend.models import Chunk, Document
from backend.retrieval import embed_query, search_chunks


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
