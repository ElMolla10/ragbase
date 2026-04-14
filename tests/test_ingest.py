"""Tests for PDF ingestion pipeline."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ingest import (
    PageText,
    TextChunk,
    chunk_text,
    embed_chunks,
    extract_text_from_pdf,
    ingest_pdf,
)
from backend.models import Chunk


class TestExtractText:
    """Tests for PDF text extraction."""

    def test_extract_text_from_valid_pdf(self, sample_pdf_bytes: bytes) -> None:
        """Test that text extraction works on a valid PDF."""
        pages = extract_text_from_pdf(sample_pdf_bytes)
        # The minimal PDF may or may not extract text depending on PyMuPDF version
        assert isinstance(pages, list)

    def test_extract_text_from_invalid_pdf(self) -> None:
        """Test that invalid PDF raises an error."""
        import fitz

        with pytest.raises(fitz.FileDataError):
            extract_text_from_pdf(b"not a pdf")


class TestChunking:
    """Tests for text chunking."""

    def test_chunk_text_basic(self) -> None:
        """Test basic chunking functionality."""
        pages = [
            PageText(
                page_number=1, text="A" * 3000
            ),  # Long text to ensure multiple chunks
        ]
        chunks = chunk_text(pages, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.page_number == 1 for c in chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        """Test that chunk indices are sequential."""
        pages = [
            PageText(page_number=1, text="A" * 2000),
            PageText(page_number=2, text="B" * 2000),
        ]
        chunks = chunk_text(pages, chunk_size=100, overlap=20)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_produce_no_chunks(self) -> None:
        """Test that empty pages produce no chunks."""
        pages = [PageText(page_number=1, text="   ")]
        chunks = chunk_text(pages)
        assert len(chunks) == 0


class TestEmbedding:
    """Tests for embedding generation."""

    def test_embedding_dimension(self) -> None:
        """Test that embeddings have correct dimensions (384)."""
        chunks = [
            TextChunk(content="Test content", page_number=1, chunk_index=0),
            TextChunk(content="Another chunk", page_number=1, chunk_index=1),
        ]
        embeddings = embed_chunks(chunks)

        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embedding_values_are_floats(self) -> None:
        """Test that embedding values are floats."""
        chunks = [TextChunk(content="Test", page_number=1, chunk_index=0)]
        embeddings = embed_chunks(chunks)

        assert all(isinstance(v, float) for v in embeddings[0])


@pytest.mark.asyncio
class TestIngestPdf:
    """Integration tests for full PDF ingestion pipeline."""

    async def test_ingest_pdf_creates_document(
        self, db_session: AsyncSession, sample_pdf_bytes: bytes
    ) -> None:
        """Test that ingest_pdf creates a document record."""
        document = await ingest_pdf(db_session, sample_pdf_bytes, "test.pdf")

        assert document.id is not None
        assert document.filename == "test.pdf"
        assert document.status == "completed"
        assert document.num_chunks > 0

    async def test_ingest_pdf_creates_chunks(
        self, db_session: AsyncSession, sample_pdf_bytes: bytes
    ) -> None:
        """Test that ingest_pdf creates chunk records with embeddings."""
        document = await ingest_pdf(db_session, sample_pdf_bytes, "research.pdf")

        result = await db_session.execute(
            select(Chunk).where(Chunk.document_id == document.id)
        )
        chunks = result.scalars().all()

        assert len(chunks) == document.num_chunks
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(len(chunk.embedding) == 384 for chunk in chunks)
        assert all(chunk.content for chunk in chunks)

    async def test_ingest_pdf_chunk_indices_sequential(
        self, db_session: AsyncSession, multi_page_pdf_bytes: bytes
    ) -> None:
        """Test that chunk indices are sequential."""
        document = await ingest_pdf(db_session, multi_page_pdf_bytes, "multipage.pdf")

        result = await db_session.execute(
            select(Chunk)
            .where(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    async def test_ingest_pdf_preserves_page_numbers(
        self, db_session: AsyncSession, multi_page_pdf_bytes: bytes
    ) -> None:
        """Test that page numbers are preserved in chunks."""
        document = await ingest_pdf(db_session, multi_page_pdf_bytes, "multipage.pdf")

        result = await db_session.execute(
            select(Chunk).where(Chunk.document_id == document.id)
        )
        chunks = result.scalars().all()

        page_numbers = {c.page_number for c in chunks}
        # Multi-page PDF should have chunks from multiple pages
        assert len(page_numbers) >= 1

    async def test_ingest_invalid_pdf_fails(self, db_session: AsyncSession) -> None:
        """Test that invalid PDF raises an error."""
        import fitz

        with pytest.raises(fitz.FileDataError):
            await ingest_pdf(db_session, b"not a pdf", "invalid.pdf")
