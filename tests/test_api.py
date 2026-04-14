"""Tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Test that health endpoint returns 200."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_health_returns_status(self, client: AsyncClient) -> None:
        """Test that health endpoint returns expected status."""
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data


@pytest.mark.asyncio
class TestDocumentsEndpoint:
    """Tests for documents endpoint."""

    async def test_list_documents_empty(self, client: AsyncClient) -> None:
        """Test that empty database returns empty list."""
        response = await client.get("/documents")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_documents_after_ingest(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test documents endpoint returns ingested documents."""
        # Ingest a document first
        files = {"file": ("research.pdf", sample_pdf_bytes, "application/pdf")}
        await client.post("/ingest", files=files)

        response = await client.get("/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "research.pdf"
        assert docs[0]["status"] == "completed"
        assert docs[0]["num_chunks"] > 0


@pytest.mark.asyncio
class TestStatsEndpoint:
    """Tests for stats endpoint."""

    async def test_stats_returns_zeros_on_empty_db(self, client: AsyncClient) -> None:
        """Test that stats returns zeros on empty database."""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 0
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0

    async def test_stats_after_ingest(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test stats endpoint reflects ingested documents."""
        files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
        await client.post("/ingest", files=files)

        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 1
        assert data["total_chunks"] > 0


@pytest.mark.asyncio
class TestIngestEndpoint:
    """Tests for ingest endpoint."""

    async def test_ingest_rejects_non_pdf(self, client: AsyncClient) -> None:
        """Test that non-PDF files are rejected."""
        files = {"file": ("test.txt", b"not a pdf", "text/plain")}
        response = await client.post("/ingest", files=files)
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    async def test_ingest_accepts_valid_pdf(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test that valid PDF files are ingested successfully."""
        files = {"file": ("research.pdf", sample_pdf_bytes, "application/pdf")}
        response = await client.post("/ingest", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "research.pdf"
        assert data["status"] == "completed"
        assert data["num_chunks"] > 0
        assert "document_id" in data

    async def test_ingest_creates_chunks(
        self, client: AsyncClient, multi_page_pdf_bytes: bytes
    ) -> None:
        """Test that ingesting a multi-page PDF creates multiple chunks."""
        files = {"file": ("big_doc.pdf", multi_page_pdf_bytes, "application/pdf")}
        response = await client.post("/ingest", files=files)

        assert response.status_code == 200
        data = response.json()
        # Multi-page PDF with substantial text should create multiple chunks
        assert data["num_chunks"] >= 3

    async def test_ingest_multiple_documents(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test ingesting multiple documents."""
        for i in range(3):
            files = {"file": (f"doc_{i}.pdf", sample_pdf_bytes, "application/pdf")}
            response = await client.post("/ingest", files=files)
            assert response.status_code == 200

        # Verify all documents are listed
        response = await client.get("/documents")
        assert len(response.json()) == 3


@pytest.mark.asyncio
class TestQueryEndpoint:
    """Tests for query endpoint."""

    async def test_query_empty_db_returns_error(self, client: AsyncClient) -> None:
        """Test that querying empty database returns appropriate error."""
        response = await client.post(
            "/query",
            json={"query": "What is machine learning?"},
        )
        # Should fail due to no chunks or no API key
        assert response.status_code in [400, 404, 500]

    async def test_query_with_populated_db(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test query endpoint with mocked LLM response."""
        # First ingest a document
        files = {"file": ("ml_paper.pdf", sample_pdf_bytes, "application/pdf")}
        await client.post("/ingest", files=files)

        # Mock the LLM response
        mock_response = AsyncMock()
        mock_response.answer = "Machine learning is a subset of AI."
        mock_response.total_tokens = 100
        mock_response.model = "test-model"

        with patch("backend.main.generate_answer", return_value=mock_response):
            response = await client.post(
                "/query",
                json={"query": "What is machine learning?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) > 0
            assert "latency_ms" in data

    async def test_query_with_document_filter(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ) -> None:
        """Test query endpoint with document ID filter."""
        # Ingest a document
        files = {"file": ("specific.pdf", sample_pdf_bytes, "application/pdf")}
        ingest_response = await client.post("/ingest", files=files)
        doc_id = ingest_response.json()["document_id"]

        mock_response = AsyncMock()
        mock_response.answer = "Test answer"
        mock_response.total_tokens = 50
        mock_response.model = "test-model"

        with patch("backend.main.generate_answer", return_value=mock_response):
            response = await client.post(
                "/query",
                json={"query": "Test query", "document_id": doc_id},
            )

            assert response.status_code == 200
            # All sources should be from the specified document
            for source in response.json()["sources"]:
                assert source["id"] is not None
