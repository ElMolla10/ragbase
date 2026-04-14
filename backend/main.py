"""FastAPI application for RAGBase."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session, init_db
from backend.ingest import ingest_pdf
from backend.llm import generate_answer
from backend.logger import log_query
from backend.models import Chunk, Document, QueryLog
from backend.retrieval import search_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")
    yield


app = FastAPI(
    title="RAGBase",
    description="Research paper Q&A platform with ML observability",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str
    document_id: int | None = None


class SourceChunk(BaseModel):
    """A source chunk in the response."""

    id: int
    content: str
    page_number: int
    similarity_score: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: list[SourceChunk]
    latency_ms: int
    token_count: int
    model_used: str


class DocumentResponse(BaseModel):
    """Response model for document info."""

    id: int
    filename: str
    num_chunks: int
    status: str
    uploaded_at: str


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    document_id: int
    filename: str
    num_chunks: int
    status: str


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    total_queries: int
    avg_latency_ms: float
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="RAGBase is running")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    """Upload and ingest a PDF document.

    Args:
        file: PDF file to upload.
        session: Database session.

    Returns:
        Document information with chunk count.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await file.read()
        document = await ingest_pdf(session, pdf_bytes, file.filename)

        return IngestResponse(
            document_id=document.id,
            filename=document.filename,
            num_chunks=document.num_chunks,
            status=document.status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document") from e


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    """Query documents and get an answer.

    Args:
        request: Query request with question and optional document filter.
        session: Database session.

    Returns:
        Answer with sources and metrics.
    """
    start_time = time.time()

    try:
        # Retrieve relevant chunks
        chunks = await search_chunks(
            session,
            request.query,
            document_id=request.document_id,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found. Please upload documents first.",
            )

        # Generate answer
        llm_response = await generate_answer(request.query, chunks)

        latency_ms = int((time.time() - start_time) * 1000)

        # Log the query
        await log_query(
            session,
            query=request.query,
            answer=llm_response.answer,
            chunks=chunks,
            latency_ms=latency_ms,
            token_count=llm_response.total_tokens,
            model_used=llm_response.model,
        )

        sources = [
            SourceChunk(
                id=chunk.id,
                content=chunk.content[:500],  # Truncate for response
                page_number=chunk.page_number,
                similarity_score=round(chunk.similarity_score, 4),
            )
            for chunk in chunks
        ]

        return QueryResponse(
            answer=llm_response.answer,
            sources=sources,
            latency_ms=latency_ms,
            token_count=llm_response.total_tokens,
            model_used=llm_response.model,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query") from e


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> list[DocumentResponse]:
    """List all uploaded documents.

    Args:
        session: Database session.

    Returns:
        List of document information.
    """
    result = await session.execute(
        select(Document).order_by(Document.uploaded_at.desc())
    )
    documents = result.scalars().all()

    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            num_chunks=doc.num_chunks,
            status=doc.status,
            uploaded_at=doc.uploaded_at.isoformat(),
        )
        for doc in documents
    ]


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: AsyncSession = Depends(get_session),
) -> StatsResponse:
    """Get aggregate statistics.

    Args:
        session: Database session.

    Returns:
        Statistics about queries, documents, and chunks.
    """
    # Count queries and average latency
    query_stats = await session.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.coalesce(func.avg(QueryLog.latency_ms), 0).label("avg_latency"),
        )
    )
    query_row = query_stats.one()

    # Count documents
    doc_count = await session.execute(select(func.count(Document.id)))
    total_docs = doc_count.scalar() or 0

    # Count chunks
    chunk_count = await session.execute(select(func.count(Chunk.id)))
    total_chunks = chunk_count.scalar() or 0

    return StatsResponse(
        total_queries=query_row.total,
        avg_latency_ms=round(float(query_row.avg_latency), 2),
        total_documents=total_docs,
        total_chunks=total_chunks,
    )
