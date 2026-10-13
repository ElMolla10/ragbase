"""FastAPI application for RAGBase."""

import json
import logging
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session, init_db
from backend.ingest import get_embedding_model, ingest_pdf
from backend.llm import generate_answer, stream_answer
from backend.logger import log_query
from backend.models import Chunk, Document, QueryLog
from backend.protection import protect_request, settings
from backend.retrieval import search_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")
    logger.info("Warming embedding model...")
    get_embedding_model()
    logger.info("Embedding model ready")
    yield


app = FastAPI(
    title="RAGBase",
    description="Research paper Q&A platform with ML observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.middleware("http")(protect_request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    document_id: int | None = None


class SourceChunk(BaseModel):
    id: int
    document_id: int
    content: str
    page_number: int
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: int
    token_count: int
    model_used: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    num_chunks: int
    status: str
    uploaded_at: str


class IngestResponse(BaseModel):
    document_id: int
    filename: str
    num_chunks: int
    status: str


class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    message: str


class DeleteResponse(BaseModel):
    message: str
    document_id: int
    filename: str


def _sources(chunks) -> list[dict]:
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content[:500],
            "page_number": chunk.page_number,
            "similarity_score": round(chunk.similarity_score, 4),
        }
        for chunk in chunks
    ]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", message="RAGBase is running")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await file.read(settings.max_upload_bytes + 1)
        if len(pdf_bytes) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413, detail="PDF exceeds upload size limit"
            )
        document = await ingest_pdf(session, pdf_bytes, file.filename)
        return IngestResponse(
            document_id=document.id,
            filename=document.filename,
            num_chunks=document.num_chunks,
            status=document.status,
        )
    except HTTPException:
        raise
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
    start_time = time.time()
    try:
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

        llm_response = await generate_answer(request.query, chunks)
        latency_ms = int((time.time() - start_time) * 1000)

        await log_query(
            session,
            query=request.query,
            answer=llm_response.answer,
            chunks=chunks,
            latency_ms=latency_ms,
            token_count=llm_response.total_tokens,
            model_used=llm_response.model,
        )

        return QueryResponse(
            answer=llm_response.answer,
            sources=_sources(chunks),
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


@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    start_time = time.time()
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context") from e

    sources = _sources(chunks)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        answer_parts: list[str] = []
        try:
            async for token in stream_answer(request.query, chunks):
                answer_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            answer = "".join(answer_parts)
            latency_ms = int((time.time() - start_time) * 1000)
            token_count = max(1, len(answer) // 4)

            await log_query(
                session,
                query=request.query,
                answer=answer,
                chunks=chunks,
                latency_ms=latency_ms,
                token_count=token_count,
                model_used=settings.llm_model,
            )

            done = {
                "type": "done",
                "latency_ms": latency_ms,
                "token_count": token_count,
                "model_used": settings.llm_model,
            }
            yield f"data: {json.dumps(done)}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> list[DocumentResponse]:
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


@app.get("/documents/{document_id}/file")
async def get_document_file(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document or not document.pdf_bytes:
        raise HTTPException(
            status_code=404,
            detail="PDF not available. Re-upload the document.",
        )

    filename = quote(document.filename or "document.pdf")
    return Response(
        content=document.pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{filename}",
            "Cache-Control": "private, max-age=3600",
        },
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: AsyncSession = Depends(get_session),
) -> StatsResponse:
    query_stats = await session.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.coalesce(func.avg(QueryLog.latency_ms), 0).label("avg_latency"),
        )
    )
    query_row = query_stats.one()

    doc_count = await session.execute(select(func.count(Document.id)))
    total_docs = doc_count.scalar() or 0

    chunk_count = await session.execute(select(func.count(Chunk.id)))
    total_chunks = chunk_count.scalar() or 0

    return StatsResponse(
        total_queries=query_row.total,
        avg_latency_ms=round(float(query_row.avg_latency), 2),
        total_documents=total_docs,
        total_chunks=total_chunks,
    )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> DeleteResponse:
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document with id {document_id} not found",
        )

    filename = document.filename
    await session.delete(document)
    await session.commit()

    logger.info(f"Deleted document '{filename}' (id={document_id})")
    return DeleteResponse(
        message="Document deleted successfully",
        document_id=document_id,
        filename=filename,
    )
