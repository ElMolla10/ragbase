"""Query logging for observability."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import QueryLog
from backend.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


async def log_query(
    session: AsyncSession,
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    latency_ms: int,
    token_count: int,
    model_used: str,
) -> QueryLog:
    """Log a query to the database for observability.

    Args:
        session: Database session.
        query: The user's query.
        answer: The generated answer.
        chunks: Retrieved chunks used for context.
        latency_ms: Total latency in milliseconds.
        token_count: Total tokens used.
        model_used: The LLM model used.

    Returns:
        Created QueryLog object.
    """
    chunk_ids = [chunk.id for chunk in chunks]

    log_entry = QueryLog(
        query=query,
        answer=answer,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
        token_count=token_count,
        model_used=model_used,
    )

    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)

    logger.info(f"Logged query with latency {latency_ms}ms, {token_count} tokens")
    return log_entry
