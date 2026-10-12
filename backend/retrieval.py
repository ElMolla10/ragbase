"""Vector search with optional PostgreSQL full-text reciprocal rank fusion."""

import logging
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.ingest import get_embedding_model
from backend.models import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score."""

    id: int
    document_id: int
    content: str
    page_number: int
    chunk_index: int
    similarity_score: float


def embed_query(query: str) -> list[float]:
    """Embed a query string.

    Args:
        query: The search query.

    Returns:
        Embedding vector (384 dimensions).
    """
    model = get_embedding_model()
    embedding = model.encode(query, show_progress_bar=False)
    return embedding.tolist()


async def search_chunks(
    session: AsyncSession,
    query: str,
    document_id: int | None = None,
    top_k: int = settings.top_k,
) -> list[RetrievedChunk]:
    """Search for similar chunks using pgvector cosine similarity.

    Args:
        session: Database session.
        query: The search query.
        document_id: Optional document ID to restrict search.
        top_k: Number of results to return.

    Returns:
        Results ranked by cosine similarity, or reciprocal rank fusion when
        hybrid search is enabled. Hybrid scores are not cosine similarities.
    """
    if top_k <= 0:
        return []

    # Embed the query
    query_embedding = embed_query(query)

    # Build the query using cosine distance (1 - cosine_similarity)
    # pgvector's <=> operator computes cosine distance
    distance = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.content,
            Chunk.page_number,
            Chunk.chunk_index,
            (1 - distance).label("similarity"),
        )
        .order_by(distance)
        .limit(top_k * 4 if settings.use_hybrid_search else top_k)
    )

    # Filter by document if specified
    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)

    result = await session.execute(stmt)
    rows = result.all()

    fused_scores: dict[int, float] = {}
    if settings.use_hybrid_search:
        # 'simple' retains English/Arabic terms without English-only stemming.
        terms = func.to_tsvector("simple", Chunk.content)
        text_query = func.plainto_tsquery("simple", query)
        lexical_stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.content,
                Chunk.page_number,
                Chunk.chunk_index,
            )
            .where(terms.bool_op("@@")(text_query))
            .order_by(func.ts_rank_cd(terms, text_query).desc(), Chunk.id)
            .limit(top_k * 4)
        )
        if document_id is not None:
            lexical_stmt = lexical_stmt.where(Chunk.document_id == document_id)
        lexical_rows = (await session.execute(lexical_stmt)).all()

        # RRF avoids mixing incompatible cosine and full-text score scales.
        candidates = {}
        for ranking in (rows, lexical_rows):
            for rank, row in enumerate(ranking, start=1):
                candidates[row.id] = row
                fused_scores[row.id] = fused_scores.get(row.id, 0.0) + 1 / (60 + rank)
        ids = sorted(candidates, key=lambda id_: (-fused_scores[id_], id_))[:top_k]
        rows = [candidates[id_] for id_ in ids]

    retrieved = [
        RetrievedChunk(
            id=row.id,
            document_id=row.document_id,
            content=row.content,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            similarity_score=(
                fused_scores[row.id]
                if settings.use_hybrid_search
                else float(row.similarity)
            ),
        )
        for row in rows
    ]

    logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
    return retrieved
