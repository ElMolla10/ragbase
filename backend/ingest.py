"""PDF ingestion pipeline: parsing, chunking, embedding, and storing."""

import logging
import re
from dataclasses import dataclass

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.models import Chunk, Document

logger = logging.getLogger(__name__)
settings = get_settings()

# Load embedding model once at module level
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        model = SentenceTransformer(settings.embedding_model)
        if model.get_sentence_embedding_dimension() != settings.embedding_dimension:
            raise ValueError("Embedding model dimension must match the database schema")
        _embedding_model = model
    return _embedding_model


def embedding_input(text: str, *, is_query: bool = False) -> str:
    """Apply E5's retrieval prefixes, including for non-English inputs."""
    model_name = settings.embedding_model.lower().rsplit("/", 1)[-1]
    if model_name.startswith(("e5-", "multilingual-e5-")):
        return ("query: " if is_query else "passage: ") + text
    return text


@dataclass
class PageText:
    """Extracted text from a single PDF page."""

    page_number: int
    text: str


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    page_number: int
    chunk_index: int


def extract_text_from_pdf(pdf_bytes: bytes) -> list[PageText]:
    """Extract text from each page of a PDF.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        List of PageText objects, one per page.
    """
    pages: list[PageText] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(PageText(page_number=page_num + 1, text=text))

    logger.info(f"Extracted text from {len(pages)} pages")
    return pages


def chunk_text(
    pages: list[PageText],
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[TextChunk]:
    """Split each page at sentence/paragraph boundaries where possible.

    Args:
        pages: List of PageText objects from PDF extraction.
        chunk_size: Approximate token budget (four characters per token).
        overlap: Maximum overlap budget in the same approximate token units.

    Returns:
        List of TextChunk objects.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        text = page.text
        # Keep legacy size units, prefer sentence/paragraph endings, and fall
        # back to words or hard cuts for oversized sentences/unbroken text.
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4

        start = 0
        while start < len(text):
            end = min(start + char_chunk_size, len(text))
            if end < len(text):
                window = text[start:end]
                boundaries = list(re.finditer(r"\n\s*\n|(?<=[.!?؟。])\s+", window))
                if not boundaries:
                    boundaries = list(re.finditer(r"\s+", window))
                if boundaries and boundaries[-1].end() > char_overlap:
                    end = start + boundaries[-1].end()
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append(
                    TextChunk(
                        content=chunk_content,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end == len(text):
                break
            next_start = max(start + 1, end - char_overlap)
            if char_overlap and next_start > 0 and not text[next_start - 1].isspace():
                boundary = re.search(r"\s+", text[next_start:end])
                if boundary:
                    next_start += boundary.end()
            start = next_start

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def embed_chunks(chunks: list[TextChunk]) -> list[list[float]]:
    """Generate embeddings for text chunks.

    Args:
        chunks: List of TextChunk objects.

    Returns:
        List of embedding vectors (384 dimensions).
    """
    model = get_embedding_model()
    texts = [embedding_input(chunk.content) for chunk in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )

    logger.info(f"Generated {len(embeddings)} embeddings")
    return [emb.tolist() for emb in embeddings]


async def ingest_pdf(
    session: AsyncSession,
    pdf_bytes: bytes,
    filename: str,
    file_path: str | None = None,
) -> Document:
    """Full ingestion pipeline: extract, chunk, embed, and store.

    Args:
        session: Database session.
        pdf_bytes: Raw PDF file bytes.
        filename: Original filename.
        file_path: Optional path where file is stored.

    Returns:
        Created Document object with all chunks.
    """
    # Create document record
    document = Document(
        filename=filename,
        file_path=file_path,
        status="processing",
    )
    session.add(document)
    await session.flush()  # Get the document ID

    try:
        # Extract text from PDF
        pages = extract_text_from_pdf(pdf_bytes)

        if not pages:
            document.status = "failed"
            document.num_chunks = 0
            await session.commit()
            raise ValueError("No text could be extracted from PDF")

        # Chunk the text
        text_chunks = chunk_text(pages)

        # Generate embeddings
        embeddings = embed_chunks(text_chunks)

        # Create chunk records
        for chunk, embedding in zip(text_chunks, embeddings, strict=True):
            db_chunk = Chunk(
                document_id=document.id,
                content=chunk.content,
                embedding=embedding,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
            )
            session.add(db_chunk)

        # Update document status
        document.num_chunks = len(text_chunks)
        document.status = "completed"

        await session.commit()
        await session.refresh(document)

        logger.info(f"Ingested document '{filename}' with {document.num_chunks} chunks")
        return document

    except Exception as e:
        document.status = "failed"
        await session.commit()
        logger.error(f"Failed to ingest document '{filename}': {e}")
        raise
