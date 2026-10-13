"""Compress and load stored PDF bytes."""

import logging
import zlib

from sqlalchemy import select

from backend.database import get_session_context
from backend.models import Document

logger = logging.getLogger(__name__)


def compress_pdf(pdf_bytes: bytes) -> bytes:
    return zlib.compress(pdf_bytes, level=6)


def load_pdf(stored: bytes) -> bytes:
    if stored.startswith(b"%PDF"):
        return stored
    return zlib.decompress(stored)


async def save_pdf_bytes(document_id: int, pdf_bytes: bytes) -> None:
    compressed = compress_pdf(pdf_bytes)
    async with get_session_context() as session:
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if not document:
            return
        document.pdf_bytes = compressed
        await session.commit()
    logger.info(
        "Stored PDF %s (%s -> %s bytes)",
        document_id,
        len(pdf_bytes),
        len(compressed),
    )
