"""Compress and load stored PDF bytes."""

import zlib


def compress_pdf(pdf_bytes: bytes) -> bytes:
    return zlib.compress(pdf_bytes, level=6)


def load_pdf(stored: bytes) -> bytes:
    if stored.startswith(b"%PDF"):
        return stored
    return zlib.decompress(stored)
