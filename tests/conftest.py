"""Pytest configuration and fixtures."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database import get_session
from backend.main import app
from backend.models import Base

# Use DATABASE_URL from environment (set by CI) or default to local test database
TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://ragbase:ragbase@localhost:5432/ragbase_test",
)

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with test_session_maker() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database override."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a valid PDF with extractable text for testing."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()

    # Insert substantial text for chunking tests
    text = """Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    Deep learning uses neural networks with many layers to model complex patterns.
    Natural language processing allows computers to understand human language.
    This document covers fundamental concepts in AI and machine learning research."""

    page.insert_text((72, 72), text, fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Create a multi-page PDF for comprehensive testing."""
    import fitz

    doc = fitz.open()

    pages_content = [
        "Chapter 1: Introduction to Machine Learning. "
        "Machine learning enables computers to learn from data. " * 50,
        "Chapter 2: Neural Networks and Deep Learning. "
        "Neural networks are inspired by biological neurons. " * 50,
        "Chapter 3: Natural Language Processing. "
        "NLP helps computers understand human language. " * 50,
    ]

    for content in pages_content:
        page = doc.new_page()
        page.insert_text((72, 72), content, fontsize=10)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
