"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://ragbase:ragbase@localhost:5432/ragbase"

    @field_validator("database_url")
    @classmethod
    def use_asyncpg(cls, value: str) -> str:
        """Accept provider URLs while keeping the async SQLAlchemy driver."""
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+asyncpg://" + value[len(prefix) :]
        return value

    # Groq API
    groq_api_key: str = ""

    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: Literal[384] = 384  # Must match Chunk.embedding's schema.

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    pdf_layout_extraction: bool = True

    # Demo protection; write API key is opt-in, queries stay public.
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    query_rate_limit: int = Field(default=30, ge=0)
    ingest_rate_limit: int = Field(default=5, ge=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)
    write_api_key: str = ""

    # Retrieval settings
    top_k: int = 3
    use_hybrid_search: bool = True

    # LLM settings
    # gpt-oss-20b is ~2x faster on Groq than 120b; quality stays solid for RAG.
    llm_model: str = "openai/gpt-oss-20b"
    llm_max_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
