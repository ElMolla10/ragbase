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

    # Legacy Groq field retained for configuration compatibility; generation uses BYOK.
    groq_api_key: str = ""
    openrouter_api_key: str = Field(default="", repr=False)

    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: Literal[384] = 384  # Must match Chunk.embedding's schema.

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    pdf_layout_extraction: bool = True

    # Demo protection; write API key is opt-in, queries stay public.
    max_upload_bytes: int = Field(default=50_000_000, gt=0)
    query_rate_limit: int = Field(default=30, ge=0)
    ingest_rate_limit: int = Field(default=5, ge=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)
    write_api_key: str = ""

    # Retrieval settings
    top_k: int = 5
    use_hybrid_search: bool = True

    # Legacy provider/base/model fields remain parseable; the shared BYOK policy
    # fixes active generation routing independently of these historical values.
    llm_provider: Literal["openrouter", "groq", "ollama"] = "openrouter"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_max_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
