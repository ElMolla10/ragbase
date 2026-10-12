"""Configuration management using Pydantic settings."""

from functools import lru_cache

from pydantic import field_validator
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
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval settings
    top_k: int = 5
    use_hybrid_search: bool = False

    # LLM settings
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
