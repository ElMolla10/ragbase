"""Tests for database URLs supplied by deployment providers."""

import pytest

from backend.config import Settings


@pytest.mark.parametrize(
    "scheme", ["postgres://", "postgresql://", "postgresql+asyncpg://"]
)
def test_database_url_uses_asyncpg_without_changing_connection_details(
    scheme: str,
) -> None:
    suffix = (
        "user:p%40ss@postgres.railway.internal:5432/ragbase?application_name=ragbase"
    )
    settings = Settings(_env_file=None, database_url=scheme + suffix)

    assert settings.database_url == "postgresql+asyncpg://" + suffix
