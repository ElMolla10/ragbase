"""Explicit enterprise-only migrations; never run by API startup."""

import asyncio
import os

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine


def configure(connection):
    # The version table needs a namespace before the first revision can run.
    connection.execute(text("CREATE SCHEMA IF NOT EXISTS enterprise"))
    connection.commit()
    context.configure(connection=connection, version_table_schema="enterprise")
    with context.begin_transaction():
        context.run_migrations()


async def run_online():
    url = os.environ.get("ENTERPRISE_MIGRATION_DATABASE_URL", "")
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("A separate enterprise migration URL is required")
    engine = create_async_engine(url, poolclass=pool.NullPool, hide_parameters=True)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(configure)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    raise RuntimeError(
        "Offline SQL rendering is not enabled for this bootstrap migration"
    )
else:
    asyncio.run(run_online())
