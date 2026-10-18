"""Idempotent synthetic workspace provisioning using migration credentials only."""

import asyncio
import os
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main():
    if os.environ.get("RAGBASE_PROFILE") != "dev-lite":
        raise RuntimeError("Synthetic provisioning is development-only")
    url = os.environ.get("ENTERPRISE_MIGRATION_DATABASE_URL", "")
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("Migration credentials required")
    engine = create_async_engine(url, hide_parameters=True)
    try:
        async with engine.begin() as conn:
            for id_, subject in [
                ("11111111-1111-4111-8111-111111111111", "synthetic:alice"),
                ("22222222-2222-4222-8222-222222222222", "synthetic:bob"),
                ("33333333-3333-4333-8333-333333333333", "synthetic:viewer"),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO enterprise.users(id,external_subject) VALUES(:id,:subject) ON CONFLICT DO NOTHING"
                    ),
                    {"id": UUID(id_), "subject": subject},
                )
            for id_, name in [
                ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "Synthetic workspace A"),
                ("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "Synthetic workspace B"),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO enterprise.workspaces(id,name) VALUES(:id,:name) ON CONFLICT DO NOTHING"
                    ),
                    {"id": UUID(id_), "name": name},
                )
            for workspace, user, role in [
                (
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "11111111-1111-4111-8111-111111111111",
                    "administrator",
                ),
                (
                    "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                    "22222222-2222-4222-8222-222222222222",
                    "contributor",
                ),
                (
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "33333333-3333-4333-8333-333333333333",
                    "viewer",
                ),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO enterprise.memberships(workspace_id,user_id,role) VALUES(:workspace,:user,:role) ON CONFLICT DO NOTHING"
                    ),
                    {"workspace": UUID(workspace), "user": UUID(user), "role": role},
                )
    finally:
        await engine.dispose()
    print("Synthetic development identities and workspaces provisioned.")


if __name__ == "__main__":
    asyncio.run(main())
