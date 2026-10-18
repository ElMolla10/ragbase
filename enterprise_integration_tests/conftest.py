"""Synthetic test identities and actual non-owner PostgreSQL serving sessions."""

import os
import secrets
from uuid import uuid4

import httpx
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.enterprise import create_app
from backend.enterprise_config import EnterpriseSettings


@pytest_asyncio.fixture
async def world(tmp_path):
    owner = create_async_engine(
        os.environ["ENTERPRISE_MIGRATION_DATABASE_URL"], hide_parameters=True
    )
    ids = {key: uuid4() for key in ("alice", "bob", "viewer", "a", "b")}
    tokens = {key: secrets.token_urlsafe(32) for key in ("alice", "bob", "viewer")}
    async with owner.begin() as conn:
        for name in ("alice", "bob", "viewer"):
            await conn.execute(
                text(
                    "INSERT INTO enterprise.users(id,external_subject) VALUES(:id,:subject)"
                ),
                {"id": ids[name], "subject": "synthetic:" + str(ids[name])},
            )
        for name in ("a", "b"):
            await conn.execute(
                text("INSERT INTO enterprise.workspaces(id,name) VALUES(:id,:name)"),
                {"id": ids[name], "name": "Synthetic " + name},
            )
        for user, workspace, role in [
            ("alice", "a", "administrator"),
            ("bob", "b", "contributor"),
            ("viewer", "a", "viewer"),
        ]:
            await conn.execute(
                text(
                    "INSERT INTO enterprise.memberships(workspace_id,user_id,role) VALUES(:workspace,:user,:role)"
                ),
                {"workspace": ids[workspace], "user": ids[user], "role": role},
            )
    settings = EnterpriseSettings(
        database_url=os.environ["ENTERPRISE_TEST_APP_URL"],
        storage_root=tmp_path / "artifacts",
        dev_tokens={tokens[key]: ids[key] for key in tokens},
        max_upload_bytes=10000,
        generation_mode="fixture",
    )
    app = create_app(settings)
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            yield {
                "ids": ids,
                "tokens": tokens,
                "owner": owner,
                "app": app,
                "client": client,
                "root": settings.storage_root,
            }
    finally:
        async with owner.begin() as conn:
            for table in (
                "answer_sources",
                "answer_runs",
                "document_chunks",
                "retrieval_indexes",
                "extracted_pages",
                "ingestion_jobs",
                "document_versions",
                "documents",
                "memberships",
            ):
                await conn.execute(
                    text(
                        f"DELETE FROM enterprise.{table} WHERE workspace_id IN (:a,:b)"
                    ),
                    {"a": ids["a"], "b": ids["b"]},
                )
            await conn.execute(
                text("DELETE FROM enterprise.workspaces WHERE id IN (:a,:b)"),
                {"a": ids["a"], "b": ids["b"]},
            )
            await conn.execute(
                text("DELETE FROM enterprise.users WHERE id IN (:a,:b,:v)"),
                {"a": ids["alice"], "b": ids["bob"], "v": ids["viewer"]},
            )
        await owner.dispose()


def headers(world, user="alice", filename="synthetic.txt"):
    return {"Authorization": "Bearer " + world["tokens"][user], "X-Filename": filename}


async def upload(world, user="alice", workspace="a", content=b"synthetic evidence"):
    return await world["client"].post(
        f"/api/workspaces/{world['ids'][workspace]}/documents",
        headers=headers(world, user),
        content=content,
    )
