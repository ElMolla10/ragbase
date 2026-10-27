"""Serving-role tests exercise API authorization, RLS and durable originals."""

import os
import subprocess
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from backend.enterprise_db import EnterpriseDatabase
from backend.storage.local import StoredArtifact
from backend.storage.reconcile import reconcile
from enterprise_integration_tests.conftest import headers, upload


async def test_authorized_upload_listing_and_original_download(world):
    content = "مصادر الشركة التجريبية".encode()
    response = await upload(world, content=content)
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["status"] == "stored"
    assert record["extraction"] == "queued"
    listing = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/documents", headers=headers(world)
    )
    assert listing.json()["documents"][0]["version_id"] == record["version_id"]
    file = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file",
        headers=headers(world),
    )
    assert file.status_code == 200
    assert file.content == content
    assert file.headers["cache-control"] == "no-store"
    assert file.headers["x-content-type-options"] == "nosniff"


async def test_no_auth_or_forged_identity_header_cannot_read(world):
    path = "/api/workspaces"
    for extra in (
        {},
        {"X-User-ID": str(world["ids"]["alice"])},
        {"Authorization": "Bearer forged"},
    ):
        assert (await world["client"].get(path, headers=extra)).status_code == 401


async def test_cross_workspace_ids_cannot_list_write_or_download(world):
    record = (await upload(world)).json()
    client = world["client"]
    target = world["ids"]["a"]
    for path in [
        f"/api/workspaces/{target}/documents",
        f"/api/workspaces/{target}/versions/{record['version_id']}/file",
        f"/api/workspaces/{world['ids']['b']}/versions/{record['version_id']}/file",
    ]:
        assert (
            await client.get(path, headers=headers(world, "bob"))
        ).status_code == 404
    assert (await upload(world, user="bob", workspace="a")).status_code == 404
    listed = (await client.get("/api/workspaces", headers=headers(world, "bob"))).json()
    assert [row["id"] for row in listed["workspaces"]] == [str(world["ids"]["b"])]


async def test_viewer_can_read_but_cannot_store(world):
    record = (await upload(world)).json()
    file = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file",
        headers=headers(world, "viewer"),
    )
    assert file.status_code == 200
    assert (await upload(world, user="viewer")).status_code == 404


async def test_revocation_takes_effect_for_same_token_and_prior_file(world):
    record = (await upload(world)).json()
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:actor"),
            {"actor": world["ids"]["alice"]},
        )
    path = f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file"
    assert (await world["client"].get(path, headers=headers(world))).status_code == 404
    assert (
        await world["client"].get("/api/workspaces", headers=headers(world))
    ).json()["workspaces"] == []


async def test_revocation_waits_for_active_authorized_operation(world):
    database = world["app"].state.database
    async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        await database.authorize(conn, world["ids"]["a"])
        with pytest.raises(DBAPIError):
            async with world["owner"].begin() as operator:
                await operator.execute(text("SET LOCAL lock_timeout='100ms'"))
                await operator.execute(
                    text("DELETE FROM enterprise.memberships WHERE user_id=:actor"),
                    {"actor": world["ids"]["alice"]},
                )
    async with world["owner"].begin() as operator:
        await operator.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:actor"),
            {"actor": world["ids"]["alice"]},
        )
    assert (await upload(world)).status_code == 404


async def test_failed_database_commit_leaves_reportable_orphan(world, monkeypatch):
    database = world["app"].state.database
    scope = database.scope

    @asynccontextmanager
    async def fail_before_commit(*args, **kwargs):
        async with scope(*args, **kwargs) as conn:
            yield conn
            raise RuntimeError("synthetic commit failure")

    monkeypatch.setattr(database, "scope", fail_before_commit)
    with pytest.raises(RuntimeError, match="synthetic commit failure"):
        await upload(world)
    async with scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        assert await conn.scalar(text("SELECT count(*) FROM enterprise.documents")) == 0
    report = reconcile(world["app"].state.storage, [])
    assert len(report["orphan_keys"]) == 1
    assert (world["root"] / report["orphan_keys"][0]).is_file()


async def test_missing_original_is_reported_without_deleting_metadata(world):
    record = (await upload(world)).json()
    path = next(world["root"].glob("*/*/*.bin"))
    key = path.relative_to(world["root"]).as_posix()
    path.unlink()
    artifact = StoredArtifact(key, record["sha256"], record["byte_size"])
    report = reconcile(world["app"].state.storage, [(world["ids"]["a"], artifact)])
    assert report["invalid_references"] == [key]
    listing = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/documents", headers=headers(world)
    )
    assert listing.json()["documents"][0]["version_id"] == record["version_id"]
    response = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file",
        headers=headers(world),
    )
    assert response.status_code == 503


async def test_disabled_identity_cannot_list_or_download(world):
    record = (await upload(world)).json()
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("UPDATE enterprise.users SET disabled=true WHERE id=:actor"),
            {"actor": world["ids"]["alice"]},
        )
    assert (
        await world["client"].get("/api/workspaces", headers=headers(world))
    ).status_code == 401
    path = f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file"
    assert (await world["client"].get(path, headers=headers(world))).status_code == 404


async def test_rls_filters_even_without_application_where_clause(world):
    first = await upload(world)
    second = await upload(world, user="bob", workspace="b")
    assert first.status_code == second.status_code == 201
    database = world["app"].state.database
    async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        rows = (
            (await conn.execute(text("SELECT workspace_id FROM enterprise.documents")))
            .scalars()
            .all()
        )
        assert rows == [world["ids"]["a"]]
    async with database.scope(world["ids"]["alice"], world["ids"]["b"]) as conn:
        assert await conn.scalar(text("SELECT count(*) FROM enterprise.documents")) == 0


async def test_pool_identity_clears_after_commit_and_rollback(world):
    assert (await upload(world)).status_code == 201
    database = EnterpriseDatabase(os.environ["ENTERPRISE_TEST_APP_URL"], pool_size=1)
    try:
        async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
            assert (
                await conn.scalar(text("SELECT count(*) FROM enterprise.documents"))
                == 1
            )
        with pytest.raises(RuntimeError, match="abort"):
            async with database.scope(world["ids"]["alice"], world["ids"]["a"]):
                raise RuntimeError("abort")
        async with database.engine.connect() as conn:
            assert await conn.scalar(text("SELECT enterprise.current_actor()")) is None
            assert (
                await conn.scalar(text("SELECT enterprise.current_workspace()")) is None
            )
            assert (
                await conn.scalar(text("SELECT count(*) FROM enterprise.documents"))
                == 0
            )
        async with database.scope(world["ids"]["bob"], world["ids"]["b"]) as conn:
            assert (
                await conn.scalar(text("SELECT count(*) FROM enterprise.documents"))
                == 0
            )
    finally:
        await database.close()


async def test_app_cannot_grant_itself_membership_or_update_immutable_versions(world):
    assert (await upload(world)).status_code == 201
    database = world["app"].state.database
    for sql in [
        "UPDATE enterprise.memberships SET role='administrator'",
        "UPDATE enterprise.memberships SET workspace_id=workspace_id",
        "UPDATE enterprise.document_versions SET status='stored'",
        "TRUNCATE enterprise.documents CASCADE",
        "CREATE TABLE enterprise.unauthorized(id int)",
    ]:
        with pytest.raises(DBAPIError):
            async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
                await conn.execute(text(sql))


async def test_cross_workspace_insert_and_forged_creator_are_rejected_by_rls(world):
    database = world["app"].state.database
    for workspace, creator in [
        (world["ids"]["b"], world["ids"]["alice"]),
        (world["ids"]["a"], world["ids"]["bob"]),
    ]:
        with pytest.raises(DBAPIError):
            async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
                await conn.execute(
                    text(
                        "INSERT INTO enterprise.documents(id,workspace_id,filename,created_by) VALUES(:id,:workspace,'fake.txt',:creator)"
                    ),
                    {"id": uuid4(), "workspace": workspace, "creator": creator},
                )


async def test_unsafe_migration_role_cannot_be_used_as_application_role(world):
    database = EnterpriseDatabase(os.environ["ENTERPRISE_MIGRATION_DATABASE_URL"])
    try:
        with pytest.raises(RuntimeError, match="Unsafe"):
            await database.check_ready()
    finally:
        await database.close()


async def test_every_data_table_has_forced_rls_and_nonowner_serving_role(world):
    database = world["app"].state.database
    await database.check_ready()
    async with database.scope(world["ids"]["alice"], world["ids"]["a"]) as conn:
        assert await conn.scalar(text("SELECT current_user")) == "ragbase_app"
        rows = (
            await conn.execute(
                text(
                    "SELECT relrowsecurity,relforcerowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='enterprise' AND c.relname IN ('users','workspaces','memberships','documents','document_versions','ingestion_jobs','extracted_pages','retrieval_indexes','document_chunks','answer_runs','answer_sources','coverage_runs','coverage_sections')"
                )
            )
        ).all()
        assert len(rows) == 13 and all(enabled and forced for enabled, forced in rows)


@pytest.mark.parametrize(
    "filename",
    [
        "../secret.txt",
        "%2e%2e%2fsecret.txt",
        "evil%0d%0aheader.txt",
        "file.xlsm",
        "encrypted.pdf",
    ],
)
async def test_unsafe_or_unimplemented_inputs_are_rejected(world, filename):
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename=filename),
        content=b"synthetic",
    )
    assert response.status_code in {400, 415}
    assert not list(world["root"].glob("*/*/*.bin"))


async def test_oversized_and_empty_body_leave_no_metadata_or_artifacts(world):
    for content in [b"", b"a" * (world["app"].state.settings.max_upload_bytes + 1)]:
        response = await upload(world, content=content)
        assert response.status_code in {400, 413}
    assert not list(world["root"].glob("*/*/*.bin"))
    async with world["app"].state.database.scope(
        world["ids"]["alice"], world["ids"]["a"]
    ) as conn:
        assert await conn.scalar(text("SELECT count(*) FROM enterprise.documents")) == 0


async def test_download_detects_corruption_and_hides_storage_path(world):
    record = (await upload(world)).json()
    path = next(world["root"].glob("*/*/*.bin"))
    path.write_bytes(b"corrupt")
    response = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file",
        headers=headers(world),
    )
    assert response.status_code == 503
    assert str(world["root"]) not in response.text
    assert "corrupt" not in response.text


async def test_arabic_filename_survives_as_safe_attachment(world):
    filename = "تقرير المبيعات.txt"
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename=quote(filename)),
        content="بيانات".encode(),
    )
    assert response.status_code == 201
    record = response.json()
    assert record["filename"] == filename
    file = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/versions/{record['version_id']}/file",
        headers=headers(world),
    )
    assert quote(filename) in file.headers["content-disposition"]


def test_reapplying_migration_head_is_safe():
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
