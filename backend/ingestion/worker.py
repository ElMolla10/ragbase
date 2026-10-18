"""One bounded extraction at a time, using explicit development identities.

No migration credentials or unrestricted workspace discovery. Production worker
identity/provisioning remains a separate release gate.
"""

import asyncio
import json
import os
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import text

from backend.enterprise_config import EnterpriseSettings
from backend.enterprise_db import EnterpriseDatabase
from backend.ingestion.extract import extract
from backend.storage.local import ArtifactError, LocalArtifactStore, StoredArtifact


async def run_once(database, store, actors):
    async with database.engine.connect() as lock:
        acquired = await lock.scalar(text("SELECT pg_try_advisory_lock(728003)"))
        await lock.commit()
        if not acquired:
            return False
        try:
            # Narrow global maintenance returns no source or job data. Revoked
            # jobs must not consume queue capacity forever or be impersonated.
            await lock.execute(text("SELECT enterprise.reap_revoked_ingestion()"))
            await lock.commit()
            for actor in actors:
                async with database.scope(actor) as conn:
                    workspaces = (
                        (
                            await conn.execute(
                                text("SELECT id FROM enterprise.workspaces ORDER BY id")
                            )
                        )
                        .scalars()
                        .all()
                    )
                for workspace in workspaces:
                    token = uuid4()
                    try:
                        async with database.scope(actor, workspace) as conn:
                            await database.authorize(conn, workspace, write=True)
                            job = await conn.scalar(
                                text("SELECT enterprise.claim_ingestion(:token)"),
                                {"token": token},
                            )
                        if not job:
                            continue
                        job = json.loads(job) if isinstance(job, str) else job
                        result = await process_claim(
                            database, store, actor, workspace, job
                        )
                        await publish(
                            database, actor, workspace, UUID(job["id"]), token, result
                        )
                        return True
                    except HTTPException:
                        # Revoked/disabled identities cannot read or publish.
                        # The next maintenance cycle cancels their pending work.
                        continue
            return False
        finally:
            await lock.execute(text("SELECT pg_advisory_unlock(728003)"))
            await lock.commit()


async def process_claim(database, store, actor, workspace, job):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace, write=True)
        artifact = StoredArtifact(job["storage_key"], job["sha256"], job["byte_size"])
        try:
            body = await asyncio.to_thread(store.read, workspace, artifact)
        except (ArtifactError, OSError):
            return {"error_code": "artifact_unavailable"}
    return await asyncio.to_thread(extract, body, job["filename"])


async def publish(database, actor, workspace, job_id, token, result):
    async with database.scope(actor, workspace) as conn:
        await database.authorize(conn, workspace, write=True)
        job = (
            (
                await conn.execute(
                    text("""
            SELECT version_id FROM enterprise.ingestion_jobs
            WHERE id=:id AND claim_token=:token AND status='extracting'
              AND lease_until>clock_timestamp() FOR UPDATE
        """),
                    {"id": job_id, "token": token},
                )
            )
            .mappings()
            .first()
        )
        if not job:
            return False
        error = result.get("error_code")
        if not error:
            await conn.execute(
                text(
                    "DELETE FROM enterprise.extracted_pages WHERE version_id=:version"
                ),
                {"version": job["version_id"]},
            )
            for page in result["pages"]:
                await conn.execute(
                    text("""
                    INSERT INTO enterprise.extracted_pages(workspace_id,version_id,ordinal,content,location,warnings)
                    VALUES(:workspace,:version,:ordinal,:content,CAST(:location AS jsonb),CAST(:warnings AS jsonb))
                """),
                    {
                        "workspace": workspace,
                        "version": job["version_id"],
                        "ordinal": page["ordinal"],
                        "content": page["content"],
                        "location": json.dumps(page["location"]),
                        "warnings": json.dumps(page["warnings"]),
                    },
                )
        await conn.execute(
            text("""
            UPDATE enterprise.ingestion_jobs SET status=:status,error_code=:error,
                warnings=CAST(:warnings AS jsonb),extractor=:extractor,
                claim_token=NULL,lease_until=NULL,updated_at=now() WHERE id=:id
        """),
            {
                "id": job_id,
                "status": "failed" if error else result["status"],
                "error": error,
                "warnings": json.dumps(result.get("warnings", [])),
                "extractor": result.get("extractor"),
            },
        )
        return True


async def main():
    settings = EnterpriseSettings.from_env()
    actors = [
        UUID(value)
        for value in os.environ.get("ENTERPRISE_WORKER_ACTORS", "").split(",")
        if value
    ]
    if not settings.database_url or not actors:
        raise RuntimeError(
            "Development worker requires explicit database and actor scopes"
        )
    database = EnterpriseDatabase(settings.database_url)
    store = LocalArtifactStore(
        settings.storage_root, max_bytes=settings.max_upload_bytes
    )
    try:
        await database.check_ready()
        while True:
            try:
                await run_once(database, store, actors)
                actors = actors[1:] + actors[:1]
            except Exception:
                # Do not log parser text, secrets, paths, SQL parameters or raw exceptions.
                print(
                    "Development ingestion cycle failed; durable jobs retained",
                    flush=True,
                )
            await asyncio.sleep(1)
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
