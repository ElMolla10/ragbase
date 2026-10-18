"""Use separate operator credentials for a read-only full storage audit."""

import asyncio
import json
import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.storage.local import LocalArtifactStore, StoredArtifact
from backend.storage.reconcile import reconcile


async def main():
    url = os.environ.get("ENTERPRISE_MIGRATION_DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "Operator migration credentials are required; never run with scoped serving credentials"
        )
    root = Path(os.environ["ENTERPRISE_STORAGE_ROOT"])
    if not root.is_dir():
        raise RuntimeError("Existing artifact root is required")
    engine = create_async_engine(url, hide_parameters=True)
    try:
        async with engine.connect() as conn:
            if await conn.scalar(text("SELECT current_user")) != "ragbase_owner":
                raise RuntimeError("Full audit requires the operator role")
            records = (
                (
                    await conn.execute(
                        text(
                            "SELECT workspace_id,storage_key,sha256,byte_size FROM enterprise.document_versions"
                        )
                    )
                )
                .mappings()
                .all()
            )
        store = LocalArtifactStore(root, max_bytes=50_000_000)
        references = [
            (
                row["workspace_id"],
                StoredArtifact(row["storage_key"], row["sha256"], row["byte_size"]),
            )
            for row in records
        ]
        # Keys are opaque generated identifiers; document content is never printed.
        print(
            json.dumps(await asyncio.to_thread(reconcile, store, references), indent=2)
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
