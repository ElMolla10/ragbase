"""Scoped transactions and explicit checks for the enterprise serving role."""

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

HEAD_REVISION = "0007_calculations"


class EnterpriseDatabase:
    def __init__(self, url: str, *, pool_size: int = 3):
        self.engine = create_async_engine(
            url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=0,
            hide_parameters=True,
            connect_args={"timeout": 3, "command_timeout": 5},
        )

    async def check_ready(self):
        async with self.engine.connect() as conn:
            role = (
                (
                    await conn.execute(
                        text("""
                SELECT r.rolsuper, r.rolbypassrls, r.rolcreaterole, r.rolcreatedb,
                       r.rolreplication, r.rolname, pg_has_role(current_user, 'ragbase_owner', 'MEMBER') AS owner_member
                FROM pg_roles r WHERE r.rolname = current_user
            """)
                    )
                )
                .mappings()
                .one()
            )
            if any(
                role[key]
                for key in (
                    "rolsuper",
                    "rolbypassrls",
                    "rolcreaterole",
                    "rolcreatedb",
                    "rolreplication",
                    "owner_member",
                )
            ):
                raise RuntimeError("Unsafe enterprise serving role")
            owner = await conn.scalar(
                text("""
                SELECT EXISTS (SELECT 1 FROM pg_tables
                WHERE schemaname='enterprise' AND pg_has_role(current_user,tableowner,'MEMBER'))
            """)
            )
            if owner:
                raise RuntimeError("Serving role must not own enterprise tables")
            revision = await conn.scalar(
                text("SELECT version_num FROM enterprise.alembic_version")
            )
            if revision != HEAD_REVISION:
                raise RuntimeError("Enterprise migration revision is not current")

    @asynccontextmanager
    async def scope(self, user_id: UUID, workspace_id: UUID | None = None):
        async with self.engine.begin() as conn:
            await conn.execute(
                text("""
                SELECT set_config('app.user_id', :actor, true),
                       set_config('app.workspace_id', :workspace, true)
            """),
                {
                    "actor": str(user_id),
                    "workspace": str(workspace_id) if workspace_id else "",
                },
            )
            yield conn

    async def authorize(self, conn, workspace_id: UUID, *, write: bool = False):
        # A row lock keeps revocation ordered with the current storage operation.
        role = await conn.scalar(
            text("""
            SELECT role FROM enterprise.memberships
            WHERE workspace_id=:workspace AND user_id=enterprise.current_actor()
            FOR SHARE
        """),
            {"workspace": workspace_id},
        )
        allowed = await conn.scalar(
            text("SELECT enterprise.has_membership(:workspace, :write)"),
            {"workspace": workspace_id, "write": write},
        )
        if not role or not allowed:
            # No distinction between a missing workspace and an inaccessible one.
            raise HTTPException(404, "Workspace or document not available")

    async def close(self):
        await self.engine.dispose()
