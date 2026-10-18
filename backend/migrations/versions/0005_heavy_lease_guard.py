"""Fence admission after a heavy worker loses its database session lock."""

from alembic import op

revision = "0005_heavy_lease_guard"
down_revision = "0004_retrieval_answers"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE FUNCTION enterprise.heavy_lease_active(current_run uuid) RETURNS boolean
        LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
          SELECT EXISTS(SELECT 1 FROM enterprise.ingestion_jobs WHERE status='extracting' AND lease_until>clock_timestamp())
            OR EXISTS(SELECT 1 FROM enterprise.answer_runs WHERE status='running' AND lease_until>clock_timestamp() AND id<>current_run)
        $$
    """)
    op.execute("REVOKE ALL ON FUNCTION enterprise.heavy_lease_active(uuid) FROM PUBLIC")
    op.execute(
        "GRANT EXECUTE ON FUNCTION enterprise.heavy_lease_active(uuid) TO ragbase_app"
    )


def downgrade():
    raise RuntimeError("Heavy admission requires an explicit recovery plan")
