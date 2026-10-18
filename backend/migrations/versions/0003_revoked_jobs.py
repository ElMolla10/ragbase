"""Cancel inaccessible queued work without granting a worker unrestricted reads."""

from alembic import op

revision = "0003_revoked_jobs"
down_revision = "0002_ingestion_jobs"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE FUNCTION enterprise.reap_revoked_ingestion() RETURNS void
        LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
            UPDATE enterprise.ingestion_jobs j SET status='cancelled',
                error_code='authorization_revoked',claim_token=NULL,lease_until=NULL,updated_at=now()
            WHERE j.status IN ('queued','extracting') AND NOT EXISTS (
                SELECT 1 FROM enterprise.memberships m JOIN enterprise.users u ON u.id=m.user_id
                WHERE m.workspace_id=j.workspace_id AND m.user_id=j.created_by
                AND NOT u.disabled AND m.role IN ('contributor','administrator'))
        $$
    """)
    op.execute("REVOKE ALL ON FUNCTION enterprise.reap_revoked_ingestion() FROM PUBLIC")
    op.execute(
        "GRANT EXECUTE ON FUNCTION enterprise.reap_revoked_ingestion() TO ragbase_app"
    )


def downgrade():
    raise RuntimeError("Revoked jobs require an explicit recovery plan")
