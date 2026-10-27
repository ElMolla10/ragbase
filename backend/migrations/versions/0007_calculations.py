"""Calculation records for authorized CSV versions. Results are not source text."""

from alembic import op

revision = "0007_calculations"
down_revision = "0006_coverage_runs"
branch_labels = None
depends_on = None


def upgrade():
    statements = [
        """CREATE TABLE enterprise.calculations (
            id uuid PRIMARY KEY,
            workspace_id uuid NOT NULL,
            created_by uuid NOT NULL REFERENCES enterprise.users(id),
            client_request_id uuid NOT NULL,
            version_id uuid NOT NULL,
            question_sha256 text CHECK (question_sha256 IS NULL OR question_sha256 ~ '^[a-f0-9]{64}$'),
            status text NOT NULL CHECK (status IN ('running','completed','clarification','failed')),
            lease_until timestamptz,
            error_code text CHECK (error_code IS NULL OR error_code ~ '^[a-z0-9_]+$'),
            dataset_sha256 text CHECK (dataset_sha256 IS NULL OR dataset_sha256 ~ '^[a-f0-9]{64}$'),
            plan jsonb NOT NULL DEFAULT '{}',
            result jsonb NOT NULL DEFAULT '{}',
            export_csv text NOT NULL DEFAULT '',
            record jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CHECK (status <> 'running' OR lease_until IS NOT NULL),
            CHECK (status <> 'completed' OR (
                dataset_sha256 IS NOT NULL AND char_length(export_csv) BETWEEN 1 AND 200000
            )),
            UNIQUE (id, workspace_id),
            UNIQUE (created_by, client_request_id),
            FOREIGN KEY (version_id, workspace_id)
                REFERENCES enterprise.document_versions(id, workspace_id))""",
        """CREATE INDEX calculations_active_idx ON enterprise.calculations(status, lease_until)
            WHERE status = 'running'""",
        "ALTER TABLE enterprise.calculations ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE enterprise.calculations FORCE ROW LEVEL SECURITY",
        "CREATE POLICY owner_admin ON enterprise.calculations TO ragbase_owner USING (true) WITH CHECK (true)",
        """CREATE POLICY scoped_read ON enterprise.calculations FOR SELECT TO ragbase_app
            USING (workspace_id = enterprise.current_workspace()
                AND enterprise.has_membership(workspace_id, false)
                AND created_by = enterprise.current_actor())""",
        """CREATE POLICY scoped_write ON enterprise.calculations FOR ALL TO ragbase_app
            USING (workspace_id = enterprise.current_workspace()
                AND enterprise.has_membership(workspace_id, false)
                AND created_by = enterprise.current_actor())
            WITH CHECK (workspace_id = enterprise.current_workspace()
                AND enterprise.has_membership(workspace_id, false)
                AND created_by = enterprise.current_actor())""",
        "GRANT SELECT, INSERT, UPDATE, DELETE ON enterprise.calculations TO ragbase_app",
        """CREATE FUNCTION enterprise.reap_calculations() RETURNS void
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
            BEGIN
                UPDATE enterprise.calculations c
                SET status = 'failed',
                    error_code = CASE
                        WHEN lease_until < clock_timestamp() THEN 'interrupted_or_revoked'
                        ELSE 'authorization_revoked' END,
                    lease_until = NULL,
                    updated_at = now()
                WHERE status = 'running' AND (
                    lease_until < clock_timestamp() OR NOT EXISTS (
                        SELECT 1 FROM enterprise.memberships m
                        JOIN enterprise.users u ON u.id = m.user_id
                        WHERE m.user_id = c.created_by AND m.workspace_id = c.workspace_id
                          AND NOT u.disabled
                    )
                );
            END $$""",
        """CREATE FUNCTION enterprise.admit_calculation() RETURNS boolean
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
            BEGIN
                IF NOT enterprise.has_membership(enterprise.current_workspace(), false) THEN
                    RETURN false;
                END IF;
                PERFORM pg_advisory_xact_lock(728006);
                PERFORM enterprise.reap_calculations();
                RETURN (
                    SELECT count(*) < 1 FROM enterprise.calculations
                    WHERE status = 'running' AND lease_until > clock_timestamp()
                ) AND NOT enterprise.heavy_lease_active('00000000-0000-0000-0000-000000000000');
            END $$""",
        """CREATE OR REPLACE FUNCTION enterprise.heavy_lease_active(current_run uuid) RETURNS boolean
            LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog AS $$
            SELECT EXISTS (
                    SELECT 1 FROM enterprise.ingestion_jobs
                    WHERE status = 'extracting' AND lease_until > clock_timestamp()
                )
                OR EXISTS (
                    SELECT 1 FROM enterprise.answer_runs
                    WHERE status = 'running' AND lease_until > clock_timestamp()
                      AND id <> current_run
                )
                OR EXISTS (
                    SELECT 1 FROM enterprise.coverage_runs
                    WHERE status = 'running' AND lease_until > clock_timestamp()
                      AND id <> current_run
                )
                OR EXISTS (
                    SELECT 1 FROM enterprise.calculations
                    WHERE status = 'running' AND lease_until > clock_timestamp()
                ) $$""",
        """CREATE OR REPLACE FUNCTION enterprise.coverage_must_yield() RETURNS boolean
            LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog AS $$
            SELECT EXISTS (
                    SELECT 1 FROM enterprise.answer_runs
                    WHERE status IN ('queued', 'running') AND lease_until > clock_timestamp()
                )
                OR EXISTS (
                    SELECT 1 FROM enterprise.ingestion_jobs
                    WHERE status = 'extracting' AND lease_until > clock_timestamp()
                )
                OR EXISTS (
                    SELECT 1 FROM enterprise.calculations
                    WHERE status = 'running' AND lease_until > clock_timestamp()
                ) $$""",
        "REVOKE ALL ON FUNCTION enterprise.reap_calculations(), enterprise.admit_calculation() FROM PUBLIC",
        "GRANT EXECUTE ON FUNCTION enterprise.reap_calculations(), enterprise.admit_calculation() TO ragbase_app",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade():
    raise RuntimeError("Calculation records require an explicit recovery plan")
