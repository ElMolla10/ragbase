"""Durable exhaustive-coverage runs and section records."""

from alembic import op

revision = "0006_coverage_runs"
down_revision = "0005_heavy_lease_guard"
branch_labels = None
depends_on = None


def upgrade():
    statements = [
        """CREATE TABLE enterprise.coverage_runs (
            id uuid PRIMARY KEY,
            workspace_id uuid NOT NULL REFERENCES enterprise.workspaces(id),
            created_by uuid NOT NULL REFERENCES enterprise.users(id),
            client_request_id uuid NOT NULL,
            question_sha256 text NOT NULL,
            version_ids uuid[] NOT NULL,
            status text NOT NULL CHECK(status IN ('queued','running','yielding','finished','failed','cancelled')),
            coverage_complete boolean NOT NULL DEFAULT false,
            lease_until timestamptz NOT NULL,
            error_code text,
            metadata jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CHECK(NOT coverage_complete OR status='finished'),
            UNIQUE(id,workspace_id),
            UNIQUE(created_by,client_request_id))""",
        """CREATE INDEX coverage_active_idx ON enterprise.coverage_runs(status,lease_until)
            WHERE status IN ('queued','running','yielding')""",
        """CREATE TABLE enterprise.coverage_sections (
            run_id uuid NOT NULL,
            workspace_id uuid NOT NULL,
            version_id uuid NOT NULL,
            ordinal integer NOT NULL CHECK(ordinal BETWEEN 1 AND 1000),
            part integer NOT NULL CHECK(part BETWEEN -10000 AND 10000),
            chunk_id uuid,
            status text NOT NULL CHECK(status IN ('pending','covered','gap')),
            gap_code text CHECK(gap_code IS NULL OR gap_code ~ '^[a-z0-9_]+$'),
            location jsonb NOT NULL,
            PRIMARY KEY(run_id,version_id,ordinal,part),
            FOREIGN KEY(run_id,workspace_id) REFERENCES enterprise.coverage_runs(id,workspace_id),
            FOREIGN KEY(version_id,workspace_id) REFERENCES enterprise.document_versions(id,workspace_id),
            CHECK((status='covered' AND gap_code IS NULL) OR (status='gap' AND gap_code IS NOT NULL) OR (status='pending' AND gap_code IS NULL)))""",
    ]
    for table in ("coverage_runs", "coverage_sections"):
        private = (
            "AND created_by=enterprise.current_actor()"
            if table == "coverage_runs"
            else "AND EXISTS (SELECT 1 FROM enterprise.coverage_runs r WHERE r.id=run_id AND r.created_by=enterprise.current_actor())"
        )
        statements += [
            f"ALTER TABLE enterprise.{table} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE enterprise.{table} FORCE ROW LEVEL SECURITY",
            f"CREATE POLICY owner_admin ON enterprise.{table} TO ragbase_owner USING(true) WITH CHECK(true)",
            f"""CREATE POLICY scoped_read ON enterprise.{table} FOR SELECT TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,false) {private})""",
            f"""CREATE POLICY scoped_write ON enterprise.{table} FOR ALL TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,false) {private})
                WITH CHECK(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,false) {private})""",
            f"GRANT SELECT,INSERT,UPDATE,DELETE ON enterprise.{table} TO ragbase_app",
        ]
    statements += [
        """CREATE FUNCTION enterprise.reap_coverage() RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
            BEGIN
                UPDATE enterprise.coverage_runs r
                SET status='failed',
                    error_code=CASE WHEN lease_until<clock_timestamp() THEN 'interrupted_or_revoked' ELSE 'authorization_revoked' END,
                    coverage_complete=false, updated_at=now()
                WHERE status IN ('queued','running','yielding') AND (lease_until<clock_timestamp() OR NOT EXISTS (
                    SELECT 1 FROM enterprise.memberships m JOIN enterprise.users u ON u.id=m.user_id
                    WHERE m.user_id=r.created_by AND m.workspace_id=r.workspace_id AND NOT u.disabled));
                UPDATE enterprise.coverage_sections s
                SET status='gap', gap_code=r.error_code
                FROM enterprise.coverage_runs r
                WHERE s.run_id=r.id AND s.status='pending' AND r.status='failed'
                  AND r.error_code IN ('interrupted_or_revoked','authorization_revoked');
            END $$""",
        """CREATE FUNCTION enterprise.admit_coverage() RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
            BEGIN
                IF NOT enterprise.has_membership(enterprise.current_workspace(),false) THEN RETURN false; END IF;
                PERFORM pg_advisory_xact_lock(728005);
                PERFORM enterprise.reap_coverage();
                RETURN (SELECT count(*)<2 FROM enterprise.coverage_runs WHERE status IN ('queued','running','yielding'))
                  AND NOT EXISTS(SELECT 1 FROM enterprise.coverage_runs
                    WHERE created_by=enterprise.current_actor() AND status IN ('queued','running','yielding'));
            END $$""",
        """CREATE FUNCTION enterprise.coverage_must_yield() RETURNS boolean LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
            SELECT EXISTS(SELECT 1 FROM enterprise.answer_runs WHERE status IN ('queued','running') AND lease_until>clock_timestamp())
                OR EXISTS(SELECT 1 FROM enterprise.ingestion_jobs WHERE status='extracting' AND lease_until>clock_timestamp()) $$""",
        """CREATE OR REPLACE FUNCTION enterprise.heavy_lease_active(current_run uuid) RETURNS boolean
            LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
            SELECT EXISTS(SELECT 1 FROM enterprise.ingestion_jobs WHERE status='extracting' AND lease_until>clock_timestamp())
                OR EXISTS(SELECT 1 FROM enterprise.answer_runs WHERE status='running' AND lease_until>clock_timestamp() AND id<>current_run)
                OR EXISTS(SELECT 1 FROM enterprise.coverage_runs WHERE status='running' AND lease_until>clock_timestamp() AND id<>current_run) $$""",
        "REVOKE ALL ON FUNCTION enterprise.reap_coverage(),enterprise.admit_coverage(),enterprise.coverage_must_yield(),enterprise.heavy_lease_active(uuid) FROM PUBLIC",
        "GRANT EXECUTE ON FUNCTION enterprise.reap_coverage(),enterprise.admit_coverage(),enterprise.coverage_must_yield(),enterprise.heavy_lease_active(uuid) TO ragbase_app",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade():
    raise RuntimeError("Coverage records require an explicit recovery plan")
