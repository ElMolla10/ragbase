"""Versioned lexical evidence and content-free answer lifecycle records."""

from alembic import op

revision = "0004_retrieval_answers"
down_revision = "0003_revoked_jobs"
branch_labels = None
depends_on = None


def upgrade():
    statements = [
        """CREATE TABLE enterprise.retrieval_indexes (
            workspace_id uuid NOT NULL, version_id uuid NOT NULL, index_version text NOT NULL,
            chunk_count integer NOT NULL CHECK(chunk_count>=0),
            PRIMARY KEY(version_id,index_version),
            FOREIGN KEY(version_id,workspace_id) REFERENCES enterprise.document_versions(id,workspace_id))""",
        """CREATE TABLE enterprise.document_chunks (
            id uuid PRIMARY KEY, workspace_id uuid NOT NULL, version_id uuid NOT NULL,
            ordinal integer NOT NULL, part integer NOT NULL, index_version text NOT NULL,
            char_start integer NOT NULL CHECK(char_start>=0), char_end integer NOT NULL,
            content text NOT NULL CHECK(length(content) BETWEEN 1 AND 600), search_text text NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple'::regconfig,search_text)) STORED,
            CHECK(char_end-char_start=length(content)),
            FOREIGN KEY(version_id,ordinal) REFERENCES enterprise.extracted_pages(version_id,ordinal),
            FOREIGN KEY(version_id,workspace_id) REFERENCES enterprise.document_versions(id,workspace_id),
            UNIQUE(version_id,index_version,ordinal,part))""",
        "CREATE INDEX chunks_scope_idx ON enterprise.document_chunks(workspace_id,index_version,version_id)",
        "CREATE INDEX chunks_search_idx ON enterprise.document_chunks USING gin(search_vector)",
        """CREATE TABLE enterprise.answer_runs (
            id uuid PRIMARY KEY, workspace_id uuid NOT NULL REFERENCES enterprise.workspaces(id),
            created_by uuid NOT NULL REFERENCES enterprise.users(id), client_request_id uuid NOT NULL,
            question_sha256 text NOT NULL,
            status text NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
            lease_until timestamptz NOT NULL, error_code text, metadata jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE(id,workspace_id), UNIQUE(created_by,client_request_id))""",
        "CREATE INDEX answer_active_idx ON enterprise.answer_runs(status,lease_until) WHERE status IN ('queued','running')",
        """CREATE TABLE enterprise.answer_sources (
            run_id uuid NOT NULL, workspace_id uuid NOT NULL, chunk_id uuid NOT NULL REFERENCES enterprise.document_chunks(id),
            PRIMARY KEY(run_id,chunk_id),
            FOREIGN KEY(run_id,workspace_id) REFERENCES enterprise.answer_runs(id,workspace_id))""",
    ]
    for table in (
        "retrieval_indexes",
        "document_chunks",
        "answer_runs",
        "answer_sources",
    ):
        private = (
            "AND created_by=enterprise.current_actor()"
            if table == "answer_runs"
            else ""
        )
        if table == "answer_sources":
            private = "AND EXISTS (SELECT 1 FROM enterprise.answer_runs r WHERE r.id=run_id AND r.created_by=enterprise.current_actor())"
        statements += [
            f"ALTER TABLE enterprise.{table} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE enterprise.{table} FORCE ROW LEVEL SECURITY",
            f"CREATE POLICY owner_admin ON enterprise.{table} TO ragbase_owner USING(true) WITH CHECK(true)",
            f"""CREATE POLICY scoped_read ON enterprise.{table} FOR SELECT TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,false) {private})""",
        ]
        write = "false" if table in {"answer_runs", "answer_sources"} else "true"
        statements += [
            f"""CREATE POLICY scoped_write ON enterprise.{table} FOR ALL TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,{write}) {private})
                WITH CHECK(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,{write}) {private})""",
            f"GRANT SELECT,INSERT,UPDATE,DELETE ON enterprise.{table} TO ragbase_app",
        ]
    statements += [
        """CREATE FUNCTION enterprise.reap_answers() RETURNS void LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
            UPDATE enterprise.answer_runs r SET status='failed',error_code='interrupted_or_revoked',updated_at=now()
            WHERE status IN ('queued','running') AND (lease_until<clock_timestamp() OR NOT EXISTS (
                SELECT 1 FROM enterprise.memberships m JOIN enterprise.users u ON u.id=m.user_id
                WHERE m.user_id=r.created_by AND m.workspace_id=r.workspace_id AND NOT u.disabled)) $$""",
        """CREATE FUNCTION enterprise.admit_answer() RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
            BEGIN
                IF NOT enterprise.has_membership(enterprise.current_workspace(),false) THEN RETURN false; END IF;
                PERFORM pg_advisory_xact_lock(728004);
                PERFORM enterprise.reap_answers();
                RETURN (SELECT count(*)<8 FROM enterprise.answer_runs WHERE status='queued')
                  AND NOT EXISTS(SELECT 1 FROM enterprise.answer_runs WHERE created_by=enterprise.current_actor() AND status IN ('queued','running'));
            END $$""",
        """CREATE FUNCTION enterprise.answers_waiting() RETURNS boolean LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
            SELECT EXISTS(SELECT 1 FROM enterprise.answer_runs WHERE status IN ('queued','running') AND lease_until>clock_timestamp()) $$""",
        "REVOKE ALL ON FUNCTION enterprise.reap_answers(),enterprise.admit_answer(),enterprise.answers_waiting() FROM PUBLIC",
        "GRANT EXECUTE ON FUNCTION enterprise.reap_answers(),enterprise.admit_answer(),enterprise.answers_waiting() TO ragbase_app",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade():
    raise RuntimeError("Evidence lineage requires an explicit recovery plan")
