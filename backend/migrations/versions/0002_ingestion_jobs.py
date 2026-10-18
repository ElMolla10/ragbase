"""Durable, scoped ingestion jobs and native page evidence."""

from alembic import op

revision = "0002_ingestion_jobs"
down_revision = "0001_workspace_storage"
branch_labels = None
depends_on = None


def upgrade():
    statements = [
        "ALTER TABLE enterprise.document_versions ADD UNIQUE(id,workspace_id)",
        """CREATE TABLE enterprise.ingestion_jobs (
            id uuid PRIMARY KEY, workspace_id uuid NOT NULL, version_id uuid NOT NULL UNIQUE,
            created_by uuid NOT NULL REFERENCES enterprise.users(id),
            status text NOT NULL DEFAULT 'queued'
                CHECK(status IN ('queued','extracting','extracted','needs_review','failed','cancelled')),
            attempts integer NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
            claim_token uuid, lease_until timestamptz, error_code text,
            warnings jsonb NOT NULL DEFAULT '[]', extractor text,
            created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY(version_id,workspace_id) REFERENCES enterprise.document_versions(id,workspace_id))""",
        "CREATE INDEX ingestion_poll_idx ON enterprise.ingestion_jobs(workspace_id,created_by,status,created_at)",
        """CREATE TABLE enterprise.extracted_pages (
            workspace_id uuid NOT NULL, version_id uuid NOT NULL,
            ordinal integer NOT NULL CHECK(ordinal BETWEEN 1 AND 100),
            content text NOT NULL CHECK(length(content)<=1000000),
            location jsonb NOT NULL, warnings jsonb NOT NULL DEFAULT '[]',
            PRIMARY KEY(version_id,ordinal),
            FOREIGN KEY(version_id,workspace_id) REFERENCES enterprise.document_versions(id,workspace_id))""",
        "CREATE INDEX pages_workspace_idx ON enterprise.extracted_pages(workspace_id,version_id)",
    ]
    for table in ("ingestion_jobs", "extracted_pages"):
        statements += [
            f"ALTER TABLE enterprise.{table} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE enterprise.{table} FORCE ROW LEVEL SECURITY",
            f"CREATE POLICY owner_admin ON enterprise.{table} TO ragbase_owner USING(true) WITH CHECK(true)",
            f"""CREATE POLICY scoped_read ON enterprise.{table} FOR SELECT TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,false))""",
            f"""CREATE POLICY scoped_update ON enterprise.{table} FOR UPDATE TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,true))
                WITH CHECK(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,true))""",
            f"""CREATE POLICY scoped_insert ON enterprise.{table} FOR INSERT TO ragbase_app
                WITH CHECK(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,true)
                {"AND created_by=enterprise.current_actor()" if table == "ingestion_jobs" else ""})""",
            f"""CREATE POLICY scoped_delete ON enterprise.{table} FOR DELETE TO ragbase_app
                USING(workspace_id=enterprise.current_workspace() AND enterprise.has_membership(workspace_id,true))""",
        ]
    statements += [
        "GRANT SELECT,INSERT ON enterprise.ingestion_jobs TO ragbase_app",
        "GRANT UPDATE(status,claim_token,lease_until,error_code,warnings,extractor,updated_at) ON enterprise.ingestion_jobs TO ragbase_app",
        "GRANT SELECT,INSERT,DELETE ON enterprise.extracted_pages TO ragbase_app",
        # The function only returns an admission decision, not other users' data.
        # Keep its transaction lock until metadata and job have committed together.
        """CREATE FUNCTION enterprise.admit_ingestion() RETURNS boolean
            LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
            BEGIN
                IF NOT enterprise.has_membership(enterprise.current_workspace(),true) THEN RETURN false; END IF;
                PERFORM pg_advisory_xact_lock(728002);
                RETURN (SELECT count(*)<8 FROM enterprise.ingestion_jobs WHERE status IN ('queued','extracting'))
                    AND NOT EXISTS (SELECT 1 FROM enterprise.ingestion_jobs
                        WHERE created_by=enterprise.current_actor() AND status='queued');
            END $$""",
        # A server-configured worker runs under a specific actor/workspace.
        # This bounded definer operation performs global admission, but only
        # returns a job belonging to that actor in the authorized workspace.
        """CREATE FUNCTION enterprise.claim_ingestion(token uuid) RETURNS jsonb
            LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
            DECLARE selected enterprise.ingestion_jobs; result jsonb;
            BEGIN
                IF token IS NULL OR NOT enterprise.has_membership(enterprise.current_workspace(),true) THEN RETURN NULL; END IF;
                PERFORM pg_advisory_xact_lock(728002);
                IF EXISTS (SELECT 1 FROM enterprise.ingestion_jobs
                    WHERE status='extracting' AND lease_until>clock_timestamp()) THEN RETURN NULL; END IF;
                UPDATE enterprise.ingestion_jobs SET status='failed',error_code='retry_limit',
                    claim_token=NULL,lease_until=NULL,updated_at=now()
                    WHERE workspace_id=enterprise.current_workspace() AND created_by=enterprise.current_actor()
                    AND status='extracting' AND lease_until<=clock_timestamp() AND attempts>=3;
                SELECT * INTO selected FROM enterprise.ingestion_jobs
                    WHERE workspace_id=enterprise.current_workspace() AND created_by=enterprise.current_actor()
                    AND (status='queued' OR (status='extracting' AND lease_until<=clock_timestamp()))
                    AND attempts<3 ORDER BY created_at,id LIMIT 1 FOR UPDATE SKIP LOCKED;
                IF NOT FOUND THEN RETURN NULL; END IF;
                UPDATE enterprise.ingestion_jobs SET status='extracting',attempts=attempts+1,
                    claim_token=token,lease_until=clock_timestamp()+interval '90 seconds',
                    error_code=NULL,updated_at=now() WHERE id=selected.id;
                SELECT jsonb_build_object('id',selected.id,'workspace_id',selected.workspace_id,
                    'version_id',selected.version_id,'storage_key',v.storage_key,'sha256',v.sha256,
                    'byte_size',v.byte_size,'filename',d.filename) INTO result
                    FROM enterprise.document_versions v JOIN enterprise.documents d ON d.id=v.document_id
                    WHERE v.id=selected.version_id AND v.workspace_id=selected.workspace_id;
                RETURN result;
            END $$""",
        "REVOKE ALL ON FUNCTION enterprise.admit_ingestion() FROM PUBLIC",
        "REVOKE ALL ON FUNCTION enterprise.claim_ingestion(uuid) FROM PUBLIC",
        "GRANT EXECUTE ON FUNCTION enterprise.admit_ingestion(),enterprise.claim_ingestion(uuid) TO ragbase_app",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade():
    raise RuntimeError("Preserve ingestion provenance; use an explicit recovery plan")
