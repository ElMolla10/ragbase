"""Initial enterprise identities, workspace membership and immutable originals."""

from alembic import op

revision = "0001_workspace_storage"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    statements = [
        "REVOKE ALL ON SCHEMA enterprise FROM PUBLIC",
        "GRANT USAGE ON SCHEMA enterprise TO ragbase_app",
        """CREATE TABLE enterprise.users (
            id uuid PRIMARY KEY, external_subject text NOT NULL UNIQUE,
            disabled boolean NOT NULL DEFAULT false)""",
        """CREATE TABLE enterprise.workspaces (
            id uuid PRIMARY KEY, name text NOT NULL CHECK(length(name) BETWEEN 1 AND 200))""",
        """CREATE TABLE enterprise.memberships (
            workspace_id uuid REFERENCES enterprise.workspaces(id) ON DELETE CASCADE,
            user_id uuid REFERENCES enterprise.users(id) ON DELETE CASCADE,
            role text NOT NULL CHECK(role IN ('viewer','contributor','administrator')),
            PRIMARY KEY (workspace_id,user_id))""",
        "CREATE INDEX memberships_user_idx ON enterprise.memberships(user_id,workspace_id)",
        """CREATE TABLE enterprise.documents (
            id uuid PRIMARY KEY,
            workspace_id uuid NOT NULL REFERENCES enterprise.workspaces(id),
            filename text NOT NULL CHECK(length(filename) BETWEEN 1 AND 240),
            created_by uuid NOT NULL REFERENCES enterprise.users(id),
            created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(id,workspace_id))""",
        "CREATE INDEX documents_workspace_idx ON enterprise.documents(workspace_id,created_at,id)",
        """CREATE TABLE enterprise.document_versions (
            id uuid PRIMARY KEY, workspace_id uuid NOT NULL, document_id uuid NOT NULL,
            version_number integer NOT NULL CHECK(version_number>0),
            storage_key text NOT NULL UNIQUE,
            sha256 text NOT NULL CHECK(sha256 ~ '^[a-f0-9]{64}$'),
            byte_size bigint NOT NULL CHECK(byte_size>0),
            status text NOT NULL DEFAULT 'stored' CHECK(status='stored'),
            CHECK(storage_key=workspace_id::text || '/' || id::text || '/' || sha256 || '.bin'),
            created_at timestamptz NOT NULL DEFAULT now(),
            FOREIGN KEY(document_id,workspace_id) REFERENCES enterprise.documents(id,workspace_id),
            UNIQUE(document_id,version_number))""",
        "CREATE INDEX versions_workspace_idx ON enterprise.document_versions(workspace_id,document_id)",
        """CREATE FUNCTION enterprise.current_actor() RETURNS uuid
            LANGUAGE sql STABLE SET search_path = pg_catalog AS $$
            SELECT nullif(current_setting('app.user_id',true),'')::uuid $$""",
        """CREATE FUNCTION enterprise.current_workspace() RETURNS uuid
            LANGUAGE sql STABLE SET search_path = pg_catalog AS $$
            SELECT nullif(current_setting('app.workspace_id',true),'')::uuid $$""",
    ]
    for statement in statements:
        op.execute(statement)
    for table in (
        "users",
        "workspaces",
        "memberships",
        "documents",
        "document_versions",
    ):
        op.execute(f"ALTER TABLE enterprise.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE enterprise.{table} FORCE ROW LEVEL SECURITY")
        # The migration/provisioning account is explicitly privileged, never served.
        op.execute(
            f"CREATE POLICY owner_admin ON enterprise.{table} TO ragbase_owner USING(true) WITH CHECK(true)"
        )
    statements = [
        """CREATE FUNCTION enterprise.has_membership(target uuid, needs_write boolean)
            RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER
            SET search_path = pg_catalog AS $$
            SELECT EXISTS (
                SELECT 1 FROM enterprise.memberships m JOIN enterprise.users u ON u.id=m.user_id
                WHERE m.workspace_id=target AND m.user_id=enterprise.current_actor()
                AND NOT u.disabled AND (NOT needs_write OR m.role IN ('contributor','administrator'))
            ) $$""",
        "REVOKE ALL ON FUNCTION enterprise.has_membership(uuid,boolean) FROM PUBLIC",
        "REVOKE ALL ON FUNCTION enterprise.current_actor() FROM PUBLIC",
        "REVOKE ALL ON FUNCTION enterprise.current_workspace() FROM PUBLIC",
        "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA enterprise TO ragbase_app",
        """CREATE POLICY own_identity ON enterprise.users FOR SELECT TO ragbase_app
            USING(id=enterprise.current_actor() AND NOT disabled)""",
        """CREATE POLICY own_memberships ON enterprise.memberships FOR SELECT TO ragbase_app
            USING(user_id=enterprise.current_actor())""",
        """CREATE POLICY member_workspaces ON enterprise.workspaces FOR SELECT TO ragbase_app
            USING(enterprise.has_membership(id,false))""",
        "GRANT SELECT ON enterprise.users,enterprise.memberships,enterprise.workspaces TO ragbase_app",
        # FOR SHARE authorization lock requires some UPDATE privilege; restrict
        # it to the immutable workspace_id column and deny UPDATE with no policy.
        "GRANT UPDATE(workspace_id) ON enterprise.memberships TO ragbase_app",
        # SELECT FOR SHARE applies UPDATE visibility as well as SELECT policy.
        # Permit locking one's own membership, but reject every new row version.
        """CREATE POLICY lock_own_membership ON enterprise.memberships FOR UPDATE TO ragbase_app
            USING(user_id=enterprise.current_actor()) WITH CHECK(false)""",
        "GRANT SELECT,INSERT ON enterprise.documents,enterprise.document_versions TO ragbase_app",
        "GRANT SELECT ON enterprise.alembic_version TO ragbase_app",
    ]
    for table in ("documents", "document_versions"):
        actor_check = (
            "AND created_by=enterprise.current_actor()" if table == "documents" else ""
        )
        statements += [
            f"""CREATE POLICY scoped_read ON enterprise.{table} FOR SELECT TO ragbase_app
                USING(workspace_id=enterprise.current_workspace()
                AND enterprise.has_membership(workspace_id,false))""",
            f"""CREATE POLICY scoped_insert ON enterprise.{table} FOR INSERT TO ragbase_app
                WITH CHECK(workspace_id=enterprise.current_workspace()
                AND enterprise.has_membership(workspace_id,true) {actor_check})""",
        ]
    for statement in statements:
        op.execute(statement)


def downgrade():
    raise RuntimeError(
        "Initial enterprise schema rollback requires an explicit restore/recovery plan"
    )
