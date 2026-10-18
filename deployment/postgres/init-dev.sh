#!/bin/sh
# Development database bootstrap. PostgreSQL passes psql variables as SQL literals.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=owner_password="$DEV_OWNER_PASSWORD" --set=app_password="$DEV_APP_PASSWORD" <<'SQL'
CREATE ROLE ragbase_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD :'owner_password';
CREATE ROLE ragbase_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD :'app_password';
ALTER DATABASE ragbase_enterprise OWNER TO ragbase_owner;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
