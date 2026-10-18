# Enterprise foundation development

This is the first foundation slice. It is not the completed dev-lite application:
there is no document storage, SSO, enterprise database, migrations, queue, or
upload/chat/analysis route yet. Generation is fixture-only in contract tests.

## Docker startup

From the repository root:

```powershell
docker compose -f compose.enterprise.yaml up -d --build
Invoke-RestMethod http://127.0.0.1:8800/health/live
Invoke-RestMethod http://127.0.0.1:8800/health/ready
```

The initial image build needs access to approved Python/container package sources;
the API runs on an internal Docker network behind a loopback-facing reverse
proxy. The proxy also has a bridge network for host access; full host/firewall
egress enforcement and internal HTTPS remain server acceptance work. This build is not an
offline release bundle. It downloads no model weights. The foundation container
runs as a non-root user with a read-only filesystem and a 256 MB memory cap;
the development proxy has a separate 96 MB cap and only forwards the two health
routes. It uses the same minimal Python image; it is not the production HTTPS
proxy and has no general URL forwarding endpoint.

Stop only this project with:

```powershell
docker compose -f compose.enterprise.yaml down
```

The legacy `docker-compose.yml` and `backend.main:app` launch the public-demo
architecture and must not be mistaken for an enterprise deployment.

## Python checks

Use Python 3.11 in a virtual environment (or a dedicated test container):

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-enterprise-dev.txt
.venv/Scripts/python scripts/build_policy.py
.venv/Scripts/python -m pytest enterprise_tests -q
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
```

`enterprise_tests` deliberately has no dependency on `tests/conftest.py`, which
imports the legacy database/model stack. Existing integration tests still require
an isolated PostgreSQL/pgvector database and appropriate embedding artifacts.
Never point destructive test fixtures at a real application database.

After an authorized constitution/template change, review mappings and version,
then run `python scripts/build_policy.py --refresh-reviewed-hashes` explicitly.
Commit the reviewed manifest with the source change; ordinary verification must
fail for stale assets. Re-run the relevant tests and update individual ledger
entries without claiming that unimplemented controls are verified.
