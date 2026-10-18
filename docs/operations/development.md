# Enterprise development storage

This profile is a development-only storage foundation. It accepts UTF-8 TXT/Markdown originals, retains immutable files and metadata, and checks workspace access on every upload/list/download. Extraction, generation, SSO, analysis and a graphical enterprise interface are pending. The legacy docker-compose.yml remains a separate public-demo application.

## Start

From the repository root, with Python available for credential generation:

```powershell
python scripts/init_enterprise_dev.py
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --build
Invoke-RestMethod http://127.0.0.1:8800/health/ready
```

Run initialization only once. If configuration already exists it stops without replacing secrets. The ignored .env.enterprise contains database credentials and development tokens; .local/dev-access.json contains tokens for synthetic Alice, Bob and Viewer accounts. Never commit, publish, print or reuse these secrets in production. They are development identities, not company SSO. No real company documents belong in this environment.

Startup orders database health, migration, synthetic provisioning, API and proxy. Three services remain running; migration/seed services exit successfully. Only the proxy publishes a host port, bound to loopback 8800. The API uses a non-owner, non-BYPASSRLS PostgreSQL role. The generation model is not loaded. The initial build requires package access and is not an offline release bundle.

## Synthetic smoke and persistence

```powershell
python scripts/smoke_enterprise_storage.py
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --force-recreate database api proxy
python scripts/smoke_enterprise_storage.py --verify-existing
```

Wait for readiness before the final command if the API is still starting. The first command uploads only a built-in synthetic fixture and stores its identifiers in ignored .local/storage-smoke.json. The second check verifies the exact earlier bytes and access controls after container replacement. Named database and artifact volumes persist. This is not a backup restoration test.

## Operator reconciliation

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml --profile operations run --rm reconcile
```

This optional job has separate operator credentials and read-only artifact access. It reports orphan keys, invalid/missing references and staged files; it never deletes or repairs originals. Investigate inconsistencies before any manual cleanup. This first reconciler is for the small storage slice and is not a benchmarked large-corpus audit.

## Checks

Use the disposable synthetic integration stack, which has no company data and no published database port:

```powershell
docker compose -f compose.enterprise-tests.yaml build tests
docker compose -f compose.enterprise-tests.yaml run --rm tests
docker compose -f compose.enterprise-tests.yaml run --rm tests python -m pytest enterprise_tests -p no:cacheprovider -q
docker compose -f compose.enterprise-tests.yaml down
```

Alternatively use a dedicated Python 3.11 environment with requirements-enterprise-dev.txt for unit tests, ruff check/format checks and scripts/build_policy.py. Integration tests need their isolated database and migration/serving credentials; never point them at application databases. The original tests/conftest.py belongs to the legacy stack and is not the enterprise fixture configuration.

After an explicitly reviewed constitution/template change, verify mappings and run python scripts/build_policy.py --refresh-reviewed-hashes. Ordinary verification must fail on stale assets. Keep the manifest, constitution, tests and rule ledger consistent.

## Capacity inventory

Run directly on each host, outside its containers:

```powershell
python scripts/discover_hardware.py --role laptop --output .local/laptop-hardware.json
```

On the company server, use --role server and a local output path. The script inventories selected CPU/RAM/Docker/disk capacity; memory bandwidth, ISA usability, cgroup/VM limits and inference capacity still require review and benchmarks. Laptop measurements do not qualify server capacity.

## Stop

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml down
```

Do not add --volumes unless intentionally deleting this development project's stored documents and database. Existing public-demo and unrelated containers are separate. Full offline network enforcement, internal TLS, production SSO, release locking, backup restoration and server qualification remain pending.
