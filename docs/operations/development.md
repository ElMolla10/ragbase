# Enterprise development storage

This profile is a development-only storage foundation. It accepts UTF-8 TXT/Markdown, PDF, and CSV originals, retains immutable files and metadata, and checks workspace access on uploads, status/evidence and downloads. A separate worker performs bounded native extraction. Versioned lexical retrieval, cited answers, exhaustive coverage, and allowlisted CSV calculations are implemented. Generation is disabled by default. The same origin serves a development browser with an Arabic/English toggle. Company SSO, OCR, charts, joins, and XLSX formulas remain pending. The legacy docker-compose.yml remains a separate public-demo application.

## Start

For hosted generation without local weights, follow the separate
[OpenRouter setup](../openrouter.md). It requires an explicit remote override and
a privately entered key; the base profile below keeps generation disabled.

From the repository root, with Python available for credential generation:

```powershell
python scripts/init_enterprise_dev.py
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --build
Invoke-RestMethod http://127.0.0.1:8800/health/ready
```

The same origin serves the development browser. Open `http://127.0.0.1:8800`
and paste a synthetic bearer token from `.local/dev-access.json`. The shell is
built into the enterprise API image. Document routes stay authenticated.

Run initialization only once. If configuration already exists it stops without replacing secrets. The ignored .env.enterprise contains database credentials and development tokens; .local/dev-access.json contains tokens for synthetic Alice, Bob and Viewer accounts. Never commit, publish, print or reuse these secrets in production. They are development identities, not company SSO. No real company documents belong in this environment.

Startup orders database health, migration, synthetic provisioning, API and proxy. Four services remain running: database, API, proxy and native-extraction worker; migration/seed services exit successfully. Only the proxy publishes a host port, bound to loopback 8800. The API uses a non-owner, non-BYPASSRLS PostgreSQL role. The generation model is not loaded. The initial build requires package access and is not an offline release bundle.

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

## Durable native extraction

New uploads automatically enqueue extraction in the metadata transaction. GET /api/workspaces/{workspace}/versions/{version}/extraction returns status, original hash, extractor and page/line locations. retrieval_ready=true means that an extracted source has its lexical index; it does not qualify answer or Arabic quality. Blank native pages require OCR/visual review. Encrypted PDFs are stored for audit but fail extraction with an explanation; provide an unlocked original. PDF text, tables, images, layout and Arabic quality still need representative review.

POST /api/workspaces/{workspace}/versions/{version}/process idempotently queues an older original with no job. It returns any existing job without resetting it. POST .../cancel prevents an active job from publishing. An already-running child may continue for its bounded deadline (up to 30 seconds). Production worker provisioning and general operator retry handling are pending.

The worker uses explicit synthetic actor scopes and the ordinary serving role. It reads original files through a read-only mount and publishes results only after rechecking permission. Pending jobs whose requester loses write permission are cancelled by narrow database maintenance; no administrator identity is impersonated. Generation is disabled unless explicitly configured; the optional local-model bridge is separate from extraction.

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml stop worker
python scripts/smoke_enterprise_ingestion.py --queue-only
docker compose --env-file .env.enterprise -f compose.enterprise.yaml up -d --force-recreate api worker proxy
python scripts/smoke_enterprise_ingestion.py --verify-existing
```

This persists a synthetic two-page PDF job while the worker is stopped, then verifies the same saved job after recreation. Safe status polling tolerates the startup gap; uploads are not automatically retried. The Linux worker enforces parser CPU/RAM/time/output limits. It has no public egress route on the internal Compose network; full offline and per-document network isolation are not yet qualified.

## Phase 3 lookup and local-model bridge

The normal Compose profile does not load a model. Search and source previews work independently. The optional laptop-only override enables the installed pinned Qwen checkpoint, with no pull endpoint:

```powershell
docker compose --env-file .env.enterprise -f compose.enterprise.yaml -f compose.enterprise-ollama.yaml up -d --build
Invoke-RestMethod http://127.0.0.1:8800/health/generation
python scripts/smoke_enterprise_answers.py --output .local/phase3-live-evidence.json
```

The relay is a fifth steady service, has no document/database access, and reaches only host.docker.internal:11434. It has an edge network for this development path; it is not a production offline boundary. It requires the preexisting local model with the exact configured digest. Missing or changed models fail closed. Do not change the checkpoint without adapter/context and quality checks. No candidate has been production-qualified.

POST /api/workspaces/{workspace}/search accepts question and version_ids. POST .../answers accepts the same fields plus an optional UUID client_request_id for duplicate protection. Unknown fields, including endpoint overrides, are rejected. The answer response is NDJSON: queued, progress, then a validated answer and done, or error. Text is released only after schema, quote and citation validation. Preserve queued run_id for POST .../answers/{run_id}/cancel. Cancellation stops publication and closes the upstream transport before releasing admission; engine-level cancellation still needs runtime qualification.

GET /api/workspaces/{workspace}/sources/{source_id} returns an authenticated plain-text preview. Source identity/hash/character offsets are returned with quotes. Search rank is a retrieval ordering signal, not confidence. A no-hit response means insufficient evidence in the selected scope, not corpus-wide absence. Dense retrieval, full cross-language retrieval and exhaustive reviews are later acceptance work.

```powershell
$version = (Get-Content .local/ingestion-smoke.json -Raw | ConvertFrom-Json).version_id
python scripts/query_enterprise.py --version $version --question "What were Quarter 2 sales?" --preview
```

Core /health/ready remains separate from /health/generation. Raw questions, prompts and answers are not stored in answer-run metadata; hashes, source dependencies and resource/provenance metadata are retained. There is no answer-history replay yet. The 4K setting and synthetic smoke do not validate a full 4K evidence load, larger contexts or target-server capacity. See docs/governance/phase3-completion.md for measured limits and remaining gates.

## Interactive enterprise test

After starting the services, run `python scripts/try_enterprise.py` from the
repository folder. It prompts for a supported file path, uploads it as the local
Alice development identity, waits up to 120 seconds for durable processing, then
accepts questions and shows citation previews. A timed-out client does not cancel
its durable job. `--version UUID` reuses a ready version without uploading again;
`--file PATH --question TEXT` runs one question. No credentials are displayed.
Use nonconfidential files only with hosted generation enabled. The same origin
also serves the development browser. Prefix a question with `/calculate` for one
ready CSV version. The browser displays the server record and does not calculate.
