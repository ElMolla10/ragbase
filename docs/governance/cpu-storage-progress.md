# CPU-only implementation progress — storage milestone

Updated: 18 October 2026. Constitution 2.0.0; execution plan 2.0.

Execution resumed on the project owner's instruction. This is a local development milestone, not a production release or completion of P0–P6.

## Implemented and running

The enterprise development API now supports authenticated workspace listing, TXT/Markdown UTF-8 original uploads, metadata listing and permission-checked downloads. Originals use immutable generated keys, SHA-256 integrity checks and a durable Docker volume. PostgreSQL metadata uses explicit Alembic migrations, a separate owner account and a non-owner/non-BYPASSRLS serving account. Forced row-level security scopes documents by user membership and transaction-local workspace identity.

Synthetic development identities have separate workspaces and read/write roles. Tokens stay in ignored local configuration and are never supplied by document content. Production SSO is still pending. Storage failures return generic errors; originals are downloaded as attachments with no-store/nosniff headers. No extraction, retrieval, chat or numerical analysis is exposed by this enterprise slice.

Fixed an actual integration defect: SELECT FOR SHARE also needed UPDATE row visibility. A restricted membership policy now permits locking the user's own row while WITH CHECK(false) rejects every attempted membership update. The serving role cannot change roles, grant membership, mutate immutable versions, create tables or truncate data. A concurrent revocation test verifies an authorized operation holds its membership lock until its transaction ends, after which revocation takes effect.

Added a read-only operator reconciliation command that reports orphaned files, missing/corrupt references and staged files without deleting data. Tests exercise an injected database rollback after file publication and preserve the resulting orphan for review.

## Measured evidence

- 79 enterprise unit/contract tests and 24 real PostgreSQL integration tests: **103 passed**. These ran together against read-only mounted current sources on an internal Docker network.
- The rebuilt packaged test image independently passed all **24 integration tests** against the actual serving role.
- Ruff lint, formatting and constitution-derived instruction verification passed; policy tests now require constitution 2.0.0 and the current CPU shortlist. One dependency deprecation warning remains in the development TestClient stack.
- A synthetic upload through http://127.0.0.1:8800 passed authorized reader, viewer, cross-workspace, anonymous and write-permission checks.
- Recreated the API, proxy and PostgreSQL containers, retaining their named volumes. The same prior upload remained readable and hash-correct with the same access restrictions. This tests container-replacement durability, **not backup restoration**.
- The running operator reconciliation reported one referenced original, zero orphans, zero invalid references/paths and zero staged files.
- One idle resource snapshot after recreation: API 57.39 MiB, proxy 40.11 MiB, database 23.53 MiB. These are development observations, not load peaks or server sizing evidence.

## Hardware discovery

Read-only laptop inventory: Intel i7-1165G7, 4 physical cores / 8 logical processors, Windows 11 Pro, about 15.71 GiB visible RAM. Free host memory was about 2.21 GiB at collection time. Docker reported about 7.61 GiB available to its Linux VM. No company-server measurements are available.

Added scripts/discover_hardware.py, runnable on Windows or Linux without model dependencies. It collects selected capacity information and avoids hostnames, usernames, serial numbers, network addresses and environment variables. It does not measure memory bandwidth or inference speed. Run it on the actual server before choosing CPU thread counts or model/context budgets.

Gemma 4 12B remains a registered comparison candidate. No model weights were downloaded, no model inference was initiated, and existing unrelated model sessions/services were left alone. The laptop's observed free RAM did not justify loading Gemma alongside the running applications.

## Running development services

Three steady services: PostgreSQL, API and development proxy. Migration and synthetic seed containers exit after success; reconciliation is an optional operator command. The database and API have no host-published ports. The development proxy is available only on 127.0.0.1:8800. Existing public-demo and unrelated containers were not modified.

See docs/operations/development.md for startup, checks and recovery commands. The current API is a storage foundation and has no graphical enterprise interface yet.

## Remaining gates and next work

P0 remains open for company-server hardware, operational requirements and reviewer-provided evaluation data. P1 storage/authorization development checks pass, but company SSO, deployment-wide offline enforcement, backup/restore, production provisioning and wider product authorization remain pending. No complete constitution rule group is declared production-verified.

Next implementation package is P2: durable PostgreSQL jobs/admission, bounded native PDF extraction with page lineage, authorized retrieval, and evidence-backed answers through a replaceable local gateway. Use synthetic fixtures and fixture inference while developing; qualify a real model only under a measured resource budget. P3–P6 formats, Arabic OCR, calculations, complete-document workflows, optional vision, model comparisons and pilot acceptance remain pending.

All changes remain local until reviewed for publication. No server deployment or GitHub publication was performed.
