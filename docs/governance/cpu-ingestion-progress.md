# CPU-only implementation progress — durable native extraction

Updated: 18 October 2026. Constitution 2.0.0; execution plan 2.0. This supersedes the storage-only status report for current functionality.

## Result

The local development stack now stores ingestion jobs in PostgreSQL in the same transaction as uploaded document metadata, then processes native PDF/TXT/Markdown text in a separate, bounded worker. Extracted pages retain workspace, immutable document/version, original SHA-256, extractor version and page/line locations. Original files and earlier uploads remain unchanged.

This completes the durable-job/native-extraction part of P2. Retrieval, embeddings, cited model answers and the graphical enterprise interface are still pending. The API explicitly reports retrieval_ready=false. A successful native extraction does not establish complete visual, table, reading-order or Arabic PDF accuracy.

## Implemented behavior

- New uploads accept UTF-8 TXT/Markdown and PDF-header originals, up to 5,000,000 bytes. Full PDF validation occurs in the worker. Encrypted, malformed and over-limit PDFs fail with stable codes and no partial published pages; encrypted PDFs require an unlocked copy.
- PostgreSQL retains queued/extracting/extracted/needs_review/failed/cancelled job states. The separate original-storage state remains stored. Migration 0002 adds jobs/pages and scoped RLS; migration 0003 cancels inaccessible pending work without exposing its contents.
- One PostgreSQL session advisory lock admits one extraction worker globally. A 90-second claim lease and unique claim token prevent stale or cancelled workers from publishing. Interrupted claims are recovered after lease expiry, with at most three attempts. Finished pages and terminal status publish atomically. Admission permits at most eight outstanding ingestion jobs and one queued job per requester; excess uploads fail before creating files.
- Workers use the ordinary non-owner serving account with explicit server-configured synthetic actor scopes. They recheck write permission before reading originals and before publishing. They receive neither migration credentials nor development bearer tokens. A narrow definer maintenance function cancels work whose requester is disabled, revoked or no longer allowed to write; it returns no document/job data. Production worker identity/provisioning remains pending.
- An isolated Python child runs native extraction with no database credentials in its environment. Linux limits are 384 MiB address space, 20 CPU seconds, 30-second wall deadline and 8 MiB output file limit. PDF limits are 100 pages, 8,000,000 decoded content-stream bytes per page, 100,000 characters per PDF page and 1,000,000 total extracted characters. Excess text is rejected rather than silently truncated. The worker container has a 768 MiB RAM/no-swap limit, one CPU quota, restricted process count, read-only filesystem and read-only original-file volume.
- Empty native-text pages become needs_review with an OCR/visual-review warning. Every PDF result warns that native text does not validate layout, tables, images or OCR. Scanned PDFs are not represented as fully understood. Existing OCR text layers are not independently verified.
- Permission-checked endpoints expose extraction status/page text and cancel active work. Cancellation immediately prevents publication; an already-running bounded parser may continue until its process ends or reaches its 30-second deadline. No immediate child-kill guarantee is claimed.
- A process endpoint idempotently queues pre-migration originals that have no job. Repeated requests return the existing job without resetting completed evidence. Ordinary failed/cancelled jobs are not silently retried through this endpoint; operator handling is still needed.

## Verification

- Full current-source run: 132 tests passed, comprising 92 unit/contract/parser tests and 40 PostgreSQL integration tests using the actual serving role. Lint, formatting and constitution-derived policy checks passed.
- Rebuilt packaged test image independently passed the same 132 tests. One existing TestClient dependency deprecation warning remains.
- Tests cover Arabic UTF-8 and instruction-like text preservation, two-page native PDF locations, encrypted/blank/malformed PDFs, page limits, compressed-stream expansion, parser deadlines, queue admission, atomic rollback, file integrity, expired-lease recovery, retry exhaustion, cancelled/stale publication, revocation before read/publication, cross-workspace RLS and idempotent processing requests.
- Live smoke: stopped the worker, uploaded a synthetic two-page PDF through the loopback proxy, observed queued state, recreated API/worker/proxy, then verified the same saved job resumed with page 1 value 120 and page 2 value 180, matching original hash, authorized viewer access and cross-workspace denial. The smoke initially encountered the brief startup gap; its safe GET polling now tolerates startup availability failures without retrying uploads.
- The earlier storage smoke still reads its exact original after both migrations and service recreation. Container replacement and queue restart checks are not backup restoration tests.
- An idle snapshot after extraction measured API 57.78 MiB, worker 52.36 MiB, proxy 42.68 MiB and database 33.1 MiB. This is neither peak parser memory nor a company-server performance benchmark.

All fixtures are synthetic. No company files, model weights, inference requests, server deployment or GitHub publication were used. Existing public-demo and unrelated services were left unchanged.

## Run and inspect

The standard enterprise Compose command now starts four steady services: database, API, proxy and worker. Migrations and synthetic seed jobs exit after success. The only published enterprise port remains 127.0.0.1:8800. Use docs/operations/development.md for startup and checks.

For a synthetic queue/restart check, stop the worker, run scripts/smoke_enterprise_ingestion.py --queue-only, recreate/start the API/worker/proxy and run the script with --verify-existing. Identifiers are saved only under ignored .local/; bearer tokens are not printed.

## Remaining work

Next: versioned chunks, authorized retrieval, source preview/citation validation, and fixture-backed answer integration through the existing replaceable gateway. Then qualify a small real-model configuration within measured resources. Hybrid retrieval, embeddings/reranking, Arabic PDF gold tests, OCR, Office files, calculations, whole-document coverage, image reasoning, company SSO, production scheduling, comprehensive offline enforcement, backup restoration and server qualification remain pending. No complete P2 or production-readiness claim is made.

The Linux container is required for parser resource enforcement; native Windows parser execution is not qualified. The child process has internal-network access through its worker, so this is not a separate per-document network namespace. Full host/network isolation remains a release gate. The current small-queue audit and scheduling paths have not been benchmarked on a large corpus.

## Primary implementation references

[pypdf extraction guidance](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) documents native extraction and its limitations, including memory expansion and the distinction from OCR. [PostgreSQL locking](https://www.postgresql.org/docs/16/explicit-locking.html) documents transaction/session advisory locks. Artifact extraction and model quality remain separate qualification tasks.
