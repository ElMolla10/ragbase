# Enterprise foundation — implementation checkpoint

Date: 18 October 2026. Governing constitution: 1.0.0.

## Delivered

- Reviewed upstream baseline and separate local enterprise/feature branches.
- Canonical constitution, execution plan, agent instructions, decision record,
  baseline inventory, backlog, and a ledger covering all 55 rules.
- Deterministic policy assembly with version/hash checks, mandatory core, task
  sections, known rule validation, and rejection of stale/malformed assets.
- Model-independent streaming request/event contracts, deterministic fake gateway,
  and HTTP adapter scaffold with separate instructions/evidence, explicit errors,
  unknown-versus-real usage, timeouts, cancellation, and capability rejection.
- Three candidate manifests with no claimed capabilities, no benchmark scores,
  and production disabled. The application alias stays `company-assistant`.
- Minimal enterprise runtime without Torch, sentence-transformers, Groq SDK,
  SQLAlchemy, model weights, or database-dependent startup.
- Isolated Compose development project: internal API plus health-only reverse
  proxy exposed at `127.0.0.1:8800`. Both run non-root with read-only filesystems
  and resource limits. Existing local RAGBase services were left untouched.
- Separate foundation CI workflow requiring neither model downloads nor databases.

## Verified

| Check | Result |
|---|---|
| Original upstream suite on isolated test database | 63 passed; one existing embedding API deprecation warning |
| New foundation tests in clean minimal image | 55 passed; one test-client deprecation warning |
| Foundation tests with networking disabled | Passed under a read-only filesystem and 256 MB memory cap |
| Ruff lint and format | Passed, 48 Python files checked for formatting |
| Policy verification | Core plus retrieval, coverage, analysis, and vision profiles validate |
| Compose configuration and clean runtime image build | Passed |
| Host loopback health/live and health/ready | HTTP 200; development/fixture status explicitly reported |
| Runtime heavyweight dependency check | Torch, sentence-transformers, Groq, SQLAlchemy absent |
| Development idle memory snapshot | API about 34 MiB; proxy about 37 MiB; not a load benchmark |

The test fixtures exercise fragmented Arabic UTF-8/SSE output, incomplete and
malformed streams, HTTP failures/redirects, timeouts, cancellation/disconnect,
unsupported tools/images, missing usage, input byte limits, instruction/evidence
separation, policy drift, profile endpoint changes, and production-profile rejection.
They do not establish that any real model follows the constitution or that a
real model's internal reasoning channels are fully qualified.

## Current boundary

Only health/status endpoints are exposed. There is no enterprise document/chat
UI, database schema/migration, durable storage, workspace authorization, SSO,
queue, parser pipeline, or calculation service yet. A1 is in progress; A2 onward
is not implemented. The old demo remains legacy code with its documented gaps.

The HTTP adapter is a development text-streaming scaffold, not a production
gateway. Actual tokenizer input budgeting, model credentials/configuration,
nonstreaming normalization, capability negotiation, structured tools/vision,
resource hardening, and live candidate compatibility still need implementation
and validation. Its byte ceiling is not a model context-token check.

All real Arabic, 64K/128K context, GPU memory, throughput, OCR on representative
internal data, actual model swap, production egress/TLS, offline cold-start, and
restore acceptance remain pending. No company documents were processed and no
candidate model weights were downloaded for this batch.

The development image build uses package registries. It is not the production
offline artifact bundle, and the proxy bridge is not a production firewall.
An attempted additional proxy-image pull failed certificate validation; no TLS
checks were bypassed. The development proxy instead uses the already validated
minimal Python runtime and only forwards fixed health paths.

## Next work

Finish the A1 configuration/injection and migration foundation, then implement
A2 workspace identity, durable storage, and actual serving-role RLS checks before
adding document ingestion or other data routes. Track all deferred server checks
in the ledger/backlog. This checkpoint has not been pushed or deployed remotely.
