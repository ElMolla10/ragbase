# Baseline and dependency inventory

> Historical foundation record. The CPU-only execution plan v2.0 and constitution v2.0.0 supersede earlier hardware, model, context and startup assumptions. Existing source WIP must be reviewed and tested before these older startup instructions are reused.

Date: 18 October 2026.

## Source and environment

- Repository: `https://github.com/ElMolla10/ragbase`.
- Reviewed and current upstream base: `2f0a0204a14f6659b97a061168ccb0bdd8c061f7`.
- Separate checkout created under this chat's `work/ragbase` directory.
- Local `enterprise-local` base branch created; feature work is isolated locally.
- Host Python: 3.14.7. Validation used Python 3.11.17 Linux containers to match
  the existing application's supported deployment baseline.
- Docker engine available. Existing RAGBase services occupy ports 8000, 8501,
  and 5432; they were neither reused nor altered.
- Production server/GPU access remains unavailable; no GPU measurements made.

## Existing behavior and risks

The demo uses FastAPI, async SQLAlchemy/asyncpg, PostgreSQL/pgvector, PyMuPDF,
sentence-transformers, a Groq SDK or Ollama HTTP path, and an HTML frontend.
`requirements.txt` also includes Streamlit and broad unbounded dependency ranges.
The production API currently imports the embedding framework and warms it on
startup; database schema changes use `create_all` plus an ad hoc alteration.

The frontend loads Tailwind, Marked, and Google Fonts from public services and
renders generated Markdown/filenames through HTML paths requiring sanitization.
The rate limiter covers `/query` but not `/query/stream`; administration and
document delivery lack enterprise workspace authorization. Limits are per process.
Generation logs include question prefixes. There is no enterprise SSO, RLS,
durable job/outbox framework, normalized Office datasets, or constitutional policy
enforcement in those legacy paths.

These findings are implementation backlog items, not newly introduced regressions.
The new foundation does not expose legacy routes and is not a company-data app.

## Baseline checks before enterprise changes

- Ruff lint: passed.
- Ruff formatting: passed, 29 Python files.
- Original test suite: **63 passed**, one existing embedding API deprecation
  warning, in 12.17 seconds of pytest execution.
- Tests used a separate temporary PostgreSQL/pgvector container on a private
  internal network with no host database port. Test credentials were development
  fixtures only. The existing local database was not touched.
- A pre-existing local test image supplied Python/test dependencies and cached
  embedding artifacts; Hugging Face offline flags were set and the test network
  had no public egress. This is not a clean offline-release validation.

## Dependency strategy

Keep legacy requirements/deployment unchanged initially. The foundation has its
own small direct requirements: FastAPI, Uvicorn, HTTPX, plus pytest/asyncio/Ruff
for development. Versions were observed and then installed/tested in a fresh
foundation image. No Torch, sentence-transformers, Groq SDK, or SQLAlchemy is
installed in the foundation runtime image.

The base image is Python 3.11.17 slim; its build resolved an immutable digest,
but a production manifest and fully locked, hashed transitive dependencies are
still required before release. Parser/OCR, database/migration, and worker groups
will be introduced separately as those work packages are implemented.

## Initial decisions

See `docs/decisions/0001-enterprise-foundation.md`. No model weights were downloaded
for this batch. All three candidate manifests remain production-disabled with
null benchmark results. The runtime protocol scaffold has only text streaming;
full token counting, capability qualification, nonstreaming/tools/vision, and
actual local-model swapping remain open work.
