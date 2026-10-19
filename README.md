# RAGBase

Upload PDFs, ask questions in English, Arabic or mixed language, and inspect the
source pages used for an answer. RAGBase combines local multilingual embeddings,
PostgreSQL/pgvector retrieval and hosted generation through OpenRouter BYOK.
Language support is a capability to evaluate on your documents, not a guarantee
of business accuracy.

**Public demo:** [ragbase.up.railway.app](https://ragbase.up.railway.app)  
**Branch:** `master` — public application  
**Updated:** 19 October 2026

For the separate access-controlled implementation and its roadmap, see the
[enterprise branch](https://github.com/ElMolla10/ragbase/tree/enterprise).
Enterprise storage, permissions and quote-validation controls are not part of
this public app.

## What the app does

- Extracts native PDF text, preserving page numbers and using layout-aware
  handling for common columns and ruled tables.
- Creates page-scoped chunks and multilingual E5 embeddings locally.
- Combines vector similarity and PostgreSQL full-text retrieval by default,
  optionally restricted to a selected document.
- Answers with retrieved context, page citations and recent conversation history.
- Streams answer text and exposes the retrieved passages and source PDFs.
- Stores documents, compressed PDF originals, extracted chunks, embeddings, and
  query/answer records in PostgreSQL. Older uploads without stored PDF bytes
  must be uploaded again to enable their PDF preview.
- Reports query counts, latency and document/chunk totals. Metabase is available
  as an optional database dashboard.

## Generation and fallback

| Role | Model ID | Provider |
|---|---|---|
| Primary | `google/gemini-3.8-flash` | Google AI Studio |
| Fallback | `openai/gpt-oss-120b` | Groq |

The backend uses the OpenAI Python SDK with
`https://openrouter.ai/api/v1`. Every completion includes the model fallback list
and provider `order`, `only` and `allow_fallbacks`. The approved providers are
`google-ai-studio` and `groq`; Mistral and other providers are excluded.
Both routes use low reasoning. Gemini requires reasoning enabled; reasoning text
is excluded from the requested response.

Fallback happens inside the same request when OpenRouter cannot serve the
primary. It is not continuation of a partially delivered reply. The SDK adds no
separate automatic retry loop. No local generation model or GPU is needed;
embedding inference still runs locally on CPU.

In OpenRouter's **same workspace as the application API key**:

1. Add valid Google AI Studio and Groq keys as prioritized BYOK provider keys.
2. Allow the corresponding model on each key if using model filters.
3. Set **Never use shared capacity for any model on this provider** for both.

The provider allowlist alone does not enforce that workspace setting. Keep the
provider accounts on the intended free plans. Free tiers have quotas, and
OpenRouter BYOK fees/allowances are separate: this is not unlimited or guaranteed
zero-cost inference. See [the BYOK setup guide](docs/openrouter-byok.md).

## Run locally with Docker

Requirements: Git, Docker with Compose, Python 3.11+ for the hidden key prompt,
and an OpenRouter workspace configured as above.

```powershell
git clone --branch master https://github.com/ElMolla10/ragbase.git
cd ragbase
Copy-Item .env.example .env
python scripts/configure_openrouter.py
docker compose up -d --build postgres backend frontend
```

On an existing checkout, keep your `.env` instead of overwriting it. On Linux or
macOS, use `cp .env.example .env` for the initial copy.

The setup script asks for the **OpenRouter application key**, not a Google or
Groq key, and saves it in ignored `.local/openrouter-api-key`. Compose mounts that
file only in the backend as `/run/secrets/openrouter_api_key`. The file is required
by this Compose configuration even if an environment key is supplied. Git and
Docker builds exclude local secrets.

The first backend build installs CPU PyTorch and downloads the embedding model;
it needs internet access and enough RAM/disk space for those dependencies. No
generation weights are downloaded. PostgreSQL data uses a persistent volume.

| Service | Local URL |
|---|---|
| Frontend | `http://localhost:8501` |
| API documentation | `http://localhost:8000/docs` |
| API health | `http://localhost:8000/health` |
| Optional Metabase | `http://localhost:3000` |

Start the optional dashboard with `docker compose up -d metabase`. Stop services
with `docker compose down`; avoid `down -v` unless you intend to delete the data
volumes. Uploaded public-demo data is shared across visitors.

## Deploy on Railway

Use three services: pgvector-enabled PostgreSQL, the backend and the frontend.
Select the `master` branch for both application services. Keep the repository
root as their build context.

### Database

Use a pgvector-enabled PostgreSQL template or `pgvector/pgvector:pg16` with a
persistent volume. Standard PostgreSQL does not gain pgvector just by running
SQL; the extension's files must already be installed.

For a manually configured image, set database credentials, mount the database
volume, and use `PGDATA=/var/lib/postgresql/data/pgdata` when the volume is mounted
at `/var/lib/postgresql/data`. Use the database's private connection URL for the
backend. Startup enables the `vector` extension and creates application tables;
the database role must have the necessary permissions.

### Backend

Create a repository service named `backend` and set its variables:

| Variable | Value |
|---|---|
| `RAILWAY_DOCKERFILE_PATH` | `docker/Dockerfile.backend` |
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}`, adapting the database service name |
| `OPENROUTER_API_KEY` | Your OpenRouter application key |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small`, or leave unset |

**Set the OpenRouter key in Railway itself.** A key saved on your laptop is not
sent by GitHub pushes, and Railway does not receive the local Compose secret.
Open the backend service's Variables, add `OPENROUTER_API_KEY`, apply the change,
and redeploy. Keep provider keys in OpenRouter BYOK settings.

Leave the start command unset so the Dockerfile handles startup and Railway's
`PORT`. Set the healthcheck path to `/health`, generate a backend domain and
check `https://YOUR_BACKEND_DOMAIN/health`. Health checks do not verify model
credentials or free-tier capacity.

### Frontend

Create another repository service named `frontend`:

| Variable | Value |
|---|---|
| `RAILWAY_DOCKERFILE_PATH` | `docker/Dockerfile.frontend` |
| `BACKEND_URL` | `https://YOUR_BACKEND_DOMAIN`, without a trailing slash |

Leave its start command unset and use `/health` for its healthcheck. Generate a
frontend domain, then upload a nonconfidential PDF and test a question.
The frontend injects `BACKEND_URL` for browser requests, so the backend URL must
be publicly reachable. Visitors do not supply API keys; generation uses the
backend owner's provider quotas. Never put `OPENROUTER_API_KEY` in frontend
variables or browser code.

Back up existing PostgreSQL data before a database cutover. Preserve documents,
PDF bytes, chunks and query logs together, and verify the destination before
retiring the old deployment. Keep source PDFs separately as well.

## Configuration

Copy [.env.example](.env.example) for local settings. On Railway, use backend
service variables rather than committing an `.env` file.

| Setting | Default / purpose |
|---|---|
| `OPENROUTER_API_KEY` | Application credential; takes precedence over a key file |
| `OPENROUTER_API_KEY_FILE` | Optional private key file; Compose sets the mounted path |
| `DATABASE_URL` | PostgreSQL connection; common postgres URL schemes are accepted |
| `BACKEND_URL` | Frontend setting pointing to the API |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small`, 384 dimensions |
| `USE_HYBRID_SEARCH` | `true`; set `false` for vector-only retrieval |
| `PDF_LAYOUT_EXTRACTION` | `true`; set `false` for sorted plain-text extraction |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` approximate units, four characters per unit |
| `MAX_UPLOAD_BYTES` | `50000000` — 50 MB file limit |
| `QUERY_RATE_LIMIT` / `INGEST_RATE_LIMIT` | `30` / `5` per client per configured window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` |
| `WRITE_API_KEY` | Optional `X-API-Key` protection for uploads and document deletion |

The generation model/provider chain is owned by
[backend/openrouter_client.py](backend/openrouter_client.py). Legacy Groq/Ollama
configuration fields remain parseable but no longer select generation routes.
Changing the generation model does not require re-embedding existing documents.
Changing the embedding model requires re-ingestion; equal dimensions do not make
different embedding spaces compatible. Other vector dimensions require a schema
change. Re-ingest to apply new chunking or extraction settings as well.

## API

Interactive schemas are available at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Application health |
| POST | `/ingest` | Upload a PDF as multipart field `file` |
| GET | `/documents` | List documents |
| GET | `/documents/{document_id}/file` | View a stored source PDF |
| DELETE | `/documents/{document_id}` | Delete a document and its chunks |
| POST | `/query` | Return an answer, sources, latency, usage and selected model |
| POST | `/query/stream` | Stream source, token, completion or error events over SSE |
| GET | `/stats` | Aggregate query/document statistics |
| DELETE | `/stats` | Clear query logs |

For example, from a shell with curl:

```bash
curl -X POST http://localhost:8000/ingest -F "file=@paper.pdf"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the main finding?","document_id":1}'
```

Use the actual integer ID returned by ingestion. Omit `document_id` to search
across documents. The optional `history` field accepts user/assistant turns.
On Windows PowerShell, use `curl.exe` for these curl examples.

## Data, privacy and limits

This is a public demo, not an enterprise authorization boundary. Queries,
document listings and source PDFs are publicly accessible. Optional write-key
protection does not add workspace isolation, SSO or private per-user collections.
Do not upload confidential company data.

PDF parsing and embeddings run in the backend. Questions, recent conversation
history and retrieved passages are sent to OpenRouter and its selected provider.
Query/answer text and retrieved chunk IDs are also stored in `query_logs` locally
in the database. Provider privacy and retention policies still apply to BYOK.

Native PDFs are supported; scans require OCR, which is not implemented here.
Complex layouts and tables can be extracted incorrectly. There is no reranker,
exhaustive document-coverage workflow or trusted spreadsheet calculation engine.
Page citations are generated from retrieved evidence; this app does not use the
enterprise branch's exact-quote validation before releasing answers.

The synchronous answer path records actual returned model/usage when available.
Streaming completion metadata currently uses the configured primary model and an
approximate answer-length token count, even if fallback occurred. Use SDK logs
for the actual selected streaming model; do not treat UI statistics as a billing
or tokenizer measurement.

Rate limits are in memory per worker, reset at restart and are not shared across
replicas. Current query throttling covers `/query`; `/query/stream` does not use
that limiter. `WRITE_API_KEY` protects upload/document deletion, not stats reset.
Proxy addresses must be trusted correctly for client-IP limits. Use an ingress
body limit for strict protection before arbitrary uploads are fully read.

## Verification and development

The BYOK port passed **15 focused tests** covering SDK payloads, provider
restrictions, sync/async and streaming paths, and generation return contracts.
Repository lint, formatting and Compose validation passed. Shared routing was
also live-tested with both providers and same-request fallback using synthetic
data; HTTP 200 does not prove free-only billing, Arabic accuracy or production
reliability. These results do not replace the full application test suite.

Run the lightweight mocked tests without model downloads:

```powershell
python -m pytest tests_byok -q
ruff check .
ruff format --check .
```

For synthetic live probes from a local Python environment with dependencies:

```powershell
$env:OPENROUTER_API_KEY_FILE = ".local/openrouter-api-key"
python scripts/test_openrouter_fallbacks.py
```

The probes cover the standard chain, Groq alone, and a Gemini-primary request
restricted to Groq to force fallback. They do not inject a real provider 429.

The full application tests need installed dependencies, the embedding model and
an isolated pgvector database. **Use a disposable database:** fixtures create and
drop application tables. Never point them at the running app's database.

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ragbase_test \
  python -m pytest tests/ tests_byok/ -v
```

GitHub Actions is configured to run lint/format checks and both test directories
against its own pgvector service on pushes/PRs to `master`. The small retrieval
regression and [evaluation corpus](eval/README.md) are development evidence, not
company acceptance benchmarks.

## Troubleshooting

| Symptom | Check |
|---|---|
| `OPENROUTER_API_KEY is not configured` on Railway | Set the variable on the **backend** service and redeploy; local files are not transferred |
| Same error locally | Run the hidden setup prompt for Compose; direct Python needs `OPENROUTER_API_KEY` or `OPENROUTER_API_KEY_FILE` |
| Provider returns `Invalid API Key` | Replace that provider's key in OpenRouter BYOK settings; it is separate from the application key |
| `429` / unavailable generation | Check provider quotas and model access; fallback cannot create more quota when both providers are exhausted |
| Frontend cannot reach API | Verify its public `BACKEND_URL`, backend health and service deployment |
| PostgreSQL reports missing `vector` | Use a pgvector-enabled database image and a role allowed to enable the extension |
| PDF preview unavailable for an older upload | Re-upload the original; earlier records may not contain PDF bytes |
| Bad or missing evidence | Inspect extraction, selected document and retrieved passages; scans/layouts and cross-language recall need separate review |

## Project map

| Path | Responsibility |
|---|---|
| `backend/main.py` | Public API and SSE responses |
| `backend/ingest.py` | Native PDF extraction, chunking and embedding |
| `backend/retrieval.py` | Vector/hybrid retrieval |
| `backend/openrouter_client.py` | Fixed BYOK route and SDK clients |
| `backend/llm.py` | Context/history assembly and generation contracts |
| `backend/models.py`, `backend/database.py` | PostgreSQL persistence |
| `backend/logger.py` | Query/answer observability |
| `frontend/` | HTML frontend and Python static server |
| `scripts/` | Credential setup, routing probes and evaluation helpers |
| `tests/`, `tests_byok/` | Application and focused adapter tests |
| `docker/`, `docker-compose.yml` | Images and local deployment |

Author: [Mohamed El Molla](https://github.com/ElMolla10) ·
[LinkedIn](https://www.linkedin.com/in/mohamed-el-molla/)
