# RAGBase

> Enterprise work starts at [the execution plan](docs/execution-plan.md) and
> [development instructions](docs/operations/development.md). The enterprise
> foundation is separate from the legacy demo described below. It currently
> exposes development health/status only; company-data features are not ready.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Railway](https://img.shields.io/badge/Deployment-Railway-black?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)
![pgvector](https://img.shields.io/badge/pgvector-Postgres%2016-336791?style=flat-square&logo=postgresql)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-181717?style=flat-square&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A production RAG (Retrieval-Augmented Generation) platform that lets you upload research papers and chat with them. Ask questions in **any language** — Arabic, English, or mixed — and get cited answers pulled from the actual document content, with full ML observability built in.

**Live demo:** [RAGBase on Railway](https://ragbase.up.railway.app)

---

## Quick Start with Docker

Install Docker with Compose, then run:

```bash
git clone https://github.com/ElMolla10/ragbase.git
cd ragbase
./start.sh
```

The script creates `.env` from `.env.example` only when it is missing, preserves
existing configuration, and builds/starts the services. Set `GROQ_API_KEY` in
`.env` and run `./start.sh` again to enable answers. The UI is at
http://localhost:8501, API docs at http://localhost:8000/docs, and optional
Metabase at http://localhost:3000. The first build downloads the embedding model.

---

## Features

- **Multilingual questions** — ask in Arabic, English, or mixed language and get cited answers. Retrieval uses multilingual E5-small; quality varies by language and corpus.
- **PDF ingestion** — upload any PDF, chunks are embedded and stored in pgvector
- **Hybrid search (on by default)** — combines pgvector cosine similarity and PostgreSQL full-text search using reciprocal rank fusion
- **Cited answers** — every answer references the exact source pages it used
- **ML observability** — every query is logged: latency, token count, timestamp, sources
- **Live stats** — total queries, avg latency, documents, and chunks shown in real time
- **Metabase dashboard** — visual monitoring of query patterns and system health

---

## Architecture

```text
Browser → HTML/Tailwind frontend (Railway public HTTPS URL)
                ↓ BACKEND_URL
          FastAPI (Railway) → Groq API (server-side key)
                ↓
          Postgres + pgvector (Railway)
          documents · chunks · query_logs
                ↑
          Metabase (optional)
```

PDFs are processed in memory. Filenames, metadata, extracted text, embeddings,
and query logs persist in Postgres; original PDF files are not stored.
The nullable `file_path` column remains for compatibility with existing databases.
External PDF storage is optional and is not part of this deployment.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, async SQLAlchemy |
| Embeddings | sentence-transformers (intfloat/multilingual-e5-small) |
| Vector DB | pgvector on Railway Postgres |
| LLM | Groq API — llama-3.3-70b-versatile |
| Frontend | HTML/Tailwind + Python static server |
| Observability | Metabase + query_logs table |
| Infrastructure | Railway backend + frontend + Postgres |
| Containers | Docker Compose |
| CI | GitHub Actions (ruff lint + pytest) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload PDF — returns document_id and chunk count |
| POST | `/query` | Ask a question — returns answer, sources, latency |
| GET | `/documents` | List all uploaded documents |
| GET | `/stats` | Total queries, avg latency, document count |

### Example: Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"
```

Response:
```json
{
  "document_id": "abc123",
  "filename": "paper.pdf",
  "num_chunks": 31
}
```

### Example: Query (Arabic)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي المنهجية المستخدمة في هذا البحث؟"}'
```

### Example: Query (English)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main contribution of this paper?"}'
```

Response:
```json
{
  "answer": "The paper introduces a two-phase screening tool...",
  "sources": [
    {"document": "paper.pdf", "page": 8, "score": 0.91},
    {"document": "paper.pdf", "page": 3, "score": 0.87}
  ],
  "latency_ms": 1749,
  "tokens_used": 312
}
```

---

## Local Setup

### Prerequisites

- Docker + Docker Compose
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Run locally

```bash
git clone https://github.com/ElMolla10/ragbase.git
cd ragbase

# Create environment file
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Start all services
docker compose up -d --build

# Services:
# localhost:8000  → FastAPI (API + docs at /docs)
# localhost:8501  → HTML/Tailwind UI
# localhost:3000  → Metabase dashboard
```

---

## Deploy to Railway

1. Create an **Empty Project** in [Railway](https://railway.com).
2. Add a **Postgres service with pgvector installed**, using a pgvector template
   from Railway's template marketplace. Name it `Postgres`.
   Railway's standard Postgres image does **not** include pgvector; running SQL
   alone cannot install a missing extension. Keep the database's persistent volume.
   Alternatively, use the official `pgvector/pgvector:pg16` image, mount a volume
   at `/var/lib/postgresql/data`, and set `PGDATA=/var/lib/postgresql/data/pgdata`
   so initialization does not conflict with the volume's `lost+found` directory.
   Set `POSTGRES_USER=ragbase`, `POSTGRES_DB=ragbase`, a strong
   `POSTGRES_PASSWORD`, and the service's private `DATABASE_URL` using those
   credentials and `${{Postgres.RAILWAY_PRIVATE_DOMAIN}}:5432`.
3. Connect to that database with a SQL client or `railway connect Postgres`
   (requires the Railway CLI and `psql`), then run:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   SELECT extversion FROM pg_extension WHERE extname = 'vector';
   ```

   Backend startup also enables the extension and creates `documents`, `chunks`,
   and `query_logs` via `init_db()`. The database role must have permission to
   create the extension. No separate schema setup script is required.
4. Add a GitHub service from this repository, name it `backend`, and select the
   branch containing these changes. Keep **Root Directory** at the repository root
   so Docker can copy `requirements.txt` and `backend/`.
   Set these service variables:

   | Variable | Value |
   |----------|-------|
   | `RAILWAY_DOCKERFILE_PATH` | `docker/Dockerfile.backend` |
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (adjust the service name if different) |
   | `GROQ_API_KEY` | Your real Groq API key |

   Use the private database URL. The app accepts `postgres://`, `postgresql://`,
   and `postgresql+asyncpg://` URLs and selects the asyncpg driver automatically.
   Leave the start command unset to use the Dockerfile.
   Set the deployment healthcheck path to `/health`.
   Generate a public domain under **Settings → Networking** and verify
   `https://<backend-domain>/health` returns `"status": "healthy"`.
5. Add a second GitHub service from the same repository and branch, name it
   `frontend`, and keep its Root Directory at the repository root.
   Set these service variables:

   | Variable | Value |
   |----------|-------|
   | `RAILWAY_DOCKERFILE_PATH` | `docker/Dockerfile.frontend` |
   | `BACKEND_URL` | `https://<backend-domain>` (no trailing slash) |

   Leave the start command unset. Both Dockerfiles bind to `0.0.0.0` and use
   Railway's `PORT` at runtime. Compose exposes the API on 8000 and the UI on
   8501 (the frontend container defaults to 8080).
   Set the frontend healthcheck path to `/health`.
   Generate a public domain for the frontend.
6. Open the frontend HTTPS URL, upload a PDF, ask a question, and check that
   citations and query statistics appear. Share that URL with your friend.
   **Visitors do not need a Groq API key**: only the backend has the deployer's key;
   The browser sends queries to the backend via the injected `BACKEND_URL`.
   Groq usage uses the backend key's quota.

The backend runs sentence-transformers and CPU PyTorch locally, using
`intfloat/multilingual-e5-small`; allocate enough backend memory for model loading and ingestion.
The model is downloaded during the Docker build. No GPU or embedding API key is
required. Metabase is optional: deploy it separately and connect it to the same
Postgres database if you want dashboards.

Service-specific variables select the two Dockerfiles, so no shared
`railway.toml` is needed. See Railway's
[Dockerfile configuration](https://docs.railway.com/builds/dockerfiles),
[Postgres extension guidance](https://docs.railway.com/databases/postgresql), and
[variable references](https://docs.railway.com/variables).

### Existing data and cutover

A new Railway database starts empty. To keep existing documents, chunks, and
query logs, back up the source Postgres database and restore it into the
pgvector-enabled destination using `pg_dump` / `pg_restore` before switching URLs.
Pause writes during the final copy, verify row counts and a document query, then
share the new frontend URL. Original PDFs in external storage are not migrated or
deleted by this app. Keep the source database and deployment until the new
deployment and data have been verified; switching back to the old URL remains
possible during that period.

---

## CI Pipeline

GitHub Actions runs on every push to master:

- **Lint** — ruff (format + style check)
- **Test** — pytest with a live Postgres/pgvector service container

```
✅ lint  (ruff check + format)
✅ test  (unit tests + Postgres/pgvector integration tests)
```

---

## Known Limitations & Next Steps

- **Embeddings:** the default is `intfloat/multilingual-e5-small` (384 dimensions),
  with normalized vectors and automatic `query: ` / `passage: ` prefixes for E5.
  `EMBEDDING_MODEL` remains configurable; non-E5 models receive unprefixed text.
  Model loading rejects dimensions that do not match `EMBEDDING_DIMENSION`.
  Changing dimensions requires migrating the existing pgvector column and indexes
  and updating the schema guard in config.py; other dimensions are rejected.
  **After upgrading from MiniLM, delete and re-ingest all existing documents**
  from your original PDFs before querying. Equal dimensions do not mean compatible
  embeddings. PDFs are not retained, so keep your source copies. Query logs can stay.
  For Railway, remove an old `EMBEDDING_MODEL` override or set the new model name,
  then rebuild the backend. Compose passes the setting at build and runtime.
  Custom Railway models may download on first use unless the build argument matches.
  Multilingual support improves coverage, but quality varies by language and corpus.
- **Chunking:** splitting now prefers paragraph/sentence endings, falls back to
  whitespace or hard cuts for long sentences, and keeps every chunk on its source
  page. `CHUNK_SIZE` and `CHUNK_OVERLAP` retain the legacy approximate token units
  (four characters per unit); overlap is bounded and may be smaller to avoid
  starting inside a word. This is not semantic chunking.
  Existing chunks stay unchanged; re-ingest documents to use the new splitter.
- **PDF layout:** PyMuPDF block coordinates improve common two-column reading
  order and preserve full-width headings. Detected ruled tables keep cells on
  each row separated by ` | `. Table detection errors fall back to text.
  Set `PDF_LAYOUT_EXTRACTION=false` for sorted plain-text extraction.
  Scanned PDFs still need OCR; complex layouts and merged/borderless tables may
  remain imperfect. Re-ingest existing PDFs to use improved extraction.
- **No reranking yet:** a cross-encoder reranker remains a possible next step.
- **Hybrid search is enabled by default:** set `USE_HYBRID_SEARCH=false` for
  pure cosine search. Hybrid mode combines
  pgvector and PostgreSQL full-text search using reciprocal rank fusion, with
  document filters applied to both. Returned scores are fusion scores rather
  than cosine similarities. It uses the `simple` text-search configuration,
  without Arabic stemming or language-specific normalization; it does not
  translate queries. Full-text vectors are computed at query time; a GIN index
  is a future optimization for larger collections. No schema migration is needed.

## Demo protection and evaluation

Uploads default to a 50 MB file limit (`MAX_UPLOAD_BYTES=50000000`). Oversized files
return HTTP 413; multipart Content-Length is also checked before parsing with
64 KiB allowed for form overhead. Requests without Content-Length still receive
the file check after multipart parsing; use an ingress body limit for strict
stream-level protection against arbitrary bodies.

`QUERY_RATE_LIMIT=30` and `INGEST_RATE_LIMIT=5` allow that many requests per client
in `RATE_LIMIT_WINDOW_SECONDS=60`. HTTP 429 includes `Retry-After`; set a limit to
0 to disable it. Limits are in memory per worker, reset on restart, and are not
shared across replicas. Client identity uses the ASGI client address; behind a
proxy, configure Uvicorn's trusted proxy addresses (`FORWARDED_ALLOW_IPS`) for
accurate per-client limits. Only trust known ingress proxies. Otherwise visitors
behind the same proxy may share one allowance.

The public demo needs no new variables. Optional `WRITE_API_KEY` protects uploads
and document deletion via `X-API-Key`, while queries and health remain public.
Leave it empty to keep browser uploads usable. When enabled, use an API client
with the header for writes; the frontend does not contain or inject this secret.

After changing these settings, redeploy the backend (or rebuild Compose).
No re-ingestion is needed for protection changes.

Run `ruff check .`, `ruff format --check .`, and `pytest tests/` against a
**disposable** pgvector database: test fixtures create and drop application tables.
The tiny harness `pytest tests/test_eval.py -q` loads two sample texts and checks
expected keywords in top-1 retrieved chunks for English and Arabic questions,
using both vector-only and hybrid search. It uses the real embedding model and
Postgres, without Groq calls. This is a regression smoke test, not a benchmark.

## Observability

Every query is logged to the `query_logs` table:

```sql
SELECT query, latency_ms, token_count, created_at
FROM query_logs
ORDER BY created_at DESC;
```

Connect Metabase to the same Postgres instance to build dashboards on top of this data.

---

## Project Structure

```
ragbase/
├── backend/
│   ├── main.py        # FastAPI app, 6 endpoints
│   ├── ingest.py      # PDF parsing, chunking, embedding
│   ├── retrieval.py   # pgvector similarity search
│   ├── llm.py         # Groq API call with context injection
│   ├── logger.py      # Query logging to Postgres
│   ├── models.py      # SQLAlchemy models (documents, chunks, query_logs)
│   ├── database.py    # Async Postgres + pgvector setup
│   └── config.py      # Environment variable config
├── frontend/
│   ├── index.html     # HTML/Tailwind UI
│   └── serve.py       # Static server + BACKEND_URL injection
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── init.sql       # pgvector extension + table creation
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── docker-compose.yml
└── requirements.txt
```

---

## Environment Variables

```bash
# .env.example
GROQ_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@host:5432/ragbase
BACKEND_URL=http://localhost:8000
POSTGRES_USER=ragbase
POSTGRES_PASSWORD=ragbase
POSTGRES_DB=ragbase
```

---

## Author

**Mohamed El Molla** — Final Year AI Student @ ESLSCA University (Graduating June 2026) Focus: MLOps, RAG Architecture, and Cloud Deployment - [LinkedIn](https://www.linkedin.com/in/mohamed-el-molla/) | [GitHub](https://github.com/ElMolla10)
