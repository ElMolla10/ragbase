# RAGBase

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20RDS%20%2B%20S3-orange?style=flat-square&logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)
![pgvector](https://img.shields.io/badge/pgvector-Postgres%2016-336791?style=flat-square&logo=postgresql)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-181717?style=flat-square&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A production RAG (Retrieval-Augmented Generation) platform that lets you upload research papers and chat with them. Ask questions in **any language** — Arabic, English, or mixed — and get cited answers pulled from the actual document content, with full ML observability built in.

**Live demo:** `http://3.75.213.162:8501`

---

## Features

- **Cross-lingual retrieval** — ask in Arabic, get answers in Arabic. The embedding model (all-MiniLM-L6-v2) maps Arabic and English into the same 384-dimensional latent space. No translation layer. No Arabic-specific index. Cosine similarity handles the rest.
- **PDF ingestion** — upload any PDF, chunks are embedded and stored in pgvector
- **Semantic search** — queries use cosine similarity to retrieve the most relevant chunks
- **Cited answers** — every answer references the exact source pages it used
- **ML observability** — every query is logged: latency, token count, timestamp, sources
- **Live stats** — total queries, avg latency, documents, and chunks shown in real time
- **Metabase dashboard** — visual monitoring of query patterns and system health

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User                             │
│                   (Streamlit UI)                        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                       │
│   /ingest  /query  /documents  /stats  /health          │
└──────┬─────────────────────────────┬────────────────────┘
       │                             │
┌──────▼──────────┐       ┌──────────▼──────────┐
│   AWS RDS       │       │    Groq API          │
│   Postgres 16   │       │  llama-3.3-70b       │
│   + pgvector    │       │  (LLM inference)     │
│                 │       └─────────────────────┘
│  - documents    │
│  - chunks       │       ┌─────────────────────┐
│  - query_logs   │       │    AWS S3            │
└─────────────────┘       │  (PDF storage)       │
                          └─────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   Metabase Dashboard                    │
│         Query monitoring, latency tracking              │
└─────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, async SQLAlchemy |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | pgvector on AWS RDS Postgres 16 |
| LLM | Groq API — llama-3.3-70b-versatile |
| Frontend | Streamlit |
| Observability | Metabase + query_logs table |
| Infrastructure | AWS EC2 (t4g.small) + RDS + S3 |
| Containers | Docker Compose |
| CI | GitHub Actions (ruff lint + pytest) |

---

## Radical Pragmatism: The t4g.small Story

The entire stack runs on a **t4g.small instance — 2GB RAM, $5/month**.

During deployment, the PyTorch installation inside the Docker build consumed every byte of available memory and froze the OS. The standard move would have been to upgrade to a t3.medium ($40/month). Instead:

1. Force rebooted the instance to clear the deadlock
2. Connected via EC2 Instance Connect (browser terminal) immediately after reboot while the system was quiet
3. Created a 4GB swap file on the SSD:

```bash
sudo dd if=/dev/zero of=/swapfile bs=128M count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

4. Made it persistent via `/etc/fstab` so it survives reboots

Result: 2GB RAM + 4GB SSD-backed swap = 6GB effective memory. The build finished. The swap is still active.

```
Mem:   1.8Gi   used: 1.4Gi
Swap:  4.0Gi   used: 314Mi
```

A $5/month server doing the work of a $40/month server. Sometimes the right answer is infrastructure engineering, not a bigger budget.

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
curl -X POST http://3.75.213.162:8000/ingest \
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
curl -X POST http://3.75.213.162:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي المنهجية المستخدمة في هذا البحث؟"}'
```

### Example: Query (English)

```bash
curl -X POST http://3.75.213.162:8000/query \
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
docker-compose up -d

# Services:
# localhost:8000  → FastAPI (API + docs at /docs)
# localhost:8501  → Streamlit UI
# localhost:3000  → Metabase dashboard
```

---

## AWS Deployment

This project is deployed on AWS free tier. Here's the full setup:

### Infrastructure

| Resource | Service | Config |
|----------|---------|--------|
| Compute | EC2 t4g.small | Amazon Linux 2023, arm64 |
| Database | RDS Postgres 16 | db.t3.micro, 20GB, pgvector enabled |
| Storage | S3 | PDF storage |
| Region | eu-central-1 | Frankfurt |

### Deploy from scratch

1. **Create RDS instance**
```bash
aws rds create-db-instance \
  --db-instance-identifier ragbase-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 16.6 \
  --master-username ragbase \
  --master-user-password <password> \
  --allocated-storage 20 \
  --publicly-accessible \
  --db-name ragbase \
  --region eu-central-1
```

2. **Create S3 bucket**
```bash
aws s3api create-bucket \
  --bucket ragbase-pdfs-<account-id> \
  --region eu-central-1 \
  --create-bucket-configuration LocationConstraint=eu-central-1
```

3. **Launch EC2 instance** (t4g.small, arm64, Amazon Linux 2023)

4. **SSH in and deploy**
```bash
ssh -i ~/.ssh/ragbase-key.pem ec2-user@<ec2-ip>
git clone https://github.com/ElMolla10/ragbase.git
cd ragbase
# Create .env with RDS connection string and GROQ_API_KEY
docker-compose up -d --build
```

5. **Add swap space** (required on t4g.small for PyTorch install)
```bash
sudo dd if=/dev/zero of=/swapfile bs=128M count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

6. **Enable auto-restart**
```bash
docker update --restart unless-stopped ragbase-backend ragbase-frontend ragbase-metabase
```

---

## CI Pipeline

GitHub Actions runs on every push to master:

- **Lint** — ruff (format + style check)
- **Test** — pytest with a live Postgres/pgvector service container

```
✅ lint  (ruff check + format)
✅ test  (9 unit tests + 22 integration tests)
```

---

## Observability

Every query is logged to the `query_logs` table:

```sql
SELECT query_text, latency_ms, tokens_used, created_at
FROM query_logs
ORDER BY created_at DESC;
```

Connect Metabase to the same Postgres instance to build dashboards on top of this data.

---

## Project Structure

```
ragbase/
├── backend/
│   ├── main.py        # FastAPI app, 5 endpoints
│   ├── ingest.py      # PDF parsing, chunking, embedding
│   ├── retrieval.py   # pgvector similarity search
│   ├── llm.py         # Groq API call with context injection
│   ├── logger.py      # Query logging to Postgres
│   ├── models.py      # SQLAlchemy models (documents, chunks, query_logs)
│   ├── database.py    # Async Postgres + pgvector setup
│   └── config.py      # Environment variable config
├── frontend/
│   └── app.py         # Streamlit UI (Chat + Documents pages)
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
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/ragbase
POSTGRES_USER=ragbase
POSTGRES_PASSWORD=ragbase
POSTGRES_DB=ragbase
```

---

## Author

**Mohamed El Molla** — [LinkedIn](https://www.linkedin.com/in/mohamed-el-molla/) | [GitHub](https://github.com/ElMolla10)
