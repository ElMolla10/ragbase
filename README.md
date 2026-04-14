# RAGBase - Research Paper Q&A Platform

A production-grade research paper Q&A system with full ML observability. Upload PDFs, ask questions, get cited answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAGBase                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Streamlit  │────▶│   FastAPI    │────▶│   Postgres   │    │
│  │   Frontend   │     │   Backend    │     │  + pgvector  │    │
│  │   :8501      │     │   :8000      │     │   :5432      │    │
│  └──────────────┘     └──────┬───────┘     └──────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                       ┌──────────────┐                          │
│                       │   Groq API   │                          │
│                       │ llama-3.3-70b│                          │
│                       └──────────────┘                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Metabase   │────────────────────────────────────────────┐ │
│  │   Dashboard  │     Query logs, latency, token usage       │ │
│  │   :3000      │◀───────────────────────────────────────────┘ │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **PDF Upload**: Upload research papers, automatically chunked and embedded
- **Semantic Search**: pgvector-powered similarity search (all-MiniLM-L6-v2)
- **LLM Answers**: Groq-powered answers with source citations
- **Observability**: Every query logged with latency, tokens, sources
- **Dashboard**: Metabase dashboard for metrics visualization

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd ragbase
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Start services

```bash
docker-compose up -d
```

### 3. Access the application

- **Chat UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Metabase**: http://localhost:3000

## API Documentation

### POST /ingest

Upload and process a PDF document.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"
```

Response:
```json
{
  "document_id": 1,
  "filename": "paper.pdf",
  "num_chunks": 42,
  "status": "completed"
}
```

### POST /query

Query documents and get an answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main contribution of this paper?"}'
```

Response:
```json
{
  "answer": "The main contribution is...",
  "sources": [
    {
      "id": 5,
      "content": "...",
      "page_number": 3,
      "similarity_score": 0.8542
    }
  ],
  "latency_ms": 1234,
  "token_count": 567,
  "model_used": "llama-3.3-70b-versatile"
}
```

### GET /documents

List all uploaded documents.

```bash
curl http://localhost:8000/documents
```

### GET /stats

Get aggregate statistics.

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "total_queries": 150,
  "avg_latency_ms": 1432.5,
  "total_documents": 12,
  "total_chunks": 1847
}
```

### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

## Observability

### Metabase Dashboard

Access Metabase at http://localhost:3000 (first-time setup required).

**Tracked Metrics:**
- Total queries over time
- Average latency (ms)
- Token usage per query
- Documents uploaded
- Most queried documents
- Query latency distribution

### Query Logs Schema

```sql
SELECT 
  query,
  latency_ms,
  token_count,
  model_used,
  created_at
FROM query_logs
ORDER BY created_at DESC;
```

## Local Development

### Without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Postgres with pgvector (requires local install)
# Or use: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=ragbase pgvector/pgvector:pg16

# Run backend
uvicorn backend.main:app --reload

# Run frontend (separate terminal)
streamlit run frontend/app.py
```

### Running Tests

```bash
# Ensure test database exists
createdb ragbase_test

# Run tests
pytest tests/ -v
```

## AWS Deployment

### Prerequisites

- AWS CLI configured
- EC2 instance (t3.medium or larger)
- RDS PostgreSQL with pgvector
- S3 bucket for PDF storage (optional)

### Step 1: Set up RDS

```bash
# Create RDS instance with pgvector
aws rds create-db-instance \
  --db-instance-identifier ragbase-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 16.1 \
  --master-username ragbase \
  --master-user-password <password> \
  --allocated-storage 20

# Enable pgvector extension (connect to RDS)
psql -h <rds-endpoint> -U ragbase -d ragbase
CREATE EXTENSION vector;
```

### Step 2: Set up EC2

```bash
# SSH into EC2 instance
ssh -i key.pem ec2-user@<ec2-ip>

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and configure
git clone <repo-url>
cd ragbase
cp .env.example .env
# Edit .env with RDS connection string and GROQ_API_KEY
```

### Step 3: Deploy

```bash
# Build and start services (without postgres, using RDS)
docker-compose up -d backend frontend metabase

# Or for production, use a reverse proxy like nginx
```

### Step 4: Configure Security Groups

- Allow inbound 8000, 8501, 3000 from your IP
- Allow EC2 to connect to RDS on 5432

## Project Structure

```
ragbase/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── ingest.py        # PDF parsing, chunking, embedding
│   ├── retrieval.py     # pgvector similarity search
│   ├── llm.py           # Groq API integration
│   ├── logger.py        # Query logging
│   ├── models.py        # SQLAlchemy models
│   ├── database.py      # Database setup
│   └── config.py        # Environment configuration
├── frontend/
│   └── app.py           # Streamlit UI
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── init.sql
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Screenshots

### Chat Interface
```
[Screenshot: Chat UI with question, answer, and expandable sources]
```

### Document Upload
```
[Screenshot: Document management page with upload and list]
```

### Metabase Dashboard
```
[Screenshot: Dashboard showing query metrics, latency charts]
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | Required |
| `DATABASE_URL` | Postgres connection string | `postgresql+asyncpg://ragbase:ragbase@localhost:5432/ragbase` |
| `POSTGRES_USER` | Postgres username | `ragbase` |
| `POSTGRES_PASSWORD` | Postgres password | `ragbase` |
| `POSTGRES_DB` | Database name | `ragbase` |

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), Pydantic v2
- **Database**: PostgreSQL 16 + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Frontend**: Streamlit
- **Observability**: Metabase
- **Infrastructure**: Docker Compose, GitHub Actions

## License

MIT
