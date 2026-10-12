#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -e .env ]]; then
    cp .env.example .env
    echo "Created .env. Set GROQ_API_KEY there to enable answer generation."
fi
docker compose up -d --build
echo "RAGBase UI: http://localhost:8501 | API: http://localhost:8000/docs"
