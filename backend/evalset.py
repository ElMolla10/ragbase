"""Golden-set scoring for retrieval and optional generation."""

import json
import re
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from backend.ingest import ingest_pdf
from backend.llm import generate_answer
from backend.retrieval import search_chunks

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "eval" / "corpus"
GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"

REFUSAL_MARKERS = (
    "not in the",
    "not found",
    "cannot answer",
    "can't answer",
    "no relevant",
    "does not mention",
    "doesn't mention",
    "لا توجد",
    "غير موجود",
    "لا أستطيع",
)


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def page_hit(pages: list[int], expected: list[int]) -> bool:
    return bool(set(pages) & set(expected))


def cited_pages(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"page\s+(\d+)", answer, flags=re.I)]


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


async def ingest_corpus(session: AsyncSession, corpus_dir: Path = CORPUS_DIR) -> dict:
    mapping = {}
    for path in sorted(corpus_dir.glob("*.pdf")):
        document = await ingest_pdf(session, path.read_bytes(), path.name)
        mapping[path.name] = document.id
    return mapping


async def run_eval(
    session: AsyncSession,
    *,
    generate: bool = False,
    golden_path: Path = GOLDEN_PATH,
    corpus_dir: Path = CORPUS_DIR,
) -> dict:
    golden = load_golden(golden_path)
    docs = await ingest_corpus(session, corpus_dir)
    cases = []
    retrieval_hits = 0
    retrieval_total = 0
    citation_hits = 0
    citation_total = 0
    refusal_hits = 0
    refusal_total = 0

    for item in golden:
        expected = item["expected_pages"]
        document_id = docs[item["doc"]]
        chunks = await search_chunks(
            session,
            item["query"],
            document_id=document_id,
            top_k=5,
        )
        pages = [chunk.page_number for chunk in chunks]
        hit = page_hit(pages, expected) if expected else None
        if expected:
            retrieval_total += 1
            retrieval_hits += int(bool(hit))

        answer = None
        citation_ok = None
        refused = None
        if generate and chunks:
            response = await generate_answer(item["query"], chunks)
            answer = response.answer
            if expected:
                citation_total += 1
                citation_ok = page_hit(cited_pages(answer), expected)
                citation_hits += int(citation_ok)
            else:
                refusal_total += 1
                refused = is_refusal(answer)
                refusal_hits += int(refused)

        cases.append(
            {
                "query": item["query"],
                "doc": item["doc"],
                "type": item["type"],
                "lang": item["lang"],
                "expected_pages": expected,
                "retrieved_pages": pages,
                "recall_hit": hit,
                "answer": answer,
                "citation_hit": citation_ok,
                "refused": refused,
            }
        )

    def rate(hits: int, total: int) -> float | None:
        return round(hits / total, 4) if total else None

    return {
        "metrics": {
            "recall_at_5": rate(retrieval_hits, retrieval_total),
            "citation_accuracy": rate(citation_hits, citation_total),
            "refusal_rate": rate(refusal_hits, refusal_total),
            "n_retrieval": retrieval_total,
            "n_citation": citation_total,
            "n_refusal": refusal_total,
        },
        "cases": cases,
    }
