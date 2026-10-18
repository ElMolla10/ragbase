"""Preserve original text; normalize only a separate search representation."""

import re
import unicodedata
from uuid import UUID, uuid5

from sqlalchemy import text

INDEX_VERSION = "native-lexical-v1"
NAMESPACE = UUID("f7de6777-c404-4e4b-88dd-ae54f1c5b433")
STOP = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "and",
    "how",
    "much",
    "which",
    "was",
    "were",
    "ما",
    "هو",
    "هي",
    "في",
    "من",
    "علي",
    "على",
    "عن",
    "ايه",
    "كم",
    "ماذا",
    "هل",
}


def normalize(value):
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", value)
    return value.translate(str.maketrans("أإآى٠١٢٣٤٥٦٧٨٩", "اااي0123456789"))


def query_terms(question):
    return list(
        dict.fromkeys(
            term
            for term in re.findall(r"[^\W_]+", normalize(question))
            if term not in STOP
        )
    )[:20]


def chunks(version, pages):
    for page in pages:
        content = page["content"]
        for part, start in enumerate(range(0, len(content), 448)):
            snippet = content[start : start + 512]
            if not snippet.strip():
                continue
            yield {
                "id": uuid5(
                    NAMESPACE, f"{version}:{page['ordinal']}:{start}:{INDEX_VERSION}"
                ),
                "ordinal": page["ordinal"],
                "part": part,
                "char_start": start,
                "char_end": start + len(snippet),
                "content": snippet,
                "search_text": normalize(snippet),
            }


async def index_pages(conn, workspace, version, pages):
    # Publishing pages and their index inside one transaction prevents partial
    # retrieval. The manifest marks empty documents without repeatedly indexing.
    await conn.execute(
        text(
            "DELETE FROM enterprise.document_chunks WHERE version_id=:v AND index_version=:i"
        ),
        {"v": version, "i": INDEX_VERSION},
    )
    count = 0
    for chunk in chunks(version, pages):
        await conn.execute(
            text("""
            INSERT INTO enterprise.document_chunks(id,workspace_id,version_id,index_version,ordinal,part,char_start,char_end,content,search_text)
            VALUES(:id,:workspace,:version,:index,:ordinal,:part,:char_start,:char_end,:content,:search_text)
        """),
            {
                **chunk,
                "workspace": workspace,
                "version": version,
                "index": INDEX_VERSION,
            },
        )
        count += 1
    await conn.execute(
        text("""
        INSERT INTO enterprise.retrieval_indexes(workspace_id,version_id,index_version,chunk_count)
        VALUES(:w,:v,:i,:n) ON CONFLICT(version_id,index_version) DO UPDATE SET chunk_count=excluded.chunk_count
    """),
        {"w": workspace, "v": version, "i": INDEX_VERSION, "n": count},
    )


async def index_pending(conn, workspace):
    version = await conn.scalar(
        text("""
        SELECT j.version_id FROM enterprise.ingestion_jobs j WHERE j.workspace_id=:w
        AND j.status IN ('extracted','needs_review') AND NOT EXISTS (
            SELECT 1 FROM enterprise.retrieval_indexes i WHERE i.version_id=j.version_id AND i.index_version=:i)
        ORDER BY j.created_at LIMIT 1
    """),
        {"w": workspace, "i": INDEX_VERSION},
    )
    if not version:
        return False
    pages = (
        (
            await conn.execute(
                text(
                    "SELECT ordinal,content FROM enterprise.extracted_pages WHERE version_id=:v ORDER BY ordinal"
                ),
                {"v": version},
            )
        )
        .mappings()
        .all()
    )
    await index_pages(conn, workspace, version, pages)
    return True
