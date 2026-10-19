"""Authorize selected versions before indexed search; never filter after leakage."""

from fastapi import HTTPException
from sqlalchemy import text

from backend.search.index import INDEX_VERSION, query_terms


async def check_versions(conn, workspace, versions):
    scope = [str(value) for value in versions]
    rows = (
        (
            await conn.execute(
                text("""
        SELECT v.id,j.status,i.chunk_count FROM enterprise.document_versions v
        LEFT JOIN enterprise.ingestion_jobs j ON j.version_id=v.id
        LEFT JOIN enterprise.retrieval_indexes i ON i.version_id=v.id AND i.index_version=:index
        WHERE v.workspace_id=:workspace AND v.id=ANY(CAST(:versions AS uuid[]))
    """),
                {"workspace": workspace, "versions": scope, "index": INDEX_VERSION},
            )
        )
        .mappings()
        .all()
    )
    if len(rows) != len(versions):
        raise HTTPException(404, "Selected source not available")
    if any(row["status"] != "extracted" or row["chunk_count"] is None for row in rows):
        raise HTTPException(
            409, "Selected sources need extraction, indexing or review before questions"
        )
    return scope


async def retrieve(conn, workspace, versions, question, *, limit=10):
    scope = await check_versions(conn, workspace, versions)
    terms = query_terms(question)
    if not terms:
        return []
    query = " | ".join("'" + term + "'" for term in terms)
    hits = (
        (
            await conn.execute(
                text("""
        SELECT c.id,c.version_id,c.ordinal,c.char_start,c.char_end,c.content,c.index_version,
               v.sha256,d.id AS document_id,d.filename,p.location,j.warnings,
               ts_rank_cd(c.search_vector,CAST(:query AS tsquery)) AS rank
        FROM enterprise.document_chunks c JOIN enterprise.document_versions v ON v.id=c.version_id
        JOIN enterprise.documents d ON d.id=v.document_id
        JOIN enterprise.extracted_pages p ON p.version_id=c.version_id AND p.ordinal=c.ordinal
        JOIN enterprise.ingestion_jobs j ON j.version_id=c.version_id
        WHERE c.workspace_id=:workspace AND c.version_id=ANY(CAST(:versions AS uuid[]))
          AND c.index_version=:index AND j.status='extracted' AND c.search_vector @@ CAST(:query AS tsquery)
        ORDER BY rank DESC,c.version_id,c.ordinal,c.part LIMIT :limit
    """),
                {
                    "workspace": workspace,
                    "versions": scope,
                    "query": query,
                    "index": INDEX_VERSION,
                    "limit": limit,
                },
            )
        )
        .mappings()
        .all()
    )
    seen, result = set(), []
    for hit in hits:
        key = (hit["version_id"], hit["content"])
        if key not in seen:
            result.append(
                {
                    **dict(hit),
                    "source_id": str(hit["id"]),
                    "preview_url": f"/api/workspaces/{workspace}/sources/{hit['id']}",
                }
            )
            seen.add(key)
    return result


async def document_context(conn, workspace, versions):
    """Bounded overview candidates across beginning/end and intervening pages.

    All queries stay inside the authorized immutable-version scope. Coverage
    totals accompany the selection; sampling is never a complete-review claim.
    """
    scope = await check_versions(conn, workspace, versions)
    rows = (
        (
            await conn.execute(
                text("""
            WITH scoped AS (
                SELECT c.id,c.version_id,c.ordinal,c.part,c.char_start,c.char_end,
                       c.content,c.index_version,v.sha256,d.id AS document_id,
                       d.filename,p.location,j.warnings,
                       max(c.ordinal) OVER(PARTITION BY c.version_id) AS last_ordinal,
                       count(*) OVER() AS total_chunks
                FROM enterprise.document_chunks c
                JOIN enterprise.document_versions v ON v.id=c.version_id
                JOIN enterprise.documents d ON d.id=v.document_id
                JOIN enterprise.extracted_pages p ON p.version_id=c.version_id AND p.ordinal=c.ordinal
                JOIN enterprise.ingestion_jobs j ON j.version_id=c.version_id
                WHERE c.workspace_id=:w AND c.version_id=ANY(CAST(:versions AS uuid[]))
                  AND c.index_version=:i AND j.status='extracted'
            )
            SELECT * FROM scoped
            ORDER BY CASE WHEN ordinal<=2 THEN 0 WHEN ordinal>=last_ordinal-1 THEN 1 ELSE 2 END,
                     part,ordinal,version_id LIMIT 128
            """),
                {"w": workspace, "versions": scope, "i": INDEX_VERSION},
            )
        )
        .mappings()
        .all()
    )
    total = rows[0]["total_chunks"] if rows else 0
    return [
        {
            **{
                k: v
                for k, v in dict(row).items()
                if k not in {"total_chunks", "last_ordinal"}
            },
            "source_id": str(row["id"]),
            "preview_url": f"/api/workspaces/{workspace}/sources/{row['id']}",
        }
        for row in rows
    ], total
