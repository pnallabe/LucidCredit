"""
backend/app/agent/tools/vector_tool.py
=======================================
Tool: semantic search against the ``policy_docs`` pgvector table.

Public API
----------
    retrieve_policy_docs(query, top_k, min_similarity) -> list[RetrievedChunk]

Embeds the query via Azure OpenAI text-embedding-3-large, runs a cosine-distance
ANN scan over the pgvector index, and returns chunks above *min_similarity*.
All errors are caught and logged — the function never raises so the agent can
handle insufficient retrieval gracefully.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from app.agent.state import RetrievedChunk
from app.db.session import AsyncSessionLocal
from app.rag.embedder import embed_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL — cosine similarity (1 - cosine distance) via pgvector <=> operator
# CAST syntax used for asyncpg compatibility (no :: after bind params)
# ---------------------------------------------------------------------------
_VECTOR_SEARCH_SQL = text(
    """
    SELECT
        id::text            AS chunk_id,
        source_path,
        chunk_index,
        content,
        source_type,
        doc_type,
        version_tag,
        metadata_json,
        1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity_score
    FROM policy_docs
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_k
    """
)


async def retrieve_policy_docs(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.30,
) -> list[RetrievedChunk]:
    """
    Embed *query* and return the *top_k* most-similar ``policy_docs`` rows
    whose cosine similarity exceeds *min_similarity*.

    Returns ``[]`` on any error — caller handles insufficient retrieval.
    """
    try:
        vec = await embed_text(query)
    except Exception as exc:
        log.warning("vector_tool.embed_failed error=%s", exc)
        return []

    embedding_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                _VECTOR_SEARCH_SQL,
                {"query_embedding": embedding_str, "top_k": top_k},
            )
            rows = result.mappings().all()
    except Exception as exc:
        log.warning("vector_tool.db_failed error=%s", exc)
        return []

    chunks: list[RetrievedChunk] = []
    for row in rows:
        score = float(row["similarity_score"])
        if score < min_similarity:
            continue

        meta: dict[str, Any] = {}
        raw_meta = row.get("metadata_json")
        if raw_meta:
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
            except Exception:
                pass

        source_ref = (
            f"policy_docs:{row['source_path']}#chunk{row['chunk_index']}"
        )
        chunks.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                source_type="vector_doc",
                source_ref=source_ref,
                content=row["content"],
                relevance="AMBIGUOUS",  # graded later by grade_documents_node
                similarity_score=score,
            )
        )

    log.debug("vector_tool retrieved %d chunks for query len=%d", len(chunks), len(query))
    return chunks
