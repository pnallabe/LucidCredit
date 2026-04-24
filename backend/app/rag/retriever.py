"""
backend/app/rag/retriever.py
=============================
Hybrid retriever: dense vector search re-ranked with BM25.

Public API
----------
    hybrid_retrieve(query, top_k, min_similarity) -> list[RetrievedChunk]

Algorithm
---------
1. Dense retrieval — embed query, fetch top_k * DENSE_MULTIPLIER candidates
   from pgvector using cosine similarity.
2. BM25 re-ranking — score candidates by BM25 against the query tokens.
3. Score fusion — final_score = DENSE_WEIGHT * dense_score + BM25_WEIGHT * bm25_norm
4. Return the top_k results sorted by final_score descending.

The ``rank_bm25`` package is used when available; otherwise a minimal
TF-IDF-style scorer is used as a fallback (no extra install required).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Sequence

from app.agent.state import RetrievedChunk
from app.agent.tools.vector_tool import retrieve_policy_docs

log = logging.getLogger(__name__)

# Tuning constants
DENSE_MULTIPLIER = 3    # over-fetch by this factor before BM25 re-ranking
DENSE_WEIGHT = 0.65     # weight for cosine-similarity score
BM25_WEIGHT = 0.35      # weight for BM25 score


# ---------------------------------------------------------------------------
# Minimal BM25 fallback (no external dep)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(
    query_tokens: list[str],
    corpus: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return a BM25 score for each document in *corpus* against *query_tokens*."""
    if not corpus or not query_tokens:
        return [0.0] * len(corpus)

    tokenized = [_tokenize(doc) for doc in corpus]
    avg_dl = sum(len(d) for d in tokenized) / len(tokenized) if tokenized else 1.0

    # IDF — log( (N - df + 0.5) / (df + 0.5) + 1 )
    N = len(corpus)
    df: dict[str, int] = {}
    for tok_doc in tokenized:
        for tok in set(tok_doc):
            df[tok] = df.get(tok, 0) + 1

    scores: list[float] = []
    for tok_doc in tokenized:
        tf_map = Counter(tok_doc)
        dl = len(tok_doc)
        score = 0.0
        for q_tok in query_tokens:
            if q_tok not in df:
                continue
            idf = math.log((N - df[q_tok] + 0.5) / (df[q_tok] + 0.5) + 1)
            tf = tf_map.get(q_tok, 0)
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * norm_tf
        scores.append(score)
    return scores


def _try_rank_bm25(
    query_tokens: list[str],
    corpus: list[str],
) -> list[float] | None:
    """Use rank_bm25 if installed; returns None if not available."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        tokenized = [_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        return list(bm25.get_scores(query_tokens))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def hybrid_retrieve(
    query: str,
    top_k: int = 8,
    min_similarity: float = 0.25,
) -> list[RetrievedChunk]:
    """
    Return the *top_k* most relevant chunks for *query* using hybrid retrieval.

    Falls back to pure dense retrieval (no re-ranking) when the candidate pool
    is empty or BM25 scoring fails.
    """
    # Step 1 — Dense over-fetch
    candidates = await retrieve_policy_docs(
        query=query,
        top_k=top_k * DENSE_MULTIPLIER,
        min_similarity=min_similarity,
    )

    if not candidates:
        return []

    if len(candidates) <= top_k:
        # Not enough candidates to benefit from re-ranking
        return candidates[:top_k]

    # Step 2 — BM25 on candidate content
    query_tokens = _tokenize(query)
    corpus = [c["content"] for c in candidates]

    raw_bm25 = _try_rank_bm25(query_tokens, corpus) or _bm25_scores(query_tokens, corpus)

    # Step 3 — Normalize BM25 scores to [0, 1]
    bm25_max = max(raw_bm25) if max(raw_bm25) > 0 else 1.0
    bm25_norm = [s / bm25_max for s in raw_bm25]

    # Step 4 — Fuse scores and sort
    fused: list[tuple[float, RetrievedChunk]] = []
    for chunk, bm25_s, dense_s in zip(
        candidates, bm25_norm, [c["similarity_score"] for c in candidates]
    ):
        final = DENSE_WEIGHT * dense_s + BM25_WEIGHT * bm25_s
        fused.append((final, chunk))

    fused.sort(key=lambda x: x[0], reverse=True)

    result = []
    for final_score, chunk in fused[:top_k]:
        result.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                source_type=chunk["source_type"],
                source_ref=chunk["source_ref"],
                content=chunk["content"],
                relevance=chunk["relevance"],
                similarity_score=final_score,
            )
        )

    log.debug(
        "hybrid_retrieve: %d candidates → %d results for query len=%d",
        len(candidates), len(result), len(query),
    )
    return result
