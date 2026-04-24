"""
backend/app/rag/embedder.py
============================
Thin async wrapper around Azure OpenAI text-embedding-3-large.

Public API
----------
    embed_text(text: str) -> list[float]
    batch_embed(texts: list[str], batch_size: int = 64) -> list[list[float]]

Both functions use the Azure OpenAI endpoint via app.llm.provider.get_embedding_client().
The embedding model is always text-embedding-3-large (3072 dimensions).

IMPORTANT: The EMBEDDING_DIM constant below must match the pgvector column definition.
If the embedding provider or model ever changes, a full corpus re-ingestion is required
because existing vectors will be incompatible with new vectors (dimension or space mismatch).
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.config import get_settings
from app.llm.provider import get_embedding_client

log = logging.getLogger(__name__)

# Dimension of text-embedding-3-large — must match the pgvector column definition exactly.
# WARNING: Changing this constant alone is not sufficient — a full corpus re-ingestion is
# required if the embedding model or provider ever changes.
EMBEDDING_DIM: int = 3072


async def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a float vector of length EMBEDDING_DIM."""
    settings = get_settings()
    client = get_embedding_client()
    response = await client.embeddings.create(
        input=[text],
        model=settings.azure_openai_deployment_embedding,
    )
    return response.data[0].embedding


async def batch_embed(
    texts: Sequence[str],
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Embed *texts* in batches of *batch_size* to stay within rate limits.

    Returns a list of float vectors in the same order as *texts*.
    Raises :class:`openai.OpenAIError` on API failure (caller should retry or abort).
    """
    if not texts:
        return []

    settings = get_settings()
    client = get_embedding_client()
    results: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        log.debug(
            "Embedding batch %d/%d (%d texts)",
            i // batch_size + 1,
            -(-len(texts) // batch_size),
            len(batch),
        )
        response = await client.embeddings.create(
            input=batch,
            model=settings.azure_openai_deployment_embedding,
        )
        # The API guarantees order matches input
        batch_vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]

        for vec in batch_vectors:
            if len(vec) != EMBEDDING_DIM:
                log.debug(
                    "Unexpected embedding dimension %d (expected %d) — "
                    "model mismatch? DB insert will fail if schema is not updated.",
                    len(vec),
                    EMBEDDING_DIM,
                )

        results.extend(batch_vectors)

    return results
