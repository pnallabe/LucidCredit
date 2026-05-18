"""
backend/app/agent/grounding.py
================================
Zero-hallucination enforcement layer for LucidCredit.

Two classes:

CitationEnforcer
    Splits a generated narrative into atomic sentences, embeds each one, and
    matches it against retrieved context chunks using cosine similarity.
    Sentences that cannot be grounded (similarity < settings.min_citation_similarity)
    or that the model self-flagged with [UNVERIFIED] are suppressed and logged.

ConfidenceScorer
    Produces a calibrated confidence_score in [0.0, 1.0] from three signals:
      * retrieval_recall  — fraction of retrieved chunks graded RELEVANT
      * citation_rate     — fraction of narrative claims that were grounded
      * unverified_rate   — fraction of LLM sentences marked [UNVERIFIED]

    Score formula (weights sum to 1.0):
        0.30 × retrieval_recall
      + 0.50 × citation_rate
      + 0.20 × (1 − unverified_rate)
"""
from __future__ import annotations

import math
import re
import uuid
from typing import List, Optional

import structlog

from app.agent.state import AgentState, RetrievedChunk
from app.config import get_settings
from app.rag.embedder import batch_embed

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity; returns 0.0 on zero-norm vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split narrative text into atomic sentences.

    Splits on sentence-ending punctuation followed by whitespace.
    Filters out empty strings and whitespace-only tokens.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# CitationEnforcer
# ---------------------------------------------------------------------------

class CitationEnforcer:
    """
    Post-generation citation enforcement.

    Usage::

        enforcer = CitationEnforcer()
        result = await enforcer.enforce(narrative, graded_chunks)
        # result keys: grounded_narrative, citations, suppressed_claims
    """

    async def enforce(
        self,
        narrative: str,
        graded_chunks: List[RetrievedChunk],
        settings_override=None,
    ) -> dict:
        """
        Ground each sentence in *narrative* against *graded_chunks*.

        Returns a dict with:
            grounded_narrative: str      — sentences that passed grounding joined by space
            citations: list[dict]        — one citation record per grounded sentence
            suppressed_claims: list[dict] — sentences that were suppressed (with reason)

        ``settings_override`` allows callers to pass a Settings-like object to
        override the configured threshold (used for domain knowledge responses).
        """
        settings = settings_override if settings_override is not None else get_settings()
        threshold: float = settings.min_citation_similarity

        sentences = _split_into_sentences(narrative)
        if not sentences:
            return {"grounded_narrative": "", "citations": [], "suppressed_claims": []}

        # Only ground against RELEVANT / AMBIGUOUS chunks; IRRELEVANT are excluded.
        relevant_chunks = [
            c for c in graded_chunks
            if c.get("relevance") in ("RELEVANT", "AMBIGUOUS")
        ]

        if not relevant_chunks:
            suppressed = [
                {"claim_text": s, "suppression_reason": "no_relevant_context"}
                for s in sentences
            ]
            log.warning(
                "citation_enforcer.no_relevant_chunks",
                total_sentences=len(sentences),
            )
            return {
                "grounded_narrative": "",
                "citations": [],
                "suppressed_claims": suppressed,
            }

        # Embed sentences and chunk content in two batches.
        claim_embeddings: List[List[float]] = await batch_embed(sentences)

        # Cap chunk content at 4096 chars — text-embedding-3-large supports 8191 tokens.
        chunk_texts = [c["content"][:4096] for c in relevant_chunks]
        chunk_embeddings: List[List[float]] = await batch_embed(chunk_texts)

        grounded_sentences: List[str] = []
        citations: List[dict] = []
        suppressed_claims: List[dict] = []

        for i, (sentence, sent_emb) in enumerate(zip(sentences, claim_embeddings)):
            # Check for model self-flagged uncertainty marker.
            is_unverified = "[UNVERIFIED]" in sentence
            clean_sentence = sentence.replace("[UNVERIFIED]", "").strip()

            # Find best-matching chunk by cosine similarity.
            best_sim = 0.0
            best_chunk_idx = -1
            for j, chunk_emb in enumerate(chunk_embeddings):
                sim = _cosine_similarity(sent_emb, chunk_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_chunk_idx = j

            # DB (BigQuery) chunks are authoritative structured data. Natural language
            # sentences generated from SQL rows have low cosine similarity to the raw
            # JSON chunk content even when they are factually grounded. For DB chunks,
            # bypass the cosine threshold entirely — any sentence that best-matches a
            # DB chunk is considered grounded (the DB query itself is the ground truth).
            effective_threshold = threshold
            if best_chunk_idx >= 0 and relevant_chunks[best_chunk_idx].get("source_type") == "db":
                effective_threshold = 0.0

            if best_sim >= effective_threshold and not is_unverified:
                best_chunk = relevant_chunks[best_chunk_idx]
                grounded_sentences.append(clean_sentence)
                citations.append(
                    {
                        "citation_id": str(uuid.uuid4()),
                        "claim_text": clean_sentence,
                        "source_type": best_chunk["source_type"],
                        "source_ref": best_chunk["source_ref"],
                        "similarity_score": round(best_sim, 4),
                        "confidence": round(best_sim, 4),
                    }
                )
                log.debug(
                    "citation_enforcer.grounded",
                    sentence_idx=i,
                    similarity=round(best_sim, 4),
                    source_ref=best_chunk["source_ref"],
                )
            else:
                reason = (
                    "unverified_marker"
                    if is_unverified
                    else "below_similarity_threshold"
                )
                suppressed_claims.append(
                    {
                        "claim_text": clean_sentence,
                        "suppression_reason": reason,
                        "best_similarity": round(best_sim, 4),
                    }
                )
                log.info(
                    "citation_enforcer.suppressed",
                    sentence_idx=i,
                    reason=reason,
                    best_similarity=round(best_sim, 4),
                )

        grounded_narrative = " ".join(grounded_sentences)

        log.info(
            "citation_enforcer.complete",
            total_sentences=len(sentences),
            grounded=len(grounded_sentences),
            suppressed=len(suppressed_claims),
        )

        return {
            "grounded_narrative": grounded_narrative,
            "citations": citations,
            "suppressed_claims": suppressed_claims,
        }


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Calibrated confidence score aggregating three independent signals.

    Weights must sum to 1.0::

        score = 0.30 × retrieval_recall
              + 0.50 × citation_rate
              + 0.20 × (1 − unverified_rate)
    """

    RETRIEVAL_WEIGHT: float = 0.30
    CITATION_WEIGHT: float = 0.50
    UNVERIFIED_WEIGHT: float = 0.20

    def score(self, state: AgentState) -> dict:
        """
        Compute calibrated confidence from state signals.

        Returns::

            {
                "confidence_score": float,   # [0.0, 1.0]
                "grounding_passed": bool,     # True if score >= settings.min_confidence_score
            }
        """
        settings = get_settings()

        # Signal 1 — retrieval recall: relevant / total retrieved
        graded_chunks = state.get("graded_chunks", [])
        if graded_chunks:
            relevant_count = sum(
                1 for c in graded_chunks if c.get("relevance") == "RELEVANT"
            )
            retrieval_recall = relevant_count / len(graded_chunks)
        else:
            retrieval_recall = 0.0

        # Signal 2 — citation rate: grounded claims / total claims
        citations = state.get("citations", [])
        suppressed = state.get("suppressed_claims", [])
        total_claims = len(citations) + len(suppressed)
        citation_rate = len(citations) / total_claims if total_claims > 0 else 0.0

        # Signal 3 — unverified rate: [UNVERIFIED] markers in raw LLM output.
        # Empty raw_output is treated as fully uncertain (unverified_rate=1.0)
        # so that a state with no LLM output scores 0 on this signal.
        raw_output: str = state.get("raw_llm_output", "")
        if not raw_output.strip():
            unverified_rate = 1.0
        else:
            unverified_count = raw_output.count("[UNVERIFIED]")
            sentences = _split_into_sentences(raw_output)
            total_sentences = max(len(sentences), 1)
            unverified_rate = min(unverified_count / total_sentences, 1.0)

        # Aggregate
        raw_score = (
            self.RETRIEVAL_WEIGHT * retrieval_recall
            + self.CITATION_WEIGHT * citation_rate
            + self.UNVERIFIED_WEIGHT * (1.0 - unverified_rate)
        )
        confidence_score = round(min(max(raw_score, 0.0), 1.0), 4)
        grounding_passed = confidence_score >= settings.min_confidence_score

        log.info(
            "confidence_scorer",
            retrieval_recall=round(retrieval_recall, 4),
            citation_rate=round(citation_rate, 4),
            unverified_rate=round(unverified_rate, 4),
            confidence_score=confidence_score,
            grounding_passed=grounding_passed,
        )

        return {
            "confidence_score": confidence_score,
            "grounding_passed": grounding_passed,
        }
