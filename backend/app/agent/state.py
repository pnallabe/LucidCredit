# backend/app/agent/state.py
from __future__ import annotations

from typing import Literal, Optional, TypedDict
from uuid import UUID


class RetrievedChunk(TypedDict):
    chunk_id: str
    source_type: Literal["vector_doc", "api", "db", "clarification"]
    source_ref: str
    content: str
    relevance: Literal["RELEVANT", "IRRELEVANT", "AMBIGUOUS"]
    similarity_score: float


class AgentState(TypedDict):
    # Input
    session_id: UUID
    query: str
    intent: Literal["explain_decision", "analyst_query", "applicant_comms", "portfolio_brief"]
    audience: Literal["analyst", "applicant"]
    context_payload: dict  # raw API inputs (decision_id, application_id, etc.)

    # Context ingestion — clarification round-trip
    # clarification_items: list of {id, question, options} pending user answer
    # clarifications:      answers from the user, keyed by item id
    clarification_items: list[dict]
    clarifications: dict  # e.g. {"product_type": "Personal loans", "time_period": "2023"}

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    graded_chunks: list[RetrievedChunk]  # RELEVANT only after grading
    retrieval_sufficient: bool

    # Generation
    rendered_prompt: str
    raw_llm_output: str
    provider_used: str  # e.g. "azure:gpt-4.1-2025-04-14" — stored in session

    # Grounding
    grounded_narrative: str
    citations: list[dict]
    suppressed_claims: list[dict]
    confidence_score: float
    grounding_passed: bool  # True if confidence_score >= settings.min_confidence_score

    # Compliance (applicant audience only)
    compliance_flags: list[str]
    compliance_passed: bool

    # Output
    final_output: dict
    error: Optional[str]
