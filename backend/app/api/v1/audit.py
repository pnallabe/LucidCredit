"""
backend/app/api/v1/audit.py
============================
Audit log retrieval endpoint.

GET /v1/audit/{session_id}
  Returns the full audit record for a completed copilot session, including:
  - All LangGraph intermediate state fields (rendered_prompt, raw_llm_output,
    retrieved chunks, confidence score, suppressed claims)
  - Structured citation records (grounded claim → source traceability)
  - Compliance flags generated during the session
  - SR 11-7 model identity fields (provider_model, provider_fallback_used)

This endpoint is the primary surface for:
  1. Human review of AI-generated outputs prior to delivery.
  2. Model risk audits under SR 11-7.
  3. ECOA / FCRA adverse action audit trails.
  4. Internal QA / regression debugging.

The endpoint is intentionally read-only (GET) — audit records are immutable
after session completion.  No mutation of session state is supported here.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.citation import Citation
from app.models.session import CopilotSession

log = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CitationRecord(BaseModel):
    citation_id: str
    claim_text: str
    source_type: str
    source_ref: str
    similarity_score: Optional[float] = None
    confidence: Optional[float] = None


class AuditResponse(BaseModel):
    session_id: str
    query_text: str
    intent: str
    audience: str
    source_system: Optional[str] = None
    user_id: Optional[str] = None
    retrieved_chunks: Optional[Any] = Field(
        default=None,
        description="Raw retrieved chunk payloads stored during retrieval node.",
    )
    rendered_prompt: Optional[str] = Field(
        default=None,
        description="Full system + user prompt rendered and sent to the LLM.",
    )
    raw_llm_output: Optional[str] = Field(
        default=None,
        description="Verbatim LLM response before grounding enforcement.",
    )
    grounded_narrative: Optional[str] = Field(
        default=None,
        description="LLM output after citation enforcement and claim suppression.",
    )
    confidence_score: Optional[float] = None
    suppressed_claims: Optional[Any] = Field(
        default=None,
        description="Claims removed by the citation enforcer as unverifiable.",
    )
    compliance_flags: Optional[Any] = Field(
        default=None,
        description="ECOA/FCRA/SR 11-7 compliance flags generated during the session.",
    )
    provider_model: str = Field(
        description="LLM model identifier — SR 11-7 traceability.",
    )
    provider_fallback_used: bool = Field(
        description="True if the secondary (fallback) LLM provider was used.",
    )
    context_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    created_at: datetime
    citations: List[CitationRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/{session_id}", response_model=AuditResponse)
async def get_audit_record(session_id: str) -> AuditResponse:
    """
    Retrieve the full audit record for a copilot session.

    Returns all intermediate LangGraph state fields and associated citation
    records.  Returns 404 if no session with the given ID exists.

    The session_id must be a valid UUID string.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid session_id format: '{session_id}' is not a valid UUID.",
        )

    async with AsyncSessionLocal() as db:
        session_row: Optional[CopilotSession] = await db.get(
            CopilotSession, session_uuid
        )

        if session_row is None:
            log.warning("audit.session_not_found", session_id=session_id)
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        # Fetch associated citations ordered by creation (approximated by PK)
        result = await db.execute(
            select(Citation)
            .where(Citation.session_id == session_uuid)
            .order_by(Citation.citation_id)
        )
        citation_rows: List[Citation] = list(result.scalars().all())

    citation_records = [
        CitationRecord(
            citation_id=str(c.citation_id),
            claim_text=c.claim_text,
            source_type=c.source_type,
            source_ref=c.source_ref,
            similarity_score=c.similarity_score,
            confidence=c.confidence,
        )
        for c in citation_rows
    ]

    log.info(
        "audit.retrieved",
        session_id=session_id,
        citation_count=len(citation_records),
        audience=session_row.audience,
    )

    return AuditResponse(
        session_id=str(session_row.session_id),
        query_text=session_row.query_text,
        intent=session_row.intent,
        audience=session_row.audience,
        source_system=session_row.source_system,
        user_id=session_row.user_id,
        retrieved_chunks=session_row.retrieved_chunks,
        rendered_prompt=session_row.rendered_prompt,
        raw_llm_output=session_row.raw_llm_output,
        grounded_narrative=session_row.grounded_narrative,
        confidence_score=session_row.confidence_score,
        suppressed_claims=session_row.suppressed_claims,
        compliance_flags=session_row.compliance_flags,
        provider_model=session_row.provider_model,
        provider_fallback_used=session_row.provider_fallback_used,
        context_token_count=session_row.context_token_count,
        output_token_count=session_row.output_token_count,
        created_at=session_row.created_at,
        citations=citation_records,
    )
