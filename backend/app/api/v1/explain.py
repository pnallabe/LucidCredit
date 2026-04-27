"""
backend/app/api/v1/explain.py
================================
FastAPI router for POST /v1/explain/decision.

Runs the full corrective-RAG agent graph and returns a grounded narrative
explaining a specific credit decision for either an analyst or an applicant.
"""
from __future__ import annotations

import uuid as _uuid
from typing import List, Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import get_graph

log = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExplainDecisionRequest(BaseModel):
    decision_id: UUID
    source: Literal["credit-risk-platform", "thinfile"] = "credit-risk-platform"
    audience: Literal["analyst", "applicant"] = "analyst"
    language: str = "en"
    include_counterfactual: bool = True
    include_shap_narrative: bool = True


class CitationSchema(BaseModel):
    claim: str
    source_type: str
    source_ref: str
    confidence: float


class ExplainDecisionResponse(BaseModel):
    session_id: UUID
    narrative: str
    citations: List[CitationSchema] = Field(default_factory=list)
    confidence_score: float
    counterfactual: Optional[dict] = None
    adverse_action_codes: List[str] = Field(default_factory=list)
    compliance_flags: List[str] = Field(default_factory=list)
    audience: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/decision", response_model=ExplainDecisionResponse)
async def explain_decision(
    request: ExplainDecisionRequest,
) -> ExplainDecisionResponse:
    """
    Retrieve grounded context for *decision_id* and generate an audience-appropriate
    explanation narrative via the corrective-RAG agent.

    - ``analyst`` audience: technical narrative with SHAP values, PD bands, DTI ratios.
    - ``applicant`` audience: Plain English ECOA/FCRA-compliant adverse action notice.

    Returns HTTP 503 if the primary LLM is unavailable for applicant audience.
    Returns HTTP 422 if confidence < threshold or compliance check fails.
    """
    session_id = _uuid.uuid4()
    bound_log = log.bind(
        session_id=str(session_id),
        decision_id=str(request.decision_id),
        audience=request.audience,
    )

    query = (
        f"Explain credit decision {request.decision_id} for "
        f"{'an applicant' if request.audience == 'applicant' else 'an analyst'}. "
        f"Source: {request.source}. "
        f"{'Include counterfactual guidance.' if request.include_counterfactual else ''} "
        f"{'Include SHAP narrative.' if request.include_shap_narrative else ''}"
    ).strip()

    initial_state = {
        "session_id": session_id,
        "query": query,
        "intent": "explain_decision",
        "audience": request.audience,
        "context_payload": {
            "decision_id": str(request.decision_id),
            "source": request.source,
            "language": request.language,
            "include_counterfactual": request.include_counterfactual,
            "include_shap_narrative": request.include_shap_narrative,
        },
    }

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": str(session_id)}}
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        bound_log.error("explain_graph_error", error=str(exc))
        raise HTTPException(status_code=500, detail={"error": "GRAPH_ERROR", "message": str(exc)})

    output: dict = final_state.get("final_output", {})

    # Error routing
    if "error" in output:
        error_code = output["error"]
        if error_code == "PRIMARY_UNAVAILABLE":
            raise HTTPException(
                status_code=503,
                headers={"Retry-After": "30"},
                detail={"error": error_code, "message": "Primary LLM unavailable."},
            )
        if error_code in ("INSUFFICIENT_GROUNDING", "COMPLIANCE_FAILED"):
            raise HTTPException(
                status_code=422,
                detail={"error": error_code, "message": "Output failed quality/compliance gate."},
            )
        raise HTTPException(status_code=500, detail={"error": error_code})

    bound_log.info(
        "explain_decision_ok",
        confidence=output.get("confidence_score", 0.0),
    )

    raw_citations = output.get("citations") or []
    citations = [
        CitationSchema(
            claim=c.get("claim", ""),
            source_type=c.get("source_type", ""),
            source_ref=c.get("source_ref", ""),
            confidence=float(c.get("confidence", 1.0)),
        )
        for c in raw_citations
    ]

    return ExplainDecisionResponse(
        session_id=session_id,
        narrative=output.get("narrative", output.get("grounded_narrative", "")),
        citations=citations,
        confidence_score=float(output.get("confidence_score", 0.0)),
        counterfactual=output.get("counterfactual"),
        adverse_action_codes=output.get("adverse_action_codes", []),
        compliance_flags=final_state.get("compliance_flags") or [],
        audience=request.audience,
    )
