"""
backend/app/api/v1/applicant.py
=================================
FastAPI router for POST /v1/applicant/communication.

REGULATORY CONSTRAINT — hard 503 gate:
  Applicant communications are ECOA/FCRA-regulated adverse action notices.
  If the primary LLM (Azure OpenAI GPT-4.1) is unavailable, this endpoint
  returns HTTP 503 with Retry-After: 30 and does NOT fall back to any other
  model. A degraded or non-compliant AAN must never be delivered to an applicant.
"""
from __future__ import annotations

import uuid as _uuid
from typing import Literal
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


class ApplicantCommunicationRequest(BaseModel):
    application_id: UUID
    source: Literal["thinfile", "credit_risk_platform"]
    communication_type: Literal["decline", "approve", "counteroffer", "incomplete"]
    channel: Literal["email", "sms", "portal", "letter"]
    language: str = "en"


class ApplicantCommunicationResponse(BaseModel):
    session_id: UUID
    subject_line: str
    body: str
    adverse_action_notice: dict
    compliance_validated: bool
    citations: list[dict]
    confidence_score: float


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/communication", response_model=ApplicantCommunicationResponse)
async def applicant_communication(
    request: ApplicantCommunicationRequest,
) -> ApplicantCommunicationResponse:
    """
    Generate a compliant applicant communication (adverse action notice, approval
    letter, counteroffer, or incomplete application notice).

    Returns HTTP 503 with Retry-After: 30 if the primary LLM is unavailable.
    Returns HTTP 422 if the generated output fails ECOA/FCRA compliance validation.

    NEVER falls back to a secondary model — regulatory compliance requires the
    specific GPT-4.1 deployment whose outputs have been validated.
    """
    session_id = _uuid.uuid4()
    bound_log = log.bind(
        session_id=str(session_id),
        application_id=str(request.application_id),
        communication_type=request.communication_type,
    )

    initial_state = {
        "session_id": session_id,
        "query": (
            f"Generate a {request.communication_type} communication for application "
            f"{request.application_id} via {request.channel} in {request.language}."
        ),
        "intent": "applicant_comms",
        "audience": "applicant",
        "context_payload": {
            "application_id": str(request.application_id),
            "source": request.source,
            "communication_type": request.communication_type,
            "channel": request.channel,
            "language": request.language,
        },
        "retrieved_chunks": [],
        "graded_chunks": [],
        "retrieval_sufficient": False,
        "rendered_prompt": "",
        "raw_llm_output": "",
        "provider_used": "",
        "grounded_narrative": "",
        "citations": [],
        "suppressed_claims": [],
        "confidence_score": 0.0,
        "grounding_passed": False,
        "compliance_flags": [],
        "compliance_passed": False,
        "final_output": {},
        "error": None,
    }

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": str(session_id)}}
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        bound_log.error("graph_invocation_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_unavailable",
                "message": "Internal error during applicant communication generation.",
                "retry_after": 30,
            },
            headers={"Retry-After": "30"},
        ) from exc

    # --- Hard 503 gate: primary LLM unavailable ---
    if final_state.get("error") in ("PRIMARY_UNAVAILABLE", "ALL_PROVIDERS_UNAVAILABLE"):
        bound_log.error(
            "applicant_503_gate_triggered",
            error=final_state.get("error"),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_unavailable",
                "message": (
                    "Primary inference provider unavailable. Applicant communication "
                    "cannot be generated without GPT-4.1. Please retry."
                ),
                "retry_after": 30,
            },
            headers={"Retry-After": "30"},
        )

    # --- Other graph errors ---
    if final_state.get("error"):
        bound_log.error("graph_error", error=final_state.get("error"))
        raise HTTPException(
            status_code=503,
            detail={
                "error": final_state["error"],
                "message": "Failed to generate applicant communication.",
                "retry_after": 30,
            },
            headers={"Retry-After": "30"},
        )

    # --- ECOA/FCRA compliance gate ---
    if not final_state.get("compliance_passed", False):
        flags = final_state.get("compliance_flags", [])
        bound_log.error("compliance_failure", flags=flags)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "compliance_failure",
                "flags": flags,
                "message": (
                    "Generated communication failed ECOA/FCRA validation. "
                    "Review compliance flags."
                ),
            },
        )

    final_output: dict = final_state.get("final_output", {})

    return ApplicantCommunicationResponse(
        session_id=final_state.get("session_id", session_id),
        subject_line=final_output.get("subject_line", ""),
        body=final_output.get(
            "narrative",
            final_state.get("grounded_narrative", final_state.get("raw_llm_output", "")),
        ),
        adverse_action_notice=final_output.get("adverse_action_notice", {}),
        compliance_validated=final_state.get("compliance_passed", False),
        citations=final_state.get("citations", []),
        confidence_score=final_state.get("confidence_score", 0.0),
    )
