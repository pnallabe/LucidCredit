"""
backend/app/api/v1/gateway.py
==============================
AgentHive Integration Gateway Adapter

Exposes two endpoints that the AgentHiveHQ proxy calls:

  GET  /v1/gateway/health   — liveness probe (no auth required)
  POST /v1/gateway/invoke   — unified invocation endpoint

The /invoke endpoint accepts AgentHiveHQ's standard InvokeRequest shape and
returns a NormalisedResponse so it can be displayed natively in the
AgentHiveHQ Lab → Integration Gateway UI.

Authentication
--------------
The proxy injects the shared secret as:
  Authorization: Bearer <LUCIDCREDIT_API_KEY>
  X-AgentHive-Key: <LUCIDCREDIT_API_KEY>

The key must match AGENTHIVE_INCOMING_KEY in LucidCredit's .env.

Session types (driven by inputs.session_type)
----------------------------------------------
  analyst   — analyst Q&A via the corrective-RAG graph
  applicant — applicant communication (decline/approve) via the agent graph
  briefing  — portfolio briefing narrative

Demo scenarios (when inputs.demo_mode=true or demo_mode=true in body)
----------------------------------------------------------------------
  analyst_decision_explanation — demo analyst Q&A about a high-risk decision
  applicant_decline_notice     — demo ECOA-compliant adverse action notice
  applicant_approve_notice     — demo approval communication
"""
from __future__ import annotations

import secrets
import time
import uuid as _uuid
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.agent.graph import get_graph
from app.config import get_settings

log = structlog.get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def _verify_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> None:
    """Validate the API key from Authorization header or X-AgentHive-Key header."""
    settings = get_settings()
    expected = settings.agenthive_incoming_key

    # If no key is configured, allow all (useful in development)
    if not expected:
        return

    token: str | None = None
    if credentials is not None:
        token = credentials.credentials
    if token is None:
        token = request.headers.get("X-AgentHive-Key")

    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "type": "https://lucidcredit.ai/errors/unauthorized",
                "title": "Unauthorized",
                "status": 401,
                "detail": "Invalid or missing API key.",
                "instance": "/v1/gateway/invoke",
            },
        )


# ---------------------------------------------------------------------------
# Gateway request / response schemas
# (Mirror AgentHiveHQ's InvokeRequest / NormalisedResponse contracts)
# ---------------------------------------------------------------------------


class GatewayInvokeRequest(BaseModel):
    """Exactly matches AgentHiveHQ integrations.schemas.InvokeRequest."""

    inputs: Dict[str, Any] = Field(..., description="Inputs forwarded from the AgentHiveHQ proxy")
    demo_mode: bool = Field(default=False)
    scenario_tag: Optional[str] = Field(default=None, max_length=64)


class AuditBlock(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_version: str = "lucidcredit:0.1.0"
    timestamp: str = ""
    latency_ms: float = 0.0


class ClarificationItem(BaseModel):
    """A single pending clarifying question surfaced to the caller."""
    id: str = ""
    question: str
    options: List[str] = Field(default_factory=list)


class NormalisedResponse(BaseModel):
    """Exactly matches AgentHiveHQ integrations.schemas.NormalisedResponse."""

    decision_id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    decision: str
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)
    explanations: Dict[str, Any] = Field(default_factory=dict)
    reason_codes: List[str] = Field(default_factory=list)
    audit: AuditBlock = Field(default_factory=AuditBlock)
    source: str = "lucidcredit-copilot"
    product_id: str = "lucidcredit-copilot-v1"
    raw_extra: Dict[str, Any] = Field(default_factory=dict)
    # Context ingestion — set when the agent needs user clarification before answering
    needs_clarification: bool = False
    clarification_items: List[ClarificationItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health probe (no auth)
# ---------------------------------------------------------------------------


@router.get("/health")
async def gateway_health() -> dict:
    """Liveness probe consumed by AgentHiveHQ health monitor."""
    return {"status": "ok", "product_id": "lucidcredit-copilot-v1"}


# ---------------------------------------------------------------------------
# Demo response factory
# ---------------------------------------------------------------------------

_DEMO_RESPONSES: Dict[str, NormalisedResponse] = {
    "analyst_decision_explanation": NormalisedResponse(
        decision="analyst_response",
        summary=(
            "Application APP-DEMO-001 was declined. "
            "The top driver was debt-to-income ratio (0.62), which exceeds the 0.45 guideline. "
            "Secondary factors: thin credit history (14 months) and recent hard inquiry surge (+3 in 90 days)."
        ),
        details={
            "application_id": "APP-DEMO-001",
            "decision": "reject",
            "confidence_score": 0.91,
            "top_features": {
                "dti_ratio": -0.41,
                "credit_history_months": -0.18,
                "hard_inquiries_90d": -0.12,
                "income": 0.09,
            },
            "model_score": 0.34,
            "risk_segment": "high_risk",
        },
        explanations={
            "dti_ratio": -0.41,
            "credit_history_months": -0.18,
            "hard_inquiries_90d": -0.12,
        },
        reason_codes=["HIGH_DTI", "THIN_FILE", "RECENT_INQUIRIES"],
        audit=AuditBlock(
            model_version="lucidcredit:0.1.0-demo",
            latency_ms=420.0,
        ),
        raw_extra={"demo": True, "scenario_tag": "analyst_decision_explanation"},
    ),
    "applicant_decline_notice": NormalisedResponse(
        decision="reject",
        summary=(
            "Your credit application (ref APP-DEMO-002) was not approved. "
            "The primary reason is a debt-to-income ratio that exceeds our current guidelines. "
            "You have the right to request the specific reasons within 60 days under the Equal Credit Opportunity Act."
        ),
        details={
            "application_id": "APP-DEMO-002",
            "communication_type": "decline",
            "channel": "portal",
            "compliance_validated": True,
            "ecoa_compliant": True,
            "fcra_compliant": True,
        },
        explanations={},
        reason_codes=["HIGH_DTI", "THIN_FILE"],
        audit=AuditBlock(
            model_version="lucidcredit:0.1.0-demo",
            latency_ms=380.0,
        ),
        raw_extra={"demo": True, "scenario_tag": "applicant_decline_notice"},
    ),
    "applicant_approve_notice": NormalisedResponse(
        decision="approve",
        summary=(
            "Congratulations! Your application (ref APP-DEMO-003) has been approved for a $12,000 credit line at 18.99% APR. "
            "Your strong repayment history and stable income were the primary factors. "
            "You can activate your account within 30 days."
        ),
        details={
            "application_id": "APP-DEMO-003",
            "communication_type": "approve",
            "credit_limit": 12000,
            "apr": 0.1899,
            "channel": "portal",
            "compliance_validated": True,
        },
        explanations={"income": 0.32, "payment_history": 0.28, "credit_utilization": 0.15},
        reason_codes=[],
        audit=AuditBlock(
            model_version="lucidcredit:0.1.0-demo",
            latency_ms=310.0,
        ),
        raw_extra={"demo": True, "scenario_tag": "applicant_approve_notice"},
    ),
}


# ---------------------------------------------------------------------------
# Invoke endpoint
# ---------------------------------------------------------------------------


@router.post("/invoke", response_model=NormalisedResponse)
async def gateway_invoke(
    body: GatewayInvokeRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> NormalisedResponse:
    """
    Unified invocation endpoint consumed by the AgentHiveHQ secure proxy.

    Expected inputs keys:
      session_type  : "analyst" | "applicant" | "briefing"  (required)
      query         : free-text question (analyst / briefing sessions)
      application_id: UUID string (applicant sessions)
      source        : "thinfile" | "credit_risk_platform" (applicant sessions)
      communication_type: "decline" | "approve" | "counteroffer" (applicant sessions)
      channel       : "portal" | "email" | "sms" | "letter" (applicant sessions)
      language      : ISO 639-1 language code (default "en")
    """
    _verify_key(request, credentials)

    inputs = body.inputs
    demo_mode: bool = body.demo_mode or bool(inputs.get("demo_mode", False))
    scenario_tag: str | None = body.scenario_tag or inputs.get("scenario_tag")  # type: ignore[assignment]

    bound_log = log.bind(
        demo_mode=demo_mode,
        scenario_tag=scenario_tag,
        session_type=inputs.get("session_type"),
    )

    # ── Demo fast-path ────────────────────────────────────────────────────────
    if demo_mode and scenario_tag:
        template = _DEMO_RESPONSES.get(scenario_tag)
        if template is not None:
            # Return a fresh copy with a new decision_id on every call so that
            # AgentHiveHQ's audit table (decisions.id PK) never gets a duplicate.
            return template.model_copy(update={"decision_id": str(_uuid.uuid4())})
        # Unknown demo scenario → fall through to live path with a safe default query

    # ── Route by session_type ─────────────────────────────────────────────────
    session_type: str = str(inputs.get("session_type", "analyst")).lower()

    t0 = time.monotonic()

    try:
        if session_type == "applicant":
            return await _invoke_applicant(inputs, bound_log, t0)
        elif session_type == "briefing":
            return await _invoke_briefing(inputs, bound_log, t0)
        else:
            # Default: analyst Q&A
            return await _invoke_analyst(inputs, bound_log, t0)

    except HTTPException:
        raise
    except Exception as exc:
        bound_log.error("gateway_invoke_error", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "type": "https://lucidcredit.ai/errors/internal",
                "title": "Internal Gateway Error",
                "status": 500,
                "detail": str(exc),
                "instance": "/v1/gateway/invoke",
            },
        )


# ---------------------------------------------------------------------------
# Session-type handlers
# ---------------------------------------------------------------------------


async def _invoke_analyst(
    inputs: Dict[str, Any],
    bound_log: Any,
    t0: float,
) -> NormalisedResponse:
    """Run the corrective-RAG analyst agent and normalise the result."""
    query: str = str(inputs.get("query", "Provide a summary of the latest credit portfolio performance."))
    session_id = _uuid.uuid4()

    initial_state = {
        "session_id": session_id,
        "query": query,
        "intent": "analyst_query",
        "audience": "analyst",
        "context_payload": {
            "data_scope": inputs.get("data_scope", []),
            "sql_query": inputs.get("sql_query", ""),
            # Context ingestion: pass caller-supplied clarification answers so
            # ask_analytics() can skip ambiguity detection on second+ turns.
            "clarifications": inputs.get("clarifications") or {},
        },
    }

    graph = await get_graph()
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = await graph.ainvoke(initial_state, config=config)

    latency_ms = (time.monotonic() - t0) * 1000
    final_output: dict = final_state.get("final_output") or {}
    error_code: str | None = final_output.get("error") or final_state.get("error")

    # final_output["narrative"] holds the grounded answer; fall back through
    # grounded_narrative → raw_llm_output for debugging when grounding fails.
    answer: str = (
        final_output.get("narrative")
        or final_state.get("grounded_narrative")
        or final_state.get("raw_llm_output")
        or ""
    )
    citations: list = final_output.get("citations") or final_state.get("citations") or []
    confidence: float = float(final_output.get("confidence_score") or final_state.get("confidence_score") or 0.0)
    suppressed: list = final_output.get("suppressed_claims") or final_state.get("suppressed_claims") or []
    provider_used: str = final_state.get("provider_used") or ""

    reason_codes: list[str] = []
    if error_code:
        reason_codes.append(error_code)
    elif confidence < 0.75:
        reason_codes.append("LOW_CONFIDENCE")
    if suppressed:
        reason_codes.append("CLAIMS_SUPPRESSED")

    model_ver = provider_used or get_settings().model_version_hash(
        "azure", get_settings().azure_openai_deployment_analyst
    )

    # --- Context ingestion: extract pending clarification questions if present ---
    raw_clarification_items: list = final_output.get("clarification_items") or []
    needs_clarification: bool = bool(final_output.get("needs_clarification", False))
    clarification_items = [
        ClarificationItem(
            id=item.get("id", ""),
            question=item.get("question", ""),
            options=item.get("options", []),
        )
        for item in raw_clarification_items
    ]

    # When clarification is needed, use the narrative as the summary
    # (it will be the agent's phrased follow-up question to the user).
    summary_text = (
        answer[:500]
        if answer
        else (f"Agent error: {error_code}" if error_code else "No answer generated.")
    )

    return NormalisedResponse(
        decision_id=str(session_id),
        decision="clarification_needed" if needs_clarification else "analyst_response",
        summary=summary_text,
        details={
            "full_answer": answer,
            "confidence_score": confidence,
            "intent": final_output.get("intent", ""),
        },
        explanations={
            c.get("source_ref", f"cite_{i}"): c.get("confidence", 0.0)
            for i, c in enumerate(citations[:10])
        },
        reason_codes=reason_codes,
        audit=AuditBlock(
            model_version=model_ver,
            timestamp=_now_iso(),
            latency_ms=round(latency_ms, 1),
        ),
        raw_extra={
            "session_id": str(session_id),
            "citations": citations,
            "suppressed_claims": suppressed,
            "provider_used": provider_used,
        },
        needs_clarification=needs_clarification,
        clarification_items=clarification_items,
    )


async def _invoke_applicant(
    inputs: Dict[str, Any],
    bound_log: Any,
    t0: float,
) -> NormalisedResponse:
    """Run the applicant communication agent and normalise the result."""
    application_id_raw = inputs.get("application_id", str(_uuid.uuid4()))
    try:
        application_id = UUID(str(application_id_raw))
    except ValueError:
        application_id = _uuid.uuid4()

    source: Literal["thinfile", "credit_risk_platform"] = inputs.get("source", "credit_risk_platform")  # type: ignore[assignment]
    communication_type: Literal["decline", "approve", "counteroffer", "incomplete"] = inputs.get("communication_type", "decline")  # type: ignore[assignment]
    channel: Literal["email", "sms", "portal", "letter"] = inputs.get("channel", "portal")  # type: ignore[assignment]
    language: str = str(inputs.get("language", "en"))
    session_id = _uuid.uuid4()

    initial_state = {
        "session_id": session_id,
        "query": (
            f"Generate a {communication_type} communication for application {application_id} "
            f"from {source} via {channel} channel in {language}."
        ),
        "intent": "applicant_communication",
        "audience": "applicant",
        "context_payload": {
            "application_id": str(application_id),
            "source": source,
            "communication_type": communication_type,
            "channel": channel,
            "language": language,
        },
    }

    graph = await get_graph()
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = await graph.ainvoke(initial_state, config=config)

    latency_ms = (time.monotonic() - t0) * 1000
    final_output: dict = final_state.get("final_output") or {}
    error_code: str | None = final_output.get("error") or final_state.get("error")

    body_text: str = (
        final_output.get("narrative")
        or final_state.get("grounded_narrative")
        or final_state.get("raw_llm_output")
        or ""
    )
    compliance_validated: bool = bool(final_state.get("compliance_passed", not bool(error_code)))
    compliance_flags: list = final_output.get("compliance_flags") or final_state.get("compliance_flags") or []
    citations: list = final_output.get("citations") or final_state.get("citations") or []
    provider_used: str = final_state.get("provider_used") or ""

    decision_map = {"decline": "reject", "approve": "approve", "counteroffer": "review", "incomplete": "review"}
    decision = decision_map.get(communication_type, "review")

    reason_codes: list[str] = [c.get("code", str(c)) if isinstance(c, dict) else str(c) for c in compliance_flags]
    if error_code and not reason_codes:
        reason_codes = [error_code]

    model_ver = provider_used or get_settings().model_version_hash(
        "azure", get_settings().azure_openai_deployment_applicant
    )

    return NormalisedResponse(
        decision_id=str(session_id),
        decision=decision,
        summary=body_text[:500] if body_text else (f"Agent error: {error_code}" if error_code else "No communication generated."),
        details={
            "application_id": str(application_id),
            "full_body": body_text,
            "compliance_validated": compliance_validated,
            "compliance_flags": compliance_flags,
            "channel": channel,
            "communication_type": communication_type,
        },
        explanations={},
        reason_codes=reason_codes,
        audit=AuditBlock(
            model_version=model_ver,
            timestamp=_now_iso(),
            latency_ms=round(latency_ms, 1),
        ),
        raw_extra={
            "session_id": str(session_id),
            "citations": citations,
            "compliance_validated": compliance_validated,
            "provider_used": provider_used,
        },
    )


async def _invoke_briefing(
    inputs: Dict[str, Any],
    bound_log: Any,
    t0: float,
) -> NormalisedResponse:
    """Run the portfolio briefing agent and normalise the result."""
    query: str = str(
        inputs.get("query", "Generate a Q1 2026 credit portfolio executive briefing.")
    )
    session_id = _uuid.uuid4()

    initial_state = {
        "session_id": session_id,
        "query": query,
        "intent": "portfolio_briefing",
        "audience": "analyst",
        "context_payload": {
            "data_scope": inputs.get("data_scope", []),
            "period": inputs.get("period", "Q1 2026"),
        },
    }

    graph = await get_graph()
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = await graph.ainvoke(initial_state, config=config)

    latency_ms = (time.monotonic() - t0) * 1000
    final_output: dict = final_state.get("final_output") or {}
    error_code: str | None = final_output.get("error") or final_state.get("error")

    narrative: str = (
        final_output.get("narrative")
        or final_state.get("grounded_narrative")
        or final_state.get("raw_llm_output")
        or ""
    )
    citations: list = final_output.get("citations") or final_state.get("citations") or []
    confidence: float = float(final_output.get("confidence_score") or final_state.get("confidence_score") or 0.0)
    provider_used: str = final_state.get("provider_used") or ""

    reason_codes: list[str] = [error_code] if error_code else (["LOW_CONFIDENCE"] if confidence < 0.75 else [])
    model_ver = provider_used or get_settings().model_version_hash(
        "azure", get_settings().azure_openai_deployment_analyst
    )

    return NormalisedResponse(
        decision_id=str(session_id),
        decision="briefing_generated",
        summary=narrative[:500] if narrative else (f"Agent error: {error_code}" if error_code else "No briefing generated."),
        details={
            "full_narrative": narrative,
            "period": inputs.get("period", "Q1 2026"),
            "confidence_score": confidence,
        },
        explanations={
            c.get("source_ref", f"cite_{i}"): c.get("confidence", 0.0)
            for i, c in enumerate(citations[:10])
        },
        reason_codes=reason_codes,
        audit=AuditBlock(
            model_version=model_ver,
            timestamp=_now_iso(),
            latency_ms=round(latency_ms, 1),
        ),
        raw_extra={
            "session_id": str(session_id),
            "citations": citations,
            "provider_used": provider_used,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()
