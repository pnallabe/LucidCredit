"""
backend/app/api/v1/briefing.py
================================
FastAPI router for the analyst briefing endpoints.

POST /v1/briefing/generate         — generate a portfolio/segment briefing
POST /v1/briefing/evaluate-flash   — run RAGAS gate and enable Gemini 2.0 Flash if it passes

Flash A/B gate policy:
  Gemini 2.0 Flash is never activated by a manual toggle. It can only be enabled
  by POST /v1/briefing/evaluate-flash, which runs the full RAGAS evaluation harness
  and sets the flag only if ALL thresholds pass. The gate state is persisted in Redis
  so it survives process restarts. Closing the gate requires setting the Redis key to
  "false" — there is no UI toggle.
"""
from __future__ import annotations

import uuid as _uuid
from typing import Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from app.agent.graph import get_graph
from app.config import get_settings

log = structlog.get_logger(__name__)

router = APIRouter()

# Simple API-key header for the admin evaluate-flash endpoint
_api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)


async def _require_admin_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Dependency that validates the analyst-admin API key."""
    settings = get_settings()
    # The admin key is stored in settings (sourced from env var API_KEY_SALT or a
    # dedicated ADMIN_API_KEY env var — falls back to a disabled state when empty).
    admin_key: str = getattr(settings, "admin_api_key", "") or ""
    if not admin_key:
        raise HTTPException(
            status_code=503,
            detail={"error": "admin_not_configured", "message": "Admin API key not set."},
        )
    if api_key != admin_key:
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden", "message": "Invalid admin API key."},
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

BriefingSection = Literal[
    "executive_summary",
    "risk_distribution",
    "key_drivers",
    "fairness_indicators",
    "recommendations",
]


class BriefingRequest(BaseModel):
    scope: Literal["segment", "portfolio", "cohort", "product"]
    filters: dict  # date_range, decision, employment_status, etc.
    sections: list[BriefingSection]
    audience_role: Literal["CRO", "analyst", "compliance_officer", "board"]


class BriefingResponse(BaseModel):
    session_id: UUID
    sections: dict[str, str]
    citations: list[dict]
    confidence_score: float
    provider_used: str
    sql_queries_executed: list[str]


class FlashGateResponse(BaseModel):
    gate_opened: bool
    scores: dict[str, float]


# ---------------------------------------------------------------------------
# POST /v1/briefing/generate
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=BriefingResponse)
async def briefing_generate(request: BriefingRequest) -> BriefingResponse:
    """
    Generate an analyst portfolio/segment briefing.

    Provider selection (Azure GPT-4.1 vs Gemini 2.0 Flash) is determined inside
    reason_node based on settings.briefing_use_flash. The Flash gate must have
    passed RAGAS thresholds before Flash can be used.
    """
    session_id = _uuid.uuid4()
    settings = get_settings()

    query_parts = [f"Generate a {request.scope} briefing"]
    if request.filters:
        query_parts.append(f"with filters: {request.filters}")
    query_parts.append(f"Sections: {', '.join(request.sections)}.")
    query_parts.append(f"Audience: {request.audience_role}.")
    query = " ".join(query_parts)

    initial_state = {
        "session_id": session_id,
        "query": query,
        "intent": "portfolio_brief",
        "audience": "analyst",
        "context_payload": {
            "scope": request.scope,
            "filters": request.filters,
            "sections": list(request.sections),
            "audience_role": request.audience_role,
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
        log.error("briefing_graph_failed", session_id=str(session_id), error=str(exc))
        raise HTTPException(
            status_code=503,
            detail={"error": "inference_unavailable", "message": str(exc), "retry_after": 30},
            headers={"Retry-After": "30"},
        ) from exc

    if final_state.get("error") in ("PRIMARY_UNAVAILABLE", "ALL_PROVIDERS_UNAVAILABLE"):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_unavailable",
                "message": "LLM provider unavailable for briefing generation.",
                "retry_after": 30,
            },
            headers={"Retry-After": "30"},
        )

    if final_state.get("error"):
        raise HTTPException(
            status_code=503,
            detail={"error": final_state["error"], "retry_after": 30},
            headers={"Retry-After": "30"},
        )

    # Build section map from the narrative (stub; Sprint 3 citation_enforcer fills this)
    narrative = final_state.get("grounded_narrative") or final_state.get("raw_llm_output", "")
    sections_map: dict[str, str] = {s: narrative for s in request.sections}

    final_output = final_state.get("final_output", {})
    if isinstance(final_output.get("sections"), dict):
        sections_map = final_output["sections"]

    return BriefingResponse(
        session_id=final_state.get("session_id", session_id),
        sections=sections_map,
        citations=final_state.get("citations", []),
        confidence_score=final_state.get("confidence_score", 0.0),
        provider_used=final_state.get("provider_used", ""),
        sql_queries_executed=final_output.get("sql_queries_executed", []),
    )


# ---------------------------------------------------------------------------
# POST /v1/briefing/evaluate-flash  (admin — protected by API key)
# ---------------------------------------------------------------------------


@router.post(
    "/evaluate-flash",
    response_model=FlashGateResponse,
    dependencies=[Depends(_require_admin_key)],
)
async def evaluate_flash_gate() -> FlashGateResponse:
    """
    Run the RAGAS evaluation harness against the golden dataset using the
    Gemini 2.0 Flash provider. If ALL thresholds pass, activate the Flash
    gate for briefing sessions and persist the flag in Redis.

    Thresholds (briefing session):
        faithfulness     >= 0.80
        answer_relevancy >= 0.75
        context_recall   >= 0.70

    The gate can only be closed by setting the Redis key
    ``lucidcredit:feature_flags:briefing_flash`` to ``"false"``.
    """
    settings = get_settings()

    # Build the Vertex Flash LLM for evaluation
    try:
        from langchain_google_vertexai import ChatVertexAI

        flash_llm = ChatVertexAI(
            model_name=settings.vertex_model_batch_briefing,
            project=settings.google_project_id,
            location=settings.google_location,
            temperature=0.0,
        )

        from langchain_openai import AzureOpenAIEmbeddings

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment_embedding,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key,  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "provider_init_failed",
                "message": f"Failed to initialise Vertex Flash client: {exc}",
            },
        ) from exc

    # Run RAGAS harness for briefing session type
    try:
        from tests.eval.ragas_harness import run_ragas_evaluation

        scores = run_ragas_evaluation(
            session_type="briefing",
            llm=flash_llm,
            embeddings=embeddings,
            provider_label="vertex_flash20",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "ragas_evaluation_failed", "message": str(exc)},
        ) from exc

    briefing_thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_recall": 0.70,
    }

    gate_opened = all(
        scores.get(metric, 0.0) >= threshold
        for metric, threshold in briefing_thresholds.items()
    )

    if gate_opened:
        # Activate in-memory for this process
        settings.briefing_use_flash = True

        # Persist to Redis so the gate survives restarts
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await r.set("lucidcredit:feature_flags:briefing_flash", "true")
            await r.aclose()
            log.info("briefing_flash_gate_opened", scores=scores)
        except Exception as exc:
            log.warning("briefing_flash_gate_redis_write_failed", error=str(exc))

    return FlashGateResponse(gate_opened=gate_opened, scores=scores)
