"""
backend/app/agent/nodes.py
============================
LangGraph node functions for LucidCredit's corrective RAG agent.

Each node is a pure async function: ``async def *_node(state: AgentState) -> dict``.
No node writes to the database — persistence is handled by persist_session_node
at the END of the graph after all logic is complete.

The only place in the codebase that invokes an LLM is ``reason_node``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

import openai
import structlog

from app.agent.state import AgentState, RetrievedChunk
from app.config import get_settings
from app.llm.provider import FallbackNotAvailableError, get_chat_client, get_fallback_client

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_type_for(
    audience: str, intent: str
) -> Literal["analyst", "applicant", "briefing"]:
    if audience == "applicant":
        return "applicant"
    if intent == "portfolio_brief":
        return "briefing"
    return "analyst"


# ---------------------------------------------------------------------------
# reason_node — the ONLY LLM call node
# ---------------------------------------------------------------------------

async def reason_node(state: AgentState) -> dict:
    """
    Invoke the primary LLM over graded_chunks + rendered_prompt.

    Fallback policy:
      - applicant: no fallback — return PRIMARY_UNAVAILABLE and let the graph
        route to a 503 response via error_node.
      - analyst/briefing: retry once with Vertex AI fallback on RateLimitError
        or 5xx APIStatusError; ALL_PROVIDERS_UNAVAILABLE if both fail.

    Logs the provider used at INFO level for SR 11-7 audit trail.
    """
    settings = get_settings()
    audience: str = state.get("audience", "analyst")
    intent: str = state.get("intent", "analyst_query")
    session_type = _session_type_for(audience, intent)

    # Render the system prompt from graded context + context_payload
    from app.agent.prompt_renderer import render_prompt as _render_prompt
    graded_chunks = state.get("graded_chunks", [])
    context_payload: dict = state.get("context_payload", {})
    prompt: str = _render_prompt(
        session_type,
        state.get("query", ""),
        graded_chunks,
        context_payload,
    )

    # --- Check circuit breaker before calling primary ---
    try:
        from app.llm.circuit_breaker import is_open as cb_is_open

        azure_open = await cb_is_open("azure")
    except Exception:
        azure_open = False

    primary_failed = azure_open  # skip primary if circuit is open

    raw_output: str | None = None
    provider_used: str | None = None

    # --- Primary attempt ---
    if not primary_failed:
        try:
            client = get_chat_client(session_type)
            result = await client.ainvoke(prompt)
            raw_output = result.content if hasattr(result, "content") else str(result)

            deployment = (
                settings.azure_openai_deployment_applicant
                if session_type == "applicant"
                else settings.azure_openai_deployment_analyst
            )
            provider = "openai_direct" if settings.llm_provider_mode == "openai_direct" else "azure"
            provider_used = settings.model_version_hash(provider, deployment)

            # Record success in circuit breaker
            try:
                from app.llm.circuit_breaker import record_success as cb_success
                await cb_success("azure")
            except Exception:
                pass

            log.info(
                "llm_call_success",
                provider=provider_used,
                session_type=session_type,
                session_id=str(state.get("session_id", "")),
            )

        except (openai.RateLimitError, openai.APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(exc, openai.RateLimitError) or (status and status >= 500):
                primary_failed = True
                # Record failure in circuit breaker
                try:
                    from app.llm.circuit_breaker import record_failure as cb_fail
                    await cb_fail("azure")
                except Exception:
                    pass
                log.warning(
                    "llm_primary_failed",
                    error=str(exc),
                    session_type=session_type,
                    session_id=str(state.get("session_id", "")),
                )
            else:
                raise

    # --- Applicant: no fallback ---
    if primary_failed and session_type == "applicant":
        log.error(
            "applicant_primary_unavailable",
            session_id=str(state.get("session_id", "")),
        )
        return {"error": "PRIMARY_UNAVAILABLE"}

    # --- Analyst/briefing: try Vertex fallback ---
    if primary_failed and session_type in ("analyst", "briefing"):
        try:
            fallback_client = get_fallback_client(session_type)
            result = await fallback_client.ainvoke(prompt)
            raw_output = result.content if hasattr(result, "content") else str(result)

            fallback_model = (
                settings.vertex_model_batch_briefing
                if (session_type == "briefing" and settings.briefing_use_flash)
                else settings.vertex_model_analyst_fallback
            )
            provider_used = settings.model_version_hash("vertex", fallback_model)

            # Record success for vertex
            try:
                from app.llm.circuit_breaker import record_success as cb_success
                await cb_success("vertex")
            except Exception:
                pass

            log.info(
                "llm_fallback_success",
                provider=provider_used,
                session_type=session_type,
                session_id=str(state.get("session_id", "")),
            )

        except (FallbackNotAvailableError, Exception) as exc:
            try:
                from app.llm.circuit_breaker import record_failure as cb_fail
                await cb_fail("vertex")
            except Exception:
                pass
            log.error(
                "llm_all_providers_failed",
                error=str(exc),
                session_id=str(state.get("session_id", "")),
            )
            return {"error": "ALL_PROVIDERS_UNAVAILABLE"}

    return {
        "raw_llm_output": raw_output or "",
        "provider_used": provider_used or "",
        "rendered_prompt": prompt,
        "error": None,
    }


# ---------------------------------------------------------------------------
# parse_intent_node
# ---------------------------------------------------------------------------

async def parse_intent_node(state: AgentState) -> dict:
    """
    Classify the user query into one of the four intent values using a
    lightweight prompt against the primary LLM.

    No fallback — if primary is unavailable, fail fast.
    """
    settings = get_settings()
    query: str = state.get("query", "")

    classification_prompt = (
        "Classify the following credit analyst query into exactly one of these categories:\n"
        "  explain_decision  — request for explanation of a specific credit decision\n"
        "  analyst_query     — analytical or data question from an internal analyst\n"
        "  applicant_comms   — communication to be sent to a loan applicant\n"
        "  portfolio_brief   — portfolio-level briefing or summary request\n\n"
        f"Query: {query}\n\n"
        "Respond with only the category name, nothing else."
    )

    try:
        client = get_chat_client("analyst")
        result = await client.ainvoke(classification_prompt)
        raw = (result.content if hasattr(result, "content") else str(result)).strip().lower()
    except (openai.RateLimitError, openai.APIStatusError):
        return {"error": "PRIMARY_UNAVAILABLE"}

    valid_intents = {"explain_decision", "analyst_query", "applicant_comms", "portfolio_brief"}
    intent = raw if raw in valid_intents else "analyst_query"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# retrieve_node — parallel fan-out to vector + CRP API + SQL tools
# ---------------------------------------------------------------------------

async def retrieve_node(state: AgentState) -> dict:
    """
    Parallel fan-out to three retrieval sources:
      1. Vector search — hybrid dense + BM25 over policy_docs
      2. CRP API — decision explanation + audit (when decision_id is present)
      3. SQL — analyst data queries embedded in context_payload

    All errors are swallowed so downstream grading can handle empty results.
    """
    from app.rag.retriever import hybrid_retrieve
    from app.agent.tools.crp_api_tool import fetch_decision_context, fetch_portfolio_metrics
    from app.agent.tools.sql_tool import execute_analyst_query, sql_result_to_chunk, SqlSecurityError, SqlExecutionError

    query: str = state.get("query", "")
    intent: str = state.get("intent", "analyst_query")
    context: dict = state.get("context_payload", {})

    async def _vector() -> list[RetrievedChunk]:
        return await hybrid_retrieve(query, top_k=8)

    async def _crp() -> list[RetrievedChunk]:
        decision_id = context.get("decision_id")
        if decision_id:
            return await fetch_decision_context(str(decision_id))
        if intent == "portfolio_brief":
            return await fetch_portfolio_metrics()
        return []

    async def _sql() -> list[RetrievedChunk]:
        sql_query: str = context.get("sql_query", "")
        if not sql_query:
            return []
        try:
            result = await execute_analyst_query(sql_query)
            return [sql_result_to_chunk(result)]
        except (SqlSecurityError, SqlExecutionError) as exc:
            log.warning("retrieve_node.sql_error error=%s", exc)
            return []

    vector_chunks, crp_chunks, sql_chunks = await asyncio.gather(
        _vector(), _crp(), _sql()
    )

    all_chunks: list[RetrievedChunk] = vector_chunks + crp_chunks + sql_chunks
    log.info(
        "retrieve_node vector=%d crp=%d sql=%d total=%d",
        len(vector_chunks), len(crp_chunks), len(sql_chunks), len(all_chunks),
    )
    return {"retrieved_chunks": all_chunks}


# ---------------------------------------------------------------------------
# grade_documents_node
# ---------------------------------------------------------------------------

async def grade_documents_node(state: AgentState) -> dict:
    """
    Grade each retrieved chunk as RELEVANT / IRRELEVANT / AMBIGUOUS using a
    single batched LLM call.

    Sets retrieval_sufficient = True when >= 3 RELEVANT chunks are found.
    """
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    if not chunks:
        return {"graded_chunks": [], "retrieval_sufficient": False}

    chunk_summaries = "\n".join(
        f"[{i}] {c['content'][:200]}" for i, c in enumerate(chunks)
    )
    query = state.get("query", "")

    grading_prompt = (
        f"You are grading retrieved context chunks for relevance to the following query:\n"
        f"Query: {query}\n\n"
        f"For each chunk below, output exactly one line in the format:\n"
        f"<index>: RELEVANT | IRRELEVANT | AMBIGUOUS\n\n"
        f"Chunks:\n{chunk_summaries}\n"
    )

    try:
        client = get_chat_client("analyst")
        result = await client.ainvoke(grading_prompt)
        raw = result.content if hasattr(result, "content") else str(result)
    except Exception:
        # On error, mark all as AMBIGUOUS and continue
        graded = [{**c, "relevance": "AMBIGUOUS"} for c in chunks]
        return {
            "graded_chunks": graded,
            "retrieval_sufficient": False,
        }

    # Parse the LLM's grading response
    relevance_map: dict[int, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            try:
                idx = int(parts[0].strip())
                label = parts[1].strip().upper()
                if label in ("RELEVANT", "IRRELEVANT", "AMBIGUOUS"):
                    relevance_map[idx] = label
            except ValueError:
                continue

    graded: list[RetrievedChunk] = []
    for i, chunk in enumerate(chunks):
        relevance = relevance_map.get(i, "AMBIGUOUS")
        graded.append({**chunk, "relevance": relevance})  # type: ignore[misc]

    relevant_count = sum(1 for c in graded if c["relevance"] == "RELEVANT")
    return {
        "graded_chunks": graded,
        "retrieval_sufficient": relevant_count >= 3,
    }


# ---------------------------------------------------------------------------
# citation_enforcer_node (stub)
# ---------------------------------------------------------------------------

async def citation_enforcer_node(state: AgentState) -> dict:
    """
    Ground every sentence in the raw LLM output against the retrieved chunks.

    Delegates to CitationEnforcer which embeds each sentence and checks
    cosine similarity against chunk embeddings.  Sentences below
    settings.min_citation_similarity (default 0.85) or marked [UNVERIFIED]
    by the model are suppressed and logged to suppressed_claims.
    """
    from app.agent.grounding import CitationEnforcer

    raw_output: str = state.get("raw_llm_output", "")
    graded_chunks = state.get("graded_chunks", [])

    if not raw_output:
        return {
            "grounded_narrative": "",
            "citations": [],
            "suppressed_claims": [],
        }

    enforcer = CitationEnforcer()
    result = await enforcer.enforce(raw_output, graded_chunks)
    return result


# ---------------------------------------------------------------------------
# confidence_score_node (stub)
# ---------------------------------------------------------------------------

async def confidence_score_node(state: AgentState) -> dict:
    """
    Compute calibrated confidence_score in [0.0, 1.0] from three signals:
      * retrieval_recall  — fraction of graded_chunks marked RELEVANT
      * citation_rate     — fraction of narrative claims that were grounded
      * unverified_rate   — fraction of LLM sentences marked [UNVERIFIED]

    Returns confidence_score and grounding_passed flag used by
    _route_after_confidence to decide whether to proceed or error.
    """
    from app.agent.grounding import ConfidenceScorer

    scorer = ConfidenceScorer()
    return scorer.score(state)


# ---------------------------------------------------------------------------
# compliance_check_node (stub)
# ---------------------------------------------------------------------------

async def compliance_check_node(state: AgentState) -> dict:
    """Stub: ECOA/FCRA validation for applicant outputs (Sprint 3)."""
    return {"compliance_passed": True, "compliance_flags": []}


# ---------------------------------------------------------------------------
# format_output_node (stub)
# ---------------------------------------------------------------------------

async def format_output_node(state: AgentState) -> dict:
    """
    Render the final audience-appropriate response payload.

    Analyst output includes full technical metadata (SHAP evidence, model ID,
    provider details, counterfactual).  Applicant output strips technical
    internals and exposes only consumer-facing fields required by ECOA/FCRA.
    """
    audience: str = state.get("audience", "analyst")
    session_id = str(state.get("session_id", ""))
    narrative: str = state.get("grounded_narrative") or state.get("raw_llm_output", "")
    citations = state.get("citations", [])
    confidence_score: float = state.get("confidence_score", 0.0)
    context_payload: dict = state.get("context_payload", {})
    compliance_flags = state.get("compliance_flags", [])

    if audience == "applicant":
        # Applicant output — consumer-facing, no technical internals
        final_output = {
            "session_id": session_id,
            "narrative": narrative,
            "citations": [
                {
                    "claim_text": c.get("claim_text", ""),
                    "source_type": c.get("source_type", ""),
                    "source_ref": c.get("source_ref", ""),
                    "confidence": c.get("confidence", 0.0),
                }
                for c in citations
            ],
            "confidence_score": confidence_score,
            "adverse_action_codes": context_payload.get("adverse_action_codes", []),
            "compliance_flags": compliance_flags,
            "audience": "applicant",
        }
    else:
        # Analyst / briefing output — full technical payload
        final_output = {
            "session_id": session_id,
            "narrative": narrative,
            "citations": citations,
            "confidence_score": confidence_score,
            "suppressed_claims": state.get("suppressed_claims", []),
            "counterfactual": context_payload.get("counterfactual"),
            "adverse_action_codes": context_payload.get("adverse_action_codes", []),
            "compliance_flags": compliance_flags,
            "audience": audience,
            "provider_used": state.get("provider_used", ""),
            "intent": state.get("intent", ""),
        }

    return {"final_output": final_output}


# ---------------------------------------------------------------------------
# persist_session_node (stub)
# ---------------------------------------------------------------------------

async def persist_session_node(state: AgentState) -> dict:
    """
    Persist the completed session to PostgreSQL.

    Writes one ``CopilotSession`` row and N ``Citation`` rows inside a single
    transaction.  Errors are caught and logged — a persistence failure must
    never cause the agent to return an error to the caller.
    """
    from app.db.session import AsyncSessionLocal
    from app.models import CopilotSession, Citation

    session_id = state.get("session_id")
    citations_list = state.get("citations", [])
    suppressed = state.get("suppressed_claims", [])
    context_payload: dict = state.get("context_payload", {})

    # Determine source_system from context_payload
    source_system: str = context_payload.get(
        "source", context_payload.get("source_system", "")
    )

    # provider_model and fallback flag from reason_node output
    provider_model: str = state.get("provider_used", "")
    provider_fallback_used: bool = (
        provider_model.startswith("vertex") or provider_model.startswith("gemini")
    )

    try:
        async with AsyncSessionLocal() as db:
            session_row = CopilotSession(
                session_id=session_id,
                query_text=state.get("query", ""),
                intent=state.get("intent", ""),
                audience=state.get("audience", "analyst"),
                retrieved_chunks=[
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "source_type": c.get("source_type", ""),
                        "source_ref": c.get("source_ref", ""),
                        "relevance": c.get("relevance", ""),
                    }
                    for c in state.get("graded_chunks", [])
                ],
                rendered_prompt=state.get("rendered_prompt", ""),
                raw_llm_output=state.get("raw_llm_output", ""),
                grounded_narrative=state.get("grounded_narrative", ""),
                confidence_score=state.get("confidence_score"),
                suppressed_claims=suppressed,
                compliance_flags=state.get("compliance_flags", []),
                source_system=source_system,
                user_id=context_payload.get("user_id"),
                provider_model=provider_model,
                provider_fallback_used=provider_fallback_used,
            )
            db.add(session_row)

            for c in citations_list:
                citation_row = Citation(
                    session_id=session_id,
                    claim_text=c.get("claim_text", ""),
                    source_type=c.get("source_type", ""),
                    source_ref=c.get("source_ref", ""),
                    similarity_score=c.get("similarity_score"),
                    confidence=c.get("confidence"),
                )
                db.add(citation_row)

            await db.commit()

        log.info(
            "persist_session.success",
            session_id=str(session_id),
            citation_count=len(citations_list),
        )
    except Exception as exc:  # pragma: no cover
        log.error(
            "persist_session.error",
            session_id=str(session_id),
            error=str(exc),
        )
        # Do not re-raise — persistence failure must not degrade the response.

    return {}


# ---------------------------------------------------------------------------
# error_node
# ---------------------------------------------------------------------------

async def error_node(state: AgentState) -> dict:
    """
    Terminal error node — formats a structured error response.
    Does NOT call any LLM.
    """
    error_code = state.get("error", "UNKNOWN_ERROR")
    return {
        "final_output": {
            "error": error_code,
            "session_id": str(state.get("session_id", "")),
            "retry_after": 30,
        }
    }
