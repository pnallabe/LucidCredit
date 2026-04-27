"""
backend/app/agent/graph.py
============================
LangGraph state machine for LucidCredit's corrective RAG agent.

Graph topology:
    START
      → parse_intent
      → retrieve
      → grade_documents
      → [retrieval_sufficient?]
           YES → reason
           NO  → error ("INSUFFICIENT_RETRIEVAL")
      → [error after reason?]
           YES → error
           NO  → citation_enforcer
      → confidence_score
      → [grounding_passed?]
           YES + applicant  → compliance_check
           YES + analyst    → format_output
           NO               → error ("INSUFFICIENT_GROUNDING")
      → [compliance_passed?]
           YES → format_output
           NO  → error ("COMPLIANCE_FAILED")
      → persist_session
      → END

Call ``get_graph()`` to obtain the compiled, Redis-checkpointed graph.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.nodes import (
    citation_enforcer_node,
    compliance_check_node,
    confidence_score_node,
    error_node,
    format_output_node,
    grade_documents_node,
    parse_intent_node,
    persist_session_node,
    reason_node,
    retrieve_node,
)
from app.config import get_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _route_after_grade(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    if state.get("retrieval_sufficient"):
        return "reason"
    return "error"


def _route_after_reason(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    return "citation_enforcer"


def _route_after_confidence(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("grounding_passed", False):
        return "error"
    if state.get("audience") == "applicant":
        return "compliance_check"
    return "format_output"


def _route_after_compliance(state: AgentState) -> str:
    if not state.get("compliance_passed", True):
        return "error"
    return "format_output"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

_graph = None


async def compile_graph():
    """
    Compile the LangGraph state machine with a Redis checkpointer for session
    continuity across requests.  Falls back to an in-memory checkpointer when
    the Redis package is not installed (local dev without Redis).

    Returns a ``CompiledGraph`` instance.
    """
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver as _AsyncRedisSaver
        _redis_available = True
    except ImportError:
        _redis_available = False

    settings = get_settings()

    # Build the state graph
    builder: StateGraph = StateGraph(AgentState)

    # Register nodes (graph node name = function name without "_node" suffix)
    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("reason", reason_node)
    builder.add_node("citation_enforcer", citation_enforcer_node)
    builder.add_node("confidence_score", confidence_score_node)
    builder.add_node("compliance_check", compliance_check_node)
    builder.add_node("format_output", format_output_node)
    builder.add_node("persist_session", persist_session_node)
    builder.add_node("error", error_node)

    # Linear edges
    builder.add_edge(START, "parse_intent")
    builder.add_edge("parse_intent", "retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # Conditional: retrieval sufficient?
    builder.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {"reason": "reason", "error": "error"},
    )

    # Conditional: error after reason?
    builder.add_conditional_edges(
        "reason",
        _route_after_reason,
        {"citation_enforcer": "citation_enforcer", "error": "error"},
    )

    builder.add_edge("citation_enforcer", "confidence_score")

    # Conditional: grounding passed + audience routing
    builder.add_conditional_edges(
        "confidence_score",
        _route_after_confidence,
        {
            "compliance_check": "compliance_check",
            "format_output": "format_output",
            "error": "error",
        },
    )

    # Conditional: compliance passed?
    builder.add_conditional_edges(
        "compliance_check",
        _route_after_compliance,
        {"format_output": "format_output", "error": "error"},
    )

    builder.add_edge("format_output", "persist_session")
    builder.add_edge("persist_session", END)
    builder.add_edge("error", END)

    # Checkpointer: Redis in production, in-memory in local dev
    checkpointer = None
    if _redis_available and settings.redis_url:
        try:
            # langgraph-checkpoint-redis >= 0.1.x returns an async context manager
            # from from_conn_string(); enter it and keep it open for the process lifetime.
            _cm = _AsyncRedisSaver.from_conn_string(settings.redis_url)
            checkpointer = await _cm.__aenter__()
            log.info("compile_graph: Redis checkpointer connected (%s)", settings.redis_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("compile_graph: Redis checkpointer failed (%s) — running without persistence", exc)
            checkpointer = None

    compiled = builder.compile(checkpointer=checkpointer)
    return compiled


async def get_graph():
    """
    Return the compiled graph, initialising it on first call.
    The compiled graph is cached as a module-level variable.
    """
    global _graph
    if _graph is None:
        _graph = await compile_graph()
    return _graph
