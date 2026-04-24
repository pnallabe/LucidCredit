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
    # Inject the error code before routing to error_node
    state["error"] = "INSUFFICIENT_RETRIEVAL"
    return "error"


def _route_after_reason(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    return "citation_enforcer"


def _route_after_confidence(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("grounding_passed", False):
        state["error"] = "INSUFFICIENT_GROUNDING"
        return "error"
    if state.get("audience") == "applicant":
        return "compliance_check"
    return "format_output"


def _route_after_compliance(state: AgentState) -> str:
    if not state.get("compliance_passed", True):
        state["error"] = "COMPLIANCE_FAILED"
        return "error"
    return "format_output"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

_graph = None


async def compile_graph():
    """
    Compile the LangGraph state machine with a Redis checkpointer for session
    continuity across requests.

    Returns a ``CompiledGraph`` instance.
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

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

    # Redis checkpointer
    checkpointer = AsyncRedisSaver.from_conn_string(settings.redis_url)

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
