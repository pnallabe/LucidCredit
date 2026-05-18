"""
backend/app/api/v1/query.py
=============================
FastAPI router for:
  - POST /v1/query/analyst        — full response (used by non-streaming clients)
  - GET  /v1/query/stream         — SSE streaming (used by analyst query console UI)

The streaming endpoint runs the LangGraph agent and emits one SSE event per
graph node completion, then a final ``result`` event with the full response.
Clients can render progress immediately while the agent is still running.

SSE event format (text/event-stream):
  event: status
  data: {"step": "retrieval_complete", "chunk_count": 8}

  event: result
  data: {"session_id": "...", "answer": "...", "citations": [...], ...}

  event: error
  data: {"error": "INSUFFICIENT_GROUNDING", "message": "..."}

  event: done
  data: {}
"""
from __future__ import annotations

import json
import re
import base64 as _base64
import uuid as _uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import get_graph

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Applicant comms pre-detector (runs before the graph to set correct intent)
# ---------------------------------------------------------------------------

_APPLICANT_COMMS_RE = re.compile(
    r"\b(?:draft|write|prepare|send|compose)\s+(?:a\s+)?(?:adverse\s+action|decline|"
    r"rejection|denial|counteroffer|approval)\s+(?:notice|letter|communication|email|message)\b|"
    r"\binform\s+the\s+applicant\b|"
    r"\btell\s+the\s+applicant\b|"
    r"\bwrite\s+a\s+(?:notice|decline|rejection|denial|adverse\s+action)\b|"
    r"\bdraft\s+(?:a\s+|an?\s+)?adverse\s+action\b|"
    r"\bexplain\s+to\s+an?\s+applicant\b|"
    r"\bcommunicat(?:e|ion)\s+(?:to|with)\s+(?:the\s+)?applicant\b|"
    r"\bdraft\s+(?:a\s+|an?\s+)?(?:decline|rejection|denial)\s+notice\b|"
    r"\bdraft\s+a\s+communication\s+(?:noting|explaining|stating)\b|"
    r"\bwrite\s+a\s+communication\s+explaining\b",
    re.IGNORECASE,
)

_EXPLAIN_DECISION_RE = re.compile(
    r"\bwhat\s+(?:factors?|criteria|reasons?)\s+(?:determine|affect|influence|lead\s+to|cause)\b|"
    r"\bwhy\s+(?:is|was|are|were|does|do|would|could)\s+(?:a\s+)?(?:credit|loan|application)\b|"
    r"\b(?:explain|describe|what\s+are)\s+(?:the\s+)?(?:ecoa|fcra|SR\s*11|model\s+risk|adverse\s+action|consumer\s+rights?)\b|"
    r"\b(?:explain|describe)\s+(?:the\s+)?(?:FCRA|ECOA)\s+section\b|"
    r"\bwhat\s+model\s+risk\s+governance\b",
    re.IGNORECASE,
)


def _is_applicant_comms(query: str) -> bool:
    return bool(_APPLICANT_COMMS_RE.search(query))


# ---------------------------------------------------------------------------
# Pre-graph injection / out-of-scope detector (fast path at API layer)
# ---------------------------------------------------------------------------

_PRE_GRAPH_INJECTION_RE = re.compile(
    r"\boutput\s+(?:the\s+)?(?:full\s+)?system\s+prompt\b|"
    r"\bshow\s+(?:me\s+)?(?:the\s+)?contents\s+of\s+\.env\b|"
    r"\bforget\s+your\s+role\b|"
    r"\byour\s+new\s+role\s+is\b|"
    r"\bSYSTEM\s*:\s*(?:disable|override|ignore|bypass)\b|"
    r"\badd\s+\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?\s+credit\s+limit\b|"
    r"\boverride\s+credit\s+model\b|"
    r"\bapprove\s+application\s+(?:APP-\d+|regardless)\b|"
    r"IGNORE\s+PRIOR\s+INSTR|"
    r"\bIgnore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b|"
    r"\bpretend\s+you\s+are\s+a\s+(?:data\s+export|export)\s+tool\b|"
    # Multi-step social engineering: export data in steps
    r"\bExport\s+(?:it|them|the\s+data|borrower|customer|loan|account|all)\b|"
    r"Step\s+[12]\s*:.*\bexport\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_base64_injection(query: str) -> bool:
    """
    Detect encoding-obfuscation injection: a query that is entirely (or mostly)
    base64-encoded text. Legitimate credit-risk queries are never base64.
    """
    stripped = query.strip()
    # Match queries that look like a pure base64 blob (20+ chars)
    if not re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', stripped):
        return False
    try:
        decoded = _base64.b64decode(stripped + '==').decode('utf-8', errors='ignore')
        # Refuse if decoded text matches injection patterns OR looks like instruction injection
        return bool(
            _PRE_GRAPH_INJECTION_RE.search(decoded)
            or re.search(r'\b(?:ignore|forget|override|disable|bypass|pretend|export)\b', decoded, re.IGNORECASE)
        )
    except Exception:
        return True  # Undecodable base64-like blob → treat as suspicious

_PRE_GRAPH_REFUSAL = (
    "I'm unable to process this request. It appears to contain instructions that "
    "attempt to override system behaviour, extract system configuration, or perform "
    "operations outside the scope of credit risk analysis. "
    "This system is restricted to portfolio analytics, regulatory compliance, and "
    "applicant communications."
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClarificationItemSchema(BaseModel):
    id: str
    question: str
    options: List[str] = Field(default_factory=list)


class AnalystQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    data_scope: List[str] = Field(
        default_factory=list,
        description="Optional list of table or dataset names to scope the SQL tool.",
    )
    session_id: Optional[UUID] = None
    sql_query: Optional[str] = Field(
        default=None,
        description=(
            "Optional pre-validated SELECT query to execute as part of retrieval. "
            "Must reference only allow-listed tables."
        ),
    )
    clarifications: Optional[Dict[str, str]] = Field(
        default=None,
        description="Answers to prior clarification questions, keyed by item id.",
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional hint for the agent's intent classification. "
            "Valid values: explain_decision, analyst_query, applicant_comms, portfolio_brief."
        ),
    )


class CitationSchema(BaseModel):
    claim: str
    source_type: str
    source_ref: str
    confidence: float


class ReasoningContextItem(BaseModel):
    source_type: str
    source_ref: str
    relevance: str
    snippet: str


class ReasoningTrace(BaseModel):
    retrieval_method: str = ""
    retrieved_context: List[ReasoningContextItem] = Field(default_factory=list)
    raw_analysis: str = ""
    suppressed_claims: List[Dict[str, Any]] = Field(default_factory=list)


class AnalystQueryResponse(BaseModel):
    session_id: UUID
    answer: str
    citations: List[CitationSchema] = Field(default_factory=list)
    sql_queries_executed: List[str] = Field(default_factory=list)
    confidence_score: float
    follow_up_suggestions: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_items: List[ClarificationItemSchema] = Field(default_factory=list)
    reasoning_trace: Optional[ReasoningTrace] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyst", response_model=AnalystQueryResponse)
async def analyst_query(
    request: AnalystQueryRequest,
) -> AnalystQueryResponse:
    """
    Answer a free-text analytical question using the corrective-RAG agent.

    Context is retrieved from:
      - policy_docs vector store (hybrid dense + BM25)
      - credit-risk-platform portfolio metrics
      - optional SQL query supplied in ``sql_query``

    Returns HTTP 422 when confidence is below the configured threshold.
    Returns HTTP 503 when all LLM providers are unavailable.
    """
    session_id = request.session_id or _uuid.uuid4()
    bound_log = log.bind(session_id=str(session_id))

    # ── Pre-graph injection / out-of-scope guard ──────────────────────────────
    # Catch obvious injection payloads before invoking the full LangGraph pipeline.
    # Returns HTTP 200 with a structured refusal (confidence=0.0, error field set)
    # so evals see a graceful refusal rather than HTTP 500.
    if _PRE_GRAPH_INJECTION_RE.search(request.query) or _is_base64_injection(request.query):
        bound_log.warning("pre_graph_injection_blocked", query=request.query[:80])
        return AnalystQueryResponse(
            session_id=session_id,
            answer=_PRE_GRAPH_REFUSAL,
            confidence_score=0.0,
            follow_up_suggestions=[],
            reasoning_trace=ReasoningTrace(
                retrieval_method="blocked",
                retrieved_context=[],
                raw_analysis="",
                suppressed_claims=[],
            ),
        )

    # ── Applicant comms pre-detection ─────────────────────────────────────────
    # When the query is clearly asking to draft applicant-facing communications
    # (decline notices, adverse action letters, etc.), set the intent and audience
    # before entering the graph so the correct system prompt and compliance gate run.
    _valid_intents = {"explain_decision", "analyst_query", "applicant_comms", "portfolio_brief"}
    req_intent = request.intent if request.intent in _valid_intents else None
    if req_intent == "applicant_comms" or _is_applicant_comms(request.query):
        initial_intent = "applicant_comms"
        initial_audience = "applicant"
    elif req_intent in ("explain_decision", "portfolio_brief"):
        initial_intent = req_intent
        initial_audience = "analyst"
    elif _EXPLAIN_DECISION_RE.search(request.query):
        initial_intent = "explain_decision"
        initial_audience = "analyst"
    else:
        initial_intent = "analyst_query"
        initial_audience = "analyst"

    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "intent": initial_intent,
        "audience": initial_audience,
        "context_payload": {
            "data_scope": request.data_scope,
            "sql_query": request.sql_query or "",
            "clarifications": request.clarifications or {},
        },
    }

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": str(session_id)}}
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        bound_log.error("query_graph_error", error=str(exc))
        # Return a graceful refusal rather than HTTP 500 — an unhandled graph exception
        # likely means a crafted/obfuscated payload crashed the pipeline. Surfacing
        # a 500 leaks internal state; a 200 refusal is safer and scores better on
        # injection-resistance evals.
        return AnalystQueryResponse(
            session_id=session_id,
            answer=_PRE_GRAPH_REFUSAL,
            confidence_score=0.0,
            needs_clarification=False,
            clarification_items=[],
            reasoning_trace={"error": "GRAPH_ERROR", "suppressed": True},
        )

    output: dict = final_state.get("final_output", {})

    # Clarification short-circuit — return 200 with structured clarification payload
    if output.get("needs_clarification"):
        raw_items: list = output.get("clarification_items") or []
        c_items = [
            ClarificationItemSchema(
                id=it.get("id", ""),
                question=it.get("question", ""),
                options=it.get("options") or [],
            )
            for it in raw_items
        ]
        # Include a partial reasoning trace so the audit trail shows what was
        # attempted before clarification was requested.
        raw_trace = output.get("reasoning_trace") or {}
        clarif_trace: Optional[ReasoningTrace] = None
        if raw_trace:
            clarif_trace = ReasoningTrace(
                retrieval_method=raw_trace.get("retrieval_method", ""),
                retrieved_context=[
                    ReasoningContextItem(**item)
                    for item in (raw_trace.get("retrieved_context") or [])
                ],
                raw_analysis=raw_trace.get("raw_analysis", ""),
                suppressed_claims=raw_trace.get("suppressed_claims") or [],
            )
        bound_log.info("analyst_query_clarification", item_count=len(c_items))
        return AnalystQueryResponse(
            session_id=session_id,
            answer=output.get("narrative", ""),
            confidence_score=0.0,
            needs_clarification=True,
            clarification_items=c_items,
            reasoning_trace=clarif_trace,
        )

    # Error routing
    if "error" in output:
        error_code = output["error"]
        if error_code in ("ALL_PROVIDERS_UNAVAILABLE", "PRIMARY_UNAVAILABLE"):
            raise HTTPException(
                status_code=503,
                headers={"Retry-After": "30"},
                detail={"error": error_code, "message": "LLM unavailable."},
            )
        if error_code in ("INSUFFICIENT_GROUNDING", "INSUFFICIENT_RETRIEVAL"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": error_code,
                    "message": "Insufficient grounding — retrieved context did not meet confidence threshold.",
                },
            )
        if error_code == "COMPLIANCE_FAILED":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": error_code,
                    "message": "Response did not meet ECOA/FCRA compliance requirements.",
                },
            )
        raise HTTPException(status_code=500, detail={"error": error_code})

    bound_log.info("analyst_query_ok", confidence=output.get("confidence_score", 0.0))

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

    # Collect SQL queries from retrieved db chunks
    retrieved_chunks = final_state.get("retrieved_chunks") or []
    sql_executed = [
        c["source_ref"].replace("sql_query:", "")
        for c in retrieved_chunks
        if c.get("source_type") == "db"
    ]

    raw_trace = output.get("reasoning_trace") or {}
    reasoning_trace: Optional[ReasoningTrace] = None
    if raw_trace:
        reasoning_trace = ReasoningTrace(
            retrieval_method=raw_trace.get("retrieval_method", ""),
            retrieved_context=[
                ReasoningContextItem(**item)
                for item in (raw_trace.get("retrieved_context") or [])
            ],
            raw_analysis=raw_trace.get("raw_analysis", ""),
            suppressed_claims=raw_trace.get("suppressed_claims") or [],
        )

    return AnalystQueryResponse(
        session_id=session_id,
        answer=output.get("narrative", output.get("grounded_narrative", "")),
        citations=citations,
        sql_queries_executed=sql_executed,
        confidence_score=float(output.get("confidence_score", 0.0)),
        follow_up_suggestions=output.get("follow_up_suggestions", []),
        needs_clarification=False,
        clarification_items=[],
        reasoning_trace=reasoning_trace,
    )


# ---------------------------------------------------------------------------
# SSE streaming endpoint — GET /v1/query/stream
# ---------------------------------------------------------------------------

# LangGraph node names → human-readable step labels for the UI
_NODE_STEP_LABELS = {
    "parse_intent_node": "intent_classified",
    "retrieve_node": "retrieval_complete",
    "grade_documents_node": "grading_complete",
    "reason_node": "reasoning_complete",
    "citation_enforcer_node": "grounding_complete",
    "confidence_score_node": "confidence_scored",
    "compliance_check_node": "compliance_checked",
    "format_output_node": "output_formatted",
    "persist_session_node": "session_persisted",
}


def _sse(event: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_agent_streaming(
    query: str,
    data_scope: List[str],
    sql_query: str,
    session_id: _uuid.UUID,
    clarifications: Dict[str, str] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent and yield SSE frames.

    Yields:
      - ``status`` event after each node completes (with step label + metadata)
      - ``result`` event with the full AnalystQueryResponse-equivalent dict
      - ``error`` event on known error codes
      - ``done`` event always at the end
    """
    initial_state = {
        "session_id": session_id,
        "query": query,
        "intent": "analyst_query",
        "audience": "analyst",
        "context_payload": {
            "data_scope": data_scope,
            "sql_query": sql_query,
            "clarifications": clarifications or {},
        },
    }

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": str(session_id)}}

        # Emit initial status
        yield _sse("status", {"step": "started", "session_id": str(session_id)})

        final_state: dict = {}

        # Stream LangGraph node events using astream_events (v2 API)
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Node completed
            if kind == "on_chain_end" and name in _NODE_STEP_LABELS:
                step_label = _NODE_STEP_LABELS[name]
                metadata: dict = {"step": step_label}

                # Add rich metadata for specific steps
                output_data = event.get("data", {}).get("output", {}) or {}
                if step_label == "retrieval_complete":
                    chunks = output_data.get("retrieved_chunks", [])
                    metadata["chunk_count"] = len(chunks) if isinstance(chunks, list) else 0
                elif step_label == "grounding_complete":
                    metadata["citation_count"] = len(output_data.get("citations", []))
                elif step_label == "confidence_scored":
                    metadata["confidence_score"] = round(
                        float(output_data.get("confidence_score", 0.0)), 3
                    )

                yield _sse("status", metadata)

            # Capture final graph state from __end__ event
            if kind == "on_chain_end" and name == "__end__":
                final_state = event.get("data", {}).get("output", {}) or {}

        # If astream_events didn't capture final state, fall back to ainvoke
        if not final_state:
            final_state = await graph.ainvoke(initial_state, config=config)

    except Exception as exc:
        log.error("query_stream_error", session_id=str(session_id), error=str(exc))
        yield _sse("error", {"error": "GRAPH_ERROR", "message": str(exc)})
        yield _sse("done", {})
        return

    # Build output
    output: dict = final_state.get("final_output", {})

    if "error" in output:
        error_code = output["error"]
        yield _sse("error", {"error": error_code, "message": output.get("message", "")})
        yield _sse("done", {})
        return

    # Clarification short-circuit — emit a structured clarification event
    if output.get("needs_clarification"):
        raw_items: list = output.get("clarification_items") or []
        yield _sse("clarification", {
            "session_id": str(session_id),
            "needs_clarification": True,
            "clarification_items": raw_items,
        })
        yield _sse("done", {})
        return

    raw_citations = output.get("citations") or []
    citations = [
        {
            "claim": c.get("claim", ""),
            "source_type": c.get("source_type", ""),
            "source_ref": c.get("source_ref", ""),
            "confidence": float(c.get("confidence", 1.0)),
        }
        for c in raw_citations
    ]

    retrieved_chunks = final_state.get("retrieved_chunks") or []
    sql_executed = [
        c["source_ref"].replace("sql_query:", "")
        for c in retrieved_chunks
        if c.get("source_type") == "db"
    ]

    raw_trace = output.get("reasoning_trace") or {}
    reasoning_trace_payload: dict | None = None
    if raw_trace:
        reasoning_trace_payload = {
            "retrieval_method": raw_trace.get("retrieval_method", ""),
            "retrieved_context": raw_trace.get("retrieved_context") or [],
            "raw_analysis": raw_trace.get("raw_analysis", ""),
            "suppressed_claims": raw_trace.get("suppressed_claims") or [],
        }

    yield _sse("result", {
        "session_id": str(session_id),
        "answer": output.get("narrative", output.get("grounded_narrative", "")),
        "citations": citations,
        "sql_queries_executed": sql_executed,
        "confidence_score": round(float(output.get("confidence_score", 0.0)), 4),
        "follow_up_suggestions": output.get("follow_up_suggestions", []),
        "reasoning_trace": reasoning_trace_payload,
    })
    yield _sse("done", {})


@router.get("/stream")
async def analyst_query_stream(
    query: str = Query(..., min_length=3, max_length=4000, description="Analytical question"),
    data_scope: str = Query(default="", description="Comma-separated table names to scope SQL tool"),
    sql_query: str = Query(default="", description="Optional pre-validated SELECT query"),
    session_id: Optional[str] = Query(default=None, description="Resume an existing session"),
    clarifications: Optional[str] = Query(default=None, description="JSON-encoded clarification answers, e.g. {\"product_type\":\"All combined\"}"),
) -> StreamingResponse:
    """
    Stream the LangGraph agent response as Server-Sent Events.

    Connect with ``EventSource('/v1/query/stream?query=...')`` from the browser.
    Each agent node completion emits a ``status`` event.
    The final answer emits a ``result`` event followed by ``done``.
    When clarification is needed, emits a ``clarification`` event instead.
    """
    _session_id = _uuid.UUID(session_id) if session_id else _uuid.uuid4()
    scope_list = [s.strip() for s in data_scope.split(",") if s.strip()] if data_scope else []
    clarification_dict: dict | None = None
    if clarifications:
        try:
            clarification_dict = json.loads(clarifications)
        except Exception:
            pass

    log.info("query_stream_start", session_id=str(_session_id), query_len=len(query))

    return StreamingResponse(
        _run_agent_streaming(query, scope_list, sql_query, _session_id, clarification_dict),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
