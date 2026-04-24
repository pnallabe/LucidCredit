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
import uuid as _uuid
from typing import AsyncGenerator, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import get_graph

log = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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


class CitationSchema(BaseModel):
    claim: str
    source_type: str
    source_ref: str
    confidence: float


class AnalystQueryResponse(BaseModel):
    session_id: UUID
    answer: str
    citations: List[CitationSchema] = Field(default_factory=list)
    sql_queries_executed: List[str] = Field(default_factory=list)
    confidence_score: float
    follow_up_suggestions: List[str] = Field(default_factory=list)


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

    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "intent": "analyst_query",
        "audience": "analyst",
        "context_payload": {
            "data_scope": request.data_scope,
            "sql_query": request.sql_query or "",
        },
    }

    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": str(session_id)}}
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        bound_log.error("query_graph_error", error=str(exc))
        raise HTTPException(status_code=500, detail={"error": "GRAPH_ERROR", "message": str(exc)})

    output: dict = final_state.get("final_output", {})

    # Error routing
    if "error" in output:
        error_code = output["error"]
        if error_code in ("ALL_PROVIDERS_UNAVAILABLE", "PRIMARY_UNAVAILABLE"):
            raise HTTPException(
                status_code=503,
                headers={"Retry-After": "30"},
                detail={"error": error_code, "message": "LLM unavailable."},
            )
        if error_code == "INSUFFICIENT_GROUNDING":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": error_code,
                    "message": "Insufficient grounding — retrieved context did not meet confidence threshold.",
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

    return AnalystQueryResponse(
        session_id=session_id,
        answer=output.get("narrative", output.get("grounded_narrative", "")),
        citations=citations,
        sql_queries_executed=sql_executed,
        confidence_score=float(output.get("confidence_score", 0.0)),
        follow_up_suggestions=output.get("follow_up_suggestions", []),
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
        },
    }

    try:
        graph = get_graph()
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

    yield _sse("result", {
        "session_id": str(session_id),
        "answer": output.get("narrative", output.get("grounded_narrative", "")),
        "citations": citations,
        "sql_queries_executed": sql_executed,
        "confidence_score": round(float(output.get("confidence_score", 0.0)), 4),
        "follow_up_suggestions": output.get("follow_up_suggestions", []),
    })
    yield _sse("done", {})


@router.get("/stream")
async def analyst_query_stream(
    query: str = Query(..., min_length=3, max_length=4000, description="Analytical question"),
    data_scope: str = Query(default="", description="Comma-separated table names to scope SQL tool"),
    sql_query: str = Query(default="", description="Optional pre-validated SELECT query"),
    session_id: Optional[str] = Query(default=None, description="Resume an existing session"),
) -> StreamingResponse:
    """
    Stream the LangGraph agent response as Server-Sent Events.

    Connect with ``EventSource('/v1/query/stream?query=...')`` from the browser.
    Each agent node completion emits a ``status`` event.
    The final answer emits a ``result`` event followed by ``done``.
    """
    _session_id = _uuid.UUID(session_id) if session_id else _uuid.uuid4()
    scope_list = [s.strip() for s in data_scope.split(",") if s.strip()] if data_scope else []

    log.info("query_stream_start", session_id=str(_session_id), query_len=len(query))

    return StreamingResponse(
        _run_agent_streaming(query, scope_list, sql_query, _session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
