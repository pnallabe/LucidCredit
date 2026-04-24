"""
backend/app/api/v1/query.py
=============================
FastAPI router for POST /v1/query/analyst.

Accepts a free-text analytical query, runs the corrective-RAG agent, and
returns a grounded answer with citations and (optionally) the SQL queries
that were executed to produce the result.
"""
from __future__ import annotations

import uuid as _uuid
from typing import List, Optional
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
