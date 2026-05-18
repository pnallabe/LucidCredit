"""
backend/app/api/v1/chat.py
============================
LucidCredit v2 — Conversational chat endpoint.

Routes
------
    POST /v1/chat          — full JSON response (non-streaming)
    POST /v1/chat/stream   — SSE streaming (text/event-stream)
    DELETE /v1/chat/{session_id} — clear session history

The chat endpoint replaces the old /v1/query/analyst pipeline for conversational
use. The legacy endpoint remains for backward compatibility with structured callers.

SSE event format (text/event-stream):
    data: <text chunk>\\n\\n
    data: [DONE]\\n\\n

Session continuity
------------------
Pass `session_id` from a prior response to continue a conversation.
Omit it to start a fresh session (a new UUID will be assigned and returned).

Clarifications
--------------
When the analytics system needs clarification (e.g. which product type), the
assistant asks conversationally in its response. Just reply naturally in the
next message — the orchestrator passes context automatically.

You can also explicitly provide clarifications:
    {"message": "Show me 2023 data", "clarifications": {"time_period": "2023"}}
"""
from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import run_turn
from app.agent.session_store import clear_history, load_history, save_history

log = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        default=None,
        description="Prior session ID for conversation continuity. Omit to start fresh.",
    )
    clarifications: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Explicit answers to a prior clarification request. "
            "e.g. {\"product_type\": \"Personal loans\", \"time_period\": \"2023\"}"
        ),
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    follow_up_suggestions: List[str] = Field(default_factory=list)


class ClearResponse(BaseModel):
    session_id: str
    cleared: bool


# ---------------------------------------------------------------------------
# Non-streaming endpoint
# ---------------------------------------------------------------------------


@router.post("/", response_model=ChatResponse, summary="Chat (non-streaming)")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a natural language question. Returns the full response once complete.
    Use /v1/chat/stream for streaming (recommended for UI).
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = await load_history(session_id)

    # Accumulate streaming response
    answer_parts: list[str] = []
    async for chunk in run_turn(
        question=request.message,
        history=history,
        clarifications=request.clarifications,
    ):
        answer_parts.append(chunk)

    answer = "".join(answer_parts)
    await save_history(session_id, history)

    # Simple follow-up suggestions based on what was answered
    suggestions = _suggest_follow_ups(request.message, answer)

    log.info("chat_response", session_id=session_id, answer_len=len(answer))
    return ChatResponse(session_id=session_id, answer=answer, follow_up_suggestions=suggestions)


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


@router.post("/stream", summary="Chat (SSE streaming)")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Answer a natural language question with SSE streaming.
    Each data event contains a text chunk. The final event is ``data: [DONE]``.

    The session_id is returned in the ``X-Session-Id`` response header.
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = await load_history(session_id)

    async def _event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in run_turn(
                question=request.message,
                history=history,
                clarifications=request.clarifications,
            ):
                # Escape newlines within a chunk to keep SSE framing intact
                safe = chunk.replace("\n", "\\n")
                yield f"data: {safe}\n\n".encode()

            await save_history(session_id, history)
            yield b"data: [DONE]\n\n"

        except Exception as exc:
            log.error("chat_stream_error", session_id=session_id, error=str(exc))
            yield f"data: [ERROR] {str(exc)[:200]}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@router.delete("/{session_id}", response_model=ClearResponse, summary="Clear session history")
async def clear_session(session_id: str) -> ClearResponse:
    """Clear the conversation history for a session."""
    await clear_history(session_id)
    log.info("session_cleared", session_id=session_id)
    return ClearResponse(session_id=session_id, cleared=True)


# ---------------------------------------------------------------------------
# Follow-up suggestion helper (lightweight, no LLM call)
# ---------------------------------------------------------------------------

_FOLLOW_UP_MAP: list[tuple[Any, list[str]]] = []

# Populated lazily to avoid import-time regex compilation
def _suggest_follow_ups(question: str, answer: str) -> list[str]:
    """Return 2-3 contextual follow-up suggestions based on the question topic."""
    import re
    q = question.lower()

    if re.search(r"delinquency|dpd|past.due", q):
        return [
            "How does this compare to 2023?",
            "Which product type has the highest delinquency rate?",
            "What's the 60+ and 90+ DPD breakdown?",
        ]
    if re.search(r"charge.off|write.off|nco", q):
        return [
            "Show the charge-off trend by quarter.",
            "Which vintage has the highest charge-off rate?",
            "How does charge-off compare across product types?",
        ]
    if re.search(r"approval|origination|application", q):
        return [
            "What's the approval rate by FICO tier?",
            "Show origination volume trend by month.",
            "What's the average loan size for approved applications?",
        ]
    if re.search(r"balance|outstanding|exposure", q):
        return [
            "Show the balance trend over the last 12 months.",
            "What's the balance breakdown by credit grade?",
            "Which state has the highest outstanding balance?",
        ]
    return [
        "Show me the trend over time.",
        "Break this down by product type.",
        "What's driving this metric?",
    ]
