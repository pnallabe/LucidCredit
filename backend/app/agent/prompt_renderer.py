"""
backend/app/agent/prompt_renderer.py
======================================
Renders the full prompt string sent to the LLM in ``reason_node``.

Loads system prompts from the ``app/agent/prompts/`` directory (Markdown files)
and assembles them with the retrieved, graded context chunks and the user query.

Public API
----------
    render_prompt(session_type, query, graded_chunks, context_payload) -> str

The returned string is stored in ``AgentState["rendered_prompt"]`` and passed
verbatim to the LLM.  It is also persisted to the ``copilot_sessions`` table for
the SR 11-7 immutable audit trail.
"""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import List, Literal

from app.agent.state import RetrievedChunk

PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

SessionType = Literal["analyst", "applicant", "briefing"]


# ---------------------------------------------------------------------------
# Template loading — cached so we only hit the filesystem once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _analyst_system() -> str:
    return _load_prompt("analyst_system.md")


def _applicant_system() -> str:
    return _load_prompt("applicant_system.md")


def _grading_system() -> str:
    return _load_prompt("grading_system.md")


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def _format_context(graded_chunks: List[RetrievedChunk]) -> str:
    """
    Format graded chunks into a numbered context block.

    Only RELEVANT and AMBIGUOUS chunks are included; IRRELEVANT are silently
    dropped so they cannot influence the model response.
    """
    relevant = [
        c for c in graded_chunks
        if c.get("relevance") in ("RELEVANT", "AMBIGUOUS")
    ]
    if not relevant:
        return "(No relevant context was retrieved for this query.)"

    parts: List[str] = []
    for i, chunk in enumerate(relevant, 1):
        source_line = f"[{i}] SOURCE_TYPE={chunk['source_type']} | REF={chunk['source_ref']}"
        # Cap content at 800 chars to stay well within token budgets.
        content = chunk["content"][:800].strip()
        parts.append(f"{source_line}\n{content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Renderers per session type
# ---------------------------------------------------------------------------

def _render_analyst(
    query: str,
    graded_chunks: List[RetrievedChunk],
    context_payload: dict,
) -> str:
    context_block = _format_context(graded_chunks)
    decision_id = context_payload.get("decision_id", "")
    decision_line = f"\n**Decision ID**: `{decision_id}`" if decision_id else ""
    application_id = context_payload.get("application_id", "")
    app_line = f"\n**Application ID**: `{application_id}`" if application_id else ""

    return (
        f"{_analyst_system()}\n\n"
        "---\n\n"
        "## Retrieved Context\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"## Analyst Query{decision_line}{app_line}\n\n"
        f"{query}\n\n"
        "## Response\n"
    )


def _render_applicant(
    query: str,
    graded_chunks: List[RetrievedChunk],
    context_payload: dict,
) -> str:
    context_block = _format_context(graded_chunks)
    application_id = context_payload.get("application_id", "")
    app_line = f"\n**Application ID**: `{application_id}`" if application_id else ""
    comm_type = context_payload.get("communication_type", "")
    comm_line = f"\n**Communication Type**: {comm_type}" if comm_type else ""

    return (
        f"{_applicant_system()}\n\n"
        "---\n\n"
        "## Retrieved Context\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"## Communication Request{app_line}{comm_line}\n\n"
        f"{query}\n\n"
        "## Response\n"
    )


def _render_briefing(
    query: str,
    graded_chunks: List[RetrievedChunk],
    context_payload: dict,
) -> str:
    context_block = _format_context(graded_chunks)
    scope = context_payload.get("scope", "")
    scope_line = f"\n**Scope**: {scope}" if scope else ""
    audience_role = context_payload.get("audience_role", "")
    role_line = f"\n**Audience Role**: {audience_role}" if audience_role else ""

    return (
        f"{_analyst_system()}\n\n"
        f"{_grading_system()}\n\n"
        "---\n\n"
        "## Retrieved Context\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"## Portfolio Briefing Request{scope_line}{role_line}\n\n"
        f"{query}\n\n"
        "## Response\n"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_prompt(
    session_type: SessionType,
    query: str,
    graded_chunks: List[RetrievedChunk],
    context_payload: dict,
) -> str:
    """
    Render the full LLM prompt for *session_type*.

    Args:
        session_type:    "analyst", "applicant", or "briefing"
        query:           The user's natural language query.
        graded_chunks:   Chunks from ``grade_documents_node`` (may include
                         RELEVANT, AMBIGUOUS, and IRRELEVANT entries;
                         IRRELEVANT are filtered internally).
        context_payload: Raw API inputs dict from the request
                         (decision_id, application_id, scope, etc.).

    Returns:
        A fully assembled prompt string ready to send to the LLM.
    """
    if session_type == "applicant":
        return _render_applicant(query, graded_chunks, context_payload)
    if session_type == "briefing":
        return _render_briefing(query, graded_chunks, context_payload)
    return _render_analyst(query, graded_chunks, context_payload)
