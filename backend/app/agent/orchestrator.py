"""
backend/app/agent/orchestrator.py
===================================
LucidCredit v2 — Conversational credit analyst agent.

Replaces the LangGraph multi-node pipeline (graph.py / nodes.py) with a simple
OpenAI function-calling loop.  The LLM decides which tools to call based on the
question and conversation history — no regex routing.

Architecture
------------
    User message + conversation history
        ↓
    GPT-4o (tool_choice="auto")
        ↓  (may call tools)
    Tool: query_portfolio(question)  →  analytics API  →  BigQuery rows
        ↓
    GPT-4o synthesises final answer (streamed)
        ↓
    Response + updated history saved to session

Design principles
-----------------
- The LLM routes, not regex.  GPT-4o understands intent far better than patterns.
- One tool (query_portfolio) covers all live-data questions.
- Conversation history is maintained as OpenAI messages across turns.
- Clarifications are handled conversationally: when the analytics API needs more
  context, the LLM asks the user naturally; the answer arrives on the next turn
  and is passed back to the analytics API automatically.
- No citation enforcement, no confidence gating — the LLM naturally attributes
  data ("The portfolio data shows...") and gracefully handles missing data.
- Streaming by default: first token in <500ms instead of blocking a 9-node pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx
import openai

from app.config import get_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (what the LLM can call)
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_portfolio",
            "description": (
                "Query live portfolio data from the credit analytics system. "
                "Use for ANY question that needs actual numbers from the portfolio: "
                "delinquency rates, outstanding balances, origination volume, approval rates, "
                "charge-off rates, FICO distributions, trend data, income statement metrics, "
                "geographic breakdowns, vintage performance, roll rates, etc. "
                "Also use for follow-up questions that need fresh data (e.g. 'show me 2023 instead'). "
                "The system handles NL→SQL internally — just pass the natural language question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The natural language question to query against portfolio data. "
                            "Be specific: include product type, time period, and metric if known."
                        ),
                    },
                    "clarifications": {
                        "type": "object",
                        "description": (
                            "Optional dimension answers from the user. "
                            "e.g. {\"product_type\": \"Personal loans\", \"time_period\": \"2023\"}. "
                            "Only populate when the user has explicitly answered a clarification request."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["question"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt — clean, no mandatory vocabulary injection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are LucidCredit, an AI analytical copilot for credit risk professionals.

You have access to live portfolio data via the query_portfolio tool. Use it freely for any \
question that needs real numbers — delinquency rates, balances, origination trends, charge-off \
rates, FICO distributions, approval rates, etc.

## How to handle questions

**Data questions** (rates, counts, trends, distributions, comparisons):
  → Call query_portfolio. Use the results to answer. Say "The portfolio data shows..." when \
citing figures.

**Mixed questions** (e.g. "what's the delinquency rate and what should we do about it?"):
  → Call query_portfolio for the data part, then combine with your credit risk expertise for \
the interpretation/recommendation.

**Conceptual/analytical questions** (why X happens, how to interpret Y, what Z means):
  → Answer directly from your expertise. No tool call needed.

**Follow-up questions in a conversation**:
  → Reference the prior answer naturally. Call query_portfolio again only if new data is needed.

## When the analytics system needs clarification

If query_portfolio returns a clarification request (e.g. asking which product type), \
ask the user conversationally:
  "To answer that, I need to know which product type you're asking about — Personal Loans, \
Mortgage, Auto Loans, or all combined?"
Then on the next turn, include the user's answer as clarifications in query_portfolio.

## If data isn't available

Say clearly that the data isn't available in this portfolio dataset. Suggest alternatives \
if possible (e.g. a related metric that is available).

## Tone and style

- Credit analysts: use precise technical language (30+ DPD, PD bands, DTI, FICO tiers).
- Be direct and concise. Lead with the answer, then provide context.
- For multi-part answers, use headers or bullet points.
- Never fabricate numbers. If you don't have data, say so.
"""

# ---------------------------------------------------------------------------
# Analytics API caller (direct HTTP, no LangGraph state)
# ---------------------------------------------------------------------------

_TIMEOUT = httpx.Timeout(timeout=40.0)


def _analytics_url() -> str:
    settings = get_settings()
    base = settings.crp_api_base_url.rstrip("/")
    # Port normalisation: crp_api_base_url may point at :8081 (CRP API) but
    # the analytics NL layer lives at :8001.
    if base.endswith(":8081"):
        base = base.replace(":8081", ":8001")
    return f"{base}/v1/analytics/s2s/ask"


def _analytics_headers() -> dict[str, str]:
    settings = get_settings()
    key = getattr(settings, "analytics_service_key", "dev-analytics-key") or "dev-analytics-key"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Service-Key": key,
    }


async def _call_analytics(question: str, clarifications: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Call the analytics API and return a structured result dict.

    Returns a dict with one of:
      {"status": "data",          "rows": [...], "row_count": N, "assumed_defaults": [...], "sql": "..."}
      {"status": "clarification", "items": [{id, question, options}, ...]}
      {"status": "no_data",       "message": "..."}
      {"status": "unavailable",   "message": "..."}
      {"status": "error",         "message": "..."}
    """
    url = _analytics_url()
    payload: dict[str, Any] = {"question": question}
    if clarifications:
        payload["clarifications"] = clarifications

    # --- Auto-inject product_type when unambiguous ---
    # If no clarifications given and question names a specific product, pre-fill it
    # so we avoid the clarification round-trip.
    if not (clarifications or {}).get("product_type"):
        inferred = _infer_product_type(question)
        if inferred:
            payload.setdefault("clarifications", {})["product_type"] = inferred  # type: ignore[index]

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=_analytics_headers())
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return {"status": "unavailable", "message": "The analytics data service is not reachable. Please try again in a moment."}
    except httpx.HTTPStatusError as exc:
        return {"status": "error", "message": f"Analytics API error {exc.response.status_code}: {exc.response.text[:200]}"}
    except Exception as exc:
        return {"status": "error", "message": f"Unexpected error calling analytics: {exc}"}

    # --- Clarification needed ---
    if data.get("needs_clarification"):
        items = data.get("clarification_items") or []
        # Auto-resolve product_type if portfolio-wide question
        if len(items) == 1 and items[0].get("id") == "product_type":
            if _is_portfolio_wide(question):
                # Retry with "All combined"
                return await _call_analytics(question, {**(clarifications or {}), "product_type": "All combined"})
        return {"status": "clarification", "items": items}

    rows = data.get("rows") or []
    if not rows:
        sql = data.get("generated_sql", "")
        msg = f"No data found for this query."
        if sql:
            msg += f" SQL attempted: {sql[:300]}"
        return {"status": "no_data", "message": msg, "assumed_defaults": data.get("assumed_defaults") or []}

    return {
        "status": "data",
        "rows": rows[:200],
        "row_count": data.get("row_count", len(rows)),
        "assumed_defaults": data.get("assumed_defaults") or [],
        "sql": data.get("generated_sql", "")[:500],
    }


_PORTFOLIO_WIDE_RE = __import__("re").compile(
    r"\b(portfolio|all borrowers?|all accounts?|all loans?|entire|across all|"
    r"overall|total portfolio|whole portfolio|average across|combined|aggregate)\b",
    __import__("re").IGNORECASE,
)

_PRODUCT_MAP: list[tuple[Any, str]] = [
    (__import__("re").compile(r"\bmortgages?\b", __import__("re").I), "Mortgage"),
    (__import__("re").compile(r"\bpersonal loans?\b", __import__("re").I), "Personal loans"),
    (__import__("re").compile(r"\bauto loans?\b", __import__("re").I), "Auto loans"),
    (__import__("re").compile(r"\bcredit cards?\b", __import__("re").I), "Credit cards"),
    (__import__("re").compile(r"\bhome equity\b|\bheloc\b", __import__("re").I), "Home equity"),
    (__import__("re").compile(r"\bstudent loans?\b", __import__("re").I), "Student loans"),
    (__import__("re").compile(r"\bbusiness loans?\b", __import__("re").I), "Business loans"),
]


def _infer_product_type(question: str) -> str | None:
    for pat, label in _PRODUCT_MAP:
        if pat.search(question):
            return label
    return None


def _is_portfolio_wide(question: str) -> bool:
    return bool(_PORTFOLIO_WIDE_RE.search(question))


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------

def _make_openai_client() -> openai.AsyncOpenAI:
    settings = get_settings()
    if settings.llm_provider_mode == "azure" and settings.azure_openai_endpoint:
        return openai.AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    return openai.AsyncOpenAI(api_key=settings.openai_api_key)


def _model_name() -> str:
    settings = get_settings()
    if settings.llm_provider_mode == "azure" and settings.azure_openai_endpoint:
        return settings.azure_openai_deployment_analyst
    return settings.openai_model or "gpt-4o"


# ---------------------------------------------------------------------------
# Core turn runner
# ---------------------------------------------------------------------------

async def run_turn(
    question: str,
    history: list[dict[str, Any]],
    clarifications: dict[str, str] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Run one conversational turn.

    Parameters
    ----------
    question:       The user's message for this turn.
    history:        Mutable list of prior OpenAI messages ({"role", "content"}).
                    Updated in-place after this turn completes.
    clarifications: Optional dict of user-provided answers to a prior clarification
                    request (e.g. {"product_type": "Personal loans"}).

    Yields
    ------
    str chunks of the assistant's response (suitable for SSE streaming).
    """
    client = _make_openai_client()
    model = _model_name()

    # Build message list for this turn
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]

    # ── Phase 1: let the LLM decide if it needs a tool ──
    first_response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        tools=_TOOLS,  # type: ignore[arg-type]
        tool_choice="auto",
        temperature=0,
        max_tokens=4096,
    )

    first_msg = first_response.choices[0].message

    # ── Phase 2: execute tool calls if any ──
    if first_msg.tool_calls:
        # Append assistant's tool-call message to the thread
        messages.append(first_msg.model_dump(exclude_unset=True))  # type: ignore[arg-type]

        for tc in first_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_question = args.get("question", question)
            tool_clarifs = args.get("clarifications") or clarifications or {}

            log.info("orchestrator: calling query_portfolio q=%r clarifications=%r", tool_question[:80], tool_clarifs)
            result = await _call_analytics(tool_question, tool_clarifs or None)

            # Format tool result as a readable string for the LLM
            tool_content = _format_tool_result(tool_question, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_content,
            })

    # ── Phase 3: stream the final response ──
    accumulated = ""

    if first_msg.tool_calls or not first_msg.content:
        # Need a fresh generation with tool results in context
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
            max_tokens=4096,
            stream=True,
        )
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if delta:
                accumulated += delta
                yield delta
    else:
        # LLM answered without calling any tool — stream what it already has
        # (this avoids a second round-trip for purely conceptual questions)
        content = first_msg.content or ""
        accumulated = content
        # Yield in small chunks to simulate streaming feel
        chunk_size = 40
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    # ── Update history in-place ──
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": accumulated})

    # Trim history to last 20 turns to prevent token blowout
    if len(history) > 40:
        history[:] = history[-40:]


# ---------------------------------------------------------------------------
# Tool result formatter
# ---------------------------------------------------------------------------

def _format_tool_result(question: str, result: dict[str, Any]) -> str:
    """Convert the analytics API result dict to a readable string for the LLM."""
    status = result.get("status")

    if status == "data":
        rows = result["rows"]
        row_count = result["row_count"]
        defaults = result.get("assumed_defaults") or []
        sql = result.get("sql", "")
        lines = [
            f"[Portfolio Data — {row_count} rows]",
            f"Question: {question}",
        ]
        if defaults:
            lines.append(f"Assumed defaults: {'; '.join(defaults)}")
        if sql:
            lines.append(f"SQL: {sql}")
        lines.append("")
        # Render rows as readable key=value pairs
        for i, row in enumerate(rows[:100]):
            pairs = ", ".join(f"{k}={v}" for k, v in row.items())
            lines.append(f"  {pairs}")
        if row_count > 100:
            lines.append(f"  ... ({row_count - 100} more rows truncated)")
        return "\n".join(lines)

    elif status == "clarification":
        items = result.get("items") or []
        lines = ["[Clarification Needed]", f"Question: {question}", ""]
        for item in items:
            opts = ", ".join(item.get("options") or [])
            lines.append(f"- {item.get('question', '')} (options: {opts})")
        lines.append("")
        lines.append(
            "Please ask the user for the above clarification(s) in a natural conversational way. "
            "On the next turn, call query_portfolio again with the clarifications dict populated."
        )
        return "\n".join(lines)

    elif status == "no_data":
        defaults = result.get("assumed_defaults") or []
        msg = result.get("message", "No data found.")
        lines = [f"[No Data Found]", f"Question: {question}", msg]
        if defaults:
            lines.append(f"Assumed defaults: {'; '.join(defaults)}")
        return "\n".join(lines)

    elif status == "unavailable":
        return f"[Analytics Service Unavailable]\n{result.get('message', '')}"

    else:  # error
        return f"[Analytics Error]\n{result.get('message', 'Unknown error')}"
