"""
backend/app/agent/tools/analytics_api_tool.py
===============================================
Async client for the credit-risk-platform Analytics API semantic layer.

LucidCredit sends a **natural language question** to
``POST /v1/analytics/s2s/ask``.  The analytics API handles NL→SQL internally
(using Azure OpenAI) and executes the query against BigQuery.

BQ table schema and credentials are entirely hidden from LucidCredit.

Context Ingestion
-----------------
When a question is ambiguous (e.g. "applications" without a product type,
or "recently" without a year), the analytics API returns ``needs_clarification=True``
with a list of ``clarification_items`` — each item has an ``id``, a ``question``,
and ``options``.

LucidCredit surfaces these to the user as a follow-up.  On the next turn the
caller passes ``clarifications: dict[str, str]`` keyed by item id so the
analytics API can resolve every dimension before running BQ.

Auto product_type fallback
--------------------------
When the analytics API requests clarification on ``product_type`` AND the
original question contains portfolio-wide language ("portfolio", "all borrowers",
"all accounts", "all loans", "entire", "across all"), this module automatically
retries with ``{"product_type": "All combined"}`` so the user doesn't need a
second turn for common aggregate questions.

Hallucination guard
-------------------
When the API returns rows, ``_validate_rows_for_question()`` checks whether the
returned column names are plausibly related to the question's subject matter.
If the columns look like an unrelated query result (e.g. LTV columns returned
for a "prepayment rate" question), the tool returns a structured "no_matching_field"
chunk instead of letting the LLM fabricate an answer from irrelevant data.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.agent.state import RetrievedChunk
from app.config import get_settings

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(timeout=35.0)   # NL2SQL + BQ round-trip
_DEFAULT_SERVICE_KEY = "dev-analytics-key"


def _analytics_base_url() -> str:
    settings = get_settings()
    base = settings.crp_api_base_url.rstrip("/")
    if base.endswith(":8081"):
        base = base.replace(":8081", ":8001")
    return base


def _make_headers() -> dict[str, str]:
    settings = get_settings()
    service_key = (
        getattr(settings, "analytics_service_key", _DEFAULT_SERVICE_KEY)
        or _DEFAULT_SERVICE_KEY
    )
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Service-Key": service_key,
    }


def _rows_to_chunk(
    question: str,
    rows: list[dict[Any, Any]],
    row_count: int,
    truncated: bool,
    assumed_defaults: list[str] | None = None,
) -> RetrievedChunk:
    header = f"[Credit Risk Analytics Data]\nQuestion: {question}\nRows: {row_count}"
    if truncated:
        header += " (truncated to 200)"
    if assumed_defaults:
        header += f"\nAssumed defaults: {'; '.join(assumed_defaults)}"
    body = json.dumps(rows, default=str)[:6000]
    # Natural-language row summary improves citation-enforcer embedding similarity
    # between raw SQL result data and the LLM's narrative response sentences.
    nl_lines: list[str] = []
    for row in rows[:20]:
        pairs = [f"{k} is {v}" for k, v in row.items()]
        nl_lines.append("Data: " + ", ".join(pairs) + ".")
    nl_summary = "\n".join(nl_lines)
    return RetrievedChunk(
        chunk_id="analytics_bq_ask",
        source_type="db",
        source_ref="bigquery:credit-risk-platform",
        content=f"{header}\n\n{body}\n\n{nl_summary}",
        relevance="RELEVANT",
        similarity_score=1.0,
    )


def _clarification_to_chunk(
    question: str,
    items: list[dict[str, Any]],
) -> RetrievedChunk:
    """
    Wrap multiple clarification questions into a single RetrievedChunk so they
    flow through the agent graph unchanged.

    ``items`` is the raw ``clarification_items`` list from the analytics API
    response, each element being ``{id, question, options}``.

    The structured items are embedded as JSON in ``source_ref`` so that
    ``grade_documents_node`` can recover them losslessly without text parsing.
    """
    lines = [
        f"[Clarification Needed]\n"
        f"The question \"{question}\" cannot be answered precisely without more context.\n"
        f"Please ask the user the following question(s) before running the analysis:\n"
    ]
    for item in items:
        opts = " | ".join(item.get("options", []))
        lines.append(f"• [{item.get('id', '')}] {item['question']}\n  Options: {opts}")

    content = "\n".join(lines)
    # Embed raw items as JSON in source_ref for lossless recovery in grade_documents_node
    source_ref = "analytics:clarification:" + json.dumps(items, separators=(",", ":"))
    return RetrievedChunk(
        chunk_id="analytics_clarification",
        source_type="clarification",
        source_ref=source_ref,
        content=content,
        relevance="RELEVANT",
        similarity_score=1.0,
    )


# ---------------------------------------------------------------------------
# Portfolio-wide question detection (for auto product_type fallback)
# ---------------------------------------------------------------------------

_PORTFOLIO_WIDE_PATTERNS = re.compile(
    r"\b(portfolio|all borrowers?|all accounts?|all loans?|entire|across all|"
    r"overall|total portfolio|whole portfolio|every borrower|every account|"
    r"average across|combined|aggregate)\b",
    re.IGNORECASE,
)


def _is_portfolio_wide(question: str) -> bool:
    """Return True when the question clearly intends a cross-product aggregate."""
    return bool(_PORTFOLIO_WIDE_PATTERNS.search(question))


# ---------------------------------------------------------------------------
# Hallucination guard: validate returned columns against question subject
# ---------------------------------------------------------------------------

# Maps subject keywords in the question to expected column name fragments.
# If NONE of the expected fragments appear in any returned column name,
# the result is likely from an unrelated query — refuse rather than fabricate.
_SUBJECT_COLUMN_MAP: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bprepayment\b", re.I), ["prepay", "early_payoff", "payoff_date"]),
    (re.compile(r"\bltv\b|loan.to.value", re.I), ["ltv", "loan_to_value", "collateral"]),
    (re.compile(r"\bmortgage\b", re.I), ["mortgage", "ltv", "property", "lien"]),
    (re.compile(r"\bcharge.off\b|charge_off\b|nco\b", re.I), ["charge_off", "chargeoff", "written_off", "nco"]),
    (re.compile(r"\bindustry\b", re.I), ["industry", "sector", "sic"]),
    (re.compile(r"\bemployment\b", re.I), ["employment", "employer", "job", "occupation"]),
    (re.compile(r"\bchannel\b", re.I), ["channel", "origination_channel", "source"]),
    (re.compile(r"\bgeograph|state\b|texas\b|region\b", re.I), ["state", "geo", "region", "city", "zip"]),
]

# Metrics where a uniform value of 0 (or all-null) across ALL rows indicates
# the metric is not tracked — not that it's genuinely zero.
_ZERO_IS_ABSENT_METRICS: list[re.Pattern[str]] = [
    re.compile(r"\bprepayment\b", re.I),
    re.compile(r"\bltv\b|loan.to.value", re.I),
    re.compile(r"\bnet.charge.off\b|\bnco\b", re.I),
    re.compile(r"\bbenchmark\b", re.I),
    re.compile(r"\bmarket.rate\b", re.I),
]


def _detect_fabricated_zero(
    question: str, rows: list[dict[Any, Any]]
) -> str | None:
    """
    Return a reason string if ALL numeric values in the returned rows are 0
    (or None) for a metric category that cannot legitimately be zero.

    This catches the pattern where the analytics API fabricates 0.0 from an
    empty aggregate column rather than returning 'no data'.
    """
    for pattern in _ZERO_IS_ABSENT_METRICS:
        if not pattern.search(question):
            continue
        # Collect all numeric values from result rows
        numeric_values: list[float] = []
        for row in rows[:20]:
            for v in row.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_values.append(float(v))
        if numeric_values and all(abs(v) < 1e-9 for v in numeric_values):
            return (
                f"All returned values are 0 for a metric '{pattern.pattern}' "
                "that cannot genuinely be zero. "
                "The field is likely absent from this dataset."
            )
    return None


def _validate_rows_for_question(
    question: str, rows: list[dict[Any, Any]]
) -> str | None:
    """
    Check whether the returned row columns are plausibly related to the question.

    Returns None when everything looks fine, or a short reason string when the
    columns look unrelated (hallucination risk).
    """
    if not rows:
        return None  # handled elsewhere

    # Collect all column names from the first few rows
    col_names: set[str] = set()
    for row in rows[:5]:
        col_names.update(str(k).lower() for k in row.keys())

    cols_joined = " ".join(col_names)

    for subject_re, expected_fragments in _SUBJECT_COLUMN_MAP:
        if subject_re.search(question):
            if not any(frag in cols_joined for frag in expected_fragments):
                return (
                    f"Question asks about '{subject_re.pattern}' but returned columns "
                    f"({', '.join(sorted(col_names)[:8])}) contain none of the expected "
                    f"fields ({', '.join(expected_fragments)}). "
                    "The data is unrelated to the question."
                )
    return None


def _no_matching_field_chunk(question: str, reason: str) -> RetrievedChunk:
    """Return a structured 'field not available' chunk to prevent hallucination."""
    return RetrievedChunk(
        chunk_id="analytics_no_matching_field",
        source_type="db",
        source_ref="bigquery:credit-risk-platform",
        content=(
            f"[Credit Risk Analytics Data]\n"
            f"Question: {question}\n"
            f"Result: This metric is not available in this dataset. "
            f"Status: not available.\n"
            f"Reason: {reason}\n"
            "The requested data is not available and cannot be computed from this portfolio dataset. "
            "Do not fabricate or estimate the answer. "
            "Report that this metric is not available."
        ),
        relevance="RELEVANT",
        similarity_score=0.9,
    )


# ---------------------------------------------------------------------------
# Core async request helper
# ---------------------------------------------------------------------------

async def _call_analytics_api(
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute a single POST to the analytics API. Returns parsed JSON or None on error."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=_make_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning(
            "analytics_api_tool: HTTP %s — %s",
            exc.response.status_code,
            exc.response.text[:300],
        )
        return None
    except Exception as exc:
        log.warning("analytics_api_tool: request failed — %s", exc)
        return None


async def ask_analytics(
    question: str,
    hint: str | None = None,
    clarifications: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    """
    Ask a natural language question against the credit-risk BigQuery datasets.

    Parameters
    ----------
    question:
        Natural language question.
    hint:
        Optional free-text extra context.
    clarifications:
        Dict of answers to prior clarification questions, keyed by ambiguity id
        (e.g. ``{"product_type": "Personal loans", "time_period": "2023"}``).
        When provided, the analytics API skips ambiguity detection and injects
        the answers directly into the SQL-generation prompt.

    Returns
    -------
    list[RetrievedChunk]
        • On success: single DB chunk with BQ rows (includes assumed_defaults).
        • On ambiguity (blocking only): single clarification chunk — UNLESS the
          question is portfolio-wide, in which case auto-retries with
          product_type="All combined".
        • On no_matching_field (hallucination guard): structured refusal chunk.
        • On empty result: single no-data chunk so the LLM can respond gracefully.
        • On failure: empty list.
    """
    base_url = _analytics_base_url()
    url = f"{base_url}/v1/analytics/s2s/ask"

    payload: dict[str, Any] = {"question": question}
    if hint:
        payload["hint"] = hint
    if clarifications:
        payload["clarifications"] = clarifications

    data = await _call_analytics_api(url, payload)
    if data is None:
        return []

    # --- Clarification needed ---
    if data.get("needs_clarification"):
        items = data.get("clarification_items") or []
        if not items:
            items = [{
                "id": "product_type",
                "question": data.get("clarification_question", "Which type?"),
                "options": data.get("clarification_options") or [],
            }]

        # Auto product_type fallback: if the only clarification is product_type
        # AND the question is clearly portfolio-wide, retry with "All combined"
        # so the user doesn't need a second turn.
        pending_ids = [it.get("id") for it in items]
        if (
            pending_ids == ["product_type"]
            and _is_portfolio_wide(question)
            and not (clarifications or {}).get("product_type")
        ):
            log.info(
                "analytics_api_tool: auto product_type=All combined for portfolio-wide question"
            )
            retry_clarifications = dict(clarifications or {})
            retry_clarifications["product_type"] = "All combined"
            retry_payload: dict[str, Any] = {"question": question}
            if hint:
                retry_payload["hint"] = hint
            retry_payload["clarifications"] = retry_clarifications

            retry_data = await _call_analytics_api(url, retry_payload)
            if retry_data and not retry_data.get("needs_clarification"):
                data = retry_data
                # Fall through to normal row processing below
            else:
                # Retry also needs clarification — surface original items to user
                return [_clarification_to_chunk(question=question, items=items)]
        else:
            return [_clarification_to_chunk(question=question, items=items)]

    assumed_defaults: list[str] = data.get("assumed_defaults") or []

    rows = data.get("rows") or []
    if not rows:
        generated_sql = data.get("generated_sql", "")
        content_lines = [
            "[Credit Risk Analytics Data]",
            f"Question: {question}",
            "Result: No data found for this query.",
        ]
        if assumed_defaults:
            content_lines.append(f"Assumed defaults: {'; '.join(assumed_defaults)}")
        if generated_sql:
            content_lines.append(f"SQL executed: {generated_sql[:400]}")
        return [RetrievedChunk(
            chunk_id="analytics_bq_no_data",
            source_type="db",
            source_ref="bigquery:credit-risk-platform",
            content="\n".join(content_lines),
            relevance="RELEVANT",
            similarity_score=0.8,
        )]

    # --- Hallucination guard: validate columns against question subject ---
    mismatch_reason = _validate_rows_for_question(question, rows)
    if mismatch_reason:
        log.warning(
            "analytics_api_tool: hallucination_guard triggered — %s", mismatch_reason
        )
        return [_no_matching_field_chunk(question, mismatch_reason)]

    # --- Hallucination guard: detect fabricated-zero values ---
    zero_reason = _detect_fabricated_zero(question, rows)
    if zero_reason:
        log.warning(
            "analytics_api_tool: fabricated_zero_guard triggered — %s", zero_reason
        )
        return [_no_matching_field_chunk(question, zero_reason)]

    return [_rows_to_chunk(
        question=question,
        rows=rows,
        row_count=data.get("row_count", len(rows)),
        truncated=data.get("truncated", False),
        assumed_defaults=assumed_defaults,
    )]
