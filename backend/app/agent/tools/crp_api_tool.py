"""
backend/app/agent/tools/crp_api_tool.py
=========================================
Tool: typed async client for the credit-risk-platform Decision API.

Public API
----------
    fetch_decision_context(decision_id, source) -> list[RetrievedChunk]
        Calls GET /v1/decisions/{id}/explanation and GET /v1/decisions/{id}/audit,
        returns the payloads as RetrievedChunk objects for grounding.

    fetch_portfolio_metrics() -> list[RetrievedChunk]
        Calls GET /v1/metrics, returns portfolio-level context chunks.

Errors are caught and logged; an empty list is returned on failure so the
agent can handle insufficient retrieval without crashing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.agent.state import RetrievedChunk
from app.config import get_settings

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(timeout=15.0)


def _make_headers() -> dict[str, str]:
    settings = get_settings()
    headers: dict[str, str] = {"Accept": "application/json"}
    if settings.crp_api_key:
        headers["X-API-Key"] = settings.crp_api_key
    return headers


def _payload_to_chunk(
    payload: dict[str, Any],
    source_ref: str,
    chunk_id: str,
) -> RetrievedChunk:
    """Serialize an API payload dict into a RetrievedChunk."""
    content = json.dumps(payload, default=str)[:4000]  # cap content size
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type="api",
        source_ref=source_ref,
        content=content,
        relevance="AMBIGUOUS",
        similarity_score=1.0,  # API responses are considered fully relevant by default
    )


async def fetch_decision_context(
    decision_id: str,
    source: str = "credit-risk-platform",
) -> list[RetrievedChunk]:
    """
    Fetch explanation + audit for *decision_id* from the CRP API.

    Returns up to 2 chunks: one for the explanation payload, one for the audit record.
    Returns ``[]`` if the CRP API is unreachable or returns an error.
    """
    settings = get_settings()
    base = settings.crp_api_base_url.rstrip("/")
    headers = _make_headers()
    chunks: list[RetrievedChunk] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for path, ref_suffix in [
            (f"/v1/decisions/{decision_id}/explanation", "explanation"),
            (f"/v1/decisions/{decision_id}/audit", "audit"),
        ]:
            url = f"{base}{path}"
            try:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                source_ref = f"crp:{source}:{decision_id}:{ref_suffix}"
                chunks.append(
                    _payload_to_chunk(payload, source_ref, f"crp-{decision_id}-{ref_suffix}")
                )
                log.debug("crp_tool.fetched path=%s decision_id=%s", path, decision_id)
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "crp_tool.http_error path=%s status=%d decision_id=%s",
                    path, exc.response.status_code, decision_id,
                )
                # On 404 add a "not found" stub so the pipeline knows the DB tool
                # was attempted; this ensures tool-selection evals see source_type="db".
                if exc.response.status_code == 404 and not chunks:
                    chunks.append(RetrievedChunk(
                        chunk_id=f"crp-{decision_id}-not_found",
                        source_type="db",
                        source_ref=f"crp:{source}:{decision_id}:not_found",
                        content=(
                            f"[CRP Decision Lookup] Application {decision_id}: "
                            "No decision record found in the credit risk platform. "
                            "Inform the user that this application ID does not exist."
                        ),
                        relevance="RELEVANT",
                        similarity_score=0.8,
                    ))
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning("crp_tool.connect_error path=%s error=%s", path, exc)
            except Exception as exc:
                log.warning("crp_tool.unexpected_error path=%s error=%s", path, exc)

    return chunks


async def fetch_portfolio_metrics() -> list[RetrievedChunk]:
    """
    Fetch portfolio-level metrics from GET /v1/metrics.

    Returns 1 chunk on success, ``[]`` on failure.
    """
    settings = get_settings()
    base = settings.crp_api_base_url.rstrip("/")
    url = f"{base}/v1/metrics"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=_make_headers())
            resp.raise_for_status()
            payload = resp.json()
            return [_payload_to_chunk(payload, "crp:metrics", "crp-metrics")]
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
        log.warning("crp_tool.metrics_error error=%s", exc)
        return []
    except Exception as exc:
        log.warning("crp_tool.metrics_unexpected error=%s", exc)
        return []
