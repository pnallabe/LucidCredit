"""
backend/app/agent/tools/thinfile_tool.py
==========================================
Tool: typed async client for the ThinFile Credit Underwriting Engine API.

Public API
----------
    fetch_thin_file_context(application_id, source) -> list[RetrievedChunk]
        Calls POST /score and GET /audit/logs/{id} to retrieve a live scoring
        result and the feature snapshot used for grounding thin-file applicant
        explanations.

    fetch_adverse_action_codes(application_id) -> list[RetrievedChunk]
        Calls GET /score/{id}/adverse-action and returns the structured
        adverse-action code payload as a RetrievedChunk for compliance grounding.

All errors are caught and logged; an empty list is returned on failure so the
agent can handle insufficient retrieval without crashing.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import httpx

from app.agent.state import RetrievedChunk
from app.config import get_settings

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(timeout=20.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_headers() -> Dict[str, str]:
    settings = get_settings()
    headers: Dict[str, str] = {"Accept": "application/json"}
    if settings.thinfile_api_key:
        headers["X-API-Key"] = settings.thinfile_api_key
    return headers


def _payload_to_chunk(
    payload: Dict[str, Any],
    source_ref: str,
    chunk_id: str,
) -> RetrievedChunk:
    content = json.dumps(payload, default=str)[:4000]
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type="api",
        source_ref=source_ref,
        content=content,
        relevance="AMBIGUOUS",
        similarity_score=1.0,
    )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

async def fetch_thin_file_context(
    application_id: str,
    source: str = "thinfile",
) -> list:
    """
    Retrieve live score + feature snapshot for *application_id* from the
    ThinFile Engine.

    Calls:
      - POST /score (live scoring with application context)
      - GET /audit/logs/{application_id} (feature snapshot for grounding)

    Returns up to 2 RetrievedChunk objects, or ``[]`` on error.
    """
    settings = get_settings()
    base = settings.thinfile_api_base_url.rstrip("/")
    headers = _make_headers()
    chunks: list = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        # --- Live score -------------------------------------------------------
        score_url = f"{base}/score"
        try:
            resp = await client.post(
                score_url,
                headers=headers,
                json={"application_id": application_id},
            )
            resp.raise_for_status()
            payload = resp.json()
            chunks.append(
                _payload_to_chunk(
                    payload,
                    source_ref=f"thinfile:{source}:{application_id}:score",
                    chunk_id=f"thinfile-{application_id}-score",
                )
            )
            log.debug("thinfile_tool.score_fetched application_id=%s", application_id)
        except httpx.HTTPStatusError as exc:
            log.warning(
                "thinfile_tool.score_http_error status=%d application_id=%s",
                exc.response.status_code, application_id,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("thinfile_tool.score_connect_error error=%s", exc)
        except Exception as exc:
            log.warning("thinfile_tool.score_unexpected_error error=%s", exc)

        # --- Audit / feature snapshot -----------------------------------------
        audit_url = f"{base}/audit/logs/{application_id}"
        try:
            resp = await client.get(audit_url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            chunks.append(
                _payload_to_chunk(
                    payload,
                    source_ref=f"thinfile:{source}:{application_id}:audit",
                    chunk_id=f"thinfile-{application_id}-audit",
                )
            )
            log.debug("thinfile_tool.audit_fetched application_id=%s", application_id)
        except httpx.HTTPStatusError as exc:
            log.warning(
                "thinfile_tool.audit_http_error status=%d application_id=%s",
                exc.response.status_code, application_id,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("thinfile_tool.audit_connect_error error=%s", exc)
        except Exception as exc:
            log.warning("thinfile_tool.audit_unexpected_error error=%s", exc)

    return chunks


async def fetch_adverse_action_codes(
    application_id: str,
) -> list:
    """
    Fetch ECOA adverse-action codes for *application_id* via GET /score/{id}/adverse-action.

    Returns 1 RetrievedChunk on success, ``[]`` on failure.
    """
    settings = get_settings()
    base = settings.thinfile_api_base_url.rstrip("/")
    url = f"{base}/score/{application_id}/adverse-action"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=_make_headers())
            resp.raise_for_status()
            payload = resp.json()
            return [
                _payload_to_chunk(
                    payload,
                    source_ref=f"thinfile:adverse_action:{application_id}",
                    chunk_id=f"thinfile-{application_id}-adverse-action",
                )
            ]
    except httpx.HTTPStatusError as exc:
        log.warning(
            "thinfile_tool.adverse_action_http_error status=%d application_id=%s",
            exc.response.status_code, application_id,
        )
        return []
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        log.warning("thinfile_tool.adverse_action_connect_error error=%s", exc)
        return []
    except Exception as exc:
        log.warning("thinfile_tool.adverse_action_unexpected_error error=%s", exc)
        return []
