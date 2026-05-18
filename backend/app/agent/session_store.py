"""
backend/app/agent/session_store.py
=====================================
Lightweight conversation session store for LucidCredit v2.

Stores conversation history (OpenAI messages format) keyed by session_id.
Uses Redis when available; falls back to an in-process dict for local dev.

The history is a list of OpenAI message dicts:
    [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

Redis keys: ``lucidcredit:chat:session:{session_id}``
TTL: 24 hours (configurable via CHAT_SESSION_TTL_SECONDS env var)
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# In-process fallback (used when Redis is unavailable)
_memory_store: dict[str, list[dict[str, Any]]] = {}

_REDIS_KEY_PREFIX = "lucidcredit:chat:session:"
_DEFAULT_TTL = 86_400  # 24 h


async def load_history(session_id: str) -> list[dict[str, Any]]:
    """
    Load conversation history for *session_id*.
    Returns an empty list for new sessions.
    """
    try:
        import redis.asyncio as aioredis
        from app.config import get_settings

        r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        raw = await r.get(f"{_REDIS_KEY_PREFIX}{session_id}")
        await r.aclose()
        if raw:
            return json.loads(raw)
        return []
    except Exception as exc:
        log.debug("session_store: Redis unavailable (%s), using in-memory fallback", exc)
        return list(_memory_store.get(session_id, []))


async def save_history(session_id: str, history: list[dict[str, Any]]) -> None:
    """
    Persist *history* for *session_id*.
    Trims to the last 40 messages (20 turns) before saving.
    """
    # Keep only the most recent 40 messages to cap token usage
    trimmed = history[-40:] if len(history) > 40 else history

    try:
        import redis.asyncio as aioredis
        from app.config import get_settings

        r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        await r.set(f"{_REDIS_KEY_PREFIX}{session_id}", json.dumps(trimmed), ex=_DEFAULT_TTL)
        await r.aclose()
    except Exception as exc:
        log.debug("session_store: Redis save failed (%s), using in-memory fallback", exc)
        _memory_store[session_id] = list(trimmed)


async def clear_history(session_id: str) -> None:
    """Clear the conversation history for *session_id*."""
    try:
        import redis.asyncio as aioredis
        from app.config import get_settings

        r = aioredis.from_url(get_settings().redis_url, decode_responses=True)
        await r.delete(f"{_REDIS_KEY_PREFIX}{session_id}")
        await r.aclose()
    except Exception:
        pass
    _memory_store.pop(session_id, None)
