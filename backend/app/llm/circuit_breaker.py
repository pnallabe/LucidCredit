"""
backend/app/llm/circuit_breaker.py
=====================================
Lightweight in-process + Redis-distributed circuit breaker for LLM providers.

State machine:  CLOSED  →  OPEN  →  HALF_OPEN  →  CLOSED
Per-provider breaker keyed by provider name: "azure" | "vertex"

Configuration (hardcoded — no settings fields needed):
    FAILURE_THRESHOLD      = 3    consecutive failures to trip OPEN
    RESET_TIMEOUT_SECONDS  = 120  seconds in OPEN before trying HALF_OPEN
    HALF_OPEN_MAX_CALLS    = 1    only 1 probe call allowed in HALF_OPEN state

In production (environment == "production"), state is shared across FastAPI
worker processes via Redis. In development, in-process dict is used.

Redis key pattern:
    lucidcredit:circuit:{provider}:failures  (integer counter, TTL = RESET_TIMEOUT_SECONDS)
    lucidcredit:circuit:{provider}:state     (string: CLOSED | OPEN | HALF_OPEN)

Public API:
    async def record_success(provider: str) -> None
    async def record_failure(provider: str) -> None
    async def is_open(provider: str) -> bool       True = DO NOT USE
    async def get_state(provider: str) -> str      "CLOSED" | "OPEN" | "HALF_OPEN"
"""
from __future__ import annotations

import time
from typing import Literal

from app.config import get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_THRESHOLD: int = 3
RESET_TIMEOUT_SECONDS: int = 120
HALF_OPEN_MAX_CALLS: int = 1

ProviderState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

# ---------------------------------------------------------------------------
# In-process state (used in development / when Redis is not required)
# ---------------------------------------------------------------------------

_in_process: dict[str, dict] = {}


def _get_local(provider: str) -> dict:
    if provider not in _in_process:
        _in_process[provider] = {
            "state": "CLOSED",
            "failures": 0,
            "open_since": None,
            "half_open_calls": 0,
        }
    return _in_process[provider]


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

async def _get_redis_client():
    """Return an async Redis client. Import lazily to avoid hard dep in dev."""
    import redis.asyncio as aioredis

    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _failures_key(provider: str) -> str:
    return f"lucidcredit:circuit:{provider}:failures"


def _state_key(provider: str) -> str:
    return f"lucidcredit:circuit:{provider}:state"


def _open_since_key(provider: str) -> str:
    return f"lucidcredit:circuit:{provider}:open_since"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

async def _check_and_transition_open(provider: str) -> None:
    """
    If the provider's state is OPEN and RESET_TIMEOUT_SECONDS have elapsed,
    transition to HALF_OPEN to allow a probe call.
    """
    settings = get_settings()
    if settings.environment == "production":
        r = await _get_redis_client()
        state = await r.get(_state_key(provider)) or "CLOSED"
        if state == "OPEN":
            open_since_raw = await r.get(_open_since_key(provider))
            if open_since_raw:
                elapsed = time.time() - float(open_since_raw)
                if elapsed >= RESET_TIMEOUT_SECONDS:
                    await r.set(_state_key(provider), "HALF_OPEN")
    else:
        local = _get_local(provider)
        if local["state"] == "OPEN" and local["open_since"] is not None:
            elapsed = time.time() - local["open_since"]
            if elapsed >= RESET_TIMEOUT_SECONDS:
                local["state"] = "HALF_OPEN"
                local["half_open_calls"] = 0


async def record_success(provider: str) -> None:
    """Reset the circuit to CLOSED on successful call."""
    settings = get_settings()
    if settings.environment == "production":
        r = await _get_redis_client()
        await r.set(_state_key(provider), "CLOSED")
        await r.delete(_failures_key(provider))
        await r.delete(_open_since_key(provider))
    else:
        local = _get_local(provider)
        local["state"] = "CLOSED"
        local["failures"] = 0
        local["open_since"] = None
        local["half_open_calls"] = 0


async def record_failure(provider: str) -> None:
    """
    Increment failure counter.
    When failures >= FAILURE_THRESHOLD, trip the circuit OPEN.
    In HALF_OPEN, any failure re-opens the circuit.
    """
    settings = get_settings()
    if settings.environment == "production":
        r = await _get_redis_client()
        state = await r.get(_state_key(provider)) or "CLOSED"

        if state == "HALF_OPEN":
            # Re-open immediately
            await r.set(_state_key(provider), "OPEN")
            await r.set(_open_since_key(provider), str(time.time()))
            await r.set(_failures_key(provider), str(FAILURE_THRESHOLD))
            await r.expire(_failures_key(provider), RESET_TIMEOUT_SECONDS)
            return

        failures = int(await r.get(_failures_key(provider)) or "0") + 1
        await r.set(_failures_key(provider), str(failures))
        await r.expire(_failures_key(provider), RESET_TIMEOUT_SECONDS)

        if failures >= FAILURE_THRESHOLD:
            await r.set(_state_key(provider), "OPEN")
            await r.set(_open_since_key(provider), str(time.time()))
    else:
        local = _get_local(provider)

        if local["state"] == "HALF_OPEN":
            local["state"] = "OPEN"
            local["open_since"] = time.time()
            return

        local["failures"] += 1
        if local["failures"] >= FAILURE_THRESHOLD:
            local["state"] = "OPEN"
            local["open_since"] = time.time()


async def is_open(provider: str) -> bool:
    """
    Return True if the circuit is OPEN (i.e., do NOT use this provider).
    Also handles OPEN → HALF_OPEN transition when timeout has elapsed.
    """
    await _check_and_transition_open(provider)
    state = await get_state(provider)
    return state == "OPEN"


async def get_state(provider: str) -> ProviderState:
    """Return the current circuit state for the provider."""
    await _check_and_transition_open(provider)

    settings = get_settings()
    if settings.environment == "production":
        r = await _get_redis_client()
        raw = await r.get(_state_key(provider))
        if raw in ("CLOSED", "OPEN", "HALF_OPEN"):
            return raw  # type: ignore[return-value]
        return "CLOSED"
    else:
        return _get_local(provider)["state"]  # type: ignore[return-value]


async def get_all_states() -> dict[str, dict]:
    """
    Return circuit state for all known providers.
    Used by GET /v1/health/providers.
    """
    result: dict[str, dict] = {}
    settings = get_settings()

    for provider in ("azure", "vertex"):
        state = await get_state(provider)
        if settings.environment == "production":
            r = await _get_redis_client()
            failures = int(await r.get(_failures_key(provider)) or "0")
        else:
            failures = _get_local(provider)["failures"]

        result[provider] = {"state": state, "failures": failures}

    return result
