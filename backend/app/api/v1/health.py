"""
backend/app/api/v1/health.py
==============================
Health check endpoints.

GET  /v1/health          — basic liveness check
GET  /v1/health/providers — LLM circuit breaker state for all providers
                            (consumed by ops team during incidents)
"""
from __future__ import annotations

from fastapi import APIRouter

from app.llm.circuit_breaker import get_all_states

router = APIRouter()


@router.get("")
async def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@router.get("/providers")
async def provider_health() -> dict:
    """
    Returns the circuit breaker state for each LLM provider.

    Example response:
        {
          "azure":  {"state": "CLOSED", "failures": 0},
          "vertex": {"state": "CLOSED", "failures": 0}
        }
    """
    return await get_all_states()
