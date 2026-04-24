"""
backend/app/main.py
====================
FastAPI application entry point for LucidCredit.
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.session import init_db

log = structlog.get_logger(__name__)

app = FastAPI(
    title="LucidCredit Copilot API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    settings = get_settings()
    log.info("startup", environment=settings.environment)
    await init_db()

    # Restore the briefing Flash gate from Redis on startup
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        flag = await r.get("lucidcredit:feature_flags:briefing_flash")
        if flag == "true":
            settings.briefing_use_flash = True
            log.info("briefing_flash_gate_restored", value=True)
        await r.aclose()
    except Exception as exc:
        log.warning("briefing_flash_gate_restore_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.api.v1.health import router as health_router  # noqa: E402
from app.api.v1.applicant import router as applicant_router  # noqa: E402
from app.api.v1.audit import router as audit_router  # noqa: E402
from app.api.v1.briefing import router as briefing_router  # noqa: E402
from app.api.v1.explain import router as explain_router  # noqa: E402
from app.api.v1.query import router as query_router  # noqa: E402

app.include_router(health_router, prefix="/v1/health", tags=["health"])
app.include_router(applicant_router, prefix="/v1/applicant", tags=["applicant"])
app.include_router(audit_router, prefix="/v1/audit", tags=["audit"])
app.include_router(briefing_router, prefix="/v1/briefing", tags=["briefing"])
app.include_router(explain_router, prefix="/v1/explain", tags=["explain"])
app.include_router(query_router, prefix="/v1/query", tags=["query"])
