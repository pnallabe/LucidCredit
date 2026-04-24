"""
backend/app/models/session.py
================================
SQLAlchemy 2.0 ORM model for the copilot_sessions table.

Satisfies SR 11-7 model risk audit requirements:
- provider_model stores the exact model identifier (e.g. "azure:gpt-4.1-2025-04-14")
- provider_fallback_used flags when Vertex AI fallback was invoked
- Indexed for model-version impact analysis queries

Alembic manages this schema in staging/production.
In development only, init_db() in app.db.session calls Base.metadata.create_all.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from typing import Optional

from sqlalchemy import Boolean, Float, Index, Integer, String, Text
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class CopilotSession(Base):
    __tablename__ = "copilot_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(50), nullable=False)
    audience: Mapped[str] = mapped_column(String(20), nullable=False)
    retrieved_chunks: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    rendered_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_llm_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    grounded_narrative: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suppressed_claims: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    compliance_flags: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    source_system: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # SR 11-7 model risk columns — track exactly which model produced each output
    provider_model: Mapped[str] = mapped_column(
        String(100), nullable=False, default=""
    )
    provider_fallback_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Approximate token counts for cost attribution and context window monitoring
    context_token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        # user_id: audit access API queries sessions by user
        Index("ix_copilot_sessions_user_id", "user_id"),
        # created_at: range queries for audit log export
        Index("ix_copilot_sessions_created_at", "created_at"),
        # provider_model: SR 11-7 — query sessions by model version for impact analysis
        Index("ix_copilot_sessions_provider_model", "provider_model"),
        # confidence_score: filter sessions below threshold for review
        Index("ix_copilot_sessions_confidence_score", "confidence_score"),
    )
