"""
backend/app/models/feedback.py
================================
SQLAlchemy 2.0 ORM model for the user_feedback table.

Captures analyst corrections and rating signals for RLHF-style fine-tuning
and ongoing hallucination regression tracking.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Index, SmallInteger, Text
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    feedback_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("copilot_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)  # 1–5
    useful: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    correction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_user_feedback_session_id", "session_id"),
    )
