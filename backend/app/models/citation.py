"""
backend/app/models/citation.py
================================
SQLAlchemy 2.0 ORM model for the citations table.

Each row records one grounded claim from a copilot session, linking the
claim text to its retrieved source so the audit trail is fully traceable.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class Citation(Base):
    __tablename__ = "citations"

    citation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("copilot_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # api | db | vector_doc
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_citations_session_id", "session_id"),
    )
