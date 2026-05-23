"""add_copilot_sessions_citations_feedback

Revision ID: 001
Revises:
Create Date: 2026-04-24

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "copilot_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=50), nullable=False),
        sa.Column("audience", sa.String(length=20), nullable=False),
        sa.Column("retrieved_chunks", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("rendered_prompt", sa.Text(), nullable=True),
        sa.Column("raw_llm_output", sa.Text(), nullable=True),
        sa.Column("grounded_narrative", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("suppressed_claims", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("compliance_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("source_system", sa.String(length=50), nullable=True),
        sa.Column("user_id", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("provider_model", sa.String(length=100), nullable=False, server_default=""),
        sa.Column("provider_fallback_used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("context_token_count", sa.Integer(), nullable=True),
        sa.Column("output_token_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_copilot_sessions_user_id", "copilot_sessions", ["user_id"])
    op.create_index("ix_copilot_sessions_created_at", "copilot_sessions", ["created_at"])
    op.create_index("ix_copilot_sessions_provider_model", "copilot_sessions", ["provider_model"])
    op.create_index(
        "ix_copilot_sessions_confidence_score", "copilot_sessions", ["confidence_score"]
    )

    op.create_table(
        "citations",
        sa.Column("citation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["copilot_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("citation_id"),
    )
    op.create_index("ix_citations_session_id", "citations", ["session_id"])

    op.create_table(
        "user_feedback",
        sa.Column("feedback_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=True),
        sa.Column("useful", sa.Boolean(), nullable=True),
        sa.Column("correction", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["copilot_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index("ix_user_feedback_session_id", "user_feedback", ["session_id"])


def downgrade() -> None:
    op.drop_table("user_feedback")
    op.drop_table("citations")
    op.drop_index("ix_copilot_sessions_confidence_score", table_name="copilot_sessions")
    op.drop_index("ix_copilot_sessions_provider_model", table_name="copilot_sessions")
    op.drop_index("ix_copilot_sessions_created_at", table_name="copilot_sessions")
    op.drop_index("ix_copilot_sessions_user_id", table_name="copilot_sessions")
    op.drop_table("copilot_sessions")
