"""
backend/app/db/session.py
==========================
SQLAlchemy 2.0 async engine + session factory.

Also defines the `policy_docs` ORM model that the RAG ingestor writes to.
The pgvector extension and table are created at startup via `init_db()`.

Usage
-----
    from app.db.session import AsyncSessionLocal, init_db, PolicyDoc
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import AsyncGenerator

from sqlalchemy import Column, Date, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------

def _make_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


# Module-level singletons; re-created on first import.
_engine = _make_engine()
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# policy_docs table
# ---------------------------------------------------------------------------

class PolicyDoc(Base):
    """
    One row per text chunk from a governance Markdown document.

    The pgvector `embedding` column is handled as a raw TEXT/ARRAY column here
    so the model file has no hard pgvector Python dependency.  The ingestor
    writes the vector as a plain Python list; SQLAlchemy passes it to
    asyncpg which serialises it to the `vector` Postgres type correctly once
    the extension is active.
    """
    __tablename__ = "policy_docs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_path = Column(String(1024), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    # The actual column type is vector(3072) for text-embedding-3-large.
    # We declare it as Text here and rely on the DDL in init_db() to create
    # it with the correct pgvector type.
    embedding = Column(Text, nullable=True)

    source_type = Column(String(50), nullable=True)    # policy_manual | committee_minutes | etc.
    product_type = Column(String(50), nullable=True)   # credit_card | personal_loan | mortgage
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    version_tag = Column(String(200), nullable=True)
    doc_type = Column(String(100), nullable=True)
    metadata_json = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_policy_docs_source_path_chunk", "source_path", "chunk_index", unique=True),
        Index("ix_policy_docs_doc_type", "doc_type"),
        Index("ix_policy_docs_product_type", "product_type"),
        Index("ix_policy_docs_effective_from", "effective_from"),
    )


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_docs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path     VARCHAR(1024) NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector(3072),
    source_type     VARCHAR(50),
    product_type    VARCHAR(50),
    effective_from  DATE,
    effective_to    DATE,
    version_tag     VARCHAR(200),
    doc_type        VARCHAR(100),
    metadata_json   JSONB,
    CONSTRAINT uq_policy_docs_path_chunk UNIQUE (source_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_policy_docs_doc_type
    ON policy_docs (doc_type);
CREATE INDEX IF NOT EXISTS ix_policy_docs_product_type
    ON policy_docs (product_type);
CREATE INDEX IF NOT EXISTS ix_policy_docs_effective_from
    ON policy_docs (effective_from);
"""


async def init_db() -> None:
    """
    Bootstrap the database: enable pgvector extension and create `policy_docs`
    if it does not already exist.  Safe to call multiple times (all DDL uses
    IF NOT EXISTS).

    In development, also runs Base.metadata.create_all for the ORM models in
    app.models (copilot_sessions, citations, user_feedback).  In staging and
    production, Alembic manages schema exclusively — never call create_all there.
    """
    async with _engine.begin() as conn:
        for _stmt in _INIT_SQL.split(";"):
            _stmt = _stmt.strip()
            if _stmt:
                await conn.exec_driver_sql(_stmt)

    settings = get_settings()
    if settings.environment == "development":
        from app.models import Base as ModelsBase

        async with _engine.begin() as conn:
            await conn.run_sync(ModelsBase.metadata.create_all)
