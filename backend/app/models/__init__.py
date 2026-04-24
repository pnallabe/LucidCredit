"""
backend/app/models/__init__.py
================================
Exports all ORM models and the shared DeclarativeBase.

Import this module to register all models with Alembic's metadata scanner:
    from app.models import Base, CopilotSession, Citation, UserFeedback
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models so their metadata is registered on Base
from app.models.session import CopilotSession  # noqa: E402, F401
from app.models.citation import Citation  # noqa: E402, F401
from app.models.feedback import UserFeedback  # noqa: E402, F401

__all__ = ["Base", "CopilotSession", "Citation", "UserFeedback"]
