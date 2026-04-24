"""
backend/app/llm/provider.py
============================
LLM provider abstraction layer for LucidCredit.

Public API
----------
    get_chat_client(session_type) -> BaseChatModel
        Returns an AzureChatOpenAI (or ChatOpenAI in dev) for the given session type.

    get_fallback_client(session_type) -> BaseChatModel
        Returns a ChatVertexAI fallback. Applicant sessions explicitly forbidden.

    get_embedding_client() -> AsyncOpenAI
        Returns an Azure OpenAI async client for embeddings only.

Exceptions
----------
    FallbackNotAvailableError — raised when Vertex fallback is not configured or disabled.
"""
from __future__ import annotations

import functools
from typing import Literal

from langchain_core.language_models import BaseChatModel
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.config import get_settings


class FallbackNotAvailableError(RuntimeError):
    """Raised when the Vertex AI fallback is not available or not configured."""


# ---------------------------------------------------------------------------
# Private cached factories
# The cache key is (endpoint, deployment, api_version) so instances are reused
# across requests without being module-level singletons that break patching.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _make_azure_chat_client(
    endpoint: str,
    deployment: str,
    api_version: str,
    api_key: str,
) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,  # type: ignore[arg-type]
        temperature=0.0,
        max_retries=3,
        request_timeout=60.0,
        streaming=True,
    )


@functools.lru_cache(maxsize=4)
def _make_direct_chat_client(api_key: str, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=api_key,  # type: ignore[arg-type]
        model=model,
        temperature=0.0,
        max_retries=3,
        request_timeout=60.0,
        streaming=True,
    )


@functools.lru_cache(maxsize=4)
def _make_vertex_chat_client(
    model_name: str,
    project: str,
    location: str,
) -> BaseChatModel:
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        model_name=model_name,
        project=project,
        location=location,
        temperature=0.0,
        max_retries=2,
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_chat_client(
    session_type: Literal["analyst", "applicant", "briefing"],
    *,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Return the primary LLM client for the given session type.

    - In "azure" mode (staging/production): returns AzureChatOpenAI pointed at the
      correct dated deployment.
    - In "openai_direct" mode (development only): returns ChatOpenAI.

    ``session_type`` determines the Azure deployment name:
        "applicant"  → azure_openai_deployment_applicant
        "briefing"   → azure_openai_deployment_analyst (unless briefing_use_flash=True,
                        in which case callers should use get_fallback_client)
        "analyst"    → azure_openai_deployment_analyst
    """
    settings = get_settings()

    if settings.llm_provider_mode == "openai_direct":
        # Development only — validated by model_validator at startup
        client = _make_direct_chat_client(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        # streaming override not cached; return a copy if needed
        return client

    # Azure path
    if session_type == "applicant":
        deployment = settings.azure_openai_deployment_applicant
    else:
        # analyst and briefing (non-flash) both use the analyst deployment
        deployment = settings.azure_openai_deployment_analyst

    return _make_azure_chat_client(
        endpoint=settings.azure_openai_endpoint,
        deployment=deployment,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )


def get_fallback_client(
    session_type: Literal["analyst", "briefing"],
) -> BaseChatModel:
    """
    Return the Vertex AI fallback client for analyst or briefing sessions.

    Applicant sessions MUST NOT use this function — raise ValueError immediately.
    Raises FallbackNotAvailableError if the fallback is disabled or unconfigured.
    """
    if session_type == "applicant":  # type: ignore[comparison-overlap]
        raise ValueError(
            "Applicant sessions must not fall back to Vertex AI. Return HTTP 503 instead."
        )

    settings = get_settings()

    if not settings.analyst_fallback_enabled or settings.google_project_id == "":
        raise FallbackNotAvailableError(
            "Vertex AI fallback is not available: analyst_fallback_enabled=False "
            "or google_project_id is empty."
        )

    if session_type == "briefing" and settings.briefing_use_flash:
        model_name = settings.vertex_model_batch_briefing
    else:
        model_name = settings.vertex_model_analyst_fallback

    return _make_vertex_chat_client(
        model_name=model_name,
        project=settings.google_project_id,
        location=settings.google_location,
    )


def get_embedding_client() -> AsyncAzureOpenAI:
    """
    Return an Azure OpenAI async client for embeddings.

    Embeddings always use Azure OpenAI — never Vertex AI.
    Raises NotImplementedError if azure_openai_endpoint is not configured.
    """
    settings = get_settings()

    if not settings.azure_openai_endpoint:
        raise NotImplementedError(
            "Embeddings require Azure OpenAI. Set AZURE_OPENAI_ENDPOINT in environment."
        )

    return AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
