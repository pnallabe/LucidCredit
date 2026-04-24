"""
LucidCredit application configuration.
All values read from environment variables (via .env in development).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DEPRECATED: used ONLY when llm_provider_mode == "openai_direct" (dev only).
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"

    # === Azure OpenAI (Primary — required in staging/prod) ===
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment_analyst: str = "gpt-4.1-2025-04-14"
    azure_openai_deployment_applicant: str = "gpt-4.1-2025-04-14"
    azure_openai_deployment_embedding: str = "text-embedding-3-large"

    # === Vertex AI / Gemini (Fallback) ===
    google_project_id: str = ""
    google_location: str = "us-central1"
    vertex_model_analyst_fallback: str = "gemini-1.5-pro-002"
    vertex_model_batch_briefing: str = "gemini-2.0-flash-001"

    # === Provider routing ===
    # "azure" = Azure OpenAI (required in staging/production)
    # "openai_direct" = direct OpenAI API (development only)
    llm_provider_mode: Literal["azure", "openai_direct"] = "azure"
    analyst_fallback_enabled: bool = True
    # A/B gate for Gemini 2.0 Flash on /v1/briefing/generate — disabled until RAGAS gate passes
    briefing_use_flash: bool = False

    database_url: str = "postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit"
    vector_db_url: str = "postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit"

    redis_url: str = "redis://localhost:6380"

    crp_api_base_url: str = "http://localhost:8081"
    crp_api_key: str = ""

    thinfile_api_base_url: str = "http://localhost:8000"
    thinfile_api_key: str = ""

    min_citation_similarity: float = 0.85
    min_confidence_score: float = 0.75

    # Admin API key for protected endpoints (e.g. POST /v1/briefing/evaluate-flash)
    admin_api_key: str = ""

    # AgentHiveHQ Integration Gateway
    # Incoming shared key — must match LUCIDCREDIT_API_KEY in AgentHiveHQ's .env
    # Leave blank in development to skip auth (all requests pass through).
    agenthive_incoming_key: str = ""

    environment: str = "development"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_provider_config(self) -> "Settings":
        if self.llm_provider_mode == "azure" and self.azure_openai_endpoint == "":
            raise ValueError(
                "azure_openai_endpoint must be set when llm_provider_mode == 'azure'"
            )
        if self.environment != "development" and self.llm_provider_mode == "openai_direct":
            raise ValueError(
                "llm_provider_mode == 'openai_direct' is only valid in development environments"
            )
        if self.analyst_fallback_enabled and self.google_project_id == "":
            raise ValueError(
                "google_project_id must be set when analyst_fallback_enabled == True"
            )
        return self

    def model_version_hash(self, provider: str, deployment: str) -> str:
        """
        Return a deterministic model identifier string stored in every copilot_sessions row.

        Examples:
            model_version_hash("azure", "gpt-4.1-2025-04-14") -> "azure:gpt-4.1-2025-04-14"
            model_version_hash("vertex", "gemini-1.5-pro-002") -> "vertex:gemini-1.5-pro-002"
        """
        return f"{provider}:{deployment}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
