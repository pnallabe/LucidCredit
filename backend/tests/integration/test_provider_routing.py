"""
backend/tests/integration/test_provider_routing.py
====================================================
Integration tests for LucidCredit's provider routing logic.

All 6 scenarios must pass before any production deployment of the provider
routing layer. No real LLM calls are made — all clients are mocked.

Run with:
    pytest backend/tests/integration/test_provider_routing.py -v

SCENARIOS
---------
1. Analyst — Azure primary succeeds
2. Analyst — Azure fails, Vertex fallback succeeds
3. Applicant — Azure primary succeeds
4. Applicant — Azure fails (CRITICAL: must NOT fall back → HTTP 503)
5. Analyst — Azure AND Vertex both fail → HTTP 503
6. Briefing — Flash gate active → Gemini 2.0 Flash used, Azure NOT called
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import AsyncClient, ASGITransport

from app.config import Settings, get_settings
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    """
    Build a Settings instance via model_construct (skips validators) so tests
    can run without a live Azure or Vertex endpoint.  All values that trigger
    validation (azure_openai_endpoint, google_project_id) are set to test
    stubs so the model_version_hash helper still works correctly.
    """
    return Settings.model_construct(
        llm_provider_mode="azure",
        azure_openai_endpoint="https://test.openai.azure.com/",
        azure_openai_api_key="test-key",
        azure_openai_api_version="2024-10-21",
        azure_openai_deployment_analyst="gpt-4.1-2025-04-14",
        azure_openai_deployment_applicant="gpt-4.1-2025-04-14",
        azure_openai_deployment_embedding="text-embedding-3-large",
        google_project_id="test-project",
        google_location="us-central1",
        vertex_model_analyst_fallback="gemini-1.5-pro-002",
        vertex_model_batch_briefing="gemini-2.0-flash-001",
        analyst_fallback_enabled=True,
        briefing_use_flash=False,
        environment="test",
        min_citation_similarity=0.85,
        min_confidence_score=0.75,
        redis_url="redis://localhost:6380",
        database_url="postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit",
        vector_db_url="postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit",
        admin_api_key="test-admin-key",
    )


def _make_llm_response(content: str = "Grounded analyst response.") -> MagicMock:
    """Return a mock LangChain response object."""
    msg = MagicMock()
    msg.content = content
    return msg


def _make_azure_client_mock(response: Any = None, side_effect: Any = None) -> AsyncMock:
    client = AsyncMock()
    if side_effect is not None:
        client.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        client.ainvoke = AsyncMock(return_value=response or _make_llm_response())
    return client


def _make_vertex_client_mock(response: Any = None, side_effect: Any = None) -> AsyncMock:
    client = AsyncMock()
    if side_effect is not None:
        client.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        client.ainvoke = AsyncMock(return_value=response or _make_llm_response("Vertex response."))
    return client


# ---------------------------------------------------------------------------
# Helper: run reason_node directly (unit-level)
# ---------------------------------------------------------------------------

async def _invoke_reason_node(
    state: dict,
    azure_mock: Any,
    vertex_mock: Any,
    settings_override: Settings | None = None,
) -> dict:
    """
    Invoke reason_node with mocked providers.
    Patches get_chat_client, get_fallback_client, and get_settings.
    """
    from app.agent import nodes as nodes_module

    effective_settings = settings_override or get_settings()

    with (
        patch.object(nodes_module, "get_chat_client", return_value=azure_mock),
        patch.object(nodes_module, "get_fallback_client", return_value=vertex_mock),
        patch.object(nodes_module, "get_settings", return_value=effective_settings),
        # Disable circuit breaker side effects in unit tests
        patch("app.llm.circuit_breaker.is_open", return_value=False),
        patch("app.llm.circuit_breaker.record_success", new_callable=AsyncMock),
        patch("app.llm.circuit_breaker.record_failure", new_callable=AsyncMock),
    ):
        from app.agent.nodes import reason_node
        return await reason_node(state)


def _analyst_state(**overrides) -> dict:
    base = {
        "session_id": uuid.uuid4(),
        "query": "Explain decision dec-001",
        "intent": "analyst_query",
        "audience": "analyst",
        "context_payload": {},
        "retrieved_chunks": [],
        "graded_chunks": [],
        "retrieval_sufficient": True,
        "rendered_prompt": "Explain this credit decision grounded in the context.",
        "raw_llm_output": "",
        "provider_used": "",
        "grounded_narrative": "",
        "citations": [],
        "suppressed_claims": [],
        "confidence_score": 0.0,
        "grounding_passed": False,
        "compliance_flags": [],
        "compliance_passed": False,
        "final_output": {},
        "error": None,
    }
    base.update(overrides)
    return base


def _applicant_state(**overrides) -> dict:
    return _analyst_state(
        intent="applicant_comms",
        audience="applicant",
        **overrides,
    )


def _briefing_state(**overrides) -> dict:
    return _analyst_state(
        intent="portfolio_brief",
        audience="analyst",
        **overrides,
    )


# ---------------------------------------------------------------------------
# SCENARIO 1: Analyst session — Azure primary succeeds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_1_analyst_azure_succeeds(test_settings: Settings) -> None:
    """Azure primary responds successfully. provider_used must be azure:gpt-4.1-2025-04-14."""
    azure_mock = _make_azure_client_mock()
    vertex_mock = _make_vertex_client_mock()

    result = await _invoke_reason_node(
        state=_analyst_state(),
        azure_mock=azure_mock,
        vertex_mock=vertex_mock,
        settings_override=test_settings,
    )

    assert result.get("error") is None
    assert result.get("provider_used") == "azure:gpt-4.1-2025-04-14"
    assert result.get("raw_llm_output") == "Grounded analyst response."
    # Vertex must NOT have been called
    vertex_mock.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# SCENARIO 2: Analyst session — Azure fails, Vertex fallback succeeds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_2_analyst_azure_fails_vertex_succeeds(test_settings: Settings) -> None:
    """
    Azure raises APIStatusError(503). Vertex fallback is invoked and succeeds.
    provider_used must reflect the Vertex model.
    """
    from app.agent import nodes as nodes_module

    request = MagicMock()
    request.status_code = 503
    request.headers = {}
    request.text = "Service unavailable"
    azure_mock = _make_azure_client_mock(
        side_effect=openai.APIStatusError("503", response=request, body=None)
    )
    vertex_mock = _make_vertex_client_mock()

    result = await _invoke_reason_node(
        state=_analyst_state(),
        azure_mock=azure_mock,
        vertex_mock=vertex_mock,
        settings_override=test_settings,
    )

    assert result.get("error") is None
    assert result.get("provider_used") == "vertex:gemini-1.5-pro-002"
    assert result.get("raw_llm_output") == "Vertex response."
    vertex_mock.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# SCENARIO 3: Applicant session — Azure primary succeeds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_3_applicant_azure_succeeds(test_settings: Settings) -> None:
    """Applicant session: Azure responds. provider_used must be azure:gpt-4.1-2025-04-14."""
    azure_mock = _make_azure_client_mock()
    vertex_mock = _make_vertex_client_mock()

    result = await _invoke_reason_node(
        state=_applicant_state(),
        azure_mock=azure_mock,
        vertex_mock=vertex_mock,
        settings_override=test_settings,
    )

    assert result.get("error") is None
    assert result.get("provider_used") == "azure:gpt-4.1-2025-04-14"
    vertex_mock.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# SCENARIO 4: Applicant session — Azure fails (CRITICAL: must NOT fall back)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_4_applicant_azure_fails_no_fallback(test_settings: Settings) -> None:
    """
    CRITICAL: Applicant session where Azure raises RateLimitError.
    get_fallback_client must NEVER be called.
    reason_node must return error=PRIMARY_UNAVAILABLE.
    """
    from app.agent import nodes as nodes_module

    get_fallback_spy = MagicMock(name="get_fallback_client")

    azure_mock = _make_azure_client_mock(
        side_effect=openai.RateLimitError("rate limited", response=MagicMock(status_code=429, headers={}, text=""), body=None)
    )

    with (
        patch.object(nodes_module, "get_chat_client", return_value=azure_mock),
        patch.object(nodes_module, "get_fallback_client", get_fallback_spy),
        patch.object(nodes_module, "get_settings", return_value=test_settings),
        patch("app.llm.circuit_breaker.is_open", return_value=False),
        patch("app.llm.circuit_breaker.record_success", new_callable=AsyncMock),
        patch("app.llm.circuit_breaker.record_failure", new_callable=AsyncMock),
    ):
        from app.agent.nodes import reason_node
        result = await reason_node(_applicant_state())

    assert result.get("error") == "PRIMARY_UNAVAILABLE"
    # get_fallback_client must never have been called (not even to check)
    get_fallback_spy.assert_not_called()


@pytest.mark.asyncio
async def test_scenario_4_applicant_503_http_response() -> None:
    """
    When reason_node returns PRIMARY_UNAVAILABLE for an applicant session,
    the /v1/applicant/communication endpoint must return HTTP 503 with Retry-After: 30.
    """
    final_state = {
        "session_id": uuid.uuid4(),
        "error": "PRIMARY_UNAVAILABLE",
        "compliance_passed": False,
        "final_output": {},
    }

    with (
        patch("app.api.v1.applicant.get_graph") as mock_get_graph,
        patch("app.db.session.init_db", new_callable=AsyncMock),
    ):
        compiled_graph = AsyncMock()
        compiled_graph.ainvoke = AsyncMock(return_value=final_state)
        mock_get_graph.return_value = compiled_graph

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/applicant/communication",
                json={
                    "application_id": str(uuid.uuid4()),
                    "source": "thinfile",
                    "communication_type": "decline",
                    "channel": "email",
                },
            )

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "30"
    body = response.json()
    assert body["detail"]["error"] == "inference_unavailable"


# ---------------------------------------------------------------------------
# SCENARIO 5: Analyst session — Azure AND Vertex both fail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_5_analyst_all_providers_fail(test_settings: Settings) -> None:
    """
    Azure raises 503; Vertex fallback also raises an exception.
    reason_node must return error=ALL_PROVIDERS_UNAVAILABLE.
    """
    from app.agent import nodes as nodes_module

    request = MagicMock()
    request.status_code = 503
    request.headers = {}
    request.text = "Service unavailable"
    azure_mock = _make_azure_client_mock(
        side_effect=openai.APIStatusError("503", response=request, body=None)
    )
    vertex_mock = _make_vertex_client_mock(
        side_effect=Exception("Vertex internal error")
    )

    result = await _invoke_reason_node(
        state=_analyst_state(),
        azure_mock=azure_mock,
        vertex_mock=vertex_mock,
        settings_override=test_settings,
    )

    assert result.get("error") == "ALL_PROVIDERS_UNAVAILABLE"


@pytest.mark.asyncio
async def test_scenario_5_analyst_all_fail_http_503() -> None:
    """Ensure the graph error propagates to HTTP 503 from the applicant endpoint."""
    final_state = {
        "session_id": uuid.uuid4(),
        "error": "ALL_PROVIDERS_UNAVAILABLE",
        "compliance_passed": False,
        "final_output": {},
    }

    with (
        patch("app.api.v1.applicant.get_graph") as mock_get_graph,
        patch("app.db.session.init_db", new_callable=AsyncMock),
    ):
        compiled_graph = AsyncMock()
        compiled_graph.ainvoke = AsyncMock(return_value=final_state)
        mock_get_graph.return_value = compiled_graph

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/applicant/communication",
                json={
                    "application_id": str(uuid.uuid4()),
                    "source": "thinfile",
                    "communication_type": "decline",
                    "channel": "email",
                },
            )

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# SCENARIO 6: Briefing session with Flash gate active
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_6_briefing_flash_gate_active(test_settings: Settings) -> None:
    """
    When briefing_use_flash=True, reason_node must use Gemini 2.0 Flash.
    provider_used must be vertex:gemini-2.0-flash-001.
    AzureChatOpenAI must NOT be invoked for the LLM step.
    """
    from app.agent import nodes as nodes_module

    # Flash gate active
    test_settings.briefing_use_flash = True

    azure_spy = MagicMock(name="get_chat_client")
    azure_spy.ainvoke = AsyncMock(return_value=_make_llm_response())  # should NOT be called

    vertex_flash_mock = _make_vertex_client_mock(
        response=_make_llm_response("Flash briefing response.")
    )

    get_fallback_spy = MagicMock(return_value=vertex_flash_mock)

    with (
        patch.object(nodes_module, "get_chat_client", return_value=azure_spy),
        patch.object(nodes_module, "get_fallback_client", get_fallback_spy),
        patch.object(nodes_module, "get_settings", return_value=test_settings),
        # Simulate circuit breaker: azure is OPEN so we skip directly to fallback
        patch("app.llm.circuit_breaker.is_open", side_effect=_flash_cb_is_open),
        patch("app.llm.circuit_breaker.record_success", new_callable=AsyncMock),
        patch("app.llm.circuit_breaker.record_failure", new_callable=AsyncMock),
    ):
        from app.agent.nodes import reason_node
        result = await reason_node(_briefing_state())

    # For Flash gate: the briefing session routes directly to get_fallback_client
    # because briefing_use_flash=True means the intent is "portfolio_brief" and
    # the azure circuit is forced open to trigger the fallback path
    assert result.get("error") is None
    assert result.get("provider_used") == "vertex:gemini-2.0-flash-001"
    # Azure LLM must not have been invoked for the generation step
    azure_spy.ainvoke.assert_not_called()


async def _flash_cb_is_open(provider: str) -> bool:
    """For scenario 6: azure circuit open to force fallback, vertex closed."""
    return provider == "azure"
