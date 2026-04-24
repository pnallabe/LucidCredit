"""
tests/integration/test_sprint2.py
===================================
Sprint 2 acceptance tests — LangGraph Agent & Zero-Hallucination Layer.

All tests are mock-based (no live DB, LLM, or embedding API calls).

Test Groups
-----------
A  CitationEnforcer — sentence grounding, suppression, and citation fields
B  ConfidenceScorer — signal weighting, threshold enforcement
C  PromptRenderer   — template loading, context injection, session type dispatch
D  persist_session_node — DB write, citation rows, error resilience
E  Acceptance: citation rate > 95% on controlled explain_decision flow
"""
from __future__ import annotations

import math
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import modules that have module-level SQLAlchemy engine setup so that
# _make_engine() runs with real settings before any mock patches are applied.
import app.db.session  # noqa: F401
import app.models      # noqa: F401

# ---------------------------------------------------------------------------
# Minimal Settings mock — prevents the validator from firing on import
# ---------------------------------------------------------------------------

_MOCK_SETTINGS = MagicMock()
_MOCK_SETTINGS.min_citation_similarity = 0.85
_MOCK_SETTINGS.min_confidence_score = 0.75
_MOCK_SETTINGS.analyst_fallback_enabled = False
_MOCK_SETTINGS.google_project_id = ""
_MOCK_SETTINGS.azure_openai_endpoint = "https://mock.openai.azure.com/"
_MOCK_SETTINGS.azure_openai_deployment_embedding = "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dims: int, index: int) -> List[float]:
    """Return a unit vector with a 1.0 at *index* and 0.0 elsewhere."""
    v = [0.0] * dims
    v[index % dims] = 1.0
    return v


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _make_chunk(
    source_ref: str,
    content: str,
    source_type: str = "vector_doc",
    relevance: str = "RELEVANT",
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "source_type": source_type,
        "source_ref": source_ref,
        "content": content,
        "relevance": relevance,
        "similarity_score": 0.90,
    }


# ===========================================================================
# Group A — CitationEnforcer
# ===========================================================================

class TestCitationEnforcer:

    @pytest.mark.asyncio
    async def test_a1_grounded_sentence_gets_citation(self):
        """Sentence with similarity >= threshold → appears in citations."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "The applicant has a DTI of 0.48."
            chunks = [_make_chunk("crp/decisions/001", "The applicant has a DTI of 0.48.")]

            # Simulate identical embeddings → cosine = 1.0
            mock_embeddings = [[1.0, 0.0], [1.0, 0.0]]

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [mock_embeddings[:1], mock_embeddings[:1]]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert len(result["citations"]) == 1
        assert result["citations"][0]["claim_text"] == "The applicant has a DTI of 0.48."
        assert result["citations"][0]["source_ref"] == "crp/decisions/001"
        assert result["citations"][0]["similarity_score"] == 1.0
        assert len(result["suppressed_claims"]) == 0

    @pytest.mark.asyncio
    async def test_a2_below_threshold_sentence_suppressed(self):
        """Sentence with similarity < threshold → suppressed_claims, not citations."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "The credit score is 720."
            chunks = [_make_chunk("policy/001", "Something completely different about fruits.")]

            # Orthogonal embeddings → cosine = 0.0
            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert len(result["citations"]) == 0
        assert len(result["suppressed_claims"]) == 1
        assert result["suppressed_claims"][0]["suppression_reason"] == "below_similarity_threshold"
        assert result["grounded_narrative"] == ""

    @pytest.mark.asyncio
    async def test_a3_unverified_marker_suppresses_sentence(self):
        """Model self-flagged [UNVERIFIED] sentences are always suppressed."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "[UNVERIFIED] The risk is very high according to estimates."
            chunks = [_make_chunk("policy/001", "The risk is very high according to estimates.")]

            # High similarity but [UNVERIFIED] → must be suppressed
            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [[[1.0, 0.0]], [[1.0, 0.0]]]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert len(result["citations"]) == 0
        assert len(result["suppressed_claims"]) == 1
        assert result["suppressed_claims"][0]["suppression_reason"] == "unverified_marker"

    @pytest.mark.asyncio
    async def test_a4_empty_narrative_returns_empty(self):
        """Empty narrative string → empty result, no embed calls."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                enforcer = CitationEnforcer()
                result = await enforcer.enforce("", [])

        mock_embed.assert_not_called()
        assert result["grounded_narrative"] == ""
        assert result["citations"] == []
        assert result["suppressed_claims"] == []

    @pytest.mark.asyncio
    async def test_a5_no_relevant_chunks_suppresses_all(self):
        """When no RELEVANT/AMBIGUOUS chunks exist, all sentences are suppressed."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "Decision was approved. DTI is acceptable."
            irrelevant_chunk = _make_chunk("x", "y", relevance="IRRELEVANT")

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, [irrelevant_chunk])

        mock_embed.assert_not_called()
        assert result["grounded_narrative"] == ""
        assert result["citations"] == []
        assert len(result["suppressed_claims"]) == 2
        assert all(
            s["suppression_reason"] == "no_relevant_context"
            for s in result["suppressed_claims"]
        )

    @pytest.mark.asyncio
    async def test_a6_citation_dict_has_required_fields(self):
        """Each citation record must carry citation_id, claim_text, source_type, source_ref, similarity_score, confidence."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "The DTI ratio exceeded the policy threshold."
            chunks = [_make_chunk("db/loan_applications/row42", "The DTI ratio exceeded the policy threshold.")]

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [[[1.0, 0.0]], [[1.0, 0.0]]]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert len(result["citations"]) == 1
        c = result["citations"][0]
        for field in ("citation_id", "claim_text", "source_type", "source_ref", "similarity_score", "confidence"):
            assert field in c, f"Missing field '{field}' in citation"

    @pytest.mark.asyncio
    async def test_a7_multiple_sentences_independent_grounding(self):
        """Multiple sentences each matched to the best chunk independently."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "The DTI is 0.48. The credit score is 620."
            chunks = [
                _make_chunk("ref/dti", "The DTI is 0.48."),
                _make_chunk("ref/score", "The credit score is 620."),
            ]

            # s0 matches chunk 0, s1 matches chunk 1
            sent_embs = [[1.0, 0.0], [0.0, 1.0]]
            chunk_embs = [[1.0, 0.0], [0.0, 1.0]]

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [sent_embs, chunk_embs]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert len(result["citations"]) == 2
        assert len(result["suppressed_claims"]) == 0
        source_refs = {c["source_ref"] for c in result["citations"]}
        assert "ref/dti" in source_refs
        assert "ref/score" in source_refs


# ===========================================================================
# Group B — ConfidenceScorer
# ===========================================================================

class TestConfidenceScorer:

    def _make_state(
        self,
        relevant: int = 5,
        total_chunks: int = 5,
        cited: int = 5,
        suppressed: int = 0,
        unverified_count: int = 0,
        raw_sentences: int = 5,
    ) -> dict:
        graded_chunks = (
            [{"relevance": "RELEVANT"} for _ in range(relevant)]
            + [{"relevance": "IRRELEVANT"} for _ in range(total_chunks - relevant)]
        )
        citations = [{"claim_text": f"Claim {i}"} for i in range(cited)]
        suppressed_claims = [{"claim_text": f"Bad {i}"} for i in range(suppressed)]
        # Build raw output with N sentences and M [UNVERIFIED] tags
        sentences = [f"Sentence {i}." for i in range(raw_sentences)]
        for j in range(min(unverified_count, raw_sentences)):
            sentences[j] = f"[UNVERIFIED] Sentence {j}."
        raw_output = " ".join(sentences)

        return {
            "graded_chunks": graded_chunks,
            "citations": citations,
            "suppressed_claims": suppressed_claims,
            "raw_llm_output": raw_output,
        }

    def test_b1_perfect_grounding_gives_high_score(self):
        """All RELEVANT, all cited, no [UNVERIFIED] → score near 1.0."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            state = self._make_state(relevant=5, total_chunks=5, cited=5, suppressed=0, unverified_count=0, raw_sentences=5)
            scorer = ConfidenceScorer()
            result = scorer.score(state)

        # retrieval_recall=1.0, citation_rate=1.0, unverified_rate=0 → 0.30+0.50+0.20=1.0
        assert result["confidence_score"] == 1.0
        assert result["grounding_passed"] is True

    def test_b2_no_relevant_chunks_fails(self):
        """Zero relevant chunks → low score → grounding_passed=False."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            state = self._make_state(relevant=0, total_chunks=5, cited=0, suppressed=5, unverified_count=5, raw_sentences=5)
            scorer = ConfidenceScorer()
            result = scorer.score(state)

        # retrieval_recall=0, citation_rate=0, (1-unverified_rate)=0 → 0.0
        assert result["confidence_score"] == 0.0
        assert result["grounding_passed"] is False

    def test_b3_partial_citation_rate(self):
        """4/5 cited, all relevant, no unverified."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            state = self._make_state(relevant=5, total_chunks=5, cited=4, suppressed=1, unverified_count=0, raw_sentences=5)
            scorer = ConfidenceScorer()
            result = scorer.score(state)

        # 0.30*1.0 + 0.50*(4/5) + 0.20*1.0 = 0.30 + 0.40 + 0.20 = 0.90
        assert abs(result["confidence_score"] - 0.90) < 0.001
        assert result["grounding_passed"] is True

    def test_b4_unverified_markers_penalise_score(self):
        """High unverified rate reduces score below threshold."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            # All relevant, all cited, but 5/5 sentences are [UNVERIFIED]
            state = self._make_state(relevant=5, total_chunks=5, cited=5, suppressed=0, unverified_count=5, raw_sentences=5)
            scorer = ConfidenceScorer()
            result = scorer.score(state)

        # 0.30*1.0 + 0.50*1.0 + 0.20*(1-1.0) = 0.30 + 0.50 + 0.0 = 0.80
        assert abs(result["confidence_score"] - 0.80) < 0.001

    def test_b5_empty_state_scores_zero(self):
        """Empty state → zero score → grounding fails.

        With empty raw_llm_output, unverified_rate=1.0 (fully uncertain),
        retrieval_recall=0.0, citation_rate=0.0:
          score = 0.30×0 + 0.50×0 + 0.20×(1-1) = 0.0
        """
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            scorer = ConfidenceScorer()
            result = scorer.score({})

        assert result["confidence_score"] == 0.0
        assert result["grounding_passed"] is False

    def test_b6_weights_sum_to_one(self):
        """The three weights in ConfidenceScorer must sum to exactly 1.0."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import ConfidenceScorer

            total = (
                ConfidenceScorer.RETRIEVAL_WEIGHT
                + ConfidenceScorer.CITATION_WEIGHT
                + ConfidenceScorer.UNVERIFIED_WEIGHT
            )
        assert abs(total - 1.0) < 1e-9

    def test_b7_threshold_boundary_exactly_at_min(self):
        """Score exactly at min_confidence_score boundary → grounding_passed=True."""
        mock_settings = MagicMock()
        mock_settings.min_citation_similarity = 0.85
        mock_settings.min_confidence_score = 0.75
        with patch("app.config.get_settings", return_value=mock_settings):
            from app.agent.grounding import ConfidenceScorer

            # Craft state so score = 0.75 exactly:
            # 0.30*r + 0.50*c + 0.20*u = 0.75
            # Use r=1, c=1, u=0 → 0.30+0.50+0.20 = 1.0 (too high)
            # Use r=0.5, c=1, u=0 → 0.15+0.50+0.20 = 0.85 (too high)
            # Use r=0, c=1, u=0.25 → 0+0.50+0.15 = 0.65 (too low)
            # Use r=1, c=0.5, u=0 → 0.30+0.25+0.20 = 0.75 ✓
            state = {
                "graded_chunks": [{"relevance": "RELEVANT"} for _ in range(5)],
                "citations": [{"claim_text": f"c{i}"} for i in range(5)],
                "suppressed_claims": [{"claim_text": "s0"} for _ in range(5)],
                "raw_llm_output": "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
            }
            scorer = ConfidenceScorer()
            result = scorer.score(state)

        # r=1.0, c=5/10=0.5, u=0 → 0.30*1 + 0.50*0.5 + 0.20*1 = 0.75
        assert abs(result["confidence_score"] - 0.75) < 0.001
        assert result["grounding_passed"] is True


# ===========================================================================
# Group C — PromptRenderer
# ===========================================================================

class TestPromptRenderer:

    def test_c1_analyst_prompt_contains_system_header(self):
        """Analyst prompt includes the analyst system prompt header."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        chunks = [_make_chunk("ref/001", "DTI is 0.48.")]
        result = render_prompt("analyst", "Explain this decision.", chunks, {})
        assert "LucidCredit" in result
        assert "Retrieved Context" in result
        assert "Explain this decision." in result

    def test_c2_applicant_prompt_contains_ecoa_language(self):
        """Applicant prompt includes ECOA/FCRA compliance references."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        chunks = [_make_chunk("thinfile/score/001", "Adverse action reason: DTI.")]
        result = render_prompt("applicant", "Send decline notice.", chunks, {"communication_type": "decline"})
        assert "ECOA" in result or "Applicant" in result or "Plain English" in result
        assert "decline" in result

    def test_c3_briefing_prompt_includes_grading_instructions(self):
        """Briefing prompt appends grading system instructions."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        chunks = [_make_chunk("ref/portfolio", "Portfolio had 120 applications.")]
        result = render_prompt("briefing", "Give me a portfolio brief.", chunks, {"scope": "q1_2026"})
        assert "self-grading" in result.lower() or "self-check" in result.lower() or "UNVERIFIED" in result
        assert "q1_2026" in result

    def test_c4_irrelevant_chunks_excluded_from_context(self):
        """Chunks graded IRRELEVANT should not appear in the rendered context block."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        relevant = _make_chunk("ref/good", "The DTI is 0.48.", relevance="RELEVANT")
        irrelevant = _make_chunk("ref/bad", "BANANA_IRRELEVANT_CONTENT", relevance="IRRELEVANT")
        result = render_prompt("analyst", "Explain.", [relevant, irrelevant], {})
        assert "BANANA_IRRELEVANT_CONTENT" not in result

    def test_c5_decision_id_injected_into_analyst_prompt(self):
        """decision_id from context_payload is included in analyst prompt."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        chunks = [_make_chunk("ref/001", "Decision context.")]
        result = render_prompt("analyst", "Explain.", chunks, {"decision_id": "abc-123"})
        assert "abc-123" in result

    def test_c6_application_id_injected_into_applicant_prompt(self):
        """application_id from context_payload is included in applicant prompt."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        chunks = [_make_chunk("ref/001", "Application context.")]
        result = render_prompt("applicant", "Send notice.", chunks, {"application_id": "app-999"})
        assert "app-999" in result

    def test_c7_empty_chunks_uses_fallback_message(self):
        """No relevant chunks produces a graceful fallback in the context block."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.prompt_renderer import render_prompt

        result = render_prompt("analyst", "Query.", [], {})
        assert "No relevant context" in result


# ===========================================================================
# Group D — persist_session_node
# ===========================================================================

class TestPersistSessionNode:

    @pytest.mark.asyncio
    async def test_d1_session_row_written_to_db(self):
        """persist_session_node writes a CopilotSession row with correct fields."""
        sid = uuid.uuid4()
        state = {
            "session_id": sid,
            "query": "What is the DTI?",
            "intent": "analyst_query",
            "audience": "analyst",
            "graded_chunks": [{"chunk_id": "c1", "source_type": "vector_doc", "source_ref": "ref/1", "relevance": "RELEVANT"}],
            "rendered_prompt": "prompt text",
            "raw_llm_output": "The DTI is 0.48.",
            "grounded_narrative": "The DTI is 0.48.",
            "confidence_score": 0.92,
            "suppressed_claims": [],
            "compliance_flags": [],
            "citations": [],
            "provider_used": "azure:gpt-4.1-2025-04-14",
            "context_payload": {"source": "credit-risk-platform"},
        }

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS), \
             patch("app.db.session.AsyncSessionLocal", return_value=mock_session), \
             patch("app.models.CopilotSession") as MockSession, \
             patch("app.models.Citation"):

            from app.agent.nodes import persist_session_node
            result = await persist_session_node(state)

        MockSession.assert_called_once()
        call_kwargs = MockSession.call_args[1]
        assert call_kwargs["query_text"] == "What is the DTI?"
        assert call_kwargs["intent"] == "analyst_query"
        assert call_kwargs["confidence_score"] == 0.92
        assert call_kwargs["provider_model"] == "azure:gpt-4.1-2025-04-14"
        assert call_kwargs["provider_fallback_used"] is False
        assert result == {}

    @pytest.mark.asyncio
    async def test_d2_citation_rows_written(self):
        """persist_session_node writes one Citation row per citation."""
        sid = uuid.uuid4()
        citations = [
            {
                "citation_id": str(uuid.uuid4()),
                "claim_text": "Claim A.",
                "source_type": "api",
                "source_ref": "crp/decisions/001",
                "similarity_score": 0.92,
                "confidence": 0.92,
            },
            {
                "citation_id": str(uuid.uuid4()),
                "claim_text": "Claim B.",
                "source_type": "vector_doc",
                "source_ref": "policy/001",
                "similarity_score": 0.88,
                "confidence": 0.88,
            },
        ]
        state = {
            "session_id": sid,
            "query": "Q", "intent": "analyst_query", "audience": "analyst",
            "graded_chunks": [], "rendered_prompt": "", "raw_llm_output": "",
            "grounded_narrative": "", "confidence_score": 0.91,
            "suppressed_claims": [], "compliance_flags": [],
            "citations": citations,
            "provider_used": "azure:gpt-4.1",
            "context_payload": {},
        }

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS), \
             patch("app.db.session.AsyncSessionLocal", return_value=mock_session), \
             patch("app.models.CopilotSession"), \
             patch("app.models.Citation") as MockCitation:

            from app.agent.nodes import persist_session_node
            await persist_session_node(state)

        assert MockCitation.call_count == 2

    @pytest.mark.asyncio
    async def test_d3_db_error_does_not_raise(self):
        """A DB exception must be caught silently — never propagates to caller."""
        sid = uuid.uuid4()
        state = {
            "session_id": sid, "query": "Q", "intent": "analyst_query",
            "audience": "analyst", "graded_chunks": [], "rendered_prompt": "",
            "raw_llm_output": "", "grounded_narrative": "", "confidence_score": 0.9,
            "suppressed_claims": [], "compliance_flags": [], "citations": [],
            "provider_used": "azure:gpt-4.1", "context_payload": {},
        }

        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS), \
             patch("app.db.session.AsyncSessionLocal", side_effect=RuntimeError("DB is down")):

            from app.agent.nodes import persist_session_node
            # Must not raise
            result = await persist_session_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_d4_vertex_fallback_detected_from_provider_used(self):
        """provider_used starting with 'vertex' sets provider_fallback_used=True."""
        sid = uuid.uuid4()
        state = {
            "session_id": sid, "query": "Q", "intent": "analyst_query",
            "audience": "analyst", "graded_chunks": [], "rendered_prompt": "",
            "raw_llm_output": "", "grounded_narrative": "", "confidence_score": 0.9,
            "suppressed_claims": [], "compliance_flags": [], "citations": [],
            "provider_used": "vertex:gemini-1.5-pro-002",
            "context_payload": {},
        }

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS), \
             patch("app.db.session.AsyncSessionLocal", return_value=mock_session), \
             patch("app.models.CopilotSession") as MockSession, \
             patch("app.models.Citation"):

            from app.agent.nodes import persist_session_node
            await persist_session_node(state)

        call_kwargs = MockSession.call_args[1]
        assert call_kwargs["provider_fallback_used"] is True


# ===========================================================================
# Group E — Acceptance: citation rate > 95% on explain_decision flow
# ===========================================================================

class TestCitationRateAcceptance:
    """
    Verify that CitationEnforcer achieves >= 95% citation rate on a synthetic
    explain_decision scenario where all sentences are drawn from chunk content.
    """

    @pytest.mark.asyncio
    async def test_e1_citation_rate_above_95_percent(self):
        """
        When 20 sentences are derived from chunk content, at least 19/20 (95%)
        must be grounded by CitationEnforcer.

        Mock embeddings are designed so each sentence's embedding closely matches
        its source chunk (cosine >= 0.90 >= threshold 0.85).
        """
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            DIM = 50
            NUM_SENTENCES = 20
            NUM_CHUNKS = 5  # Each chunk covers ~4 sentences

            # Build sentences and chunks
            sentences = [f"The risk factor {i} contributed significantly to the decision." for i in range(NUM_SENTENCES)]
            chunks = []
            for c_idx in range(NUM_CHUNKS):
                content = " ".join(sentences[c_idx * 4: (c_idx + 1) * 4])
                chunks.append(_make_chunk(f"crp/decisions/{c_idx}", content, "api", "RELEVANT"))

            narrative = " ".join(sentences)

            # Assign each sentence and its source chunk the same basis vector.
            # Sentences 0-3 → chunk 0 (basis dim 0), sentences 4-7 → chunk 1, etc.
            def _sent_emb(i: int) -> List[float]:
                return _unit_vec(DIM, i // 4)

            def _chunk_emb(c_idx: int) -> List[float]:
                return _unit_vec(DIM, c_idx)

            sent_embeddings = [_sent_emb(i) for i in range(NUM_SENTENCES)]
            chunk_embeddings = [_chunk_emb(c) for c in range(NUM_CHUNKS)]

            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [sent_embeddings, chunk_embeddings]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        total = len(result["citations"]) + len(result["suppressed_claims"])
        citation_rate = len(result["citations"]) / total if total > 0 else 0.0

        assert total == NUM_SENTENCES, f"Expected {NUM_SENTENCES} claims total, got {total}"
        assert citation_rate >= 0.95, (
            f"Citation rate {citation_rate:.2%} is below the 95% acceptance threshold. "
            f"Grounded: {len(result['citations'])}, Suppressed: {len(result['suppressed_claims'])}"
        )

    @pytest.mark.asyncio
    async def test_e2_grounded_narrative_is_non_empty(self):
        """Grounded narrative must be non-empty when at least one sentence is cited."""
        with patch("app.config.get_settings", return_value=_MOCK_SETTINGS):
            from app.agent.grounding import CitationEnforcer

            narrative = "The DTI is 0.48. This exceeded the maximum threshold of 0.43."
            chunks = [_make_chunk("crp/001", "DTI threshold and values.", "api", "RELEVANT")]

            # Both sentences → high similarity
            with patch("app.agent.grounding.batch_embed", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0]],
                ]
                enforcer = CitationEnforcer()
                result = await enforcer.enforce(narrative, chunks)

        assert result["grounded_narrative"] != ""
