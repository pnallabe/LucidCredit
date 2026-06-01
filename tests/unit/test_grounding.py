"""
tests/unit/test_grounding.py
==============================
Unit tests for grounding utilities: ConfidenceScorer, _cosine_similarity,
and _split_into_sentences.

No LLM calls, no embeddings, no database.
All tests are deterministic and fast (<1 s).
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agent.grounding import (
    ConfidenceScorer,
    _cosine_similarity,
    _split_into_sentences,
)

# Minimal settings stub — avoids pydantic validation requiring Azure credentials
_DUMMY_SETTINGS = SimpleNamespace(
    min_confidence_score=0.75,
    advisory_confidence_score=0.35,
)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# _split_into_sentences
# ---------------------------------------------------------------------------

class TestSplitIntoSentences:
    def test_simple_two_sentences(self):
        result = _split_into_sentences("The rate is 4.2%. This is above average.")
        assert len(result) == 2

    def test_empty_string(self):
        result = _split_into_sentences("")
        assert result == []

    def test_single_sentence_no_trailing_period(self):
        result = _split_into_sentences("The portfolio is healthy")
        assert len(result) == 1

    def test_filters_empty_tokens(self):
        result = _split_into_sentences("Hello.  World.")
        assert all(s.strip() for s in result)


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Patch get_settings for every test in this module."""
    monkeypatch.setattr("app.agent.grounding.get_settings", lambda: _DUMMY_SETTINGS)


class TestConfidenceScorer:
    SCORER = ConfidenceScorer()

    def _make_state(
        self,
        *,
        graded_chunks=None,
        citations=None,
        suppressed=None,
        raw_output="",
    ):
        return {
            "graded_chunks": graded_chunks or [],
            "citations": citations or [],
            "suppressed_claims": suppressed or [],
            "raw_llm_output": raw_output,
        }

    def test_perfect_score_all_relevant_all_grounded(self):
        state = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}, {"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}, {"claim_text": "y"}],
            suppressed=[],
            raw_output="Sentence one. Sentence two.",
        )
        result = self.SCORER.score(state)
        # Perfect signals → score well above 0.5
        assert result["confidence_score"] > 0.5

    def test_zero_score_empty_state(self):
        state = self._make_state()
        result = self.SCORER.score(state)
        assert result["confidence_score"] == 0.0

    def test_grounding_passed_flag_above_zero_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent.grounding.get_settings",
            lambda: SimpleNamespace(
                min_confidence_score=0.0,
                advisory_confidence_score=0.35,
            ),
        )
        state = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}],
            suppressed=[],
            raw_output="One clean sentence.",
        )
        result = self.SCORER.score(state)
        assert result["grounding_passed"] is True

    def test_unverified_markers_lower_score(self):
        state_clean = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}],
            raw_output="Normal sentence.",
        )
        state_unverified = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}],
            raw_output="[UNVERIFIED] Uncertain claim.",
        )
        result_clean = self.SCORER.score(state_clean)
        result_unverified = self.SCORER.score(state_unverified)
        assert result_clean["confidence_score"] >= result_unverified["confidence_score"]

    def test_suppressed_claims_lower_citation_rate(self):
        state_all_cited = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}, {"claim_text": "y"}],
            suppressed=[],
            raw_output="Sentence one. Sentence two.",
        )
        state_half_suppressed = self._make_state(
            graded_chunks=[{"relevance": "RELEVANT"}],
            citations=[{"claim_text": "x"}],
            suppressed=[{"claim_text": "y", "suppression_reason": "low_similarity"}],
            raw_output="Sentence one. Sentence two.",
        )
        result_all = self.SCORER.score(state_all_cited)
        result_half = self.SCORER.score(state_half_suppressed)
        assert result_all["confidence_score"] >= result_half["confidence_score"]

    def test_irrelevant_chunks_reduce_retrieval_recall(self):
        state_all_relevant = self._make_state(
            graded_chunks=[
                {"relevance": "RELEVANT"},
                {"relevance": "RELEVANT"},
            ],
            citations=[{"claim_text": "x"}],
            raw_output="A sentence.",
        )
        state_half_irrelevant = self._make_state(
            graded_chunks=[
                {"relevance": "RELEVANT"},
                {"relevance": "IRRELEVANT"},
            ],
            citations=[{"claim_text": "x"}],
            raw_output="A sentence.",
        )
        result_all = self.SCORER.score(state_all_relevant)
        result_half = self.SCORER.score(state_half_irrelevant)
        assert result_all["confidence_score"] >= result_half["confidence_score"]

    def test_score_bounded_between_zero_and_one(self):
        """Confidence score must always be in [0.0, 1.0]."""
        for raw_output in ("", "Clean output.", "[UNVERIFIED] bad claim."):
            state = self._make_state(raw_output=raw_output)
            result = self.SCORER.score(state)
            assert 0.0 <= result["confidence_score"] <= 1.0
