"""
backend/tests/eval/test_hallucination_regression.py
======================================================
Hallucination regression and adversarial hardening tests.

These tests do NOT require a live LLM or database connection.
They validate the zero-hallucination layers statically:

  1. CitationEnforcer — verifies that claims lacking grounding are suppressed
  2. ConfidenceScorer — verifies that low-confidence contexts produce correct scores
  3. ECOA compliance check — verifies that prohibited-basis language is rejected
  4. Adversarial context patterns — verifies that out-of-context numerics and
     invented regulations are marked as suppressed in the response
  5. Golden dataset structural integrity — validates all 50 Q&A pairs have
     required fields and session_type labels

Run with:
    pytest backend/tests/eval/test_hallucination_regression.py -v

No markers are required — these run in the standard test suite.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

GOLDEN_PATH = Path(__file__).parent / "golden" / "analyst_qa.json"

# ---------------------------------------------------------------------------
# 1. Golden dataset structural integrity
# ---------------------------------------------------------------------------


class TestGoldenDatasetIntegrity:
    """Validates the 50 Q&A golden pairs are structurally sound."""

    @pytest.fixture(scope="class")
    def golden_items(self) -> list[dict[str, Any]]:
        with open(GOLDEN_PATH) as f:
            return json.load(f)  # type: ignore[return-value]

    def test_exactly_50_items(self, golden_items: list[dict[str, Any]]) -> None:
        assert len(golden_items) == 50, (
            f"Expected 50 golden Q&A pairs, got {len(golden_items)}"
        )

    def test_all_required_fields_present(self, golden_items: list[dict[str, Any]]) -> None:
        required = {"id", "session_type", "question", "ground_truth", "context"}
        for item in golden_items:
            missing = required - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing fields: {missing}"

    def test_unique_ids(self, golden_items: list[dict[str, Any]]) -> None:
        ids = [item["id"] for item in golden_items]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in golden dataset"

    def test_session_type_distribution(self, golden_items: list[dict[str, Any]]) -> None:
        counts: dict[str, int] = {}
        for item in golden_items:
            st = item["session_type"]
            counts[st] = counts.get(st, 0) + 1
        assert counts.get("analyst", 0) >= 20, (
            f"Expected >= 20 analyst items, got {counts.get('analyst', 0)}"
        )
        assert counts.get("briefing", 0) >= 8, (
            f"Expected >= 8 briefing items, got {counts.get('briefing', 0)}"
        )
        assert counts.get("adversarial", 0) >= 10, (
            f"Expected >= 10 adversarial items, got {counts.get('adversarial', 0)}"
        )

    def test_context_is_non_empty_list(self, golden_items: list[dict[str, Any]]) -> None:
        for item in golden_items:
            assert isinstance(item["context"], list), (
                f"Item {item['id']}: context must be a list"
            )
            assert len(item["context"]) >= 1, (
                f"Item {item['id']}: context must have at least one entry"
            )

    def test_adversarial_items_have_suppressed_or_requires_retrieval_ground_truth(
        self, golden_items: list[dict[str, Any]]
    ) -> None:
        """Adversarial Q&As must have ground_truth that acknowledges suppression."""
        adversarial = [i for i in golden_items if i["session_type"] == "adversarial"]
        for item in adversarial:
            gt = item["ground_truth"]
            assert any(
                marker in gt for marker in ["[SUPPRESSED]", "[REQUIRES_RETRIEVAL]", "No."]
            ), (
                f"Adversarial item {item['id']} ground_truth should indicate suppression "
                f"or refusal. Got: {gt[:80]}"
            )

    def test_ids_are_sequential(self, golden_items: list[dict[str, Any]]) -> None:
        ids = [item["id"] for item in golden_items]
        pattern = re.compile(r"^qa_(\d+)$")
        nums = []
        for id_ in ids:
            m = pattern.match(id_)
            assert m, f"ID '{id_}' does not match pattern qa_NNN"
            nums.append(int(m.group(1)))
        assert nums == sorted(nums), "IDs are not in ascending order"
        assert nums[0] == 1 and nums[-1] == 50, (
            f"Expected IDs qa_001 through qa_050, got {nums[0]}–{nums[-1]}"
        )


# ---------------------------------------------------------------------------
# 2. CitationEnforcer unit regression
# ---------------------------------------------------------------------------


class TestCitationEnforcerRegression:
    """
    Tests the CitationEnforcer logic for suppressing uncited claims.
    Patches the embedding similarity check to return controlled scores.
    """

    def _make_enforcer(self) -> Any:
        from app.agent.grounding import CitationEnforcer

        return CitationEnforcer()

    def test_fully_grounded_narrative_passes(self) -> None:
        """A narrative where every claim has high-similarity context should pass as-is."""
        enforcer = self._make_enforcer()
        narrative = "The DTI ratio for this applicant is 0.48."
        context_chunks = [
            {"text": "SHAP explanation: feature=dti_ratio value=0.48", "source_ref": "crp:dec-001"}
        ]

        with patch.object(
            enforcer,
            "_similarity",
            return_value=0.92,  # above threshold
        ):
            result = enforcer.enforce(narrative, context_chunks)

        assert result.grounded_narrative == narrative
        assert result.suppressed_claims == []

    def test_uncited_numeric_is_suppressed(self) -> None:
        """A sentence containing a specific number not in context should be suppressed."""
        enforcer = self._make_enforcer()
        narrative = "The applicant's FICO score is 542."
        context_chunks = [
            {"text": "No FICO score data was retrieved.", "source_ref": "system"}
        ]

        with patch.object(
            enforcer,
            "_similarity",
            return_value=0.40,  # below threshold
        ):
            result = enforcer.enforce(narrative, context_chunks)

        assert "542" not in result.grounded_narrative
        assert len(result.suppressed_claims) >= 1

    def test_invented_regulation_is_suppressed(self) -> None:
        """A citation to a regulation not in context should be suppressed."""
        enforcer = self._make_enforcer()
        narrative = "Under Reg Z Section 1026.999, the creditor must provide a 3-day notice."
        context_chunks = [
            {
                "text": "No Reg Z Section 1026.999 content was retrieved.",
                "source_ref": "system",
            }
        ]

        with patch.object(
            enforcer,
            "_similarity",
            return_value=0.35,
        ):
            result = enforcer.enforce(narrative, context_chunks)

        assert len(result.suppressed_claims) >= 1

    def test_unverified_tag_in_llm_output_is_stripped(self) -> None:
        """Claims marked [UNVERIFIED] by the LLM must be stripped from the final narrative."""
        enforcer = self._make_enforcer()
        narrative = (
            "The DTI is 0.48. [UNVERIFIED] The applicant's net worth exceeds $2 million."
        )
        context_chunks = [
            {"text": "SHAP: dti_ratio=0.48", "source_ref": "crp:dec-001"}
        ]

        with patch.object(
            enforcer,
            "_similarity",
            side_effect=lambda *_: 0.92,
        ):
            result = enforcer.enforce(narrative, context_chunks)

        assert "[UNVERIFIED]" not in result.grounded_narrative
        assert "net worth" not in result.grounded_narrative

    def test_empty_narrative_returns_empty(self) -> None:
        enforcer = self._make_enforcer()
        result = enforcer.enforce("", [])
        assert result.grounded_narrative == ""
        assert result.suppressed_claims == []


# ---------------------------------------------------------------------------
# 3. ConfidenceScorer regression
# ---------------------------------------------------------------------------


class TestConfidenceScorerRegression:
    """Tests the ConfidenceScorer for correct score banding."""

    def _make_scorer(self) -> Any:
        from app.agent.grounding import ConfidenceScorer

        return ConfidenceScorer()

    def test_perfect_retrieval_gives_high_score(self) -> None:
        scorer = self._make_scorer()
        score = scorer.score(
            relevant_chunks=10,
            total_chunks=10,
            cited_claims=5,
            total_claims=5,
            model_uncertainty=0.05,
        )
        assert score >= 0.90, f"Expected score >= 0.90, got {score}"

    def test_zero_citations_gives_low_score(self) -> None:
        scorer = self._make_scorer()
        score = scorer.score(
            relevant_chunks=5,
            total_chunks=10,
            cited_claims=0,
            total_claims=5,
            model_uncertainty=0.3,
        )
        assert score < 0.75, f"Expected score < 0.75 (below threshold), got {score}"

    def test_empty_retrieval_gives_minimum_score(self) -> None:
        scorer = self._make_scorer()
        score = scorer.score(
            relevant_chunks=0,
            total_chunks=0,
            cited_claims=0,
            total_claims=0,
            model_uncertainty=1.0,
        )
        assert score == pytest.approx(0.0, abs=0.05), (
            f"Expected score ~0.0 for empty retrieval, got {score}"
        )

    def test_score_is_between_zero_and_one(self) -> None:
        scorer = self._make_scorer()
        for r, t, cc, tc, u in [
            (3, 10, 2, 8, 0.5),
            (0, 5, 0, 3, 0.9),
            (10, 10, 10, 10, 0.0),
        ]:
            score = scorer.score(
                relevant_chunks=r,
                total_chunks=t,
                cited_claims=cc,
                total_claims=tc,
                model_uncertainty=u,
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_below_threshold_is_correctly_detected(self) -> None:
        from app.config import get_settings

        scorer = self._make_scorer()
        settings = get_settings()
        score = scorer.score(
            relevant_chunks=1,
            total_chunks=10,
            cited_claims=1,
            total_claims=8,
            model_uncertainty=0.8,
        )
        assert score < settings.min_confidence_score, (
            f"Expected score < {settings.min_confidence_score}, got {score}"
        )


# ---------------------------------------------------------------------------
# 4. ECOA compliance check regression
# ---------------------------------------------------------------------------


class TestEcoaComplianceRegression:
    """Tests the ECOA validator for rejection of prohibited-basis language."""

    def _make_validator(self) -> Any:
        from app.compliance.ecoa_validator import EcoaValidator

        return EcoaValidator()

    @pytest.mark.parametrize(
        "prohibited_text",
        [
            "The applicant's race was considered as a factor.",
            "We declined based on national origin.",
            "The decision was influenced by the applicant's religion.",
            "Applicants of this sex have a higher default rate.",
            "We factored in marital status in our underwriting.",
        ],
    )
    def test_prohibited_basis_language_is_flagged(self, prohibited_text: str) -> None:
        validator = self._make_validator()
        result = validator.validate(prohibited_text, audience="applicant")
        errors = [f for f in result.flags if f.severity == "error"]
        assert len(errors) >= 1, (
            f"Expected at least one ECOA error for: '{prohibited_text}'. Got: {result.flags}"
        )
        rule_ids = [f.rule_id for f in errors]
        assert "ECOA-001" in rule_ids, (
            f"Expected ECOA-001 flag for prohibited-basis language. Got: {rule_ids}"
        )

    def test_waiver_of_rights_language_is_flagged(self) -> None:
        validator = self._make_validator()
        text = "By accepting this decision, you waive your right to contest the adverse action."
        result = validator.validate(text, audience="applicant")
        rule_ids = [f.rule_id for f in result.flags]
        assert "ECOA-004" in rule_ids, (
            f"Expected ECOA-004 flag for rights-waiver language. Got: {rule_ids}"
        )

    def test_vague_reason_code_is_flagged_as_warning(self) -> None:
        validator = self._make_validator()
        text = "Your application was declined because it did not meet our standards."
        result = validator.validate(text, audience="applicant")
        warnings = [f for f in result.flags if f.severity == "warning"]
        rule_ids = [f.rule_id for f in warnings]
        assert "ECOA-003" in rule_ids, (
            f"Expected ECOA-003 warning for vague reason. Got: {result.flags}"
        )

    def test_compliant_decline_notice_passes(self) -> None:
        validator = self._make_validator()
        text = (
            "Your application was declined due to the following specific reasons: "
            "(1) Debt-to-income ratio exceeds guideline maximum. "
            "(2) Insufficient credit history (fewer than 24 months of tradeline activity). "
            "You have the right to request the specific reasons within 60 days."
        )
        result = validator.validate(text, audience="applicant")
        errors = [f for f in result.flags if f.severity == "error"]
        assert errors == [], f"Expected no errors for compliant notice. Got: {errors}"

    def test_sr117_disclosure_not_applied_to_applicant_audience(self) -> None:
        """SR 11-7 disclosures are for analyst/briefing only."""
        from app.compliance.sr117_disclosures import Sr117Disclosures

        disclosures = Sr117Disclosures()
        text = "The model scored the application at 0.34."
        result = disclosures.inject(
            narrative=text,
            provider_model="azure:gpt-4.1-2025-04-14",
            session_id="test-session",
            audience="applicant",
        )
        # For applicant audience, no SR 11-7 footer should be appended
        assert "SR 11-7" not in result, (
            "SR 11-7 disclosure should NOT appear in applicant-audience output"
        )

    def test_sr117_disclosure_applied_to_analyst_audience(self) -> None:
        from app.compliance.sr117_disclosures import Sr117Disclosures

        disclosures = Sr117Disclosures()
        text = "The model scored the application at 0.34."
        result = disclosures.inject(
            narrative=text,
            provider_model="azure:gpt-4.1-2025-04-14",
            session_id="test-session",
            audience="analyst",
        )
        assert "SR 11-7" in result, (
            "SR 11-7 disclosure MUST appear in analyst-audience output"
        )


# ---------------------------------------------------------------------------
# 5. Adversarial narrative patterns
# ---------------------------------------------------------------------------


class TestAdversarialNarrativePatterns:
    """
    Tests that the CitationEnforcer correctly suppresses adversarial patterns
    derived from the adversarial golden dataset.
    """

    def _make_enforcer(self) -> Any:
        from app.agent.grounding import CitationEnforcer

        return CitationEnforcer()

    def test_fico_conversion_from_model_score_is_suppressed(self) -> None:
        """Model scores must not be naively mapped to FICO scores."""
        enforcer = self._make_enforcer()
        narrative = "A model score of 0.34 corresponds to a FICO score of approximately 580."
        context_chunks = [
            {
                "text": "The model output is an internal probability-of-default score, not a FICO score.",
                "source_ref": "model_card:crp",
            }
        ]

        with patch.object(enforcer, "_similarity", return_value=0.30):
            result = enforcer.enforce(narrative, context_chunks)

        assert "580" not in result.grounded_narrative

    def test_future_credit_score_prediction_is_suppressed(self) -> None:
        """Future credit score predictions are not grounded and must be suppressed."""
        enforcer = self._make_enforcer()
        narrative = "If the applicant reduces their DTI, their credit score will be 710 in 6 months."
        context_chunks = [
            {
                "text": "Counterfactual: reducing DTI may improve the model score. "
                "No temporal forecast is provided.",
                "source_ref": "crp:counterfactual",
            }
        ]

        with patch.object(enforcer, "_similarity", return_value=0.38):
            result = enforcer.enforce(narrative, context_chunks)

        assert "710" not in result.grounded_narrative

    def test_invented_regulation_not_in_corpus_is_suppressed(self) -> None:
        """A regulation not in the retrieved corpus must not appear in the narrative."""
        enforcer = self._make_enforcer()
        narrative = (
            "Under the 2025 CFPB Credit Access Reform Act Section 12, "
            "all lenders must offer a 6-month grace period."
        )
        context_chunks = [
            {
                "text": "No 2025 CFPB Credit Access Reform Act was found in the regulatory corpus.",
                "source_ref": "system",
            }
        ]

        with patch.object(enforcer, "_similarity", return_value=0.25):
            result = enforcer.enforce(narrative, context_chunks)

        # The invented regulation name must not appear
        assert "Credit Access Reform Act" not in result.grounded_narrative

    def test_zip_code_proxy_language_triggers_ecoa_flag(self) -> None:
        """Using zip code as proxy for creditworthiness must trigger a compliance flag."""
        from app.compliance.ecoa_validator import EcoaValidator

        validator = EcoaValidator()
        text = (
            "Applications from zip codes 90210 and 10001 have historically lower default rates "
            "and are therefore more likely to be approved."
        )
        result = validator.validate(text, audience="applicant")
        # Should flag as ECOA risk (geographic redlining proxy)
        # At minimum, must not pass with zero flags
        assert len(result.flags) >= 1, (
            "Expected at least one compliance flag for geographic proxy language"
        )

    def test_out_of_context_income_calculation_is_suppressed(self) -> None:
        """Derived financial calculations not grounded in retrieved data are suppressed."""
        enforcer = self._make_enforcer()
        narrative = (
            "The applicant earns $120,000 annually, so their monthly take-home "
            "after federal tax is approximately $7,600."
        )
        context_chunks = [
            {
                "text": "Application: income_stated=120000_annual. No tax data retrieved.",
                "source_ref": "application:app-001",
            }
        ]

        with patch.object(enforcer, "_similarity", return_value=0.32):
            result = enforcer.enforce(narrative, context_chunks)

        # The derived calculation ($7,600) must not survive
        assert "7,600" not in result.grounded_narrative

    def test_ungrounded_sr117_amendment_is_suppressed(self) -> None:
        """
        A false claim about an SR 11-7 amendment not in the corpus must be suppressed.
        """
        enforcer = self._make_enforcer()
        narrative = (
            "SR 11-7 was updated in 2024 to require real-time model monitoring dashboards."
        )
        context_chunks = [
            {
                "text": "SR 11-7 (April 2011): No 2024 amendment was found in the corpus.",
                "source_ref": "regulatory:sr117",
            }
        ]

        with patch.object(enforcer, "_similarity", return_value=0.28):
            result = enforcer.enforce(narrative, context_chunks)

        assert "2024" not in result.grounded_narrative or "updated in 2024" not in result.grounded_narrative
