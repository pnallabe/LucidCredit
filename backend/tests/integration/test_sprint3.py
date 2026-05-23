"""
tests/integration/test_sprint3.py
===================================
Sprint 3 acceptance tests — API Endpoints & Compliance Layer.

All tests are mock-based (no live DB, LLM, or embedding API calls).

Test Groups
-----------
A  EcoaValidator       — prohibited basis blocking, specific reasons required,
                         FCRA disclosure gating, vague reason warnings,
                         analyst passthrough
B  AdverseActionNotice — reason code mapping, FCRA disclosure present,
                         unknown code handling, render_text format
C  Sr117Disclosures    — footer injected for analyst/briefing, not applicant,
                         idempotency check via has_disclosure
D  compliance_check_node — ECOA violation → compliance_passed=False,
                            SR 11-7 injection for analyst
E  Audit endpoint      — 404 on missing session, 422 on bad UUID,
                         valid session returns AuditResponse with citations

Load-test performance targets (documented, not exercised here — deferred to Sprint 5):
  - p99 /v1/explain/decision < 8 seconds (end-to-end, LLM included)
  - p99 /v1/briefing/generate < 30 seconds
  - p99 /v1/audit/{session_id} < 200ms (DB read only, no LLM)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import to prevent _make_engine() from firing with mock DB URL
import app.db.session  # noqa: F401
import app.models      # noqa: F401

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    source_ref: str = "policy_doc_001",
    content: str = "Policy content.",
    source_type: str = "vector_doc",
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "source_type": source_type,
        "source_ref": source_ref,
        "content": content,
        "relevance": "RELEVANT",
        "similarity_score": 0.88,
    }


def _applicant_payload(
    comm_type: str = "decline",
    source: str = "thinfile",
    aa_codes: list | None = None,
) -> dict:
    return {
        "communication_type": comm_type,
        "source": source,
        "adverse_action_codes": aa_codes or [],
    }


# ===========================================================================
# Group A — EcoaValidator
# ===========================================================================

class TestEcoaValidator:

    def setup_method(self):
        from app.compliance.ecoa_validator import EcoaValidator
        self.validator = EcoaValidator()

    # -----------------------------------------------------------------------
    # A1 — Prohibited basis → ERROR flag ECOA-001
    # -----------------------------------------------------------------------
    def test_A1_prohibited_race(self):
        narrative = (
            "Your application was declined due to race as a factor in our model."
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(),
            citations=[],
        )
        assert result.passed is False
        codes = [f.rule_code for f in result.flags]
        assert "ECOA-001" in codes

    def test_A2_prohibited_sex(self):
        narrative = "Declined because of sex and income criteria."
        result = self.validator.validate(narrative, _applicant_payload())
        assert result.passed is False
        assert any(f.rule_code == "ECOA-001" for f in result.flags)

    def test_A3_prohibited_marital_status(self):
        narrative = "Marital status was considered in the credit decision."
        result = self.validator.validate(narrative, _applicant_payload())
        assert result.passed is False
        assert any(f.rule_code == "ECOA-001" for f in result.flags)

    def test_A4_prohibited_age(self):
        narrative = "Declined because your age is insufficient for this product."
        result = self.validator.validate(narrative, _applicant_payload())
        assert result.passed is False
        assert any(f.rule_code == "ECOA-001" for f in result.flags)

    # -----------------------------------------------------------------------
    # A5 — Specific reason required for decline (ECOA-002)
    # -----------------------------------------------------------------------
    def test_A5_decline_no_specific_reason_error(self):
        narrative = "We are unable to approve your application at this time."
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", aa_codes=[]),
        )
        assert result.passed is False
        assert any(f.rule_code == "ECOA-002" for f in result.flags)

    def test_A6_decline_with_aa_code_in_context_passes_reason_check(self):
        narrative = "Your application was declined based on our review."
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", aa_codes=["AA-023"]),
        )
        # May still fail for FCRA-001 if thinfile source, but NOT ECOA-002
        codes = [f.rule_code for f in result.flags]
        assert "ECOA-002" not in codes

    def test_A7_decline_with_specific_reason_in_text_passes(self):
        narrative = (
            "Your application was declined because your debt-to-income ratio "
            "exceeds our policy maximum. "
            "You have the right to a free copy of your consumer report from the "
            "consumer reporting agency within 60 days."
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", source="thinfile", aa_codes=[]),
        )
        assert result.passed is True

    # -----------------------------------------------------------------------
    # A8 — FCRA § 615 disclosure required when consumer report used (FCRA-001)
    # -----------------------------------------------------------------------
    def test_A8_fcra_disclosure_missing_on_decline(self):
        narrative = (
            "Your application was declined due to AA-023: excessive obligations. "
            # No FCRA disclosure here
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", source="thinfile", aa_codes=["AA-023"]),
        )
        assert result.passed is False
        assert any(f.rule_code == "FCRA-001" for f in result.flags)

    def test_A9_fcra_not_required_for_non_consumer_source(self):
        narrative = "Your application was declined due to insufficient income."
        result = self.validator.validate(
            narrative,
            # source is not a CRA
            {"communication_type": "decline", "source": "internal", "adverse_action_codes": ["AA-022"]},
        )
        # FCRA-001 should NOT fire for internal sources
        codes = [f.rule_code for f in result.flags]
        assert "FCRA-001" not in codes

    # -----------------------------------------------------------------------
    # A10 — Vague reasons → WARNING (not ERROR)
    # -----------------------------------------------------------------------
    def test_A10_vague_reason_is_warning_not_error(self):
        narrative = (
            "Your application was declined because you did not meet our standards. "
            "You have the right to a free copy of your consumer report from the "
            "consumer reporting agency within 60 days. "
            "Your debt-to-income ratio was a factor."
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", source="thinfile", aa_codes=["AA-030"]),
        )
        vague_flags = [f for f in result.flags if f.rule_code == "ECOA-003"]
        # Vague reasons produce warnings, not errors
        for vf in vague_flags:
            assert vf.severity == "WARNING"
        # Result may still pass (warnings don't block)
        error_flags = [f for f in result.flags if f.severity == "ERROR"]
        assert result.passed == (len(error_flags) == 0)

    # -----------------------------------------------------------------------
    # A11 — Rights waiver language → ERROR (ECOA-004)
    # -----------------------------------------------------------------------
    def test_A11_rights_waiver_blocked(self):
        narrative = (
            "By accepting, you agree to waive your rights under FCRA. "
            "Your credit score was insufficient."
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="decline", aa_codes=["AA-014"]),
        )
        assert result.passed is False
        assert any(f.rule_code == "ECOA-004" for f in result.flags)

    # -----------------------------------------------------------------------
    # A12 — Analyst audience always passes through
    # -----------------------------------------------------------------------
    def test_A12_analyst_always_passes(self):
        narrative = "Race and sex and marital status were all key factors. DTI = 55%."
        result = self.validator.validate(
            narrative,
            # analyst audience — no consumer rules apply
            {"communication_type": "analyst_query", "source": "thinfile", "adverse_action_codes": []},
        )
        # EcoaValidator still fires for any narrative regardless of audience.
        # For analyst, we expect the calling code (compliance_check_node) to
        # skip EcoaValidator entirely; the validator itself has no audience filter.
        # This test confirms the validator DOES flag prohibited basis in any text.
        assert any(f.rule_code == "ECOA-001" for f in result.flags)

    # -----------------------------------------------------------------------
    # A13 — to_state_flags produces compact strings
    # -----------------------------------------------------------------------
    def test_A13_to_state_flags_format(self):
        narrative = "Application declined. Race was a factor."
        result = self.validator.validate(narrative, _applicant_payload())
        flags = self.validator.to_state_flags(result)
        assert isinstance(flags, list)
        for flag_str in flags:
            # Must contain ":" separating code : severity : description
            parts = flag_str.split(":")
            assert len(parts) >= 3

    # -----------------------------------------------------------------------
    # A14 — Non-decline communication type skips specific reason check
    # -----------------------------------------------------------------------
    def test_A14_approval_skips_specific_reason_check(self):
        narrative = (
            "Congratulations! Your application has been approved. "
            "You have the right to a free copy of your consumer report."
        )
        result = self.validator.validate(
            narrative,
            _applicant_payload(comm_type="approve", source="thinfile", aa_codes=[]),
        )
        codes = [f.rule_code for f in result.flags]
        assert "ECOA-002" not in codes


# ===========================================================================
# Group B — AdverseActionNotice
# ===========================================================================

class TestAdverseActionNotice:

    def setup_method(self):
        from app.compliance.adverse_action import AdverseActionNoticeBuilder
        self.builder = AdverseActionNoticeBuilder()

    # -----------------------------------------------------------------------
    # B1 — Known codes mapped to Reg B reason text
    # -----------------------------------------------------------------------
    def test_B1_known_codes_mapped(self):
        notice = self.builder.build(
            application_id="app-001",
            adverse_action_codes=["AA-023", "AA-030"],
            source_system="thinfile",
        )
        assert len(notice.specific_reasons) == 2
        assert "AA-023" in notice.specific_reasons[0]
        assert "AA-030" in notice.specific_reasons[1]
        assert len(notice.unknown_codes) == 0

    # -----------------------------------------------------------------------
    # B2 — Unknown codes tracked separately
    # -----------------------------------------------------------------------
    def test_B2_unknown_codes_tracked(self):
        notice = self.builder.build(
            application_id="app-002",
            adverse_action_codes=["AA-999"],
            source_system="thinfile",
        )
        assert "AA-999" in notice.unknown_codes
        assert len(notice.specific_reasons) == 0

    # -----------------------------------------------------------------------
    # B3 — FCRA disclosure present for thinfile source
    # -----------------------------------------------------------------------
    def test_B3_fcra_disclosure_for_thinfile(self):
        notice = self.builder.build(
            application_id="app-003",
            adverse_action_codes=["AA-007"],
            source_system="thinfile",
        )
        assert notice.fcra_disclosure is not None
        assert "60 days" in notice.fcra_disclosure

    # -----------------------------------------------------------------------
    # B4 — FCRA disclosure absent for internal source
    # -----------------------------------------------------------------------
    def test_B4_no_fcra_for_internal_source(self):
        notice = self.builder.build(
            application_id="app-004",
            adverse_action_codes=["AA-007"],
            source_system="internal",
        )
        assert notice.fcra_disclosure is None

    # -----------------------------------------------------------------------
    # B5 — ECOA notice always present
    # -----------------------------------------------------------------------
    def test_B5_ecoa_notice_always_present(self):
        notice = self.builder.build(
            application_id="app-005",
            adverse_action_codes=[],
            source_system="thinfile",
        )
        assert len(notice.ecoa_notice) > 50
        assert "Equal Credit Opportunity Act" in notice.ecoa_notice

    # -----------------------------------------------------------------------
    # B6 — render_text includes key sections
    # -----------------------------------------------------------------------
    def test_B6_render_text_contains_required_sections(self):
        notice = self.builder.build(
            application_id="app-006",
            adverse_action_codes=["AA-001"],
            source_system="thinfile",
            narrative="Your credit history shows late payments.",
        )
        text = self.builder.render_text(notice)
        assert "NOTICE OF CREDIT ACTION" in text
        assert "AA-001" in text
        assert "Equal Credit Opportunity Act" in text
        assert "Fair Credit Reporting Act" in text
        assert "app-006" in text

    # -----------------------------------------------------------------------
    # B7 — notice_date defaults to today
    # -----------------------------------------------------------------------
    def test_B7_notice_date_defaults_to_today(self):
        today = datetime.now(timezone.utc).date().isoformat()
        notice = self.builder.build(
            application_id="app-007",
            adverse_action_codes=[],
            source_system="thinfile",
        )
        assert notice.notice_date == today

    # -----------------------------------------------------------------------
    # B8 — to_dict serializable
    # -----------------------------------------------------------------------
    def test_B8_to_dict_serializable(self):
        notice = self.builder.build(
            application_id="app-008",
            adverse_action_codes=["AA-005"],
            source_system="credit-risk-platform",
        )
        d = notice.to_dict()
        assert isinstance(d, dict)
        assert d["application_id"] == "app-008"
        assert isinstance(d["specific_reasons"], list)

    # -----------------------------------------------------------------------
    # B9 — Codes from narrative text extracted as fallback
    # -----------------------------------------------------------------------
    def test_B9_code_extracted_from_narrative(self):
        notice = self.builder.build(
            application_id="app-009",
            adverse_action_codes=[],   # empty — codes must come from narrative
            source_system="thinfile",
            narrative="The main factor was AA-023: high obligations ratio.",
        )
        assert len(notice.specific_reasons) == 1
        assert "AA-023" in notice.specific_reasons[0]


# ===========================================================================
# Group C — Sr117Disclosures
# ===========================================================================

class TestSr117Disclosures:

    def setup_method(self):
        from app.compliance.sr117_disclosures import Sr117Disclosures
        self.sr117 = Sr117Disclosures()

    # -----------------------------------------------------------------------
    # C1 — Footer injected for analyst
    # -----------------------------------------------------------------------
    def test_C1_analyst_receives_footer(self):
        narrative = "DTI ratio is 42%. Risk score: 620."
        result = self.sr117.inject(narrative, "gpt-4.1", "sess-001", "analyst")
        assert "MODEL RISK DISCLOSURE" in result
        assert "SR 11-7" in result
        assert narrative.strip() in result

    # -----------------------------------------------------------------------
    # C2 — Footer injected for briefing
    # -----------------------------------------------------------------------
    def test_C2_briefing_receives_footer(self):
        narrative = "Portfolio delinquency rate is 2.3%."
        result = self.sr117.inject(narrative, "gpt-4.1", "sess-002", "briefing")
        assert "MODEL RISK DISCLOSURE" in result

    # -----------------------------------------------------------------------
    # C3 — Applicant gets no footer
    # -----------------------------------------------------------------------
    def test_C3_applicant_no_footer(self):
        narrative = "Your application was reviewed and denied."
        result = self.sr117.inject(narrative, "gpt-4.1", "sess-003", "applicant")
        assert result == narrative
        assert "MODEL RISK DISCLOSURE" not in result

    # -----------------------------------------------------------------------
    # C4 — Footer contains model identifier
    # -----------------------------------------------------------------------
    def test_C4_footer_contains_model_id(self):
        result = self.sr117.inject(
            "Some analysis.", "azure/gpt-4.1-2025-04-14", "sess-004", "analyst"
        )
        assert "azure/gpt-4.1-2025-04-14" in result

    # -----------------------------------------------------------------------
    # C5 — Footer contains session_id
    # -----------------------------------------------------------------------
    def test_C5_footer_contains_session_id(self):
        result = self.sr117.inject("Analysis.", "gpt-4", "my-session-xyz", "analyst")
        assert "my-session-xyz" in result

    # -----------------------------------------------------------------------
    # C6 — has_disclosure idempotency check
    # -----------------------------------------------------------------------
    def test_C6_has_disclosure_detects_footer(self):
        once = self.sr117.inject("Analysis.", "gpt-4", "s-1", "analyst")
        assert self.sr117.has_disclosure(once) is True
        assert self.sr117.has_disclosure("No footer here.") is False

    # -----------------------------------------------------------------------
    # C7 — format_disclosure returns standalone footer
    # -----------------------------------------------------------------------
    def test_C7_format_disclosure_standalone(self):
        footer = self.sr117.format_disclosure("gpt-4.1", "sess-007")
        assert "MODEL RISK DISCLOSURE" in footer
        assert "gpt-4.1" in footer
        assert "sess-007" in footer

    # -----------------------------------------------------------------------
    # C8 — Limitations text present
    # -----------------------------------------------------------------------
    def test_C8_limitations_text_present(self):
        result = self.sr117.inject("Analysis.", "gpt-4", "s-2", "analyst")
        assert "limitations" in result.lower()
        assert "Validation Status" in result


# ===========================================================================
# Group D — compliance_check_node
# ===========================================================================

class TestComplianceCheckNode:

    # -----------------------------------------------------------------------
    # D1 — Applicant + prohibited basis → compliance_passed=False
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_D1_applicant_violation_fails(self):
        from app.agent.nodes import compliance_check_node

        state = {
            "audience": "applicant",
            "grounded_narrative": (
                "Your application was declined based on race. "
                "DTI was a factor."
            ),
            "context_payload": {
                "communication_type": "decline",
                "source": "thinfile",
                "adverse_action_codes": ["AA-023"],
            },
            "citations": [],
            "provider_model": "gpt-4.1",
            "session_id": "test-session-001",
        }

        result = await compliance_check_node(state)
        assert result["compliance_passed"] is False
        assert len(result["compliance_flags"]) > 0

    # -----------------------------------------------------------------------
    # D2 — Applicant + compliant narrative → compliance_passed=True
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_D2_applicant_compliant_passes(self):
        from app.agent.nodes import compliance_check_node

        narrative = (
            "Your application was declined because your debt-to-income ratio of 58% "
            "exceeds our maximum policy threshold of 45%. "
            "You have the right to a free copy of your consumer report from the "
            "consumer reporting agency within 60 days of receiving this notice."
        )

        state = {
            "audience": "applicant",
            "grounded_narrative": narrative,
            "context_payload": {
                "communication_type": "decline",
                "source": "thinfile",
                "adverse_action_codes": ["AA-030"],
            },
            "citations": [],
            "provider_model": "gpt-4.1",
            "session_id": "test-session-002",
        }

        result = await compliance_check_node(state)
        assert result["compliance_passed"] is True

    # -----------------------------------------------------------------------
    # D3 — Analyst audience → SR 11-7 footer injected, compliance_passed=True
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_D3_analyst_sr117_injected(self):
        from app.agent.nodes import compliance_check_node

        state = {
            "audience": "analyst",
            "grounded_narrative": "The DTI is 42% and the credit score is 680.",
            "context_payload": {"communication_type": "analyst_query"},
            "citations": [],
            "provider_model": "azure/gpt-4.1",
            "session_id": "test-session-003",
        }

        result = await compliance_check_node(state)
        assert result["compliance_passed"] is True
        assert len(result["compliance_flags"]) == 0
        assert "MODEL RISK DISCLOSURE" in result["grounded_narrative"]

    # -----------------------------------------------------------------------
    # D4 — Briefing audience → SR 11-7 footer injected
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_D4_briefing_sr117_injected(self):
        from app.agent.nodes import compliance_check_node

        state = {
            "audience": "briefing",
            "grounded_narrative": "Portfolio Q3 summary: 2.3% 30-day delinquency.",
            "context_payload": {},
            "citations": [],
            "provider_model": "azure/gpt-4.1",
            "session_id": "test-session-004",
        }

        result = await compliance_check_node(state)
        assert result["compliance_passed"] is True
        assert "MODEL RISK DISCLOSURE" in result["grounded_narrative"]

    # -----------------------------------------------------------------------
    # D5 — compliance_flags returned as list of strings
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_D5_compliance_flags_are_strings(self):
        from app.agent.nodes import compliance_check_node

        state = {
            "audience": "applicant",
            "grounded_narrative": "Declined. Race was a factor in the decision.",
            "context_payload": {
                "communication_type": "decline",
                "source": "internal",
                "adverse_action_codes": ["AA-001"],
            },
            "citations": [],
            "provider_model": "gpt-4",
            "session_id": "test-session-005",
        }

        result = await compliance_check_node(state)
        for flag in result["compliance_flags"]:
            assert isinstance(flag, str)


# ===========================================================================
# Group E — Audit endpoint
# ===========================================================================

class TestAuditEndpoint:
    """
    Tests for GET /v1/audit/{session_id}.

    The DB calls are fully mocked — no live PostgreSQL required.
    """

    # -----------------------------------------------------------------------
    # E1 — 422 on malformed UUID
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_E1_invalid_uuid_returns_422(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/audit/not-a-uuid")

        assert response.status_code == 422

    # -----------------------------------------------------------------------
    # E2 — 404 when session not found
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_E2_missing_session_returns_404(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        missing_id = str(uuid.uuid4())

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock(return_value=mock_db)

        with patch("app.api.v1.audit.AsyncSessionLocal", mock_session_cls):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(f"/v1/audit/{missing_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    # -----------------------------------------------------------------------
    # E3 — 200 with full AuditResponse on valid session
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_E3_valid_session_returns_audit_response(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        session_uuid = uuid.uuid4()
        now = datetime.now(timezone.utc)

        # Build a fake CopilotSession ORM row
        mock_session_row = MagicMock()
        mock_session_row.session_id = session_uuid
        mock_session_row.query_text = "What is my risk score?"
        mock_session_row.intent = "explain_decision"
        mock_session_row.audience = "analyst"
        mock_session_row.source_system = "thinfile"
        mock_session_row.user_id = "user-abc"
        mock_session_row.retrieved_chunks = []
        mock_session_row.rendered_prompt = "System: ...\nUser: ..."
        mock_session_row.raw_llm_output = "The risk score is 680."
        mock_session_row.grounded_narrative = "The risk score is 680. [SR 11-7 footer]"
        mock_session_row.confidence_score = 0.88
        mock_session_row.suppressed_claims = []
        mock_session_row.compliance_flags = []
        mock_session_row.provider_model = "azure/gpt-4.1"
        mock_session_row.provider_fallback_used = False
        mock_session_row.context_token_count = 1200
        mock_session_row.output_token_count = 320
        mock_session_row.created_at = now

        # Build a fake Citation row
        mock_citation = MagicMock()
        mock_citation.citation_id = uuid.uuid4()
        mock_citation.claim_text = "The risk score is 680."
        mock_citation.source_type = "vector_doc"
        mock_citation.source_ref = "policy_docs/credit_policy.md"
        mock_citation.similarity_score = 0.91
        mock_citation.confidence = 0.88

        # Mock DB context manager
        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=[mock_citation])

        mock_execute_result = MagicMock()
        mock_execute_result.scalars = MagicMock(return_value=mock_scalars)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_session_row)
        mock_db.execute = AsyncMock(return_value=mock_execute_result)
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock(return_value=mock_db)

        with patch("app.api.v1.audit.AsyncSessionLocal", mock_session_cls):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(f"/v1/audit/{session_uuid}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(session_uuid)
        assert data["query_text"] == "What is my risk score?"
        assert data["audience"] == "analyst"
        assert data["provider_model"] == "azure/gpt-4.1"
        assert data["provider_fallback_used"] is False
        assert data["confidence_score"] == pytest.approx(0.88, abs=0.001)
        assert len(data["citations"]) == 1
        assert data["citations"][0]["claim_text"] == "The risk score is 680."
        assert data["citations"][0]["source_type"] == "vector_doc"

    # -----------------------------------------------------------------------
    # E4 — Session with no citations returns empty citations list
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_E4_no_citations_returns_empty_list(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        session_uuid = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_session_row = MagicMock()
        mock_session_row.session_id = session_uuid
        mock_session_row.query_text = "Portfolio overview?"
        mock_session_row.intent = "portfolio_brief"
        mock_session_row.audience = "briefing"
        mock_session_row.source_system = None
        mock_session_row.user_id = None
        mock_session_row.retrieved_chunks = None
        mock_session_row.rendered_prompt = None
        mock_session_row.raw_llm_output = None
        mock_session_row.grounded_narrative = "Portfolio summary."
        mock_session_row.confidence_score = 0.79
        mock_session_row.suppressed_claims = None
        mock_session_row.compliance_flags = None
        mock_session_row.provider_model = "azure/gpt-4.1"
        mock_session_row.provider_fallback_used = False
        mock_session_row.context_token_count = None
        mock_session_row.output_token_count = None
        mock_session_row.created_at = now

        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=[])

        mock_execute_result = MagicMock()
        mock_execute_result.scalars = MagicMock(return_value=mock_scalars)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_session_row)
        mock_db.execute = AsyncMock(return_value=mock_execute_result)
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock(return_value=mock_db)

        with patch("app.api.v1.audit.AsyncSessionLocal", mock_session_cls):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(f"/v1/audit/{session_uuid}")

        assert response.status_code == 200
        data = response.json()
        assert data["citations"] == []
        assert data["source_system"] is None
