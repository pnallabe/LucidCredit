"""
tests/unit/test_ecoa_validator.py
===================================
Unit tests for ECOA/FCRA compliance validator and PII utility functions.

No LLM calls, no database, no external services.
All tests are deterministic and fast (<1 s).
"""
import pytest

from app.compliance.ecoa_validator import (
    FCRA_615_DISCLOSURE,
    EcoaValidator,
    ValidationResult,
    inject_fcra_disclosure,
    scrub_pii_from_output,
)

VALIDATOR = EcoaValidator()


# ---------------------------------------------------------------------------
# scrub_pii_from_output
# ---------------------------------------------------------------------------

class TestScrubPiiFromOutput:
    def test_ssn_redacted(self):
        result = scrub_pii_from_output("Borrower SSN is 123-45-6789.")
        assert "123-45-6789" not in result
        assert "[REDACTED-SSN]" in result

    def test_email_redacted(self):
        result = scrub_pii_from_output("Contact john.doe@example.com for details.")
        assert "john.doe@example.com" not in result
        assert "[REDACTED-EMAIL]" in result

    def test_phone_redacted(self):
        result = scrub_pii_from_output("Call 555-123-4567 for assistance.")
        assert "555-123-4567" not in result
        assert "[REDACTED-PHONE]" in result

    def test_dob_redacted(self):
        result = scrub_pii_from_output("Date of birth: 1985-04-15.")
        assert "1985-04-15" not in result
        assert "[REDACTED-DOB]" in result

    def test_clean_text_unchanged(self):
        text = "Your application was declined due to a high debt-to-income ratio."
        assert scrub_pii_from_output(text) == text

    def test_multiple_pii_types_all_redacted(self):
        text = "Email: bob@bank.com, SSN: 987-65-4321, Phone: 800-555-1234."
        result = scrub_pii_from_output(text)
        assert "bob@bank.com" not in result
        assert "987-65-4321" not in result
        assert "800-555-1234" not in result


# ---------------------------------------------------------------------------
# inject_fcra_disclosure
# ---------------------------------------------------------------------------

class TestInjectFcraDisclosure:
    def test_disclosure_injected_when_missing(self):
        narrative = "Your application was declined due to insufficient income."
        result = inject_fcra_disclosure(narrative)
        assert "free" in result.lower()
        # Either "consumer report" or "right" must appear in the injected block
        assert (
            "consumer report" in result.lower()
            or "right" in result.lower()
        )

    def test_no_duplicate_injection_when_already_present(self):
        narrative = (
            "Your application was declined. "
            "You have the right to a free copy of your consumer report "
            "from the consumer reporting agency within 60 days."
        )
        result = inject_fcra_disclosure(narrative)
        # The narrative was already compliant — content should not grow
        # by another full disclosure block
        assert result.count("right to a free") == narrative.count("right to a free")

    def test_disclosure_appended_at_end(self):
        narrative = "We cannot approve your application at this time."
        result = inject_fcra_disclosure(narrative)
        assert result.startswith(narrative)

    def test_fcra_constant_contains_required_language(self):
        assert "60 days" in FCRA_615_DISCLOSURE
        assert "free" in FCRA_615_DISCLOSURE.lower()
        assert "consumer report" in FCRA_615_DISCLOSURE.lower()


# ---------------------------------------------------------------------------
# EcoaValidator — ECOA-001 (discriminatory attribution)
# ---------------------------------------------------------------------------

class TestEcoaValidatorDiscriminatoryBasis:
    def test_declined_because_of_race(self):
        result = VALIDATOR.validate(
            "Your application was declined because of your race.",
            {"communication_type": "decline"},
        )
        assert any(f.rule_code == "ECOA-001" for f in result.errors)

    def test_denied_due_to_age(self):
        result = VALIDATOR.validate(
            "The application was denied due to the applicant's age.",
            {"communication_type": "decline"},
        )
        assert any(f.rule_code == "ECOA-001" for f in result.errors)

    def test_non_discriminatory_equal_opportunity_passes(self):
        # Equal-treatment disclaimer must NOT trigger ECOA-001
        result = VALIDATOR.validate(
            "We do not discriminate based on race, color, religion, national origin, "
            "sex, marital status, age, or receipt of public assistance.",
            {"communication_type": "decline", "adverse_action_codes": ["AA-001"]},
        )
        assert not any(f.rule_code == "ECOA-001" for f in result.errors)

    def test_negated_basis_passes(self):
        result = VALIDATOR.validate(
            "The decision was not based on race or any protected characteristic. "
            "Your DTI ratio of 48% exceeded our maximum threshold.",
            {"communication_type": "decline", "adverse_action_codes": ["AA-003"]},
        )
        assert not any(f.rule_code == "ECOA-001" for f in result.errors)


# ---------------------------------------------------------------------------
# EcoaValidator — FCRA-001
# ---------------------------------------------------------------------------

class TestEcoaValidatorFCRA:
    def test_decline_without_fcra_fails(self):
        result = VALIDATOR.validate(
            "Your application was declined due to a low credit score.",
            {"communication_type": "decline", "source": "thinfile"},
        )
        assert any(f.rule_code == "FCRA-001" for f in result.errors)

    def test_decline_with_fcra_passes(self):
        result = VALIDATOR.validate(
            "Your application was declined. You have the right to a free copy of "
            "your consumer report from the consumer reporting agency within 60 days.",
            {
                "communication_type": "decline",
                "source": "thinfile",
                "adverse_action_codes": ["AA-001"],
            },
        )
        assert not any(f.rule_code == "FCRA-001" for f in result.errors)

    def test_approval_no_fcra_required(self):
        result = VALIDATOR.validate(
            "Congratulations! Your application has been approved.",
            {"communication_type": "approve", "source": "thinfile"},
        )
        assert not any(f.rule_code == "FCRA-001" for f in result.errors)


# ---------------------------------------------------------------------------
# EcoaValidator — PII-001
# ---------------------------------------------------------------------------

class TestEcoaValidatorPII:
    def test_ssn_in_output_fails(self):
        result = VALIDATOR.validate(
            "Your SSN 123-45-6789 was verified.",
            {"communication_type": "decline"},
        )
        assert any(f.rule_code == "PII-001" for f in result.errors)

    def test_email_in_output_fails(self):
        result = VALIDATOR.validate(
            "We emailed bob.smith@bank.com the decision.",
            {"communication_type": "decline"},
        )
        assert any(f.rule_code == "PII-001" for f in result.errors)

    def test_clean_output_passes_pii(self):
        result = VALIDATOR.validate(
            "Your application was declined due to a high DTI ratio.",
            {"communication_type": "decline", "adverse_action_codes": ["AA-003"]},
        )
        assert not any(f.rule_code == "PII-001" for f in result.errors)


# ---------------------------------------------------------------------------
# EcoaValidator — ECOA-004 (rights waiver)
# ---------------------------------------------------------------------------

class TestEcoaValidatorRightsWaiver:
    def test_waiver_language_flagged(self):
        result = VALIDATOR.validate(
            "By proceeding you agree to waive your rights under ECOA.",
            {"communication_type": "decline"},
        )
        assert any(f.rule_code == "ECOA-004" for f in result.errors)


# ---------------------------------------------------------------------------
# EcoaValidator — ECOA-002 (specific reasons)
# ---------------------------------------------------------------------------

class TestEcoaValidatorDeclineReasons:
    def test_decline_with_dti_passes(self):
        result = VALIDATOR.validate(
            "Your debt-to-income ratio of 52% exceeds our maximum threshold.",
            {
                "communication_type": "decline",
                "source": "thinfile",
                "adverse_action_codes": ["AA-003"],
            },
        )
        assert not any(f.rule_code == "ECOA-002" for f in result.errors)

    def test_decline_with_aa_codes_passes(self):
        result = VALIDATOR.validate(
            "Your application has been declined.",
            {
                "communication_type": "decline",
                "adverse_action_codes": ["AA-001", "AA-003"],
            },
        )
        assert not any(f.rule_code == "ECOA-002" for f in result.errors)
