"""
backend/app/compliance/ecoa_validator.py
==========================================
ECOA (Equal Credit Opportunity Act) and FCRA (Fair Credit Reporting Act)
output compliance validator for applicant-facing communications.

Validates that generated adverse action notices and applicant communications:
  1. Include at least one specific adverse action reason (Reg B § 202.9).
  2. Do not reference prohibited basis characteristics (race, sex, age, etc.).
  3. Include FCRA § 615 rights disclosure when a consumer report was used.
  4. Do not include discouraged language patterns (confusing, evasive, or
     impermissibly vague reasons per Reg B Commentary).
  5. Do not use language that could imply a waiver of consumer rights.

This validator is a rule-based pre-delivery gate — it does NOT use LLM
judgment.  Deterministic rules only.  All violations are logged with the
rule code that triggered them.

Usage::

    from app.compliance.ecoa_validator import EcoaValidator, ValidationResult

    validator = EcoaValidator()
    result = validator.validate(narrative, context_payload, citations)
    if not result.passed:
        # result.flags contains structured violation records
        raise ComplianceError(result.flags)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

# Reg B § 202.2(z) — prohibited basis characteristics.
# Matching any of these in a generated applicant narrative is a hard violation.
_PROHIBITED_BASIS_PATTERNS: List[str] = [
    r"\brace\b",
    r"\bcolor\b",
    r"\breligion\b",
    r"\bnational\s+origin\b",
    r"\bsex\b",
    r"\bgender\b",
    r"\bmarital\s+status\b",
    r"\bage\b",
    r"\bpregnancy\b",
    r"\bpregnant\b",
    r"\bpublic\s+assistance\b",
    r"\bwelfare\b",
    r"\bethnic\b",
    r"\bethnicity\b",
    r"\bsexual\s+orientation\b",
    r"\bdisabilit(?:y|ies)\b",
]

_PROHIBITED_BASIS_RE = re.compile(
    "|".join(_PROHIBITED_BASIS_PATTERNS), re.IGNORECASE
)

# Reg B Commentary — discouraged vague reason phrases that do not provide
# sufficient specificity to be considered a "specific reason" under § 202.9(b).
_VAGUE_REASON_PATTERNS: List[str] = [
    r"\bdid\s+not\s+meet\s+(?:our\s+)?(?:standards?|criteria|requirements?)\b",
    r"\bnot\s+(?:qualified|eligible|approved)\b",
    r"\bcredit\s+score\s+(?:was\s+)?(?:too\s+low|insufficient|not\s+sufficient)\b",
    r"\byou\s+do\s+not\s+qualify\b",
    r"\bwe\s+(?:are\s+)?unable\s+to\s+(?:approve|extend)\b",
]

_VAGUE_REASON_RE = re.compile(
    "|".join(_VAGUE_REASON_PATTERNS), re.IGNORECASE
)

# FCRA § 615(a) — required rights disclosure language (subset matching).
# The notice must state the applicant's right to a free copy of their
# consumer report within 60 days.
_FCRA_DISCLOSURE_REQUIRED_PHRASES: List[re.Pattern] = [
    re.compile(r"free\s+(?:copy\s+of\s+(?:your\s+)?)?(?:consumer\s+)?report", re.IGNORECASE),
    re.compile(r"right\s+to\s+(?:a\s+)?free", re.IGNORECASE),
    re.compile(r"consumer\s+report(?:ing)?\s+agenc", re.IGNORECASE),
]

# Adverse action reason: at least one of these patterns must be present
# in a "decline" communication (Reg B § 202.9(b)(2) — specific reasons).
_SPECIFIC_REASON_INDICATORS: List[re.Pattern] = [
    re.compile(r"debt[\s-]?to[\s-]?income", re.IGNORECASE),
    re.compile(r"\bDTI\b"),
    re.compile(r"credit\s+(?:score|rating|history|utilization|inquiries)", re.IGNORECASE),
    re.compile(r"employment\s+(?:history|status|length|stability)", re.IGNORECASE),
    re.compile(r"income\s+(?:insufficient|too\s+low|unverifiable|could\s+not\s+be\s+verified)", re.IGNORECASE),
    re.compile(r"insufficient\s+(?:income|collateral|assets)", re.IGNORECASE),
    re.compile(r"(?:delinquent|derogatory|negative)\s+(?:accounts?|items?|history)", re.IGNORECASE),
    re.compile(r"(?:bankruptcy|foreclosure|charge[\s-]?off)", re.IGNORECASE),
    re.compile(r"(?:length|time)\s+(?:at\s+)?(?:current\s+)?(?:job|address|residence|employment)", re.IGNORECASE),
    re.compile(r"AA-\d{3}", re.IGNORECASE),  # Adverse action code reference
]

# Waiver-of-rights language — must never appear in a consumer communication.
_RIGHTS_WAIVER_PATTERNS: List[str] = [
    r"\bwaive\s+(?:your\s+)?rights?\b",
    r"\bno\s+right\s+to\b",
    r"\bby\s+(?:accepting|signing|proceeding)\s+you\s+agree\s+to\s+waive\b",
]

_RIGHTS_WAIVER_RE = re.compile(
    "|".join(_RIGHTS_WAIVER_PATTERNS), re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ComplianceFlag:
    rule_code: str        # e.g. "ECOA-001"
    severity: str         # "ERROR" | "WARNING"
    description: str
    matched_text: Optional[str] = None


@dataclass
class ValidationResult:
    passed: bool
    flags: List[ComplianceFlag] = field(default_factory=list)

    @property
    def errors(self) -> List[ComplianceFlag]:
        return [f for f in self.flags if f.severity == "ERROR"]

    @property
    def warnings(self) -> List[ComplianceFlag]:
        return [f for f in self.flags if f.severity == "WARNING"]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class EcoaValidator:
    """
    Rule-based ECOA/FCRA compliance validator for applicant communications.

    All rules are deterministic — no LLM calls are made.
    """

    def validate(
        self,
        narrative: str,
        context_payload: Dict,
        citations: Optional[List[dict]] = None,
    ) -> ValidationResult:
        """
        Run all compliance rules against *narrative*.

        Args:
            narrative:       The generated narrative text to validate.
            context_payload: The agent context dict including communication_type,
                             source, and adverse_action_codes.
            citations:       Citation records from the citation enforcer (optional).

        Returns:
            ValidationResult with passed=True iff no ERROR-severity flags.
        """
        flags: List[ComplianceFlag] = []
        comm_type: str = context_payload.get("communication_type", "")
        source: str = context_payload.get("source", "")
        adverse_action_codes: List[str] = context_payload.get("adverse_action_codes", [])

        # ECOA-001: No prohibited basis characteristics
        flags.extend(self._check_prohibited_basis(narrative))

        # ECOA-002: Specific adverse action reasons required for decline
        if comm_type == "decline":
            flags.extend(self._check_specific_reasons(narrative, adverse_action_codes))

        # FCRA-001: Consumer report rights disclosure required when CRA data used
        if source in ("thinfile", "credit_risk_platform", "credit-risk-platform"):
            flags.extend(self._check_fcra_disclosure(narrative, comm_type))

        # ECOA-003: No impermissibly vague reasons
        if comm_type == "decline":
            flags.extend(self._check_vague_reasons(narrative))

        # ECOA-004: No rights-waiver language
        flags.extend(self._check_rights_waiver(narrative))

        passed = len([f for f in flags if f.severity == "ERROR"]) == 0

        log.info(
            "ecoa_validator",
            communication_type=comm_type,
            total_flags=len(flags),
            errors=len([f for f in flags if f.severity == "ERROR"]),
            warnings=len([f for f in flags if f.severity == "WARNING"]),
            passed=passed,
        )

        return ValidationResult(passed=passed, flags=flags)

    # -----------------------------------------------------------------------
    # Individual rule checks
    # -----------------------------------------------------------------------

    def _check_prohibited_basis(self, text: str) -> List[ComplianceFlag]:
        flags: List[ComplianceFlag] = []
        for match in _PROHIBITED_BASIS_RE.finditer(text):
            flags.append(
                ComplianceFlag(
                    rule_code="ECOA-001",
                    severity="ERROR",
                    description=(
                        "Prohibited basis characteristic referenced in applicant communication. "
                        "Reg B § 202.2(z) prohibits reference to race, color, religion, national "
                        "origin, sex, marital status, age, or receipt of public assistance."
                    ),
                    matched_text=match.group(0),
                )
            )
        return flags

    def _check_specific_reasons(
        self, text: str, adverse_action_codes: List[str]
    ) -> List[ComplianceFlag]:
        # Pass if at least one specific reason indicator is present in the text
        # or at least one adverse action code was provided in context.
        has_specific_reason = any(
            pattern.search(text) for pattern in _SPECIFIC_REASON_INDICATORS
        )
        has_aa_codes = bool(adverse_action_codes)

        if not has_specific_reason and not has_aa_codes:
            return [
                ComplianceFlag(
                    rule_code="ECOA-002",
                    severity="ERROR",
                    description=(
                        "Decline communication lacks specific adverse action reasons. "
                        "Reg B § 202.9(b)(2) requires specific written reasons or a "
                        "statement of the applicant's right to request them."
                    ),
                )
            ]
        return []

    def _check_fcra_disclosure(
        self, text: str, comm_type: str
    ) -> List[ComplianceFlag]:
        # Only required for decline / adverse action communications
        if comm_type not in ("decline", "counteroffer"):
            return []

        has_disclosure = any(
            pattern.search(text) for pattern in _FCRA_DISCLOSURE_REQUIRED_PHRASES
        )
        if not has_disclosure:
            return [
                ComplianceFlag(
                    rule_code="FCRA-001",
                    severity="ERROR",
                    description=(
                        "Adverse action notice does not include FCRA § 615(a) rights disclosure. "
                        "Consumer must be notified of their right to a free copy of their consumer "
                        "report from the reporting agency within 60 days."
                    ),
                )
            ]
        return []

    def _check_vague_reasons(self, text: str) -> List[ComplianceFlag]:
        flags: List[ComplianceFlag] = []
        for match in _VAGUE_REASON_RE.finditer(text):
            flags.append(
                ComplianceFlag(
                    rule_code="ECOA-003",
                    severity="WARNING",
                    description=(
                        "Vague or non-specific decline reason detected. "
                        "Reg B Commentary requires reasons to be specific enough "
                        "for the applicant to understand what action to take."
                    ),
                    matched_text=match.group(0),
                )
            )
        return flags

    def _check_rights_waiver(self, text: str) -> List[ComplianceFlag]:
        flags: List[ComplianceFlag] = []
        for match in _RIGHTS_WAIVER_RE.finditer(text):
            flags.append(
                ComplianceFlag(
                    rule_code="ECOA-004",
                    severity="ERROR",
                    description=(
                        "Rights-waiver language detected in applicant communication. "
                        "Consumer rights under ECOA and FCRA cannot be waived by agreement."
                    ),
                    matched_text=match.group(0),
                )
            )
        return flags

    def to_state_flags(self, result: ValidationResult) -> List[str]:
        """Convert ValidationResult to a list of compact flag strings for AgentState."""
        return [
            f"{f.rule_code}:{f.severity}:{f.description[:80]}"
            for f in result.flags
        ]
