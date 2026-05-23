"""
backend/app/compliance/adverse_action.py
==========================================
ECOA/Reg B adverse action notice (AAN) generator.

Constructs a structured adverse action notice from:
  1. Adverse action codes returned by the ThinFile or CRP API.
  2. The grounded narrative produced by the LangGraph agent.
  3. Source system metadata for FCRA § 615 disclosure.

The AAN structure follows the CFPB model adverse action notice form (Form C-1
through C-5 equivalents) and satisfies Reg B § 202.9 requirements:
  - Written notice within 30 days.
  - Statement of action taken.
  - Name and address of creditor.
  - ECOA notice.
  - Statement of specific reasons (or right to request them).
  - Name and address of federal agency monitoring compliance.
  - FCRA § 615(a) rights disclosure when a consumer report was used.

This module does NOT call any LLM — all content is assembled from retrieved,
validated data.  The LLM-generated narrative is appended as supplemental
context only after the structured AAN body is built.

Usage::

    from app.compliance.adverse_action import AdverseActionNoticeBuilder

    builder = AdverseActionNoticeBuilder()
    notice = builder.build(
        application_id="...",
        adverse_action_codes=["AA-007", "AA-012"],
        source_system="thinfile",
        narrative="...",
        creditor_name="LucidCredit Financial",
    )
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timezone, datetime
from typing import Dict, List, Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Adverse action reason code registry
# ---------------------------------------------------------------------------
# Maps internal codes (AA-NNN) to Reg B § 202.9(b)(2) compliant reason text.
# Codes are aligned with the CFPB/FFIEC standard adverse action codes.

_AA_CODE_DESCRIPTIONS: Dict[str, str] = {
    # Credit history
    "AA-001": "Delinquent past or present credit obligations with others.",
    "AA-002": "Garnishment or attachment of income or property.",
    "AA-003": "Foreclosure or repossession.",
    "AA-004": "Collection action or judgment.",
    "AA-005": "Bankruptcy.",
    "AA-006": "Number of recent inquiries on credit bureau report.",
    "AA-007": "Value or type of collateral not sufficient.",
    "AA-008": "Length of employment.",
    # Income / capacity
    "AA-009": "Temporary or irregular employment.",
    "AA-010": "Unable to verify employment.",
    "AA-011": "Length of residence.",
    "AA-012": "Temporary residence.",
    "AA-013": "Unable to verify residence.",
    "AA-014": "No credit file.",
    "AA-015": "Limited credit experience.",
    "AA-016": "Poor credit performance with us.",
    # Financial ratios
    "AA-017": "Delinquent with us.",
    "AA-018": "Number of accounts with satisfactory credit history.",
    "AA-019": "Too many requests for credit within past 12 months.",
    "AA-020": "Amount of loan requested not justified by income.",
    "AA-021": "Inability to verify income.",
    "AA-022": "Insufficient income for amount of credit requested.",
    "AA-023": "Excessive obligations in relation to income.",
    "AA-024": "Unable to verify credit references.",
    # Product / policy
    "AA-025": "Credit application incomplete.",
    "AA-026": "Unable to verify identity.",
    "AA-027": "Does not meet the minimum age requirement.",
    "AA-028": "Business is not eligible for this product.",
    "AA-029": "Does not qualify under current terms.",
    "AA-030": "Debt-to-income ratio exceeds policy maximum.",
}

# FCRA § 615(a) disclosure boilerplate
_FCRA_615_DISCLOSURE: str = (
    "We obtained information from a consumer reporting agency as part of our review. "
    "You have the right to obtain a free copy of your consumer report from the reporting "
    "agency if you request it within 60 days of receiving this notice. You also have "
    "the right to dispute the accuracy or completeness of any information in your "
    "consumer report by contacting the reporting agency directly."
)

# Reg B ECOA notice boilerplate (required in all adverse action notices)
_ECOA_NOTICE: str = (
    "The federal Equal Credit Opportunity Act prohibits creditors from discriminating "
    "against credit applicants on the basis of race, color, religion, national origin, "
    "sex, marital status, age (provided the applicant has the capacity to enter into a "
    "binding contract); because all or part of the applicant's income derives from any "
    "public assistance program; or because the applicant has in good faith exercised any "
    "right under the Consumer Credit Protection Act. The federal agency that administers "
    "compliance with this law concerning this creditor is the Consumer Financial "
    "Protection Bureau, 1700 G Street NW, Washington DC 20552."
)

# Federal monitoring agency per creditor type (simplified)
_FEDERAL_AGENCY: str = (
    "Consumer Financial Protection Bureau (CFPB), 1700 G Street NW, Washington, DC 20552. "
    "Website: www.consumerfinance.gov"
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AdverseActionNotice:
    application_id: str
    notice_date: str                       # ISO-8601 date
    action_taken: str                      # "Credit application denied."
    creditor_name: str
    creditor_address: str
    specific_reasons: List[str]            # Reg B § 202.9(b)(2)
    unknown_codes: List[str]               # Codes not in registry
    ecoa_notice: str
    fcra_disclosure: Optional[str]         # None if no consumer report used
    federal_agency: str
    supplemental_narrative: Optional[str]  # LLM-generated, post-AAN
    raw_codes: List[str]

    def to_dict(self) -> dict:
        return {
            "application_id": self.application_id,
            "notice_date": self.notice_date,
            "action_taken": self.action_taken,
            "creditor_name": self.creditor_name,
            "creditor_address": self.creditor_address,
            "specific_reasons": self.specific_reasons,
            "unknown_codes": self.unknown_codes,
            "ecoa_notice": self.ecoa_notice,
            "fcra_disclosure": self.fcra_disclosure,
            "federal_agency": self.federal_agency,
            "supplemental_narrative": self.supplemental_narrative,
            "raw_codes": self.raw_codes,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class AdverseActionNoticeBuilder:
    """
    Builds a structured Reg B / FCRA-compliant adverse action notice.

    No LLM calls are made.  All content is assembled from:
      - Adverse action codes (mapped to Reg B reason text)
      - Source system metadata
      - The grounded narrative (appended as supplemental context only)
    """

    def build(
        self,
        application_id: str,
        adverse_action_codes: List[str],
        source_system: str,
        narrative: str = "",
        creditor_name: str = "LucidCredit Financial",
        creditor_address: str = "123 Credit Way, San Francisco, CA 94105",
        action_taken: str = "Credit application denied.",
        notice_date: Optional[str] = None,
    ) -> AdverseActionNotice:
        """
        Build the AAN from adverse action codes and context.

        Args:
            application_id:       UUID string of the loan application.
            adverse_action_codes: List of AA-NNN codes from the upstream API.
            source_system:        "thinfile" or "credit-risk-platform" — determines
                                  whether FCRA § 615 disclosure is required.
            narrative:            LLM-generated grounded narrative (appended last).
            creditor_name:        Name of creditor for the notice header.
            creditor_address:     Address of creditor.
            action_taken:         Action statement (Reg B § 202.9(a)(1)).
            notice_date:          ISO-8601 date string; defaults to today (UTC).

        Returns:
            AdverseActionNotice with all required disclosure fields.
        """
        if notice_date is None:
            notice_date = datetime.now(timezone.utc).date().isoformat()

        # Map codes to Reg B reason text
        specific_reasons: List[str] = []
        unknown_codes: List[str] = []

        for code in adverse_action_codes:
            code_upper = code.upper().strip()
            description = _AA_CODE_DESCRIPTIONS.get(code_upper)
            if description:
                specific_reasons.append(f"{code_upper}: {description}")
            else:
                unknown_codes.append(code_upper)
                log.warning(
                    "adverse_action.unknown_code",
                    code=code_upper,
                    application_id=application_id,
                )

        # Fallback: extract any AA-NNN patterns from the narrative itself
        if not specific_reasons and not unknown_codes:
            found_in_text = re.findall(r"\bAA-\d{3}\b", narrative, re.IGNORECASE)
            for code in found_in_text:
                code_upper = code.upper()
                description = _AA_CODE_DESCRIPTIONS.get(code_upper)
                if description:
                    specific_reasons.append(f"{code_upper}: {description}")

        # FCRA § 615 disclosure — required when consumer report was used
        uses_consumer_report = source_system.lower() in (
            "thinfile", "credit_risk_platform", "credit-risk-platform"
        )
        fcra_disclosure = _FCRA_615_DISCLOSURE if uses_consumer_report else None

        notice = AdverseActionNotice(
            application_id=application_id,
            notice_date=notice_date,
            action_taken=action_taken,
            creditor_name=creditor_name,
            creditor_address=creditor_address,
            specific_reasons=specific_reasons,
            unknown_codes=unknown_codes,
            ecoa_notice=_ECOA_NOTICE,
            fcra_disclosure=fcra_disclosure,
            federal_agency=_FEDERAL_AGENCY,
            supplemental_narrative=narrative or None,
            raw_codes=adverse_action_codes,
        )

        log.info(
            "adverse_action_notice.built",
            application_id=application_id,
            reason_count=len(specific_reasons),
            unknown_count=len(unknown_codes),
            fcra_required=uses_consumer_report,
        )

        return notice

    def render_text(self, notice: AdverseActionNotice) -> str:
        """
        Render the AAN as a plain-text string suitable for email/letter delivery.
        """
        lines: List[str] = []
        lines.append(f"NOTICE OF CREDIT ACTION — {notice.notice_date}")
        lines.append(f"Application ID: {notice.application_id}")
        lines.append("")
        lines.append(f"Creditor: {notice.creditor_name}")
        lines.append(f"Address: {notice.creditor_address}")
        lines.append("")
        lines.append(f"Action Taken: {notice.action_taken}")
        lines.append("")

        if notice.specific_reasons:
            lines.append("Principal Reason(s) for Action:")
            for reason in notice.specific_reasons:
                lines.append(f"  • {reason}")
        else:
            lines.append(
                "You have the right to request the specific reasons for this action "
                "within 60 days of receiving this notice."
            )

        if notice.supplemental_narrative:
            lines.append("")
            lines.append("Additional Information:")
            lines.append(notice.supplemental_narrative)

        if notice.fcra_disclosure:
            lines.append("")
            lines.append("Your Rights Under the Fair Credit Reporting Act:")
            lines.append(notice.fcra_disclosure)

        lines.append("")
        lines.append("Equal Credit Opportunity Act Notice:")
        lines.append(notice.ecoa_notice)
        lines.append("")
        lines.append("Federal Oversight Agency:")
        lines.append(notice.federal_agency)

        return "\n".join(lines)
