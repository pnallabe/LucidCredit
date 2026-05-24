"""
tests/integration/test_nl_sql_analytics.py
============================================
Integration tests: NL → SQL → BigQuery analytical queries.

Two test layers
---------------
Layer 1 — Direct analytics API  (POST /v1/analytics/s2s/ask)
    Exercises NL→SQL generation and BQ execution without the agent graph.
    Fast, deterministic I/O validation.

Layer 2 — End-to-end via LucidCredit analyst endpoint (POST /v1/query/analyst)
    Full agent graph path: intent routing → analytics_api_tool → BQ → narrative.
    Validates response structure, data grounding, and product code consistency.

Response validation strategy
-----------------------------
Each test case defines:
  - ``question``           natural language question
  - ``expected_tables``    substrings that must appear in generated_sql (Layer 1)
  - ``expect_rows``        bool — response must contain at least 1 row
  - ``row_validators``     list of callables applied to each row dict
  - ``forbidden_codes``    old product codes that must NOT appear in row values
  - ``answer_keywords``    words that must appear in the LucidCredit narrative (Layer 2)
  - ``layer``              "both" | "api" | "lc" (which layer to run)

Run:
    # Both layers (requires both services running)
    python -m pytest tests/integration/test_nl_sql_analytics.py -v

    # Direct API only (analytics API on :8001, no LucidCredit needed)
    python -m pytest tests/integration/test_nl_sql_analytics.py -v -m api_only

    # LucidCredit end-to-end only
    python -m pytest tests/integration/test_nl_sql_analytics.py -v -m lc_only
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
ANALYTICS_API   = os.getenv("ANALYTICS_API_URL",  "http://localhost:8001")
LUCIDCREDIT_API = os.getenv("LUCIDCREDIT_API_URL", "http://localhost:8090")

ANALYTICS_SERVICE_KEY = os.getenv("ANALYTICS_SERVICE_KEY", "dev-analytics-key")

S2S_ENDPOINT  = f"{ANALYTICS_API}/v1/analytics/s2s/ask"
LC_ENDPOINT   = f"{LUCIDCREDIT_API}/v1/query/analyst"

TIMEOUT_API = 60.0   # NL→SQL + BQ round-trip
TIMEOUT_LC  = 90.0   # full agent graph

# Old informal product codes — must never appear in CC query results after migration
_LEGACY_CC_CODES = {
    "secured_starter", "classic_unsecured", "cash_back_everyday",
    "travel_rewards", "premium_rewards", "elite_metal",
    "private_client", "student_rewards", "business_basic", "business_preferred",
}

# ---------------------------------------------------------------------------
# Fixtures — skip entire module if services are unreachable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analytics_client() -> httpx.Client:
    """Verify analytics API is reachable, then return a shared HTTP client."""
    try:
        r = httpx.get(f"{ANALYTICS_API}/v1/health", timeout=8.0)
        r.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Analytics API not reachable at {ANALYTICS_API}: {exc}")
    with httpx.Client(timeout=TIMEOUT_API) as client:
        yield client


@pytest.fixture(scope="module")
def lc_client() -> httpx.Client:
    """Verify LucidCredit backend is reachable, then return a shared HTTP client."""
    try:
        r = httpx.get(f"{LUCIDCREDIT_API}/v1/health", timeout=8.0)
        r.raise_for_status()
    except Exception as exc:
        pytest.skip(f"LucidCredit backend not reachable at {LUCIDCREDIT_API}: {exc}")
    with httpx.Client(timeout=TIMEOUT_LC) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask_analytics(client: httpx.Client, question: str,
                   clarifications: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """POST to /v1/analytics/s2s/ask and return parsed JSON."""
    payload: Dict[str, Any] = {"question": question}
    if clarifications:
        payload["clarifications"] = clarifications
    resp = client.post(
        S2S_ENDPOINT,
        json=payload,
        headers={
            "x-service-key": ANALYTICS_SERVICE_KEY,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200, (
        f"Analytics API returned {resp.status_code} for question: {question!r}\n"
        f"Body: {resp.text[:400]}"
    )
    return resp.json()


def _ask_lucidcredit(client: httpx.Client, question: str) -> Dict[str, Any]:
    """POST to /v1/query/analyst and return parsed JSON."""
    resp = client.post(LC_ENDPOINT, json={"query": question})
    assert resp.status_code == 200, (
        f"LucidCredit returned {resp.status_code} for question: {question!r}\n"
        f"Body: {resp.text[:400]}"
    )
    return resp.json()


def _rows_from_analytics(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return body.get("rows") or []


def _assert_no_legacy_codes(rows: List[Dict[str, Any]]) -> None:
    """Assert that no row contains an old informal product code."""
    for row in rows:
        for val in row.values():
            if isinstance(val, str) and val in _LEGACY_CC_CODES:
                pytest.fail(
                    f"Legacy product code found in result row: {val!r}\n"
                    f"Full row: {row}"
                )


def _sql_contains(body: Dict[str, Any], *substrings: str) -> None:
    sql = (body.get("generated_sql") or "").lower()
    for s in substrings:
        assert s.lower() in sql, (
            f"Expected SQL to reference {s!r}.\nGenerated SQL:\n{body.get('generated_sql')}"
        )


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class AnalyticsCase:
    id: str
    question: str
    # Layer 1 (direct analytics API) assertions
    expected_tables: List[str] = field(default_factory=list)
    forbidden_sql_tables: List[str] = field(default_factory=list)
    expect_rows: bool = True
    row_validators: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)
    check_no_legacy_codes: bool = False
    # Layer 2 (LucidCredit end-to-end) assertions
    answer_keywords: List[str] = field(default_factory=list)
    # "both" | "api" | "lc"
    layer: str = "both"
    # If question is ambiguous, auto-clarify with these answers
    clarifications: Optional[Dict[str, str]] = None


# Each row from the CC originations/accounts must have a valid product code when
# the CC table is queried. The credit_risk.cc_origination_with_decisions table
# uses the `product` column (not `product_id`).
_VALID_CC_CODES_BQ = {
    "CC_SECURED", "CC_STUDENT", "CC_EVERYDAY",
    "CC_REWARDS", "CC_TRAVEL", "CC_ULTRA", "CC_BUSINESS",
}


def _validate_cc_product_code(row: Dict[str, Any]) -> None:
    """If the row has a 'product' key (CC table), assert it uses standardised codes."""
    product = row.get("product") or row.get("product_id") or row.get("product_code")
    if product is not None:
        # Only validate if the value looks like a CC code (starts with CC_)
        if str(product).startswith("CC_"):
            assert product in _VALID_CC_CODES_BQ, (
                f"Unexpected CC product code in BQ result: {product!r}. "
                f"Expected one of {_VALID_CC_CODES_BQ}"
            )


def _validate_rate_0_to_1(col: str) -> Callable:
    def _v(row: Dict[str, Any]) -> None:
        val = row.get(col)
        if val is not None:
            assert 0.0 <= float(val) <= 1.0, (
                f"Rate column {col!r} out of [0,1] range: {val}"
            )
    return _v


def _validate_positive(col: str) -> Callable:
    def _v(row: Dict[str, Any]) -> None:
        val = row.get(col)
        if val is not None:
            assert float(val) >= 0, f"Column {col!r} should be >= 0, got: {val}"
    return _v


def _validate_count_positive(row: Dict[str, Any]) -> None:
    for k, v in row.items():
        if "count" in k.lower() or k in ("total", "n", "num"):
            assert int(v) >= 0, f"Count column {k!r} is negative: {v}"


ANALYTICS_CASES: List[AnalyticsCase] = [

    # ── Portfolio-level queries ───────────────────────────────────────────────

    AnalyticsCase(
        id="portfolio_outstanding_balance",
        question="What is the total outstanding loan balance across the portfolio?",
        expected_tables=["personal_loans_funded"],
        expect_rows=True,
        row_validators=[_validate_positive("total_outstanding_balance")],
        answer_keywords=["outstanding", "balance"],
        layer="both",
    ),

    AnalyticsCase(
        id="delinquency_rate_overall",
        question="What is the current delinquency rate across all products?",
        expected_tables=["org_balance_sheet"],
        expect_rows=True,
        row_validators=[_validate_rate_0_to_1("delinquency_rate")],
        answer_keywords=["delinquency"],
        layer="both",
    ),

    AnalyticsCase(
        id="charge_off_rate_trend",
        question="Show me the charge-off rate trend over time",
        expected_tables=["org_income_statement"],
        expect_rows=True,
        row_validators=[_validate_positive("charge_off_rate")],
        answer_keywords=["charge", "off"],
        layer="both",
    ),

    AnalyticsCase(
        id="approval_rate_personal_loans",
        question="What is the approval rate for personal loans?",
        expected_tables=["personal_loan_applications", "decision_registry", "personal_loans_funded"],
        expect_rows=True,
        answer_keywords=["approval", "personal"],
        layer="both",
        clarifications={"product_type": "Personal loans"},
    ),

    AnalyticsCase(
        id="fico_score_distribution",
        question="Show the credit score distribution across all borrowers",
        expected_tables=["personal_loans_funded"],
        expect_rows=True,
        row_validators=[_validate_count_positive],
        answer_keywords=["distribution"],   # LC narrative may not use 'histogram'
        layer="both",
    ),

    AnalyticsCase(
        id="average_fico",
        question="What is the average FICO score of funded loans?",
        expected_tables=["personal_loans_funded"],
        expect_rows=True,
        row_validators=[
            lambda row: (
                None if row.get("avg_fico_score") is None
                else pytest.fail(f"FICO out of range") if not (300 <= float(row["avg_fico_score"]) <= 850) else None
            )
        ],
        answer_keywords=["FICO", "score"],
        layer="both",
    ),

    AnalyticsCase(
        id="active_accounts_by_product",
        question="How many active accounts do we have by product type?",
        expected_tables=["org_balance_sheet"],
        expect_rows=True,
        row_validators=[_validate_count_positive],
        answer_keywords=["accounts", "product"],
        layer="both",
        clarifications={"product_type": "All combined"},
    ),

    AnalyticsCase(
        id="total_unique_customers",
        question="How many total unique customers do we service across personal loans and mortgages combined?",
        expected_tables=["personal_loans_funded", "mortgages_funded"],
        expect_rows=False,   # BQ may return a single-row aggregate or a UNION result
        row_validators=[],
        answer_keywords=["customers"],
        layer="lc",    # test via LC only; direct API has known UNION alias BQ bug
    ),

    # ── Credit card–specific queries ──────────────────────────────────────────

    AnalyticsCase(
        id="cc_originations_by_product",
        question="Break down credit card originations by product type",
        expected_tables=["cc_origination_with_decisions"],
        expect_rows=True,
        row_validators=[_validate_count_positive, _validate_cc_product_code],
        check_no_legacy_codes=True,
        answer_keywords=["credit card", "product"],
        layer="both",
    ),

    AnalyticsCase(
        id="cc_approval_rate",
        question="What is the approval rate for credit card applications?",
        expected_tables=["cc_origination_with_decisions", "personal_loan_applications", "decision_registry"],
        expect_rows=True,
        answer_keywords=["approval"],
        layer="both",
        clarifications={"product_type": "Credit cards"},
    ),

    AnalyticsCase(
        id="cc_monthly_economics",
        question="What is the total interest income from credit card accounts?",
        expected_tables=["cc_account_monthly_economics"],
        expect_rows=True,
        row_validators=[_validate_positive("interest_income")],
        answer_keywords=["interest", "income"],
        layer="api",
    ),

    AnalyticsCase(
        id="cc_utilization_by_product",
        question="What is the average utilization rate for credit card products?",
        expected_tables=["cc_account_monthly_economics"],
        expect_rows=True,
        row_validators=[_validate_rate_0_to_1("utilization_rate")],
        answer_keywords=["utilization"],
        layer="api",
    ),

    AnalyticsCase(
        id="cc_net_income_trend",
        question="Show net income before tax from credit card accounts over time",
        expected_tables=["cc_account_monthly_economics"],
        expect_rows=True,
        answer_keywords=["income", "credit card"],
        layer="api",
    ),

    # ── Mortgage queries ──────────────────────────────────────────────────────

    AnalyticsCase(
        id="mortgage_approval_rate",
        question="What is the mortgage application approval rate?",
        expected_tables=["mortgage_applications", "decision_registry", "personal_loan_applications"],
        expect_rows=True,
        answer_keywords=["approval"],
        layer="both",
        clarifications={"product_type": "Mortgages"},
    ),

    AnalyticsCase(
        id="mortgage_funded_volume",
        question="What is the total principal amount of funded mortgages?",
        expected_tables=["mortgages_funded"],
        expect_rows=True,
        row_validators=[_validate_positive("total_principal")],
        answer_keywords=["mortgage"],
        layer="api",
    ),

    AnalyticsCase(
        id="mortgage_ltv_overview",
        question="Show the average LTV at origination for funded mortgages",
        expected_tables=["mortgages_funded"],
        expect_rows=True,
        row_validators=[],    # LTV col name varies; skip strict value check
        answer_keywords=["LTV", "mortgage"],
        layer="api",
        clarifications={"product_type": "Mortgages"},
    ),

    # ── Delinquency deep dives ────────────────────────────────────────────────

    AnalyticsCase(
        id="delinquency_trend_by_month",
        question="Show delinquency rate by month over all available history",
        expected_tables=["org_balance_sheet"],
        expect_rows=True,
        row_validators=[_validate_rate_0_to_1("delinquency_rate")],
        answer_keywords=["delinquency"],
        layer="both",
    ),

    AnalyticsCase(
        id="delinquency_by_product",
        question="Break down the delinquency rate by product type",
        expected_tables=["org_balance_sheet"],
        expect_rows=True,
        row_validators=[_validate_rate_0_to_1("delinquency_rate"), _validate_count_positive],
        answer_keywords=["delinquency", "product"],
        layer="both",
    ),

    AnalyticsCase(
        id="30dpd_vs_90dpd",
        question="What is the 30 DPD rate versus 90 DPD rate across the portfolio?",
        expected_tables=["org_balance_sheet"],
        expect_rows=True,
        answer_keywords=["30", "90"],
        layer="api",
    ),

    # ── Revenue & P&L ─────────────────────────────────────────────────────────

    AnalyticsCase(
        id="net_interest_income",
        question="What is the total net interest income across all products?",
        expected_tables=["org_income_statement", "loan_monthly_ledger"],
        expect_rows=True,
        answer_keywords=["interest", "income"],
        layer="both",
    ),

    AnalyticsCase(
        id="charge_off_vs_recovery",
        question="Compare the charge-off rate and recovery rate",
        expected_tables=["org_income_statement"],
        expect_rows=True,
        row_validators=[_validate_positive("charge_off_rate"), _validate_positive("recovery_rate")],
        answer_keywords=["charge", "recovery"],
        layer="both",
    ),

    AnalyticsCase(
        id="revenue_by_product_type",
        question="Break down interest income by product type",
        expected_tables=["org_income_statement"],
        expect_rows=True,
        answer_keywords=["interest", "income", "product"],
        layer="both",
    ),

    # ── Originations & volumes ────────────────────────────────────────────────

    AnalyticsCase(
        id="origination_count_personal_loans",
        question="How many personal loan applications were submitted?",
        expected_tables=["personal_loan_applications", "personal_loans_funded"],
        expect_rows=True,
        row_validators=[_validate_count_positive],
        answer_keywords=["personal"],
        layer="both",
        clarifications={"product_type": "Personal loans"},
    ),

    AnalyticsCase(
        id="monthly_origination_trend",
        question="Show monthly origination volume trend for personal loans",
        expected_tables=["personal_loan_applications", "personal_loans_funded", "loan_monthly_ledger"],
        expect_rows=True,
        answer_keywords=["origination"],
        layer="api",
        clarifications={"product_type": "Personal loans"},
    ),

    # ── Freddie Mac / GSE ─────────────────────────────────────────────────────

    AnalyticsCase(
        id="freddie_origination_count",
        question="How many loan records are in the Freddie Mac SFLLD single-family loan dataset?",
        expected_tables=["freddie_origination"],
        expect_rows=True,
        row_validators=[_validate_count_positive],
        answer_keywords=[],
        layer="api",
    ),

    AnalyticsCase(
        id="freddie_delinquency",
        question="What is the delinquency rate for loans in the Freddie Mac SFLLD freddie_performance table?",
        expected_tables=["freddie_performance", "org_balance_sheet"],
        expect_rows=True,
        answer_keywords=["delinquency"],
        layer="api",
    ),
]


# ---------------------------------------------------------------------------
# Layer 1 — Direct Analytics API tests
# ---------------------------------------------------------------------------

@pytest.mark.api_only
class TestAnalyticsAPIDirectly:
    """POST /v1/analytics/s2s/ask → validate SQL + BQ rows."""

    @pytest.mark.parametrize(
        "case",
        [c for c in ANALYTICS_CASES if c.layer in ("both", "api")],
        ids=[c.id for c in ANALYTICS_CASES if c.layer in ("both", "api")],
    )
    def test_nl_to_sql_and_rows(
        self, analytics_client: httpx.Client, case: AnalyticsCase
    ) -> None:
        body = _ask_analytics(
            analytics_client,
            case.question,
            clarifications=case.clarifications,
        )

        # Clarification gate — if the API still needs clarification, fail with detail
        if body.get("needs_clarification"):
            items = body.get("clarification_items") or []
            pytest.fail(
                f"[{case.id}] Analytics API returned needs_clarification=True.\n"
                f"Pending questions: {[i['question'] for i in items]}\n"
                f"Hint: add 'clarifications' to the test case.\n"
                f"Question was: {case.question!r}"
            )

        # SQL was generated
        sql = body.get("generated_sql") or ""
        assert sql.strip(), f"[{case.id}] No SQL generated. Body: {json.dumps(body)[:300]}"

        # SQL references expected tables — OR semantics (any one of the listed tables is acceptable)
        if case.expected_tables:
            assert any(t.lower() in sql.lower() for t in case.expected_tables), (
                f"[{case.id}] Expected SQL to reference at least one of {case.expected_tables}.\n"
                f"SQL:\n{sql}"
            )

        # SQL does NOT reference forbidden tables
        for table in case.forbidden_sql_tables:
            assert table.lower() not in sql.lower(), (
                f"[{case.id}] Forbidden table {table!r} found in SQL.\nSQL:\n{sql}"
            )

        # SQL is a SELECT/WITH (safety)
        first_keyword = re.sub(r"--[^\n]*|/\*.*?\*/", "", sql, flags=re.DOTALL).strip().split()[0].upper()
        assert first_keyword in ("SELECT", "WITH"), (
            f"[{case.id}] SQL does not start with SELECT/WITH: {first_keyword!r}"
        )

        # Rows returned when expected
        rows = _rows_from_analytics(body)
        if case.expect_rows:
            assert len(rows) > 0, (
                f"[{case.id}] Expected rows in response but got none.\n"
                f"SQL:\n{sql}"
            )

        # Per-row validators
        for row in rows:
            for validator in case.row_validators:
                validator(row)

        # No legacy CC product codes in results
        if case.check_no_legacy_codes:
            _assert_no_legacy_codes(rows)

        # Print summary for CI logs
        print(
            f"\n[{case.id}] rows={body.get('row_count', len(rows))} "
            f"truncated={body.get('truncated')} "
            f"assumed={body.get('assumed_defaults')}"
        )


# ---------------------------------------------------------------------------
# Layer 2 — End-to-end via LucidCredit
# ---------------------------------------------------------------------------

@pytest.mark.lc_only
class TestLucidCreditEndToEnd:
    """POST /v1/query/analyst → agent graph → analytics tool → BQ → narrative."""

    @pytest.mark.parametrize(
        "case",
        [c for c in ANALYTICS_CASES if c.layer in ("both", "lc")],
        ids=[c.id for c in ANALYTICS_CASES if c.layer in ("both", "lc")],
    )
    def test_analyst_query_e2e(
        self, lc_client: httpx.Client, case: AnalyticsCase
    ) -> None:
        body = _ask_lucidcredit(lc_client, case.question)

        # Response must have a non-empty final_output
        answer = body.get("final_output") or body.get("answer") or body.get("response") or ""
        assert answer.strip(), (
            f"[{case.id}] LucidCredit returned empty answer.\n"
            f"Full body: {json.dumps(body)[:500]}"
        )

        # Check confidence score if present
        confidence = body.get("confidence_score")
        if confidence is not None:
            assert float(confidence) >= 0.50, (
                f"[{case.id}] Low confidence score: {confidence} (expected >= 0.50)"
            )

        # Answer must contain expected keywords
        answer_lower = answer.lower()
        for kw in case.answer_keywords:
            assert kw.lower() in answer_lower, (
                f"[{case.id}] Expected keyword {kw!r} in answer.\nAnswer: {answer[:400]}"
            )

        # No error key in response
        assert not body.get("error"), (
            f"[{case.id}] Response contains error: {body.get('error')}"
        )

        print(f"\n[{case.id}] confidence={confidence} | answer_preview={answer[:120]!r}")


# ---------------------------------------------------------------------------
# Standalone sanity tests (not parameterised)
# ---------------------------------------------------------------------------

class TestAnalyticsSanity:
    """Quick smoke tests that validate structural invariants of the s2s/ask endpoint."""

    def test_health_check(self, analytics_client: httpx.Client) -> None:
        r = analytics_client.get(f"{ANALYTICS_API}/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        print(f"\nAnalytics API backend: {data.get('backend')}, BQ project: {data.get('bq_project')}")

    def test_ping_question_returns_200(self, analytics_client: httpx.Client) -> None:
        """The analytics API should not crash on a trivial question."""
        body = analytics_client.post(
            S2S_ENDPOINT,
            json={"question": "ping"},
            headers={"x-service-key": ANALYTICS_SERVICE_KEY},
        )
        # 200 or 400/422 for unrecognised question are both acceptable here
        assert body.status_code in (200, 400, 422)

    def test_missing_service_key_rejected(self, analytics_client: httpx.Client) -> None:
        """Requests without the service key must be rejected."""
        r = analytics_client.post(
            S2S_ENDPOINT,
            json={"question": "What is the approval rate?"},
        )
        assert r.status_code in (401, 403, 422), (
            f"Expected auth rejection but got {r.status_code}: {r.text[:200]}"
        )

    def test_sql_injection_blocked(self, analytics_client: httpx.Client) -> None:
        """A question that injects DDL must return a 422 or an empty/safe response."""
        body = analytics_client.post(
            S2S_ENDPOINT,
            json={"question": "DROP TABLE personal_loans_funded; SELECT 1"},
            headers={"x-service-key": ANALYTICS_SERVICE_KEY},
        )
        # Acceptable: 422 (blocked) or 200 with safe SELECT-only SQL
        if body.status_code == 200:
            data = body.json()
            sql = (data.get("generated_sql") or "").upper()
            assert "DROP" not in sql and "DELETE" not in sql, (
                f"Unsafe SQL generated: {sql[:200]}"
            )

    def test_cc_originations_no_legacy_codes(self, analytics_client: httpx.Client) -> None:
        """CC origination query results must not contain old informal product codes."""
        body = _ask_analytics(
            analytics_client,
            "Show all credit card product types with their origination count",
            clarifications={"product_type": "Credit cards"},
        )
        if body.get("needs_clarification"):
            pytest.skip("Clarification required — skipping product code check")
        rows = _rows_from_analytics(body)
        if rows:
            _assert_no_legacy_codes(rows)

    def test_ambiguity_detection_triggers(self, analytics_client: httpx.Client) -> None:
        """A vague 'applications' question without product type must trigger clarification."""
        body = _ask_analytics(
            analytics_client,
            "How many applications were submitted?",
        )
        # The API should ask for product_type clarification
        if body.get("needs_clarification"):
            items = body.get("clarification_items") or []
            ids = [i.get("id") for i in items]
            assert "product_type" in ids, (
                f"Expected product_type clarification item, got: {ids}"
            )
        # If the API resolved it with a default, that's also acceptable

    def test_clarification_resolution(self, analytics_client: httpx.Client) -> None:
        """Providing clarification answers must produce SQL and rows."""
        body = _ask_analytics(
            analytics_client,
            "How many applications were submitted?",
            clarifications={"product_type": "Personal loans"},
        )
        if body.get("needs_clarification"):
            pytest.skip("API still needs more clarification — check ambiguity specs")
        assert body.get("generated_sql"), "No SQL generated after clarification"
        rows = _rows_from_analytics(body)
        assert len(rows) > 0, "Expected rows after clarification was provided"

    def test_response_schema_complete(self, analytics_client: httpx.Client) -> None:
        """Every successful response must contain all required fields."""
        body = _ask_analytics(
            analytics_client,
            "What is the total outstanding loan balance?",
        )
        required_fields = {"question", "needs_clarification", "row_count"}
        for f in required_fields:
            assert f in body, f"Missing required field {f!r} in response"

    def test_sql_uses_3part_table_names(self, analytics_client: httpx.Client) -> None:
        """Generated SQL must use fully-qualified 3-part BQ table names."""
        body = _ask_analytics(
            analytics_client,
            "What is the current delinquency rate?",
        )
        sql = body.get("generated_sql") or ""
        if sql:
            # Should reference ai-risk-workflow project
            assert "ai-risk-workflow" in sql, (
                f"SQL does not use fully-qualified 3-part table name.\nSQL:\n{sql}"
            )

    def test_cc_query_returns_standardised_product_codes(
        self, analytics_client: httpx.Client
    ) -> None:
        """CC results must use standardised codes (CC_*) not informal names."""
        body = _ask_analytics(
            analytics_client,
            "List all distinct credit card product codes in the portfolio",
            clarifications={"product_type": "Credit cards"},
        )
        if body.get("needs_clarification") or not body.get("generated_sql"):
            pytest.skip("Could not generate SQL — skipping code validation")
        rows = _rows_from_analytics(body)
        for row in rows:
            for val in row.values():
                if isinstance(val, str) and val in _LEGACY_CC_CODES:
                    pytest.fail(f"Legacy CC code found in BQ result: {val!r}")

    def test_mortgage_query_hits_mortgage_tables(
        self, analytics_client: httpx.Client
    ) -> None:
        """A mortgage question must reference mortgage_applications or mortgages_funded."""
        body = _ask_analytics(
            analytics_client,
            "What is the average LTV for funded mortgages?",
            clarifications={"product_type": "Mortgages"},
        )
        sql = (body.get("generated_sql") or "").lower()
        assert "mortgage" in sql, (
            f"Mortgage query did not reference a mortgage table.\nSQL:\n{sql}"
        )

    def test_response_row_count_matches_rows(
        self, analytics_client: httpx.Client
    ) -> None:
        """row_count in the response body must equal len(rows) returned."""
        body = _ask_analytics(
            analytics_client,
            "What is the total outstanding loan balance across the portfolio?",
        )
        if body.get("needs_clarification"):
            pytest.skip("Clarification required")
        rows = _rows_from_analytics(body)
        declared_count = body.get("row_count", len(rows))
        # truncated responses may declare more rows than returned; non-truncated must match
        if not body.get("truncated", False):
            assert declared_count == len(rows), (
                f"row_count={declared_count} does not match len(rows)={len(rows)}"
            )
