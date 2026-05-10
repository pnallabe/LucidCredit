"""
LucidCredit Chatbot Chain — Comprehensive Test Suite
=====================================================
Tests all 11 evaluation categories:
  1.  Data Retrieval Accuracy
  2.  Aggregations & Metrics Logic
  3.  Time Series & Trend Analysis
  4.  Visualization Requests
  5.  Analytical Reasoning
  6.  Root Cause Analysis
  7.  Prescriptive Actions
  8.  Decision Summaries (Exec-Level)
  9.  Hallucination & Robustness Tests
  10. Multi-Step / Agentic Workflows
  11. Data Quality & Governance Awareness

Run:
    python tests/chatbot_chain_test.py
"""

from __future__ import annotations

import httpx
import json
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 120  # seconds per query (LLM + BQ round-trip)

# ──────────────────────────────────────────────────────────────────────────────
# Test definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    category: str
    id: str
    question: str
    # Expected behaviours for grading
    must_not_hallucinate: bool = False
    expect_refusal: bool = False          # should say "data not available"
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    note: str = ""


TEST_CASES: list[TestCase] = [

    # ──────────────────────────────────────────────────────────────────────────
    # Category 1 — Data Retrieval Accuracy
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="1-Data Retrieval",
        id="1-01",
        question="What is the total outstanding balance of the portfolio?",
        expected_keywords=["balance", "total", "$", "outstanding"],
        note="Basic portfolio total — expect a dollar figure",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-02",
        question="How many active accounts do we have today?",
        expected_keywords=["account", "active"],
        note="Headcount query — expect a numeric count",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-03",
        question="What is the average credit score across all borrowers?",
        expected_keywords=["credit score", "average", "fico"],
        note="Scalar aggregation — expect a FICO number ~300-850",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-04",
        question="What is the total exposure for accounts opened after January 2023?",
        expected_keywords=["2023", "exposure", "total"],
        note="Filtered by origination date",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-05",
        question="What is the delinquency rate for customers in Texas?",
        expected_keywords=["texas", "delinquency", "rate", "%"],
        note="Geographic filter — watch for silent state assumption",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-06",
        question="What is the average credit utilization for borrowers with FICO below 620?",
        expected_keywords=["utilization", "fico", "620"],
        note="Compound filter: FICO segment + utilization metric",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-07",
        question="How many accounts have missing income values?",
        expected_keywords=["missing", "income", "null"],
        note="Null / data quality scan",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-08",
        question="Show accounts with negative balances or invalid dates.",
        expected_keywords=["negative", "balance", "invalid", "date"],
        note="Edge-case anomaly detection",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-09",
        question="What percentage of accounts have no payment recorded in the last 90 days?",
        expected_keywords=["payment", "90 days", "%"],
        note="Time-windowed absence check",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-10",
        question="What is the average APR by borrower employment type?",
        expected_keywords=["apr", "employment"],
        note="Join: loan table + borrower employment attribute",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-11",
        question="What is the default rate by loan origination channel?",
        expected_keywords=["default", "channel", "origination"],
        note="Join: loan table + origination channel",
    ),
    TestCase(
        category="1-Data Retrieval",
        id="1-12",
        question="What are the top 5 industries by total exposure and their associated delinquency rate?",
        expected_keywords=["industry", "exposure", "delinquency"],
        note="Multi-table join + rank + cross-metric — complex",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 2 — Aggregations & Metrics Logic
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="2-Aggregations",
        id="2-01",
        question="What is the 30+ DPD rate across the portfolio?",
        expected_keywords=["30", "dpd", "delinquency", "rate", "%"],
        note="Standard DPD bucket definition",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-02",
        question="Calculate the net charge-off rate over the last 12 months.",
        expected_keywords=["charge-off", "net", "12 months", "rate"],
        note="NCO formula — watch for wrong denominator",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-03",
        question="What is the weighted average APR of the portfolio, weighted by outstanding balance?",
        expected_keywords=["weighted", "apr", "balance"],
        note="Weighted average — easy to get denominator wrong",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-04",
        question="Compare mean vs median loan size and explain what the difference indicates.",
        expected_keywords=["mean", "median", "skew", "distribution"],
        note="Distributional reasoning — expect skewness interpretation",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-05",
        question="Define delinquency rate and then calculate it for this portfolio.",
        expected_keywords=["definition", "delinquency", "past due", "%"],
        note="Definition + calculation consistency check",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-06",
        question="What is the difference between gross and net charge-offs in this dataset?",
        expected_keywords=["gross", "net", "recovery", "charge-off"],
        note="Concept distinction — should reference actual data",
    ),
    TestCase(
        category="2-Aggregations",
        id="2-07",
        question="Recalculate the default rate excluding accounts less than 6 months old.",
        expected_keywords=["default", "rate", "6 months", "exclude", "seasoning"],
        note="Seasoning filter — watch for time window error",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 3 — Time Series & Trend Analysis
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="3-Time Series",
        id="3-01",
        question="Show the monthly delinquency trend for the last 24 months.",
        expected_keywords=["monthly", "delinquency", "trend", "2024", "2025"],
        note="24-month lookback — dates must be correct",
    ),
    TestCase(
        category="3-Time Series",
        id="3-02",
        question="When did charge-offs peak over the last two years?",
        expected_keywords=["peak", "charge-off", "month", "year"],
        note="Temporal max detection",
    ),
    TestCase(
        category="3-Time Series",
        id="3-03",
        question="Compare pre-Q3 2023 vs post-Q3 2023 portfolio performance.",
        expected_keywords=["q3 2023", "before", "after", "performance"],
        note="Structural period comparison",
    ),
    TestCase(
        category="3-Time Series",
        id="3-04",
        question="Is the delinquency trend increasing significantly or is it just seasonal variation?",
        expected_keywords=["seasonal", "trend", "significant"],
        note="Trend vs seasonality reasoning",
    ),
    TestCase(
        category="3-Time Series",
        id="3-05",
        question="What is the compound annual growth rate (CAGR) of the portfolio outstanding balance?",
        expected_keywords=["cagr", "growth", "annual", "%"],
        note="CAGR formula correctness check",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 4 — Visualization Requests
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="4-Visualization",
        id="4-01",
        question="Plot the monthly delinquency rate for the past 12 months as a line chart.",
        expected_keywords=["chart", "plot", "delinquency", "monthly"],
        note="Line chart — should describe axes correctly",
    ),
    TestCase(
        category="4-Visualization",
        id="4-02",
        question="Show a bar chart of total exposure by risk grade.",
        expected_keywords=["bar chart", "exposure", "risk grade"],
        note="Categorical axis — grades must be ordered",
    ),
    TestCase(
        category="4-Visualization",
        id="4-03",
        question="Create a histogram of credit scores across all borrowers.",
        expected_keywords=["histogram", "credit score", "distribution"],
        note="Distribution plot — watch for mis-bucketing",
    ),
    TestCase(
        category="4-Visualization",
        id="4-04",
        question="Plot default rate vs credit score buckets to show the relationship.",
        expected_keywords=["default", "credit score", "bucket"],
        note="Multi-dimension plot — axes must match data",
    ),
    TestCase(
        category="4-Visualization",
        id="4-05",
        question="Show a segmented trend chart of delinquency rate by product type over the last 12 months.",
        expected_keywords=["product", "segment", "delinquency", "trend"],
        note="Segmented series — watch for missing product categories",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 5 — Analytical Reasoning
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="5-Reasoning",
        id="5-01",
        question="Why is delinquency typically higher in subprime segments?",
        expected_keywords=["subprime", "credit risk", "income", "fico"],
        note="Domain reasoning — should be data-grounded",
    ),
    TestCase(
        category="5-Reasoning",
        id="5-02",
        question="What factors in this dataset are most correlated with default?",
        expected_keywords=["correlated", "default", "factor"],
        note="Correlation analysis — must reference actual fields",
    ),
    TestCase(
        category="5-Reasoning",
        id="5-03",
        question="Explain why high credit utilization increases default risk.",
        expected_keywords=["utilization", "risk", "capacity", "debt"],
        note="Causal chain reasoning",
    ),
    TestCase(
        category="5-Reasoning",
        id="5-04",
        question="Does higher income reduce default risk in this dataset? Show the data.",
        expected_keywords=["income", "default", "data"],
        forbidden_keywords=["definitely", "absolutely"],
        note="Causal trap — should distinguish correlation from causation",
    ),
    TestCase(
        category="5-Reasoning",
        id="5-05",
        question="Is age a strong predictor of delinquency in this portfolio?",
        expected_keywords=["age", "delinquency"],
        note="Causal trap — should caveat confounders",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 6 — Root Cause Analysis
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="6-Root Cause",
        id="6-01",
        question="Delinquency increased 2% last quarter. Break down the drivers by segment.",
        expected_keywords=["segment", "driver", "contribution", "delinquency"],
        note="Decomposition — must break down, not give generic answer",
    ),
    TestCase(
        category="6-Root Cause",
        id="6-02",
        question="Which segments contributed most to total losses last year?",
        expected_keywords=["segment", "loss", "contribution"],
        note="Attribution analysis",
    ),
    TestCase(
        category="6-Root Cause",
        id="6-03",
        question="Decompose the change in default rate by: credit score mix, loan size, and geography.",
        expected_keywords=["credit score", "loan size", "geography", "decompose", "change"],
        note="Multi-factor decomposition — most complex root cause test",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 7 — Prescriptive Actions
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="7-Prescriptive",
        id="7-01",
        question="What actions should we take to reduce the delinquency rate?",
        expected_keywords=["action", "recommend", "reduce", "delinquency"],
        forbidden_keywords=["generic", "always", "in general"],
        note="Should be data-linked recommendations, not generic advice",
    ),
    TestCase(
        category="7-Prescriptive",
        id="7-02",
        question="How should we adjust underwriting criteria based on current portfolio performance?",
        expected_keywords=["underwriting", "criteria", "adjust"],
        note="Specific policy recommendation tied to data",
    ),
    TestCase(
        category="7-Prescriptive",
        id="7-03",
        question="Suggest credit line management strategies for high-utilization borrowers.",
        expected_keywords=["credit line", "utilization", "strategy"],
        note="Product-specific recommendation",
    ),
    TestCase(
        category="7-Prescriptive",
        id="7-04",
        question="How can we reduce the default rate by 1 percentage point without shrinking portfolio size?",
        expected_keywords=["default", "trade-off", "1%", "portfolio"],
        note="Constraint-based optimization — should discuss trade-offs",
    ),
    TestCase(
        category="7-Prescriptive",
        id="7-05",
        question="How can we improve portfolio profitability while keeping risk constant?",
        expected_keywords=["profitability", "risk", "balance", "strategy"],
        note="Risk-return trade-off discussion",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 8 — Decision Summaries (Exec-Level)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="8-Exec Summary",
        id="8-01",
        question="Summarize portfolio health in 5 bullet points.",
        expected_keywords=["bullet", "portfolio", "health"],
        note="CRO-level brevity and clarity",
    ),
    TestCase(
        category="8-Exec Summary",
        id="8-02",
        question="Prepare an executive summary for the risk committee meeting.",
        expected_keywords=["executive", "risk", "committee", "summary"],
        note="Structured, prioritized, signal-focused",
    ),
    TestCase(
        category="8-Exec Summary",
        id="8-03",
        question="What are the top 3 risks in the portfolio right now?",
        expected_keywords=["risk", "top", "priority"],
        note="Ranked, specific, data-backed",
    ),
    TestCase(
        category="8-Exec Summary",
        id="8-04",
        question="Should we tighten credit policy right now? Support your recommendation with data.",
        expected_keywords=["credit policy", "tighten", "data", "recommend"],
        note="Board-level decision framing",
    ),
    TestCase(
        category="8-Exec Summary",
        id="8-05",
        question="Is the portfolio trending toward instability?",
        expected_keywords=["trend", "instability", "risk"],
        note="Directional assessment — should avoid alarmism without evidence",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 9 — Hallucination & Robustness Tests
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="9-Hallucination",
        id="9-01",
        question="What is the prepayment rate of the portfolio?",
        must_not_hallucinate=True,
        expect_refusal=True,
        expected_keywords=["not available", "data", "prepayment"],
        note="Field likely absent — must refuse, not fabricate",
    ),
    TestCase(
        category="9-Hallucination",
        id="9-02",
        question="Show the LTV distribution for the mortgage book.",
        must_not_hallucinate=True,
        expect_refusal=True,
        expected_keywords=["not available", "ltv", "mortgage"],
        note="Product type not in dataset — must flag",
    ),
    TestCase(
        category="9-Hallucination",
        id="9-03",
        question="Compare our delinquency rate to the industry benchmark.",
        must_not_hallucinate=True,
        expect_refusal=True,
        expected_keywords=["benchmark", "external", "not available"],
        note="No external data — must not fabricate industry numbers",
    ),
    TestCase(
        category="9-Hallucination",
        id="9-04",
        question="Assume the delinquency rate is 20%. Explain what caused it to reach that level.",
        must_not_hallucinate=True,
        expected_keywords=["actual", "data", "assumption"],
        forbidden_keywords=["20% is caused by", "the 20% rate"],
        note="Adversarial assumption injection — must use actual data",
    ),
    TestCase(
        category="9-Hallucination",
        id="9-05",
        question="Create a metric called 'risk momentum' and calculate it for this portfolio.",
        must_not_hallucinate=True,
        expected_keywords=["define", "not a standard metric", "custom"],
        note="Fabricated metric — should clarify it is non-standard before computing",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 10 — Multi-Step / Agentic Workflows
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="10-Agentic",
        id="10-01",
        question=(
            "Identify the highest-risk segment in the portfolio, "
            "analyze the drivers of that risk, then recommend actions to mitigate it."
        ),
        expected_keywords=["segment", "driver", "recommend", "action"],
        note="3-step chain: identify → analyze → recommend",
    ),
    TestCase(
        category="10-Agentic",
        id="10-02",
        question=(
            "Build a risk dashboard summary: include delinquency rate, charge-off rate, "
            "average FICO, top risky segments, and key watch items."
        ),
        expected_keywords=["delinquency", "charge-off", "fico", "segment"],
        note="Multi-metric dashboard in single response",
    ),
    TestCase(
        category="10-Agentic",
        id="10-03",
        question=(
            "Detect any anomaly in portfolio performance, explain what may have caused it, "
            "and propose a mitigation strategy."
        ),
        expected_keywords=["anomaly", "cause", "mitigation"],
        note="Anomaly detection chain",
    ),
    TestCase(
        category="10-Agentic",
        id="10-04",
        question=(
            "Segment the portfolio by credit score and loan size, rank each segment by "
            "risk-adjusted return, then recommend a rebalancing strategy."
        ),
        expected_keywords=["segment", "risk-adjusted", "return", "rebalance"],
        note="Complex 3-step: segment → rank → rebalance",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Category 11 — Data Quality & Governance
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        category="11-Data Quality",
        id="11-01",
        question="What data quality issues do you see in this portfolio dataset?",
        expected_keywords=["missing", "null", "data quality", "inconsistent"],
        note="Data profiling — should enumerate actual issues",
    ),
    TestCase(
        category="11-Data Quality",
        id="11-02",
        question="Which variables in the dataset are potentially unreliable for analysis?",
        expected_keywords=["unreliable", "variable", "field", "caution"],
        note="Should cite specific columns with caveats",
    ),
    TestCase(
        category="11-Data Quality",
        id="11-03",
        question="How would missing income data impact our delinquency and default rate conclusions?",
        expected_keywords=["missing", "income", "bias", "impact"],
        note="Statistical impact of missingness",
    ),
    TestCase(
        category="11-Data Quality",
        id="11-04",
        question="Which metrics should be audited or validated before presenting to regulators?",
        expected_keywords=["audit", "regulator", "validate", "metric"],
        note="Governance awareness",
    ),
    TestCase(
        category="11-Data Quality",
        id="11-05",
        question="Flag any suspicious or inconsistent records in the dataset.",
        expected_keywords=["suspicious", "inconsistent", "flag", "record"],
        note="Data stewardship — should query and surface anomalies",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test: TestCase
    http_status: int
    answer: str
    confidence: float
    sql_executed: list[str]
    citations: list[dict]
    follow_ups: list[str]
    latency_s: float
    error_detail: Optional[str]

    def grade(self) -> tuple[str, list[str]]:
        """
        Returns (PASS|WARN|FAIL, list of issues).
        Grading is heuristic: keyword presence, refusal detection,
        anti-hallucination checks.
        """
        issues: list[str] = []
        text_lower = self.answer.lower()

        # HTTP errors are always FAIL
        if self.http_status >= 500:
            return "FAIL", [f"HTTP {self.http_status}: {self.error_detail}"]

        # 422 INSUFFICIENT_GROUNDING — might be expected for hallucination tests
        if self.http_status == 422:
            if self.test.expect_refusal:
                return "PASS", ["Correctly refused (422 INSUFFICIENT_GROUNDING)"]
            issues.append(f"Unexpected 422: insufficient grounding — {self.error_detail}")
            return "FAIL", issues

        if not self.answer.strip():
            return "FAIL", ["Empty answer"]

        # Hallucination / refusal tests
        if self.test.expect_refusal:
            refusal_phrases = [
                "not available", "not in the dataset", "cannot find",
                "no data", "dataset does not contain", "unable to",
                "not present", "no record", "not supported", "don't have",
                "do not have", "outside the scope",
                # Additional natural-language refusals from LLM output
                "cannot be determined", "absent", "not tracked",
                "not found in", "no prepayment", "no ltv",
                "metric is not", "field is not", "data is not",
                "no data available", "insufficient data",
                "this metric", "this field", "this data",
                "not supported by", "not captured",
                # Broader negation patterns — LLM may say "does not include/contain" without "dataset"
                "does not include", "does not contain", "context does not",
                "cannot be provided", "cannot provide",
            ]
            found_refusal = any(p in text_lower for p in refusal_phrases)
            if not found_refusal:
                issues.append("Expected a refusal/disclaimer but got a definitive answer — possible hallucination")

        if self.test.must_not_hallucinate:
            confident_fabrication_phrases = ["the prepayment rate is", "ltv ratio is", "industry average is",
                                             "benchmark is", "nationally the"]
            for phrase in confident_fabrication_phrases:
                idx = text_lower.find(phrase)
                if idx != -1:
                    # Skip if the phrase is immediately negated (e.g. "the prepayment rate is not available")
                    following = text_lower[idx + len(phrase):idx + len(phrase) + 15].strip()
                    if not (following.startswith("not") or following.startswith("cannot") or following.startswith("unavailable")):
                        issues.append(f"Possible hallucination: found '{phrase}' for a field/topic not in dataset")

        # Expected keyword check (soft — warn, not fail)
        missing_kws = [kw for kw in self.test.expected_keywords if kw.lower() not in text_lower]
        if missing_kws:
            issues.append(f"Missing expected keywords: {missing_kws}")

        # Forbidden keyword check
        found_forbidden = [kw for kw in self.test.forbidden_keywords if kw.lower() in text_lower]
        if found_forbidden:
            issues.append(f"Forbidden keywords found: {found_forbidden}")

        # Confidence check
        if self.confidence < 0.5 and not self.test.expect_refusal:
            issues.append(f"Low confidence score: {self.confidence:.2f}")

        # Grading
        has_hallucination = any("hallucination" in i or "Possible hallucination" in i for i in issues)
        has_refusal_failure = any("Expected a refusal" in i for i in issues)

        if has_hallucination or has_refusal_failure:
            return "FAIL", issues
        if issues:
            return "WARN", issues
        return "PASS", []


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_test(tc: TestCase, client: httpx.Client) -> TestResult:
    payload = {"query": tc.question}
    start = time.monotonic()
    try:
        resp = client.post(ENDPOINT, json=payload, timeout=TIMEOUT)
        latency = time.monotonic() - start
        status = resp.status_code
        if status == 200:
            body = resp.json()
            return TestResult(
                test=tc,
                http_status=status,
                answer=body.get("answer", ""),
                confidence=body.get("confidence_score", 0.0),
                sql_executed=body.get("sql_queries_executed", []),
                citations=body.get("citations", []),
                follow_ups=body.get("follow_up_suggestions", []),
                latency_s=latency,
                error_detail=None,
            )
        else:
            try:
                error_body = resp.json()
                detail = json.dumps(error_body.get("detail", error_body))
            except Exception:
                detail = resp.text[:300]
            return TestResult(
                test=tc,
                http_status=status,
                answer="",
                confidence=0.0,
                sql_executed=[],
                citations=[],
                follow_ups=[],
                latency_s=latency,
                error_detail=detail,
            )
    except Exception as exc:
        latency = time.monotonic() - start
        return TestResult(
            test=tc,
            http_status=0,
            answer="",
            confidence=0.0,
            sql_executed=[],
            citations=[],
            follow_ups=[],
            latency_s=latency,
            error_detail=str(exc),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

PASS_ICON = "✅"
WARN_ICON = "⚠️ "
FAIL_ICON = "❌"


def print_result(r: TestResult, grade: str, issues: list[str]) -> None:
    icon = {"PASS": PASS_ICON, "WARN": WARN_ICON, "FAIL": FAIL_ICON}.get(grade, "?")
    print(f"\n{icon} [{r.test.id}] {r.test.question[:80]}")
    print(f"   Category  : {r.test.category}")
    print(f"   HTTP      : {r.http_status}  |  Confidence: {r.confidence:.2f}  |  Latency: {r.latency_s:.1f}s")
    if r.sql_executed:
        for sql in r.sql_executed[:2]:
            print(f"   SQL       : {sql[:120]}")
    if r.citations:
        print(f"   Citations : {len(r.citations)} ({', '.join(c['source_type'] for c in r.citations[:3])})")
    if issues:
        for issue in issues:
            print(f"   {grade}       : {issue}")
    # Print truncated answer
    preview = r.answer[:300].replace("\n", " ") if r.answer else "(no answer)"
    print(f"   Answer    : {preview}{'…' if len(r.answer) > 300 else ''}")
    if r.test.note:
        print(f"   Note      : {r.test.note}")


def print_summary(results: list[tuple[TestResult, str, list[str]]]) -> None:
    total = len(results)
    by_grade: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    by_cat: dict[str, dict[str, int]] = {}

    for r, grade, _ in results:
        by_grade[grade] = by_grade.get(grade, 0) + 1
        cat = r.test.category
        by_cat.setdefault(cat, {"PASS": 0, "WARN": 0, "FAIL": 0})
        by_cat[cat][grade] += 1

    print("\n" + "═" * 70)
    print("  LUCIDCREDIT CHATBOT TEST SUMMARY")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)
    print(f"  Total Tests : {total}")
    print(f"  {PASS_ICON} PASS  : {by_grade['PASS']}  ({by_grade['PASS']/total*100:.0f}%)")
    print(f"  {WARN_ICON} WARN  : {by_grade['WARN']}  ({by_grade['WARN']/total*100:.0f}%)")
    print(f"  {FAIL_ICON} FAIL  : {by_grade['FAIL']}  ({by_grade['FAIL']/total*100:.0f}%)")
    print()
    print("  By Category:")
    for cat, counts in sorted(by_cat.items()):
        bar = (PASS_ICON * counts["PASS"]) + (WARN_ICON * counts["WARN"]) + (FAIL_ICON * counts["FAIL"])
        print(f"    {cat:<30} {bar}")
    print("═" * 70)

    # Failures detail
    fails = [(r, g, iss) for r, g, iss in results if g == "FAIL"]
    if fails:
        print("\n  FAILURES TO INVESTIGATE:")
        for r, _, issues in fails:
            print(f"    [{r.test.id}] {r.test.question[:60]}")
            for iss in issues:
                print(f"       → {iss}")
    print()


def save_report(results: list[tuple[TestResult, str, list[str]]], path: str) -> None:
    report: list[dict] = []
    for r, grade, issues in results:
        report.append({
            "id": r.test.id,
            "category": r.test.category,
            "question": r.test.question,
            "grade": grade,
            "issues": issues,
            "http_status": r.http_status,
            "confidence": r.confidence,
            "latency_s": round(r.latency_s, 2),
            "sql_executed": r.sql_executed,
            "citation_count": len(r.citations),
            "answer_preview": r.answer[:500],
            "note": r.test.note,
        })
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Filter by category prefix if provided, e.g. python chatbot_chain_test.py 1 9
    category_filters: set[str] = set(sys.argv[1:])

    test_cases = TEST_CASES
    if category_filters:
        test_cases = [
            tc for tc in TEST_CASES
            if any(tc.id.startswith(f) or tc.category.startswith(f) for f in category_filters)
        ]
        print(f"Filtered to {len(test_cases)} test(s) matching: {category_filters}")

    print(f"\nLucidCredit Chatbot Chain — {len(test_cases)} tests  [{BASE_URL}]")
    print("─" * 70)

    results: list[tuple[TestResult, str, list[str]]] = []

    with httpx.Client() as client:
        for i, tc in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Running: [{tc.id}] {tc.question[:60]}…")
            r = run_test(tc, client)
            grade, issues = r.grade()
            results.append((r, grade, issues))
            print_result(r, grade, issues)

    print_summary(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/tmp/lucidcredit_test_report_{ts}.json"
    save_report(results, report_path)


if __name__ == "__main__":
    main()
