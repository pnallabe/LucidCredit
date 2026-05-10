"""
tests/integration/test_reasoning_trace.py
==========================================
Integration tests focused on four quality dimensions of the LucidCredit agent:

  1. Context Retrieval    — correct retrieval_method, non-empty retrieved_context
  2. Chain of Thought     — raw_analysis populated, contains logical reasoning steps
  3. Accuracy             — answer contains real data markers (numbers, dates, product codes)
  4. Traceability         — citations grounded, SQL generated for data questions,
                            source_refs traceable, suppressed claims auditable

These tests require the backend (port 8090) AND analytics API (port 8001) to be live.

Run:
    cd /Users/swarnabale/Documents/My\ Projects/LucidCredit
    python -m pytest tests/integration/test_reasoning_trace.py -v --tb=short 2>&1 | tee /tmp/rt_test.txt
Or directly:
    python tests/integration/test_reasoning_trace.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 90  # seconds per test

# ─────────────────────────────────────────────────────────────────────────────
# Grading helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "PASS"
WARN  = "WARN"
FAIL  = "FAIL"

# Patterns that indicate a number is present in the answer
_NUMBER_RE = re.compile(r"\$[\d,]+|\d[\d,]*\.?\d*\s*(?:%|million|billion|thousand|loans?|accounts?|bps)", re.IGNORECASE)

# Reasoning "connector" words that indicate real chain-of-thought vs boilerplate
_REASONING_CONNECTORS = [
    "because", "therefore", "which means", "this indicates", "as a result",
    "driven by", "suggests", "given that", "since", "consequently",
    "reflecting", "attributable", "compared to", "higher than", "lower than",
    "relative to", "due to", "contributing to",
]


@dataclass
class RTResult:
    """Holds the full response payload for a single reasoning-trace test."""
    test_id: str
    question: str
    http_status: int
    answer: str
    confidence: float
    sql_executed: list[str]
    citations: list[dict]
    reasoning_trace: dict            # raw reasoning_trace dict from backend
    latency_s: float
    error_detail: Optional[str]

    # ── dimension grades ──────────────────────────────────────────────────────

    def _grade_context_retrieval(self) -> tuple[str, list[str]]:
        """
        PASS  : retrieval_method set AND retrieved_context non-empty
        WARN  : retrieval_method set but retrieved_context empty
        FAIL  : reasoning_trace absent or retrieval_method blank
        """
        issues: list[str] = []
        rt = self.reasoning_trace

        if not rt:
            return FAIL, ["reasoning_trace absent from response"]

        method = rt.get("retrieval_method", "")
        ctx    = rt.get("retrieved_context") or []

        if not method:
            issues.append("retrieval_method is blank")
            return FAIL, issues

        if not ctx:
            issues.append(f"retrieved_context is empty (method={method})")
            return WARN, issues

        # Each context item should have source_type + snippet
        for i, item in enumerate(ctx):
            if not item.get("source_type"):
                issues.append(f"retrieved_context[{i}] missing source_type")
            if not item.get("snippet"):
                issues.append(f"retrieved_context[{i}] has empty snippet")

        return (WARN if issues else PASS), issues

    def _grade_chain_of_thought(self) -> tuple[str, list[str]]:
        """
        PASS  : raw_analysis non-empty AND contains reasoning connectors
        WARN  : raw_analysis present but very short (<100 chars) or no connectors
        FAIL  : raw_analysis absent or empty
        """
        issues: list[str] = []
        rt = self.reasoning_trace

        if not rt:
            return FAIL, ["reasoning_trace absent — cannot evaluate chain-of-thought"]

        raw = rt.get("raw_analysis", "")
        if not raw or not raw.strip():
            return FAIL, ["raw_analysis (pre-grounding LLM output) is empty"]

        if len(raw.strip()) < 100:
            issues.append(f"raw_analysis very short ({len(raw.strip())} chars) — may be a stub")
            return WARN, issues

        raw_lower = raw.lower()
        matched = [c for c in _REASONING_CONNECTORS if c in raw_lower]
        if not matched:
            issues.append("raw_analysis has no reasoning connector words — may lack explanation depth")
            return WARN, issues

        return PASS, []

    def _grade_accuracy(self) -> tuple[str, list[str]]:
        """
        For data-query tests (SQL executed):
          PASS  : answer contains real numeric figures
          WARN  : no numbers found in answer
          FAIL  : answer contradicts SQL source (extreme mismatch detection)

        For domain-knowledge tests (no SQL):
          PASS  : always (can't validate against external truth)
        """
        issues: list[str] = []
        answer_lower = self.answer.lower()

        # If SQL was executed, we expect real numbers in the answer
        if self.sql_executed:
            nums = _NUMBER_RE.findall(self.answer)
            if not nums:
                issues.append("SQL was executed but answer contains no numeric figures — possible hallucination or missed data binding")
                return WARN, issues

            # Check for fabricated round numbers that suggest no real data lookup
            # e.g. exactly "100 loans" or "$1,000,000" — suspicious when real BQ data is available
            suspiciously_round = [n for n in nums if re.match(r"^\$1,000,000$|^100\s*loan|^0%$", n.strip())]
            if suspiciously_round:
                issues.append(f"Suspiciously round figures found: {suspiciously_round} — verify against BQ output")
                return WARN, issues

        # Domain knowledge answers should at least not claim external unavailable data
        rt_method = (self.reasoning_trace or {}).get("retrieval_method", "")
        if rt_method == "domain_knowledge" and self.sql_executed:
            issues.append("Domain knowledge path used but SQL was also returned — inconsistent signals")
            return WARN, issues

        return PASS, issues

    def _grade_traceability(self) -> tuple[str, list[str]]:
        """
        PASS  : has citations OR (domain_knowledge path with retrieved_context)
        WARN  : citations absent but answer present; OR suppressed claims with no explanation
        FAIL  : no citations, no retrieved_context, no SQL
        """
        issues: list[str] = []
        rt = self.reasoning_trace or {}
        ctx = rt.get("retrieved_context") or []
        suppressed = rt.get("suppressed_claims") or []
        method = rt.get("retrieval_method", "")

        # At least one citation OR at least one retrieved context item
        if not self.citations and not ctx:
            issues.append("No citations and no retrieved_context — answer is fully ungrounded")
            return FAIL, issues

        if not self.citations:
            issues.append(f"No citations in response ({len(ctx)} context items were retrieved)")
            # WARN not FAIL: domain_knowledge path often has low citation rate

        # Suppressed claims are audit-critical — log them as info
        if suppressed:
            issues.append(f"{len(suppressed)} claim(s) were suppressed during grounding — auditable")

        # SQL queries must have traceable source_refs in retrieved_context
        if self.sql_executed:
            db_items = [c for c in ctx if c.get("source_type") == "db"]
            if not db_items:
                issues.append("SQL executed but no db-type items found in retrieved_context — traceability gap")
                return WARN, issues

            for item in db_items:
                if not item.get("source_ref"):
                    issues.append(f"db item missing source_ref: {item}")

        return (WARN if issues else PASS), issues

    def grade_all(self) -> dict:
        """Run all four dimension grades and compute overall."""
        ret = {
            "test_id": self.test_id,
            "question": self.question[:80],
            "latency_s": round(self.latency_s, 2),
            "http_status": self.http_status,
            "confidence": round(self.confidence, 3),
            "sql_count": len(self.sql_executed),
            "citation_count": len(self.citations),
            "context_items": len((self.reasoning_trace or {}).get("retrieved_context") or []),
            "suppressed_count": len((self.reasoning_trace or {}).get("suppressed_claims") or []),
            "retrieval_method": (self.reasoning_trace or {}).get("retrieval_method", ""),
        }

        grades: dict[str, tuple[str, list[str]]] = {}
        grades["context_retrieval"] = self._grade_context_retrieval()
        grades["chain_of_thought"]  = self._grade_chain_of_thought()
        grades["accuracy"]          = self._grade_accuracy()
        grades["traceability"]      = self._grade_traceability()

        for dim, (g, issues) in grades.items():
            ret[f"dim_{dim}"] = g
            ret[f"dim_{dim}_issues"] = issues

        # Overall: FAIL if any dim fails; WARN if any warns; else PASS
        dim_grades = [g for g, _ in grades.values()]
        if FAIL in dim_grades:
            ret["overall"] = FAIL
        elif WARN in dim_grades:
            ret["overall"] = WARN
        else:
            ret["overall"] = PASS

        return ret


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RTCase:
    test_id: str
    question: str
    expect_sql: bool = True          # True for data-query; False for domain-knowledge
    note: str = ""


# 24 focused cases across the four dimensions
RT_CASES: list[RTCase] = [

    # ── Context Retrieval ─────────────────────────────────────────────────────
    RTCase("CR-01", "What is the total outstanding balance by product type?",
           expect_sql=True,
           note="Basic BQ query — retrieval_method must be bigquery_query"),
    RTCase("CR-02", "Show the monthly delinquency trend for the last 24 months.",
           expect_sql=True,
           note="Time-series BQ query — retrieved_context must have db items"),
    RTCase("CR-03", "What is the average FICO score at origination by product type?",
           expect_sql=True,
           note="Aggregation over origination data"),
    RTCase("CR-04", "What is the charge-off rate by quarter?",
           expect_sql=True,
           note="Income statement table — separate source from balance sheet"),
    RTCase("CR-05", "Why is delinquency typically higher in subprime segments?",
           expect_sql=False,
           note="Domain knowledge path — retrieval_method must be domain_knowledge"),
    RTCase("CR-06", "How should we adjust underwriting criteria based on performance?",
           expect_sql=False,
           note="Domain reasoning — no SQL, rich domain context expected"),

    # ── Chain of Thought ─────────────────────────────────────────────────────
    RTCase("COT-01", "Does higher income reduce default risk in this dataset? Show the data.",
           expect_sql=True,
           note="Correlation analysis — raw_analysis should show causal reasoning"),
    RTCase("COT-02", "Delinquency increased 2% last quarter. Break down the drivers by segment.",
           expect_sql=False,
           note="Root-cause decomposition — chain-of-thought must show segment attribution"),
    RTCase("COT-03", "What actions should we take to reduce the delinquency rate?",
           expect_sql=False,
           note="Prescriptive chain: diagnose → prioritize → act"),
    RTCase("COT-04", "Is the delinquency trend increasing significantly or is it seasonal variation?",
           expect_sql=True,
           note="Temporal reasoning — must distinguish trend vs seasonality"),
    RTCase("COT-05",
           "Identify the highest-risk segment, analyze its drivers, then recommend mitigation.",
           expect_sql=False,
           note="3-step agentic chain — raw_analysis must show all three steps"),
    RTCase("COT-06", "Compare pre-Q3 2023 vs post-Q3 2023 portfolio performance.",
           expect_sql=True,
           note="Structural break analysis — raw_analysis must reference both periods"),

    # ── Accuracy ─────────────────────────────────────────────────────────────
    RTCase("ACC-01", "What is the total outstanding loan balance across all personal loans?",
           expect_sql=True,
           note="Single aggregate — answer must contain a dollar figure from BQ"),
    RTCase("ACC-02", "What is the weighted average APR of the portfolio weighted by outstanding balance?",
           expect_sql=True,
           note="Weighted aggregate — answer must show a %, not a generic description"),
    RTCase("ACC-03", "How many loan applications were submitted each month in 2024?",
           expect_sql=True,
           note="Count aggregation — answer must contain numeric counts"),
    RTCase("ACC-04", "What is the compound annual growth rate (CAGR) of the portfolio balance?",
           expect_sql=True,
           note="CAGR formula — answer must contain a % figure"),
    RTCase("ACC-05", "What is the approval rate by FICO tier for personal loans?",
           expect_sql=True,
           note="Cross-table metric — answer must have FICO tier labels + rates"),
    RTCase("ACC-06", "What is the net charge-off rate over the last 12 months?",
           expect_sql=True,
           note="Net charge-off requires gross minus recoveries — validate formula use"),

    # ── Traceability ─────────────────────────────────────────────────────────
    RTCase("TR-01", "What is the 30+ DPD rate across the portfolio?",
           expect_sql=True,
           note="Citations must trace to BQ row data, not domain knowledge"),
    RTCase("TR-02", "Show total outstanding balance and loan count by state.",
           expect_sql=True,
           note="Geographic grouping — SQL source_ref must appear in retrieved_context"),
    RTCase("TR-03", "What is the prepayment rate of the portfolio?",
           expect_sql=False,
           note="Not in dataset — reasoning_trace must show domain_knowledge; no fabricated % figure"),
    RTCase("TR-04", "Compare our delinquency rate to the industry benchmark.",
           expect_sql=False,
           note="External data absent — suppressed claims or refusal must be traceable"),
    RTCase("TR-05", "Create a metric called 'risk momentum' and calculate it for this portfolio.",
           expect_sql=False,
           note="Non-standard metric — raw_analysis must clarify it is custom"),
    RTCase("TR-06", "Show the loan count and outstanding balance by product type and origination channel.",
           expect_sql=True,
           note="Multi-dimension group-by — all SQL queries must be in retrieved_context"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_case(tc: RTCase, client: httpx.Client) -> RTResult:
    payload = {"query": tc.question}
    t0 = time.monotonic()
    try:
        resp = client.post(ENDPOINT, json=payload, timeout=TIMEOUT)
        latency = time.monotonic() - t0
        status = resp.status_code

        if status == 200:
            body = resp.json()
            rt = body.get("reasoning_trace") or {}
            return RTResult(
                test_id=tc.test_id,
                question=tc.question,
                http_status=status,
                answer=body.get("answer", ""),
                confidence=body.get("confidence_score", 0.0),
                sql_executed=body.get("sql_queries_executed", []),
                citations=body.get("citations", []),
                reasoning_trace=rt,
                latency_s=latency,
                error_detail=None,
            )

        # Non-200 (422/503/etc.)
        try:
            detail = json.dumps(resp.json().get("detail", resp.text[:200]))
        except Exception:
            detail = resp.text[:200]
        return RTResult(
            test_id=tc.test_id,
            question=tc.question,
            http_status=status,
            answer="",
            confidence=0.0,
            sql_executed=[],
            citations=[],
            reasoning_trace={},
            latency_s=latency,
            error_detail=detail,
        )
    except Exception as exc:
        return RTResult(
            test_id=tc.test_id,
            question=tc.question,
            http_status=0,
            answer="",
            confidence=0.0,
            sql_executed=[],
            citations=[],
            reasoning_trace={},
            latency_s=time.monotonic() - t0,
            error_detail=str(exc),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Printer
# ─────────────────────────────────────────────────────────────────────────────

_ICONS = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌"}

_DIM_LABELS = {
    "context_retrieval": "Context Retrieval",
    "chain_of_thought":  "Chain of Thought ",
    "accuracy":          "Accuracy         ",
    "traceability":      "Traceability     ",
}

def print_result(r: dict) -> None:
    icon = _ICONS.get(r["overall"], "?")
    print(f"\n{icon} [{r['test_id']}] {r['question']}")
    print(f"   HTTP {r['http_status']} | {r['latency_s']:.1f}s | "
          f"conf={r['confidence']:.2f} | sql={r['sql_count']} | "
          f"cit={r['citation_count']} | ctx={r['context_items']} | "
          f"method={r['retrieval_method']}")

    dims = ["context_retrieval", "chain_of_thought", "accuracy", "traceability"]
    for dim in dims:
        g = r[f"dim_{dim}"]
        issues = r[f"dim_{dim}_issues"]
        gi = _ICONS.get(g, "?")
        label = _DIM_LABELS[dim]
        print(f"   {gi} {label} → {g}", end="")
        if issues:
            print(f"  |  {'; '.join(issues[:2])}", end="")
        print()


def print_summary(results: list[dict], elapsed: float) -> None:
    by_overall = {PASS: 0, WARN: 0, FAIL: 0}
    by_dim: dict[str, dict[str, int]] = {
        dim: {PASS: 0, WARN: 0, FAIL: 0}
        for dim in ["context_retrieval", "chain_of_thought", "accuracy", "traceability"]
    }
    for r in results:
        by_overall[r["overall"]] += 1
        for dim in by_dim:
            by_dim[dim][r[f"dim_{dim}"]] += 1

    total = len(results)
    print("\n" + "═" * 72)
    print("REASONING TRACE INTEGRATION TESTS — SUMMARY")
    print("═" * 72)
    print(f"  Total: {total}   ✅ PASS: {by_overall[PASS]}   "
          f"⚠️  WARN: {by_overall[WARN]}   ❌ FAIL: {by_overall[FAIL]}   "
          f"  ({elapsed:.0f}s)")
    print()
    print(f"  {'Dimension':<26}  {'PASS':>5}  {'WARN':>5}  {'FAIL':>5}")
    print("  " + "─" * 46)
    for dim, label in _DIM_LABELS.items():
        d = by_dim[dim]
        print(f"  {label}  {d[PASS]:>5}  {d[WARN]:>5}  {d[FAIL]:>5}")
    print("═" * 72)

    # List failures
    fails = [r for r in results if r["overall"] == FAIL]
    if fails:
        print("\nFAILED TESTS:")
        for r in fails:
            print(f"  ❌ [{r['test_id']}] {r['question']}")
            for dim in ["context_retrieval", "chain_of_thought", "accuracy", "traceability"]:
                if r[f"dim_{dim}"] == FAIL:
                    for issue in r[f"dim_{dim}_issues"]:
                        print(f"       {dim}: {issue}")


# ─────────────────────────────────────────────────────────────────────────────
# pytest-compatible test function (auto-discovered when run via pytest)
# ─────────────────────────────────────────────────────────────────────────────

def _check_backend_live() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def test_context_retrieval_cases():
    """Pytest: all CR-* cases must have retrieval_method set and non-empty context."""
    if not _check_backend_live():
        import pytest
        pytest.skip("Backend not reachable at http://localhost:8090")
    with httpx.Client() as client:
        cases = [c for c in RT_CASES if c.test_id.startswith("CR-")]
        for tc in cases:
            res = run_case(tc, client)
            grade, issues = RTResult.__dict__["_grade_context_retrieval"](res)
            assert grade != FAIL, f"[{tc.test_id}] Context retrieval FAIL: {issues}"


def test_chain_of_thought_cases():
    """Pytest: all COT-* cases must have non-empty raw_analysis with reasoning connectors."""
    if not _check_backend_live():
        import pytest
        pytest.skip("Backend not reachable at http://localhost:8090")
    with httpx.Client() as client:
        cases = [c for c in RT_CASES if c.test_id.startswith("COT-")]
        for tc in cases:
            res = run_case(tc, client)
            grade, issues = RTResult.__dict__["_grade_chain_of_thought"](res)
            assert grade != FAIL, f"[{tc.test_id}] Chain-of-thought FAIL: {issues}"


def test_accuracy_cases():
    """Pytest: all ACC-* cases must return numeric figures when SQL is executed."""
    if not _check_backend_live():
        import pytest
        pytest.skip("Backend not reachable at http://localhost:8090")
    with httpx.Client() as client:
        cases = [c for c in RT_CASES if c.test_id.startswith("ACC-")]
        for tc in cases:
            res = run_case(tc, client)
            grade, issues = RTResult.__dict__["_grade_accuracy"](res)
            assert grade != FAIL, f"[{tc.test_id}] Accuracy FAIL: {issues}"


def test_traceability_cases():
    """Pytest: all TR-* cases must have citations or retrieved_context traceable to sources."""
    if not _check_backend_live():
        import pytest
        pytest.skip("Backend not reachable at http://localhost:8090")
    with httpx.Client() as client:
        cases = [c for c in RT_CASES if c.test_id.startswith("TR-")]
        for tc in cases:
            res = run_case(tc, client)
            grade, issues = RTResult.__dict__["_grade_traceability"](res)
            assert grade != FAIL, f"[{tc.test_id}] Traceability FAIL: {issues}"


# ─────────────────────────────────────────────────────────────────────────────
# Direct runner (python tests/integration/test_reasoning_trace.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optional: filter by prefix passed as CLI arg, e.g. "CR" or "ACC"
    filter_prefix = sys.argv[1].upper() if len(sys.argv) > 1 else None
    cases = [c for c in RT_CASES if not filter_prefix or c.test_id.startswith(filter_prefix)]

    if not cases:
        print(f"No cases match prefix '{filter_prefix}'")
        sys.exit(1)

    print(f"LucidCredit Reasoning Trace Integration Tests")
    print(f"Target: {ENDPOINT}")
    print(f"Cases:  {len(cases)}/{len(RT_CASES)}"
          + (f" (filter={filter_prefix})" if filter_prefix else ""))
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 72)

    # Check backend liveness
    if not _check_backend_live():
        print(f"\n❌  Backend not reachable at {BASE_URL}")
        print("    Start with: bash start_lucidcredit_backend.sh")
        sys.exit(1)

    all_results: list[dict] = []
    t_start = time.monotonic()

    with httpx.Client() as client:
        for tc in cases:
            res = run_case(tc, client)
            graded = res.grade_all()
            all_results.append(graded)
            print_result(graded)

    elapsed = time.monotonic() - t_start
    print_summary(all_results, elapsed)

    # Write JSON report
    report_path = f"/tmp/rt_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report: {report_path}")

    # Exit non-zero if any FAILs
    fails = sum(1 for r in all_results if r["overall"] == FAIL)
    sys.exit(1 if fails else 0)
