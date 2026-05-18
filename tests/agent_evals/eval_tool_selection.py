"""
tests/agent_evals/eval_tool_selection.py
==========================================
Eval 3 — Tool Selection Quality

For each question in the TOOL_ORACLE, the eval:
  1. Calls the live backend
  2. Inspects citations[].source_type + reasoning_trace.retrieved_context[].source_type
     to determine which tools actually fired
  3. Computes precision, recall, F1 vs. the oracle's expected_tools

Tool mapping:
  vector_doc  → vector_tool.py        (policy_docs pgvector)
  api         → analytics_api_tool.py (NL2SQL → BigQuery)
  db          → sql_tool.py / crp_api_tool.py (application DB)
  thinfile    → thinfile_tool.py      (thin-file applicant bureau)
  domain_knowledge → internal domain knowledge (not an external tool)

Run:
    python tests/agent_evals/eval_tool_selection.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 90.0

# ─────────────────────────────────────────────────────────────────────────────
# Tool oracle — 20+ entries covering all tools and multi-tool combos
# ─────────────────────────────────────────────────────────────────────────────

TOOL_ORACLE: list[dict] = [
    # ── api only (NL2SQL → BigQuery) ─────────────────────────────────────────
    {
        "question": "What is the total outstanding loan balance across the portfolio?",
        "expected_tools": ["api"],
        "rationale": "quantitative portfolio aggregate → NL2SQL",
    },
    {
        "question": "What is the current delinquency rate for personal loans?",
        "expected_tools": ["api"],
        "rationale": "portfolio metric → NL2SQL BigQuery query",
    },
    {
        "question": "Show the charge-off rate by quarter for the past two years.",
        "expected_tools": ["api"],
        "rationale": "time-series financial metric → NL2SQL",
    },
    {
        "question": "How many loans were originated per month in 2023?",
        "expected_tools": ["api"],
        "rationale": "origination volume time-series → NL2SQL",
    },
    {
        "question": "What is the approval rate by FICO tier for personal loans?",
        "expected_tools": ["api"],
        "rationale": "approval rate segmented metric → NL2SQL",
    },
    {
        "question": "Show the total outstanding balance by state for personal loans.",
        "expected_tools": ["api"],
        "rationale": "geographic portfolio breakdown → NL2SQL",
    },
    {
        "question": "What is the weighted average APR of funded personal loans?",
        "expected_tools": ["api"],
        "rationale": "weighted aggregate portfolio metric → NL2SQL",
    },
    # ── vector_doc only (pgvector policy search) ──────────────────────────────
    {
        "question": "What are the ECOA adverse action notice requirements?",
        "expected_tools": ["vector_doc"],
        "rationale": "regulatory policy lookup → pgvector policy docs",
    },
    {
        "question": "What FCRA section 615 rights must be disclosed to applicants?",
        "expected_tools": ["vector_doc"],
        "rationale": "legal disclosure requirements → pgvector",
    },
    {
        "question": "What are the SR 11-7 model risk management guidelines?",
        "expected_tools": ["vector_doc"],
        "rationale": "regulatory guidance document → pgvector policy docs",
    },
    {
        "question": "What credit policy governs underwriting of balance transfer products?",
        "expected_tools": ["vector_doc"],
        "rationale": "internal credit policy lookup → pgvector",
    },
    {
        "question": "What compliance disclosures are required for an adverse action decision?",
        "expected_tools": ["vector_doc"],
        "rationale": "regulatory compliance policy → pgvector",
    },
    # ── api + vector_doc (data + policy narrative) ────────────────────────────
    {
        "question": (
            "What is our delinquency rate and how does it compare to ECOA reporting requirements?"
        ),
        "expected_tools": ["api", "vector_doc"],
        "rationale": "quantitative metric + regulatory policy → both tools",
    },
    {
        "question": (
            "Show the charge-off rate trend and explain how our credit policy defines charge-off thresholds."
        ),
        "expected_tools": ["api", "vector_doc"],
        "rationale": "financial metric + policy definition → both tools",
    },
    {
        "question": (
            "What is the average FICO score in our portfolio and what underwriting policy applies to sub-620 borrowers?"
        ),
        "expected_tools": ["api", "vector_doc"],
        "rationale": "portfolio data + underwriting policy → both tools",
    },
    # ── db (application DB / CRP direct lookup) ───────────────────────────────
    {
        "question": "Retrieve the credit decision details for application APP-2024-001.",
        "expected_tools": ["db"],
        "rationale": "single application lookup by ID → application DB",
    },
    {
        "question": "What is the status of loan application APP-20230718-4421?",
        "expected_tools": ["db"],
        "rationale": "direct application record lookup → application DB",
    },
    # ── applicant_comms (vector_doc for policy + possible db) ─────────────────
    {
        "question": (
            "Draft an adverse action notice for an applicant declined due to DTI exceeding 45%. "
            "Include all required ECOA disclosures."
        ),
        "expected_tools": ["vector_doc"],
        "rationale": "applicant communication requires policy docs for compliance language",
    },
    {
        "question": (
            "Write a decline notice citing insufficient employment history. "
            "Reference the FCRA section 615 consumer report rights."
        ),
        "expected_tools": ["vector_doc"],
        "rationale": "compliance-heavy applicant communication → vector_doc for FCRA language",
    },
    # ── explain_decision (vector_doc + possible db) ───────────────────────────
    {
        "question": "Explain why a FICO score below 620 leads to automatic loan denial.",
        "expected_tools": ["vector_doc"],
        "rationale": "policy explanation requires underwriting policy docs",
    },
    {
        "question": "Why does a debt-to-income ratio above 50% trigger an adverse action?",
        "expected_tools": ["vector_doc"],
        "rationale": "policy-driven explanation → policy docs",
    },
    # ── domain_knowledge bypass (acceptable — agent uses internal knowledge) ──
    {
        "question": "What are the top 3 risk factors in a credit portfolio?",
        "expected_tools": ["vector_doc"],
        "rationale": "risk framework question → policy docs or domain knowledge",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool extraction from response
# ─────────────────────────────────────────────────────────────────────────────

_VALID_TOOL_TYPES = {"api", "vector_doc", "db", "thinfile"}


def _extract_used_tools(response_body: dict) -> set[str]:
    """Extract source_types from citations and retrieved_context."""
    tools: set[str] = set()
    for c in response_body.get("citations") or []:
        st = c.get("source_type", "")
        if st in _VALID_TOOL_TYPES:
            tools.add(st)
    rt = response_body.get("reasoning_trace") or {}
    for item in rt.get("retrieved_context") or []:
        st = item.get("source_type", "")
        if st in _VALID_TOOL_TYPES:
            tools.add(st)
    # Infer from retrieval_method string
    method = rt.get("retrieval_method", "").lower()
    if "vector" in method:
        tools.add("vector_doc")
    if "api" in method or "nl2sql" in method or "bigquery" in method:
        tools.add("api")
    if "sql" in method and "nl2sql" not in method:
        tools.add("db")
    return tools


def _compute_f1(selected: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Returns (precision, recall, f1)."""
    if not selected and not expected:
        return 1.0, 1.0, 1.0
    intersection = selected & expected
    precision = len(intersection) / len(selected) if selected else 0.0
    recall    = len(intersection) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolTurnResult:
    question: str
    expected_tools: list[str]
    actual_tools: list[str]
    precision: float
    recall: float
    f1: float
    wrong_tool: bool  # selected ∩ expected is empty
    http_status: int
    latency_s: float


@dataclass
class ToolSelectionReport:
    macro_f1: float
    gate_status: str
    turn_results: list[ToolTurnResult]
    wrong_tool_cases: list[ToolTurnResult]

    def print_summary(self) -> None:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[self.gate_status]
        print(f"\n{'─'*60}")
        print(f"Eval 3 — Tool Selection Macro-F1: {self.macro_f1:.3f}  {icon} {self.gate_status}")
        print(f"  Gate: ≥0.85 PASS | 0.70-0.84 WARN | <0.70 FAIL\n")
        print(f"  {'P':5s} {'R':5s} {'F1':5s}  Expected         →  Actual           Question")
        for t in self.turn_results:
            wrong = " ← WRONG TOOL" if t.wrong_tool else ""
            exp_s = ",".join(sorted(t.expected_tools))[:16]
            act_s = ",".join(sorted(t.actual_tools))[:16]
            q_s   = t.question[:50]
            print(f"  {t.precision:.3f} {t.recall:.3f} {t.f1:.3f}  {exp_s:16s}  {act_s:16s}  {q_s}{wrong}")
        if self.wrong_tool_cases:
            print(f"\n  Wrong-tool cases ({len(self.wrong_tool_cases)}):")
            for t in self.wrong_tool_cases:
                print(f"    Q: {t.question[:70]}")
                print(f"       Expected: {t.expected_tools}  Got: {t.actual_tools}")
        print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ToolSelectionEval:
    oracle: list[dict] = TOOL_ORACLE

    def run(self) -> ToolSelectionReport:
        print(f"[Eval 3] Tool Selection Quality — {len(self.oracle)} oracle entries")
        print(f"  Endpoint : {ENDPOINT}")
        print(f"  Gate     : Macro-F1 ≥0.85 PASS | 0.70-0.84 WARN | <0.70 FAIL\n")

        turn_results: list[ToolTurnResult] = []

        with httpx.Client(timeout=TIMEOUT) as client:
            for i, entry in enumerate(self.oracle, 1):
                question      = entry["question"]
                expected      = set(entry["expected_tools"])
                rationale     = entry.get("rationale", "")

                print(f"  [{i:02d}/{len(self.oracle)}] {question[:65]}...", end=" ", flush=True)

                t0 = time.perf_counter()
                try:
                    resp = client.post(
                        ENDPOINT,
                        json={"query": question},
                        headers={"X-Eval-Mode": "true"},
                    )
                    body = resp.json()
                    status = resp.status_code
                except Exception as exc:
                    body = {}
                    status = 0
                latency = time.perf_counter() - t0

                if status != 200 or not body.get("answer"):
                    actual_tools: set[str] = set()
                    precision, recall, f1 = 0.0, 0.0, 0.0
                else:
                    actual_tools = _extract_used_tools(body)
                    precision, recall, f1 = _compute_f1(actual_tools, expected)

                wrong = len(actual_tools & expected) == 0 and bool(actual_tools)
                icon  = "✅" if f1 >= 0.85 else ("⚠️" if f1 >= 0.50 else "❌")
                print(f"{icon}  F1={f1:.3f}  exp={sorted(expected)}  got={sorted(actual_tools)}  {latency:.1f}s")

                turn_results.append(ToolTurnResult(
                    question=question,
                    expected_tools=sorted(expected),
                    actual_tools=sorted(actual_tools),
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    wrong_tool=wrong,
                    http_status=status,
                    latency_s=latency,
                ))

        macro_f1 = (
            sum(t.f1 for t in turn_results) / len(turn_results)
            if turn_results else 0.0
        )
        if macro_f1 >= 0.85:
            gate = "PASS"
        elif macro_f1 >= 0.70:
            gate = "WARN"
        else:
            gate = "FAIL"

        wrong_tool_cases = [t for t in turn_results if t.wrong_tool]
        report = ToolSelectionReport(
            macro_f1=macro_f1,
            gate_status=gate,
            turn_results=turn_results,
            wrong_tool_cases=wrong_tool_cases,
        )
        report.print_summary()
        return report


if __name__ == "__main__":
    eval_ = ToolSelectionEval()
    report = eval_.run()
    sys.exit(0 if report.gate_status == "PASS" else 1)
