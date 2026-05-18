"""
tests/agent_evals/eval_task_success.py
========================================
Eval 1 — Task Success Rate

Definition: fraction of requests where the agent produces a final_output
(HTTP 200, no error key, answer non-empty, confidence_score >= 0.75).

Run:
    python tests/agent_evals/eval_task_success.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 90.0

MIN_CONFIDENCE = 0.75

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Failure mode taxonomy
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_FAILED        = "RETRIEVAL_FAILED"
GROUNDING_FAILED        = "GROUNDING_FAILED"
COMPLIANCE_BLOCKED      = "COMPLIANCE_BLOCKED"
LLM_ERROR               = "LLM_ERROR"
CLARIFICATION_LOOP      = "CLARIFICATION_LOOP"
HALLUCINATION_SUPPRESSED = "HALLUCINATION_SUPPRESSED"

# ─────────────────────────────────────────────────────────────────────────────
# Test suite — 25 prompts spanning all intent types
# ─────────────────────────────────────────────────────────────────────────────

TEST_SUITE: list[dict] = [
    # ── analyst_query (10) ────────────────────────────────────────────────────
    {
        "intent": "analyst_query",
        "query": "What is the total outstanding loan balance across the portfolio?",
    },
    {
        "intent": "analyst_query",
        "query": "What is the current delinquency rate for personal loans?",
    },
    {
        "intent": "analyst_query",
        "query": "Show the charge-off rate by quarter for the past two years.",
    },
    {
        "intent": "analyst_query",
        "query": "What is the approval rate by FICO tier for personal loan applications?",
    },
    {
        "intent": "analyst_query",
        "query": "What is the average debt-to-income ratio at origination by product type?",
    },
    {
        "intent": "analyst_query",
        "query": "How many loans were originated per month in 2023?",
    },
    {
        "intent": "analyst_query",
        "query": "What is the weighted average APR across funded personal loans?",
    },
    {
        "intent": "analyst_query",
        "query": "Show total outstanding balance by state for personal loans.",
    },
    {
        "intent": "analyst_query",
        "query": "What is the 30-day delinquency rate trend over the past 12 months?",
    },
    {
        "intent": "analyst_query",
        "query": "What is the average FICO score at origination by product type?",
    },
    # ── explain_decision (5) ─────────────────────────────────────────────────
    {
        "intent": "explain_decision",
        "query": "Explain why a high debt-to-income ratio leads to loan denial.",
    },
    {
        "intent": "explain_decision",
        "query": "What are the ECOA adverse action notice requirements?",
    },
    {
        "intent": "explain_decision",
        "query": "What factors determine a credit application is declined?",
    },
    {
        "intent": "explain_decision",
        "query": "Explain the FCRA section 615 consumer rights disclosure.",
    },
    {
        "intent": "explain_decision",
        "query": "What model risk governance requirements apply under SR 11-7?",
    },
    # ── applicant_comms (5) ──────────────────────────────────────────────────
    {
        "intent": "applicant_comms",
        "query": (
            "Draft an adverse action notice for an applicant whose loan was declined "
            "due to a debt-to-income ratio of 52% exceeding our 45% threshold."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Explain to an applicant why their credit score of 580 did not meet "
            "our minimum requirement of 620."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Write a notice informing the applicant their credit history shows "
            "two derogatory accounts in the past 24 months."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Tell the applicant we used a consumer report from a credit reporting "
            "agency and explain their right to a free copy within 60 days."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Draft a communication explaining that insufficient employment history "
            "contributed to the adverse credit decision."
        ),
    },
    # ── portfolio_brief (5) ──────────────────────────────────────────────────
    {
        "intent": "portfolio_brief",
        "query": "Summarize the current portfolio risk profile for the executive team.",
    },
    {
        "intent": "portfolio_brief",
        "query": "What are the top credit risk indicators in our personal loan portfolio?",
    },
    {
        "intent": "portfolio_brief",
        "query": "Provide a brief on the delinquency trends for the board meeting.",
    },
    {
        "intent": "portfolio_brief",
        "query": "What is the overall portfolio health as of the latest reporting period?",
    },
    {
        "intent": "portfolio_brief",
        "query": "Summarize the charge-off rate trends and their key drivers.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_failure(status_code: int, response_body: dict) -> str:
    """Derive failure mode from HTTP status + response body."""
    detail = response_body.get("detail") or {}
    if isinstance(detail, str):
        error_code = detail
    else:
        error_code = detail.get("error", "") if isinstance(detail, dict) else ""

    # Map HTTP error codes to taxonomy
    if status_code == 503:
        return LLM_ERROR
    if status_code == 422:
        return GROUNDING_FAILED
    if status_code >= 500:
        return LLM_ERROR

    # 200 but failure conditions
    if status_code == 200:
        if response_body.get("needs_clarification"):
            return CLARIFICATION_LOOP
        conf = response_body.get("confidence_score", 1.0)
        answer = response_body.get("answer", "")
        rt = response_body.get("reasoning_trace") or {}
        suppressed = rt.get("suppressed_claims") or []
        retrieval_method = rt.get("retrieval_method", "")

        if not answer and suppressed:
            return HALLUCINATION_SUPPRESSED
        if conf < MIN_CONFIDENCE:
            return GROUNDING_FAILED
        if "INSUFFICIENT_RETRIEVAL" in error_code:
            return RETRIEVAL_FAILED
        if "COMPLIANCE_FAILED" in error_code:
            return COMPLIANCE_BLOCKED

    # Map by error code string if present
    code_map = {
        "INSUFFICIENT_RETRIEVAL": RETRIEVAL_FAILED,
        "RETRIEVAL_FAILED": RETRIEVAL_FAILED,
        "INSUFFICIENT_GROUNDING": GROUNDING_FAILED,
        "GROUNDING_FAILED": GROUNDING_FAILED,
        "COMPLIANCE_FAILED": COMPLIANCE_BLOCKED,
        "ALL_PROVIDERS_UNAVAILABLE": LLM_ERROR,
        "PRIMARY_UNAVAILABLE": LLM_ERROR,
        "GRAPH_ERROR": LLM_ERROR,
    }
    for key, mode in code_map.items():
        if key in error_code.upper():
            return mode

    return LLM_ERROR


def _is_success(status_code: int, response_body: dict) -> bool:
    """True iff the response meets the task success definition."""
    if status_code != 200:
        return False
    if response_body.get("needs_clarification"):
        return False
    answer = response_body.get("answer", "")
    if not answer or not answer.strip():
        return False
    conf = float(response_body.get("confidence_score", 0.0))
    if conf < MIN_CONFIDENCE:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    intent: str
    query: str
    http_status: int
    succeeded: bool
    failure_mode: Optional[str]
    error_detail: str
    confidence_score: float
    latency_s: float


@dataclass
class TaskSuccessReport:
    total: int
    succeeded: int
    failed: int
    by_intent_type: dict[str, dict]
    by_failure_mode: dict[str, int]
    task_success_rate: float
    gate_status: str  # PASS | WARN | FAIL
    turn_results: list[TurnResult]

    def print_summary(self) -> None:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[self.gate_status]
        rate_pct = self.task_success_rate * 100
        print(f"\n{'─'*60}")
        print(f"Eval 1 — Task Success Rate: {rate_pct:.1f}%  {icon} {self.gate_status}")
        print(f"  {self.succeeded}/{self.total} succeeded")
        print(f"\n  By intent:")
        for intent, stats in sorted(self.by_intent_type.items()):
            intent_rate = stats["succeeded"] / max(stats["total"], 1) * 100
            print(f"    {intent:20s}  {stats['succeeded']}/{stats['total']}  ({intent_rate:.0f}%)")
        if self.by_failure_mode:
            print(f"\n  Failure modes:")
            for mode, count in sorted(self.by_failure_mode.items(), key=lambda x: -x[1]):
                print(f"    {mode:30s}  {count}")
        print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class TaskSuccessEval:
    test_suite: list[dict] = TEST_SUITE

    def run(self) -> TaskSuccessReport:
        print(f"[Eval 1] Task Success Rate — {len(self.test_suite)} prompts")
        print(f"  Endpoint : {ENDPOINT}")
        print(f"  Gate     : ≥85% PASS | 70-84% WARN | <70% FAIL\n")

        turn_results: list[TurnResult] = []
        failures_for_fixture: list[dict] = []

        with httpx.Client(timeout=TIMEOUT) as client:
            for i, case in enumerate(self.test_suite, 1):
                intent = case["intent"]
                query  = case["query"]
                print(f"  [{i:02d}/{len(self.test_suite)}] {intent:20s}  {query[:70]}...", end=" ", flush=True)

                t0 = time.perf_counter()
                try:
                    resp = client.post(
                        ENDPOINT,
                        json={"query": query},
                        headers={"X-Eval-Mode": "true"},
                    )
                    body = resp.json()
                    status = resp.status_code
                except Exception as exc:
                    body = {"detail": str(exc)}
                    status = 0
                latency = time.perf_counter() - t0

                success = _is_success(status, body)
                failure_mode = None if success else _classify_failure(status, body)
                conf = float(body.get("confidence_score", 0.0)) if status == 200 else 0.0
                error_detail = ""
                if not success:
                    detail = body.get("detail") or ""
                    if isinstance(detail, dict):
                        error_detail = detail.get("message", str(detail))
                    else:
                        error_detail = str(detail)[:200]

                icon = "✅" if success else "❌"
                print(f"{icon}  conf={conf:.2f}  {latency:.1f}s")
                if not success:
                    print(f"         → {failure_mode}: {error_detail[:100]}")

                turn = TurnResult(
                    intent=intent,
                    query=query,
                    http_status=status,
                    succeeded=success,
                    failure_mode=failure_mode,
                    error_detail=error_detail,
                    confidence_score=conf,
                    latency_s=latency,
                )
                turn_results.append(turn)

                if not success:
                    failures_for_fixture.append({
                        "question": query,
                        "intent": intent,
                        "failure_mode": failure_mode,
                        "http_status": status,
                        "error": error_detail,
                        "confidence_score": conf,
                    })

        # Aggregate
        total = len(turn_results)
        succeeded = sum(1 for t in turn_results if t.succeeded)
        failed = total - succeeded

        by_intent: dict[str, dict] = {}
        by_failure: dict[str, int] = {}
        for t in turn_results:
            if t.intent not in by_intent:
                by_intent[t.intent] = {"total": 0, "succeeded": 0, "failed": 0}
            by_intent[t.intent]["total"] += 1
            if t.succeeded:
                by_intent[t.intent]["succeeded"] += 1
            else:
                by_intent[t.intent]["failed"] += 1
                if t.failure_mode:
                    by_failure[t.failure_mode] = by_failure.get(t.failure_mode, 0) + 1

        rate = succeeded / total if total > 0 else 0.0
        if rate >= 0.85:
            gate = "PASS"
        elif rate >= 0.70:
            gate = "WARN"
        else:
            gate = "FAIL"

        report = TaskSuccessReport(
            total=total,
            succeeded=succeeded,
            failed=failed,
            by_intent_type=by_intent,
            by_failure_mode=by_failure,
            task_success_rate=rate,
            gate_status=gate,
            turn_results=turn_results,
        )

        # Persist failure fixtures for CI regression detection
        fixture_path = FIXTURES_DIR / "task_success_failures.json"
        with open(fixture_path, "w") as f:
            json.dump(failures_for_fixture, f, indent=2)
        if failures_for_fixture:
            print(f"\n  [!] Failure fixtures written → {fixture_path}")

        report.print_summary()
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eval_ = TaskSuccessEval()
    report = eval_.run()
    sys.exit(0 if report.gate_status == "PASS" else 1)
