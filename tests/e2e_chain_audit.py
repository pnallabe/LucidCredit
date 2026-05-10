#!/usr/bin/env python3
"""
tests/e2e_chain_audit.py
========================
End-to-end audit of the full analytics chain:

  LucidCredit NL question
      → decomposition (nodes.py _decompose_broad_question)
      → analytics_api_tool.ask_analytics (HTTP POST /v1/analytics/s2s/ask)
          → credit-risk-platform NL→SQL (Azure OpenAI)
          → BigQuery execution
          → structured rows
      → reason_node (Azure OpenAI NL narrative)
      → structured answer with citations + SQL audit trail

Usage (both services must be running):
  python3 tests/e2e_chain_audit.py
  python3 tests/e2e_chain_audit.py --lucid-only   # skip analytics API stage tests
  python3 tests/e2e_chain_audit.py --api-only      # skip LucidCredit full-chain tests
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ── Service endpoints ────────────────────────────────────────────────────────
ANALYTICS_BASE = "http://localhost:8001"
LUCID_BASE     = "http://localhost:8090"
ANALYTICS_KEY  = "dev-analytics-key"

ANALYTICS_HEADERS = {
    "x-service-key": ANALYTICS_KEY,
    "Content-Type": "application/json",
}

# ── Stage definitions for analytics API (NL→SQL→BQ) ─────────────────────────
# Each stage tests a different SQL/BQ capability so failures are pinpointed.
STAGE_QUERIES: list[tuple[str, str]] = [
    (
        "1 · Simple aggregate",
        "What is the total outstanding loan balance across all funded personal loans?",
    ),
    (
        "2 · INT64 nanosecond timestamps (org_balance_sheet)",
        "Show the monthly delinquency rate trend over all available history by product type.",
    ),
    (
        "3 · Income-statement table routing (org_income_statement)",
        "Show the charge-off rate and net charge-off rate by quarter across all available history.",
    ),
    (
        "4 · Cross-table approval metric (decision_registry / applications)",
        "What is the approval rate by FICO tier for personal loan applications?",
    ),
    (
        "5 · Weighted formula derivation (weighted avg APR)",
        "What is the weighted average APR of funded personal loans weighted by current outstanding balance?",
    ),
    (
        "6 · Native DATE column origination (not INT64)",
        "How many personal loans were originated per month in 2023 and 2024?",
    ),
    (
        "7 · Multi-dimensional GROUP BY (product × channel)",
        "Show loan count and total funded amount by product type and origination channel.",
    ),
    (
        "8 · Geographic segmentation (state-level rollup)",
        "Show total outstanding balance and number of active loans by state for personal loans.",
    ),
    (
        "9 · Payment behaviour (personal_loan_payments table)",
        "Show the monthly payment count for personal loans by month over all available history.",
    ),
    (
        "10 · CAGR derivation (first vs. last period window function)",
        "What is the compound annual growth rate (CAGR) of the total outstanding portfolio balance?",
    ),
    (
        "11 · Net interest income from loan_monthly_ledger",
        "Show total net interest income per month from the loan monthly ledger for all product types.",
    ),
    (
        "12 · Vintage / cohort cumulative default",
        "Show loan count and average origination amount by origination year for personal loans, ordered oldest to newest.",
    ),
]

# ── Full-chain queries sent to LucidCredit /v1/query/analyst ─────────────────
# These exercise the entire NL→SQL→BQ→NL pipeline.
FULL_CHAIN_QUERIES: list[tuple[str, str]] = [
    (
        "Trigger: audit phrase → decomposition",
        "audit the bigquery integration and run end to end chain of analysis",
    ),
    (
        "Single: delinquency rate (INT64 timestamp table)",
        "What is the current delinquency rate across the portfolio?",
    ),
    (
        "Single: charge-off trend by quarter",
        "Show the charge-off rate trend by quarter across all available history.",
    ),
    (
        "Single: approval rate by FICO tier",
        "What is the approval rate by FICO tier for personal loans?",
    ),
    (
        "Single: total outstanding balance by product type",
        "What is the total outstanding loan balance by product type?",
    ),
]


# ── Result containers ─────────────────────────────────────────────────────────
@dataclass
class StageResult:
    stage: str
    question: str
    status: str          # "PASS" | "WARN" | "FAIL"
    http_code: int = 0
    row_count: int = 0
    sql_snippet: str = ""
    error: str = ""
    latency_ms: int = 0
    assumed_defaults: list[str] = field(default_factory=list)


@dataclass
class ChainResult:
    label: str
    question: str
    status: str
    http_code: int = 0
    answer_snippet: str = ""
    sql_count: int = 0
    sql_queries: list[str] = field(default_factory=list)
    confidence: float = 0.0
    citation_count: int = 0
    latency_ms: int = 0
    error: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(label: str, char: str = "─", width: int = 72) -> str:
    return f"\n{char * 4}  {label}  " + char * max(0, width - len(label) - 6)


def _wrap(text: str, indent: int = 6) -> str:
    return textwrap.fill(text, width=100, initial_indent=" " * indent, subsequent_indent=" " * indent)


# ── Analytics API stage tests (NL → SQL → BQ) ─────────────────────────────────

def run_stage_test(client: httpx.Client, stage: str, question: str) -> StageResult:
    t0 = time.monotonic()
    try:
        resp = client.post(
            f"{ANALYTICS_BASE}/v1/analytics/s2s/ask",
            json={"question": question},
            headers=ANALYTICS_HEADERS,
            timeout=90,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()

        if resp.status_code != 200:
            return StageResult(
                stage=stage, question=question, status="FAIL",
                http_code=resp.status_code,
                error=(data.get("detail") or str(data))[:200],
                latency_ms=latency_ms,
            )

        if data.get("needs_clarification"):
            items = [i.get("id", "?") for i in (data.get("clarification_items") or [])]
            return StageResult(
                stage=stage, question=question, status="WARN",
                http_code=200, latency_ms=latency_ms,
                error=f"needs_clarification: {items}",
            )

        rows = data.get("rows") or []
        row_count = data.get("row_count", len(rows))
        sql = (data.get("generated_sql") or "").replace("\n", " ")[:120]
        assumed = data.get("assumed_defaults") or []

        st = "PASS" if row_count > 0 else "WARN"
        err = "" if row_count > 0 else "0 rows returned (table may be empty)"
        return StageResult(
            stage=stage, question=question, status=st,
            http_code=200, row_count=row_count,
            sql_snippet=sql, latency_ms=latency_ms,
            assumed_defaults=assumed, error=err,
        )

    except Exception as exc:
        return StageResult(
            stage=stage, question=question, status="FAIL",
            latency_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc)[:200],
        )


# ── LucidCredit full-chain test (NL → SQL → BQ → NL) ─────────────────────────

def run_chain_test(client: httpx.Client, label: str, question: str) -> ChainResult:
    t0 = time.monotonic()
    try:
        resp = client.post(
            f"{LUCID_BASE}/v1/query/analyst",
            json={"question": question, "query": question},
            timeout=300,  # decomposed queries fan out to 12 BQ calls
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()

        if resp.status_code != 200:
            return ChainResult(
                label=label, question=question, status="FAIL",
                http_code=resp.status_code, latency_ms=latency_ms,
                error=(data.get("detail") or str(data))[:300],
            )

        answer = data.get("answer", "")
        sqls   = data.get("sql_queries_executed", [])
        confi  = float(data.get("confidence_score", 0))
        cites  = data.get("citations", [])

        st = "PASS" if answer and confi > 0 else "WARN"
        return ChainResult(
            label=label, question=question, status=st,
            http_code=200, latency_ms=latency_ms,
            answer_snippet=answer[:300],
            sql_count=len(sqls),
            sql_queries=sqls[:3],  # show first 3 SQL queries
            confidence=confi,
            citation_count=len(cites),
        )

    except Exception as exc:
        return ChainResult(
            label=label, question=question, status="FAIL",
            latency_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc)[:300],
        )


# ── Service health gate ───────────────────────────────────────────────────────

def check_services(client: httpx.Client) -> tuple[bool, bool]:
    analytics_ok = lucid_ok = False
    try:
        r = client.get(f"{ANALYTICS_BASE}/v1/health", timeout=5)
        analytics_ok = r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        pass
    try:
        r = client.get(f"{LUCID_BASE}/v1/health", timeout=5)
        lucid_ok = r.status_code == 200
    except Exception:
        pass
    return analytics_ok, lucid_ok


# ── Report printer ────────────────────────────────────────────────────────────

ICONS = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}


def print_stage_report(results: list[StageResult]) -> int:
    print(_bar("STAGE 1 — Analytics API  (NL → SQL → BigQuery)", "═"))
    print(f"  Endpoint : POST {ANALYTICS_BASE}/v1/analytics/s2s/ask")
    print(f"  Purpose  : Validates that each SQL capability works against real BigQuery\n")

    fails = 0
    for r in results:
        icon = ICONS[r.status]
        print(f"  {icon} {r.stage}  ({r.latency_ms} ms)")
        print(f"       Q : {r.question[:90]}")
        if r.sql_snippet:
            print(f"     SQL : {r.sql_snippet}...")
        if r.row_count:
            print(f"    ROWS : {r.row_count}")
        if r.assumed_defaults:
            print(f"DEFAULTS : {'; '.join(r.assumed_defaults)}")
        if r.error:
            print(f"   ERROR : {r.error}")
        print()
        if r.status == "FAIL":
            fails += 1
    return fails


def print_chain_report(results: list[ChainResult]) -> int:
    print(_bar("STAGE 2 — LucidCredit Full Chain  (NL → SQL → BQ → NL narrative)", "═"))
    print(f"  Endpoint : POST {LUCID_BASE}/v1/query/analyst")
    print(f"  Purpose  : Validates the complete end-to-end pipeline\n")

    fails = 0
    for r in results:
        icon = ICONS[r.status]
        print(f"  {icon} {r.label}  ({r.latency_ms} ms)")
        print(f"       Q : {r.question[:90]}")
        if r.error:
            print(f"   ERROR : {r.error}")
        else:
            print(f"  CONFI : {r.confidence:.3f}")
            print(f"   SQL# : {r.sql_count} queries executed")
            print(f"  CITES : {r.citation_count} citations")
            if r.sql_queries:
                print(f"  SQL[0]: {r.sql_queries[0][:110]}...")
            print(f" ANSWER : {r.answer_snippet[:200]}...")
        print()
        if r.status == "FAIL":
            fails += 1
    return fails


def print_summary(stage_results: list[StageResult], chain_results: list[ChainResult]) -> None:
    print(_bar("AUDIT SUMMARY", "═"))
    total = len(stage_results) + len(chain_results)
    passed = sum(1 for r in stage_results + chain_results if r.status == "PASS")  # type: ignore[operator]
    warned = sum(1 for r in stage_results + chain_results if r.status == "WARN")  # type: ignore[operator]
    failed = sum(1 for r in stage_results + chain_results if r.status == "FAIL")  # type: ignore[operator]

    print(f"\n  Total checks : {total}")
    print(f"  ✅ PASS      : {passed}")
    print(f"  ⚠️  WARN      : {warned}")
    print(f"  ❌ FAIL      : {failed}")

    if stage_results:
        avg_latency = sum(r.latency_ms for r in stage_results) // len(stage_results)
        print(f"\n  Avg analytics-API latency : {avg_latency} ms/query")
    if chain_results:
        avg_chain = sum(r.latency_ms for r in chain_results) // len(chain_results)
        print(f"  Avg full-chain latency    : {avg_chain} ms/query")

    verdict = "PASS" if failed == 0 else ("WARN" if warned and not failed else "FAIL")
    print(f"\n  Overall : {ICONS[verdict]} {verdict}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LucidCredit end-to-end chain audit")
    parser.add_argument("--lucid-only", action="store_true", help="Skip analytics API stage tests")
    parser.add_argument("--api-only",   action="store_true", help="Skip LucidCredit full-chain tests")
    args = parser.parse_args()

    print("\n" + "═" * 72)
    print("  LucidCredit — End-to-End Analytics Chain Audit")
    print("  NL question → SQL decomposition → BigQuery → NL narrative")
    print("═" * 72)

    with httpx.Client() as client:
        # ── Health gate ────────────────────────────────────────────────────
        analytics_ok, lucid_ok = check_services(client)
        print(f"\n  Analytics API  (:{8001}) : {'✅ UP' if analytics_ok else '❌ DOWN'}")
        print(f"  LucidCredit    (:{8090}) : {'✅ UP' if lucid_ok else '❌ DOWN'}\n")

        if not analytics_ok and not args.lucid_only:
            print("  ❌ Analytics API is down. Start it with: bash start_analytics_api.sh")
            sys.exit(1)
        if not lucid_ok and not args.api_only:
            print("  ❌ LucidCredit is down. Start it with: bash start_lucidcredit_backend.sh")
            sys.exit(1)

        stage_results: list[StageResult] = []
        chain_results: list[ChainResult] = []

        # ── Stage 1: analytics API (NL→SQL→BQ) ────────────────────────────
        if not args.lucid_only and analytics_ok:
            print(_bar("Running analytics API stage tests …"))
            for stage, question in STAGE_QUERIES:
                print(f"  → {stage} …", flush=True)
                result = run_stage_test(client, stage, question)
                stage_results.append(result)

        # ── Stage 2: full chain (NL→SQL→BQ→NL) ────────────────────────────
        if not args.api_only and lucid_ok:
            print(_bar("Running LucidCredit full-chain tests …"))
            for label, question in FULL_CHAIN_QUERIES:
                print(f"  → {label} …", flush=True)
                result = run_chain_test(client, label, question)
                chain_results.append(result)

    # ── Print reports ──────────────────────────────────────────────────────
    print()
    if stage_results:
        fails_s = print_stage_report(stage_results)
    else:
        fails_s = 0

    if chain_results:
        fails_c = print_chain_report(chain_results)
    else:
        fails_c = 0

    print_summary(stage_results, chain_results)

    sys.exit(1 if (fails_s + fails_c) > 0 else 0)


if __name__ == "__main__":
    main()
