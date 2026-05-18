"""
tests/agent_evals/eval_operational.py
=======================================
Eval 7 — Operational Evals: Latency & Cost

7a. Task Completion Time (end-to-end latency)
    Tier 1: Simple retrieval (vector only)    <= 4.0s p95
    Tier 2: Data query (NL2SQL + BQ)          <= 12.0s p95
    Tier 3: Multi-step (broad decomposed)     <= 25.0s p95
    Tier 4: Applicant comms (+ compliance)    <= 8.0s p95

7b. Cost per Task
    Parse structured log events "llm_token_usage" from /tmp/lucidcredit_backend.log.
    gpt-4.1-2025-04-14: $2.00/1M input, $8.00/1M output tokens
    text-embedding-3-large: $0.13/1M tokens
    Gate: mean_cost <= $0.04 PASS | $0.04-$0.08 WARN | > $0.08 FAIL
    Runaway: any single query > $0.20 = hard FAIL

Run:
    python tests/agent_evals/eval_operational.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT  = 90.0
LOG_PATH = Path("/tmp/lucidcredit_backend.log")

# ─────────────────────────────────────────────────────────────────────────────
# SLO definitions per tier
# ─────────────────────────────────────────────────────────────────────────────

TIERS: list[dict] = [
    {
        "tier": 1,
        "name": "Simple retrieval (vector only)",
        "slo_p95_s": 4.0,
        "slo_p50_s": 2.5,
        "prompts": [
            "What are the ECOA adverse action notice requirements?",
            "What FCRA section 615 rights must be disclosed to applicants?",
            "What are the SR 11-7 model risk management guidelines?",
            "What credit policy governs underwriting of personal loans?",
            "What compliance disclosures are required for adverse action?",
        ],
    },
    {
        "tier": 2,
        "name": "Data query (NL2SQL + BQ round-trip)",
        "slo_p95_s": 12.0,
        "slo_p50_s": 7.0,
        "prompts": [
            "What is the total outstanding loan balance across the portfolio?",
            "What is the current delinquency rate for personal loans?",
            "Show the charge-off rate by quarter.",
            "What is the approval rate by FICO tier?",
            "How many loans were originated per month in 2023?",
        ],
    },
    {
        "tier": 3,
        "name": "Multi-step (decomposed broad query)",
        "slo_p95_s": 25.0,
        "slo_p50_s": 15.0,
        "prompts": [
            "Provide a complete portfolio health summary including delinquency, charge-off, and balance metrics.",
            "Show a comprehensive credit quality overview: NPL ratio, average PD, 30+ and 90+ DPD rates.",
            "Give me an executive summary of portfolio risk covering delinquency, charge-off, FICO distribution, and balance.",
            "Run a full risk dashboard covering all key credit risk indicators.",
            "Summarize the overall portfolio with exposure by product, delinquency trends, and charge-off rates.",
        ],
    },
    {
        "tier": 4,
        "name": "Applicant comms (retrieval + compliance check)",
        "slo_p95_s": 8.0,
        "slo_p50_s": 5.0,
        "prompts": [
            "Draft an adverse action notice for a declined application due to DTI exceeding 45%.",
            "Write a decline notice citing credit score of 575 below our minimum of 620.",
            "Inform the applicant of their right to a free consumer report copy under FCRA section 615.",
            "Draft a decline notice citing insufficient employment history of less than 12 months.",
            "Write an adverse action notice citing derogatory accounts in the past 24 months.",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Azure OpenAI pricing (current as of May 2026)
# ─────────────────────────────────────────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-2025-04-14": {
        "input_per_1m":  2.00,
        "output_per_1m": 8.00,
    },
    "text-embedding-3-large": {
        "input_per_1m": 0.13,
        "output_per_1m": 0.0,
    },
}
_DEFAULT_MODEL_PRICING = _PRICING["gpt-4.1-2025-04-14"]
_RUNAWAY_COST_THRESHOLD = 0.20


def _price_tokens(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost for a single LLM call."""
    pricing = _PRICING.get(model_name, _DEFAULT_MODEL_PRICING)
    cost = (
        prompt_tokens * pricing["input_per_1m"] / 1_000_000
        + completion_tokens * pricing["output_per_1m"] / 1_000_000
    )
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing for token usage events
# ─────────────────────────────────────────────────────────────────────────────

_LOG_TOKEN_EVENT_RE = re.compile(r'"event"\s*:\s*"llm_token_usage"')
_LOG_SESSION_RE     = re.compile(r'"session_id"\s*:\s*"([^"]+)"')
_LOG_PROVIDER_RE    = re.compile(r'"provider"\s*:\s*"([^"]+)"')
_LOG_PROMPT_RE      = re.compile(r'"prompt_tokens"\s*:\s*(\d+)')
_LOG_COMPLETION_RE  = re.compile(r'"completion_tokens"\s*:\s*(\d+)')
_LOG_TOTAL_RE       = re.compile(r'"total_tokens"\s*:\s*(\d+)')


def _parse_token_log(session_id: str) -> Optional[dict]:
    """
    Parse /tmp/lucidcredit_backend.log for a llm_token_usage event
    matching the given session_id.
    Returns dict with prompt_tokens, completion_tokens, total_tokens, provider
    or None if not found.
    """
    if not LOG_PATH.exists():
        return None
    try:
        text = LOG_PATH.read_text(errors="replace")
        lines = text.splitlines()
    except OSError:
        return None

    for line in reversed(lines):  # most recent first
        if not _LOG_TOKEN_EVENT_RE.search(line):
            continue
        m_sid = _LOG_SESSION_RE.search(line)
        if not m_sid or session_id not in m_sid.group(1):
            continue
        m_prov  = _LOG_PROVIDER_RE.search(line)
        m_prom  = _LOG_PROMPT_RE.search(line)
        m_comp  = _LOG_COMPLETION_RE.search(line)
        m_total = _LOG_TOTAL_RE.search(line)
        return {
            "provider":          (m_prov.group(1)  if m_prov  else "unknown"),
            "prompt_tokens":     int(m_prom.group(1))  if m_prom  else 0,
            "completion_tokens": int(m_comp.group(1))  if m_comp  else 0,
            "total_tokens":      int(m_total.group(1)) if m_total else 0,
        }
    return None


def _extract_model_name(provider_str: str) -> str:
    """Extract model name from 'azure:gpt-4.1-2025-04-14' or similar."""
    for known_model in _PRICING:
        if known_model in provider_str:
            return known_model
    return "gpt-4.1-2025-04-14"  # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TierResult:
    tier: int
    name: str
    slo_p95_s: float
    slo_p50_s: float
    latencies: list[float] = field(default_factory=list)
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    slo_p95_status: str = "UNKNOWN"  # PASS | WARN | FAIL
    slo_p50_status: str = "UNKNOWN"

    def compute_percentiles(self) -> None:
        if not self.latencies:
            return
        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)
        def pct(p: float) -> float:
            idx = min(int(p / 100 * n), n - 1)
            return sorted_lat[idx]
        self.p50 = pct(50)
        self.p95 = pct(95)
        self.p99 = pct(99)
        self.slo_p95_status = "PASS" if self.p95 <= self.slo_p95_s else "WARN"
        self.slo_p50_status  = "PASS" if self.p50 <= self.slo_p50_s else "FAIL"


@dataclass
class CostResult:
    total_queries: int
    total_cost_usd: float
    mean_cost_usd: float
    max_cost_usd: float
    runaway_queries: int
    total_prompt_tokens: int
    total_completion_tokens: int
    cost_by_tier: dict[int, float] = field(default_factory=dict)
    gate: str = "UNKNOWN"

    def compute_gate(self) -> None:
        if self.runaway_queries > 0:
            self.gate = "FAIL"
        elif self.mean_cost_usd <= 0.04:
            self.gate = "PASS"
        elif self.mean_cost_usd <= 0.08:
            self.gate = "WARN"
        else:
            self.gate = "FAIL"


@dataclass
class OperationalReport:
    tier_results: list[TierResult]
    cost_result: Optional[CostResult]
    latency_gate: str  # PASS | WARN | FAIL (worst of all tier gates)
    cost_gate: str

    def print_summary(self) -> None:
        icons = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "UNKNOWN": "⚪"}
        print(f"\n{'─'*60}")
        print(f"Eval 7 — Operational: Latency & Cost")
        print(f"\n  [7a] Latency per Tier:")
        print(f"  {'Tier':6s} {'Name':35s} {'p50':6s} {'p95':6s} {'p99':6s} {'SLO_p95':8s} {'Status'}")
        for t in self.tier_results:
            slo_icon = icons.get(t.slo_p95_status, "⚪")
            p50_icon = icons.get(t.slo_p50_status, "⚪")
            print(
                f"  {t.tier:6d} {t.name[:35]:35s} "
                f"{t.p50:5.1f}s {t.p95:5.1f}s {t.p99:5.1f}s "
                f"{t.slo_p95_s:6.1f}s   {slo_icon} p95{t.slo_p95_status}  {p50_icon} p50{t.slo_p50_status}"
            )
        print(f"\n  Overall Latency: {icons.get(self.latency_gate, '⚪')} {self.latency_gate}")

        if self.cost_result:
            cr = self.cost_result
            print(f"\n  [7b] Cost per Task:")
            print(f"    Mean cost/task:    ${cr.mean_cost_usd:.4f}  (gate ≤$0.04)")
            print(f"    Max cost/task:     ${cr.max_cost_usd:.4f}")
            print(f"    Total queries:     {cr.total_queries}")
            print(f"    Total tokens:      {cr.total_prompt_tokens} prompt + {cr.total_completion_tokens} completion")
            print(f"    Runaway queries:   {cr.runaway_queries} (>${_RUNAWAY_COST_THRESHOLD:.2f} each — HARD FAIL if > 0)")
            print(f"    Gate: {icons.get(cr.gate, '⚪')} {cr.gate}")
        else:
            print(f"\n  [7b] Cost: ⚪ SKIPPED (no llm_token_usage events found in log)")
        print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class OperationalEval:
    tiers: list[dict] = TIERS

    def run(self) -> OperationalReport:
        print(f"[Eval 7] Operational — Latency & Cost")
        print(f"  Endpoint : {ENDPOINT}\n")

        tier_results: list[TierResult] = []
        all_token_data: list[dict] = []

        # ── 7a: Latency measurement ───────────────────────────────────────────
        with httpx.Client(timeout=TIMEOUT) as client:
            for tier_spec in self.tiers:
                tier_num  = tier_spec["tier"]
                tier_name = tier_spec["name"]
                slo_p95   = tier_spec["slo_p95_s"]
                slo_p50   = tier_spec["slo_p50_s"]
                prompts   = tier_spec["prompts"]

                print(f"  [Tier {tier_num}] {tier_name}")
                print(f"    SLO: p95≤{slo_p95}s / p50≤{slo_p50}s  ({len(prompts)} prompts)")

                tier_result = TierResult(
                    tier=tier_num,
                    name=tier_name,
                    slo_p95_s=slo_p95,
                    slo_p50_s=slo_p50,
                )

                for i, prompt in enumerate(prompts, 1):
                    print(f"    [{i}] {prompt[:65]}...", end=" ", flush=True)

                    t0 = time.perf_counter()
                    try:
                        resp = client.post(
                            ENDPOINT,
                            json={"query": prompt},
                            headers={"X-Eval-Mode": "true"},
                        )
                        body = resp.json()
                        status = resp.status_code
                    except Exception as exc:
                        body = {"error": str(exc)}
                        status = 0
                    latency = time.perf_counter() - t0

                    tier_result.latencies.append(latency)

                    # Collect session_id for cost tracking
                    session_id = str(body.get("session_id", ""))
                    if session_id:
                        # Wait briefly for log to flush, then parse
                        time.sleep(0.3)
                        token_data = _parse_token_log(session_id)
                        if token_data:
                            token_data["session_id"] = session_id
                            token_data["tier"] = tier_num
                            token_data["query"] = prompt
                            token_data["latency_s"] = latency
                            all_token_data.append(token_data)

                    slo_icon = "✅" if latency <= slo_p95 else "❌"
                    print(f"{slo_icon}  {latency:.2f}s  HTTP={status}")

                tier_result.compute_percentiles()
                tier_results.append(tier_result)
                print(f"    → p50={tier_result.p50:.2f}s  p95={tier_result.p95:.2f}s  p99={tier_result.p99:.2f}s\n")

        # Determine overall latency gate
        p95_gate_worst = "PASS"
        for t in tier_results:
            if t.slo_p50_status == "FAIL":
                p95_gate_worst = "FAIL"
                break
            if t.slo_p95_status == "WARN" and p95_gate_worst != "FAIL":
                p95_gate_worst = "WARN"
        latency_gate = p95_gate_worst

        # ── 7b: Cost tracking ─────────────────────────────────────────────────
        print(f"  [7b] Cost Analysis ({len(all_token_data)} token events parsed from log)...")

        cost_result: Optional[CostResult] = None

        if all_token_data:
            costs_usd: list[float] = []
            runaway = 0
            total_prompt = 0
            total_completion = 0
            cost_by_tier: dict[int, float] = {}

            for td in all_token_data:
                provider = td.get("provider", "unknown")
                model = _extract_model_name(provider)
                prompt_tok  = td.get("prompt_tokens", 0)
                comp_tok    = td.get("completion_tokens", 0)
                cost        = _price_tokens(model, prompt_tok, comp_tok)

                costs_usd.append(cost)
                total_prompt     += prompt_tok
                total_completion += comp_tok
                tier_n            = td.get("tier", 0)
                cost_by_tier[tier_n] = cost_by_tier.get(tier_n, 0.0) + cost

                if cost > _RUNAWAY_COST_THRESHOLD:
                    runaway += 1
                    print(f"    ⚠  RUNAWAY: session={td.get('session_id', '?')[:8]}  cost=${cost:.4f}  tokens={prompt_tok}+{comp_tok}")

            mean_cost = sum(costs_usd) / len(costs_usd) if costs_usd else 0.0
            max_cost  = max(costs_usd) if costs_usd else 0.0

            cost_result = CostResult(
                total_queries=len(costs_usd),
                total_cost_usd=sum(costs_usd),
                mean_cost_usd=mean_cost,
                max_cost_usd=max_cost,
                runaway_queries=runaway,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                cost_by_tier=cost_by_tier,
            )
            cost_result.compute_gate()

            print(f"    → Mean=${mean_cost:.4f}  Max=${max_cost:.4f}  Runaway={runaway}  {cost_result.gate}")
        else:
            print(f"    ⚠  No llm_token_usage log events found in {LOG_PATH}")
            print(f"       Ensure nodes.py emits structlog 'llm_token_usage' event after LLM calls.")
            print(f"       See IMPLEMENTATION RULE 9 in the eval spec.")

        cost_gate = cost_result.gate if cost_result else "WARN"

        report = OperationalReport(
            tier_results=tier_results,
            cost_result=cost_result,
            latency_gate=latency_gate,
            cost_gate=cost_gate,
        )
        report.print_summary()
        return report


if __name__ == "__main__":
    eval_ = OperationalEval()
    report = eval_.run()
    gate = "FAIL" if report.latency_gate == "FAIL" or report.cost_gate == "FAIL" else (
        "WARN" if report.latency_gate == "WARN" or report.cost_gate == "WARN" else "PASS"
    )
    sys.exit(0 if gate == "PASS" else 1)
