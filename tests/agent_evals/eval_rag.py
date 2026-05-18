"""
tests/agent_evals/eval_rag.py
================================
Eval 5 — RAG Evals: Faithfulness, Context Relevance, IR Metrics

5a. Faithfulness / Grounding
    faithfulness_rate = 1 - (suppressed_claims / total_sentences)
    Gate: >= 0.95 PASS

5b. Context Relevance & Sufficiency
    context_relevance = RELEVANT / total retrieved items
    context_sufficiency = retrieval_sufficient (inferred from answer non-empty + conf >= 0.75)
    Gate: >= 0.70 relevance AND sufficiency for all answerable questions

5c. Retrieval IR Metrics — Precision@K, Recall@K, MRR
    Uses rag_ground_truth.json oracle.
    Gate: MRR >= 0.70, Precision@5 >= 0.60

Note: cross-link with test_reasoning_trace.py dimensions (context_retrieval,
traceability) — this eval adds faithfulness and IR metrics that aren't in that file.

Run:
    python tests/agent_evals/eval_rag.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 90.0
MIN_CONFIDENCE = 0.75

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ─────────────────────────────────────────────────────────────────────────────
# 5a — Faithfulness test cases (20 analyst_query questions)
# ─────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_CASES: list[str] = [
    "What is the total outstanding loan balance across the portfolio?",
    "What is the current delinquency rate for personal loans?",
    "Show the charge-off rate by quarter.",
    "What is the approval rate by FICO tier?",
    "What is the average debt-to-income ratio at origination?",
    "How many loans were originated per month in 2023?",
    "What is the weighted average APR across personal loans?",
    "Show total outstanding balance by state for personal loans.",
    "What is the average FICO score at origination?",
    "What is the 30-day delinquency rate trend over the past 12 months?",
    "What is the net charge-off rate by product type?",
    "Show the loan count and total funded amount by product type.",
    "What is the total exposure for accounts opened after January 2023?",
    "What is the cumulative default rate by origination vintage year?",
    "What is the prepayment rate for personal loans by month?",
    "Show the monthly delinquency trend for all product types.",
    "What is the average credit score across all borrowers?",
    "What is the total outstanding balance for accounts 30+ days past due?",
    "What is the loan distribution by credit grade?",
    "What is the compound annual growth rate of the total outstanding portfolio?",
]

# ─────────────────────────────────────────────────────────────────────────────
# 5b — Context Relevance test cases (20 questions with known data coverage)
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_RELEVANCE_CASES: list[str] = [
    "What is the total outstanding loan balance?",
    "What is the delinquency rate for personal loans?",
    "What are the ECOA adverse action notice requirements?",
    "What is the charge-off rate by quarter?",
    "What FCRA section 615 rights must be disclosed?",
    "What is the approval rate by FICO tier?",
    "What is the average DTI at origination?",
    "How many loans were originated per month in 2023?",
    "What is the weighted average APR of personal loans?",
    "Show total outstanding balance by state.",
    "What is the average FICO score at origination?",
    "What is the 30-day delinquency trend over 12 months?",
    "What is the net charge-off rate by product type?",
    "What are the SR 11-7 model risk management guidelines?",
    "Show total outstanding balance by product type.",
    "What is the monthly loan origination count in 2022?",
    "What is the average credit utilization for FICO < 620 borrowers?",
    "What compliance disclosures are required for adverse action?",
    "What is the portfolio delinquency rate by credit grade?",
    "Show the loan count and funded amount by origination channel.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into atomic sentences."""
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in raw if s.strip()]


def _compute_faithfulness(response: dict) -> tuple[float, int, int]:
    """
    faithfulness_rate = 1 - (suppressed / total_sentences)
    Returns (rate, total_sentences, suppressed_count)
    """
    rt = response.get("reasoning_trace") or {}
    suppressed = rt.get("suppressed_claims") or []
    answer = response.get("answer", "") or ""

    # Total sentences = grounded answer sentences + suppressed
    answer_sentences = _split_sentences(answer)
    total = len(answer_sentences) + len(suppressed)
    if total == 0:
        return 1.0, 0, 0
    rate = 1.0 - (len(suppressed) / total)
    return rate, total, len(suppressed)


def _compute_context_relevance(response: dict) -> tuple[float, int, int]:
    """
    context_relevance = RELEVANT / total retrieved
    Returns (relevance_rate, total_items, relevant_count)
    """
    rt = response.get("reasoning_trace") or {}
    items = rt.get("retrieved_context") or []
    if not items:
        return 0.0, 0, 0
    relevant = sum(1 for item in items if item.get("relevance") == "RELEVANT")
    return relevant / len(items), len(items), relevant


def _is_retrieval_sufficient(response: dict) -> bool:
    """Infer retrieval sufficiency from answer + confidence."""
    answer = response.get("answer", "") or ""
    conf = float(response.get("confidence_score", 0.0))
    return bool(answer.strip()) and conf >= MIN_CONFIDENCE


# ─────────────────────────────────────────────────────────────────────────────
# 5c — IR Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _load_ground_truth() -> list[dict]:
    path = FIXTURES_DIR / "rag_ground_truth.json"
    with open(path) as f:
        return json.load(f)


def _extract_retrieved_refs(response: dict, k: int) -> list[str]:
    """Extract up to K retrieved chunk IDs / source_refs from the response."""
    refs: list[str] = []
    # From citations (grounded claims)
    for c in (response.get("citations") or [])[:k]:
        ref = c.get("source_ref", "")
        if ref and ref not in refs:
            refs.append(ref)
    # From reasoning_trace.retrieved_context
    rt = response.get("reasoning_trace") or {}
    for item in (rt.get("retrieved_context") or []):
        ref = item.get("source_ref", "")
        if ref and ref not in refs:
            refs.append(ref)
        if len(refs) >= k:
            break
    return refs[:k]


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """P@K: fraction of top-K retrieved that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if any(rel in r or r in rel for rel in relevant))
    return hits / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """R@K: fraction of relevant items that appear in top-K."""
    if not relevant:
        return 1.0  # vacuously true
    top_k = retrieved[:k]
    hits = sum(1 for rel in relevant if any(rel in r or r in rel for rel in top_k))
    return hits / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    """MRR: 1/rank of first relevant result (0 if none)."""
    for rank, ref in enumerate(retrieved, 1):
        if any(rel in ref or ref in rel for rel in relevant):
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGReport:
    # 5a
    faithfulness_rate: float
    faithfulness_gate: str
    faithfulness_details: list[dict]
    # 5b
    mean_context_relevance: float
    context_relevance_gate: str
    sufficiency_rate: float
    # 5c
    mean_precision_at5: float
    mean_precision_at10: float
    mean_recall_at10: float
    mean_mrr: float
    ir_gate: str
    ir_details: list[dict]

    @property
    def overall_gate(self) -> str:
        gates = [self.faithfulness_gate, self.context_relevance_gate, self.ir_gate]
        if all(g == "PASS" for g in gates):
            return "PASS"
        if any(g == "FAIL" for g in gates):
            return "FAIL"
        return "WARN"

    def print_summary(self) -> None:
        icons = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}
        print(f"\n{'─'*60}")
        print(f"Eval 5 — RAG Evals")
        print(
            f"  5a Faithfulness Rate:    {self.faithfulness_rate*100:.1f}%  "
            f"(gate ≥95%)  {icons[self.faithfulness_gate]} {self.faithfulness_gate}"
        )
        print(
            f"  5b Context Relevance:    {self.mean_context_relevance*100:.1f}%  "
            f"(gate ≥70%)  {icons[self.context_relevance_gate]} {self.context_relevance_gate}"
        )
        print(f"     Sufficiency Rate:    {self.sufficiency_rate*100:.1f}%  (should be 100% for answerable)")
        print(
            f"  5c MRR:                  {self.mean_mrr:.3f}  (gate ≥0.70)  "
            f"Precision@5: {self.mean_precision_at5:.3f}  (gate ≥0.60)  "
            f"{icons[self.ir_gate]} {self.ir_gate}"
        )
        print(f"     Precision@10:        {self.mean_precision_at10:.3f}")
        print(f"     Recall@10:           {self.mean_recall_at10:.3f}")
        print(f"\n  Overall: {icons[self.overall_gate]} {self.overall_gate}")
        print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEval:
    faithfulness_cases: list[str] = FAITHFULNESS_CASES
    context_cases: list[str] = CONTEXT_RELEVANCE_CASES

    def run(self) -> RAGReport:
        print(f"[Eval 5] RAG Evals — Faithfulness, Context Relevance, IR Metrics")
        print(f"  Endpoint : {ENDPOINT}\n")

        # ── 5a: Faithfulness ─────────────────────────────────────────────────
        print(f"  [5a] Faithfulness ({len(self.faithfulness_cases)} cases)...")
        faith_rates: list[float] = []
        faith_details: list[dict] = []

        with httpx.Client(timeout=TIMEOUT) as client:
            for i, q in enumerate(self.faithfulness_cases, 1):
                print(f"    [{i:02d}] {q[:65]}...", end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    resp = client.post(ENDPOINT, json={"query": q}, headers={"X-Eval-Mode": "true"})
                    body = resp.json()
                    status = resp.status_code
                except Exception as exc:
                    body = {}
                    status = 0
                latency = time.perf_counter() - t0

                if status != 200:
                    faith_rates.append(0.0)
                    print(f"HTTP {status} — skipped")
                    continue

                rate, total_s, suppressed = _compute_faithfulness(body)
                faith_rates.append(rate)
                faith_details.append({
                    "question": q,
                    "faithfulness_rate": rate,
                    "total_sentences": total_s,
                    "suppressed_count": suppressed,
                })
                icon = "✅" if rate >= 0.95 else ("⚠️" if rate >= 0.80 else "❌")
                print(f"{icon}  faith={rate:.3f}  total_s={total_s}  suppressed={suppressed}  {latency:.1f}s")

        mean_faith = sum(faith_rates) / len(faith_rates) if faith_rates else 0.0
        faith_gate = "PASS" if mean_faith >= 0.95 else ("WARN" if mean_faith >= 0.80 else "FAIL")
        print(f"    → Mean Faithfulness: {mean_faith*100:.1f}%  {faith_gate}")

        # ── 5b: Context Relevance & Sufficiency ──────────────────────────────
        print(f"\n  [5b] Context Relevance ({len(self.context_cases)} cases)...")
        relevance_rates: list[float] = []
        sufficiency_flags: list[bool] = []

        with httpx.Client(timeout=TIMEOUT) as client:
            for i, q in enumerate(self.context_cases, 1):
                print(f"    [{i:02d}] {q[:65]}...", end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    resp = client.post(ENDPOINT, json={"query": q}, headers={"X-Eval-Mode": "true"})
                    body = resp.json()
                    status = resp.status_code
                except Exception as exc:
                    body = {}
                    status = 0
                latency = time.perf_counter() - t0

                if status != 200:
                    relevance_rates.append(0.0)
                    sufficiency_flags.append(False)
                    print(f"HTTP {status} — skipped")
                    continue

                rel_rate, total_ctx, relevant_count = _compute_context_relevance(body)
                sufficient = _is_retrieval_sufficient(body)
                relevance_rates.append(rel_rate)
                sufficiency_flags.append(sufficient)
                icon = "✅" if rel_rate >= 0.70 and sufficient else ("⚠️" if rel_rate >= 0.50 else "❌")
                print(f"{icon}  rel={rel_rate:.2f}  ctx={total_ctx}(rel={relevant_count})  sufficient={'Y' if sufficient else 'N'}  {latency:.1f}s")

        mean_rel = sum(relevance_rates) / len(relevance_rates) if relevance_rates else 0.0
        suf_rate = sum(sufficiency_flags) / len(sufficiency_flags) if sufficiency_flags else 0.0
        rel_gate = "PASS" if mean_rel >= 0.70 and suf_rate >= 0.90 else ("WARN" if mean_rel >= 0.50 else "FAIL")
        print(f"    → Mean Relevance: {mean_rel*100:.1f}%  Sufficiency: {suf_rate*100:.1f}%  {rel_gate}")

        # ── 5c: IR Metrics — Precision@K, Recall@K, MRR ──────────────────────
        ground_truth = _load_ground_truth()
        # Filter out placeholder entries
        gt_usable = [
            entry for entry in ground_truth
            if not any("PLACEHOLDER" in cid for cid in entry.get("relevant_chunk_ids", []))
        ]

        print(f"\n  [5c] IR Metrics ({len(gt_usable)} usable ground-truth entries)...")

        p_at_5_list: list[float] = []
        p_at_10_list: list[float] = []
        r_at_10_list: list[float] = []
        mrr_list: list[float] = []
        ir_details: list[dict] = []

        if not gt_usable:
            print("    ⚠  No usable ground-truth entries (all are PLACEHOLDER).")
            print("    ⚠  Run the agent once and populate fixtures/rag_ground_truth.json")
            print("       with actual chunk IDs from citations[].source_ref")
            # Score 0 but note this is a fixture gap, not a system failure
            mean_p5 = 0.0
            mean_p10 = 0.0
            mean_r10 = 0.0
            mean_mrr = 0.0
            ir_gate = "WARN"  # Not FAIL — fixture needs population
        else:
            with httpx.Client(timeout=TIMEOUT) as client:
                for i, entry in enumerate(gt_usable, 1):
                    q = entry["question"]
                    relevant_ids = set(entry.get("relevant_chunk_ids", []))
                    print(f"    [{i:02d}] {q[:65]}...", end=" ", flush=True)

                    t0 = time.perf_counter()
                    try:
                        resp = client.post(ENDPOINT, json={"query": q}, headers={"X-Eval-Mode": "true"})
                        body = resp.json()
                        status = resp.status_code
                    except Exception as exc:
                        body = {}
                        status = 0
                    latency = time.perf_counter() - t0

                    if status != 200:
                        p_at_5_list.append(0.0)
                        p_at_10_list.append(0.0)
                        r_at_10_list.append(0.0)
                        mrr_list.append(0.0)
                        print(f"HTTP {status} — skipped")
                        continue

                    retrieved_refs = _extract_retrieved_refs(body, k=10)
                    p5  = _precision_at_k(retrieved_refs, relevant_ids, 5)
                    p10 = _precision_at_k(retrieved_refs, relevant_ids, 10)
                    r10 = _recall_at_k(retrieved_refs, relevant_ids, 10)
                    mrr = _mrr(retrieved_refs, relevant_ids)

                    p_at_5_list.append(p5)
                    p_at_10_list.append(p10)
                    r_at_10_list.append(r10)
                    mrr_list.append(mrr)
                    ir_details.append({"question": q, "P@5": p5, "P@10": p10, "R@10": r10, "MRR": mrr})
                    icon = "✅" if mrr >= 0.70 and p5 >= 0.60 else ("⚠️" if mrr >= 0.40 else "❌")
                    print(f"{icon}  P@5={p5:.2f}  P@10={p10:.2f}  R@10={r10:.2f}  MRR={mrr:.2f}  {latency:.1f}s")

            mean_p5  = sum(p_at_5_list)  / len(p_at_5_list)  if p_at_5_list  else 0.0
            mean_p10 = sum(p_at_10_list) / len(p_at_10_list) if p_at_10_list else 0.0
            mean_r10 = sum(r_at_10_list) / len(r_at_10_list) if r_at_10_list else 0.0
            mean_mrr = sum(mrr_list)     / len(mrr_list)     if mrr_list     else 0.0
            ir_gate = "PASS" if mean_mrr >= 0.70 and mean_p5 >= 0.60 else (
                "WARN" if mean_mrr >= 0.40 else "FAIL"
            )

        print(f"    → MRR={mean_mrr:.3f}  P@5={mean_p5:.3f}  P@10={mean_p10:.3f}  R@10={mean_r10:.3f}  {ir_gate}")

        report = RAGReport(
            faithfulness_rate=mean_faith,
            faithfulness_gate=faith_gate,
            faithfulness_details=faith_details,
            mean_context_relevance=mean_rel,
            context_relevance_gate=rel_gate,
            sufficiency_rate=suf_rate,
            mean_precision_at5=mean_p5,
            mean_precision_at10=mean_p10,
            mean_recall_at10=mean_r10,
            mean_mrr=mean_mrr,
            ir_gate=ir_gate,
            ir_details=ir_details,
        )
        report.print_summary()
        return report


if __name__ == "__main__":
    eval_ = RAGEval()
    report = eval_.run()
    sys.exit(0 if report.overall_gate == "PASS" else 1)
