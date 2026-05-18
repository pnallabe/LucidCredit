"""
tests/agent_evals/run_all_evals.py
=====================================
Eval 8 — Master Runner & Scorecard

Runs all 7 evals sequentially, prints a human-readable scorecard,
writes a structured JSON report, and appends to EVAL_HISTORY.md.

Any Eval 6 (compliance/security) FAIL = overall FAIL regardless of others.

Run:
    python tests/agent_evals/run_all_evals.py
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_HISTORY_PATH = Path(__file__).parent / "EVAL_HISTORY.md"

BASE_URL_BACKEND  = "http://localhost:8090"
BASE_URL_ANALYTICS = "http://localhost:8001"

# ─────────────────────────────────────────────────────────────────────────────
# Service health check
# ─────────────────────────────────────────────────────────────────────────────

def _check_services() -> tuple[bool, list[str]]:
    """Returns (all_healthy, list_of_issues)."""
    issues: list[str] = []
    with httpx.Client(timeout=8.0) as client:
        # Backend health — LucidCredit mounts router at /v1/health
        try:
            r = client.get(f"{BASE_URL_BACKEND}/v1/health")
            if r.status_code not in (200, 204):
                issues.append(f"LucidCredit backend unhealthy: HTTP {r.status_code}")
        except Exception as exc:
            issues.append(f"LucidCredit backend not reachable ({BASE_URL_BACKEND}/v1/health): {exc}")

        # Analytics API — basic connectivity (best-effort, 5s budget)
        try:
            r = client.post(
                f"{BASE_URL_ANALYTICS}/v1/analytics/s2s/ask",
                json={"question": "ping"},
                headers={
                    "x-service-key": "dev-analytics-key",
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
            if r.status_code not in (200, 400, 422):
                issues.append(f"Analytics API unhealthy: HTTP {r.status_code}")
        except Exception as exc:
            issues.append(f"Analytics API not reachable ({BASE_URL_ANALYTICS}): {exc}")

    return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────────────────────
# Individual eval runner helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_eval(name: str, eval_fn) -> tuple[Optional[Any], Optional[str]]:
    """
    Run a single eval function, catching all exceptions.
    Returns (report_object, error_string).
    """
    print(f"\n{'═'*62}")
    print(f"  Running: {name}")
    print(f"{'═'*62}")
    try:
        return eval_fn(), None
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n  ❌ EVAL CRASHED: {exc}")
        print(f"  {tb[:500]}")
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Report serialiser helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gate(report, *attrs: str) -> str:
    """Safely extract gate status from a report object via attribute path."""
    obj = report
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return "UNKNOWN"
    return str(obj)


def _score(report, *attrs: str, fmt: str = ".3f") -> str:
    """Safely extract a numeric score from a report object."""
    obj = report
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return "N/A"
    try:
        return format(float(obj), fmt)
    except (TypeError, ValueError):
        return str(obj)


def _pct(report, *attrs: str) -> str:
    obj = report
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return "N/A"
    try:
        return f"{float(obj)*100:.1f}%"
    except (TypeError, ValueError):
        return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_scorecard(rows: list[dict], overall: str) -> str:
    """
    Renders the scorecard table.
    Each row: {name, score, gate_label, status}
    """
    icons = {"PASS": "✅ PASS", "WARN": "⚠️  WARN", "FAIL": "❌ FAIL", "UNKNOWN": "⚪ N/A", "CRASH": "💥 CRASH"}
    W = 60
    lines = [
        f"╔{'═'*W}╗",
        f"║{'LucidCredit AI Agent Eval Scorecard':^{W}}║",
        f"╠{'═'*W}╣",
        f"║ {'Eval':<34} {'Score':>8}  {'Gate':>8}   {'Status':<12}║",
        f"╠{'═'*W}╣",
    ]
    for row in rows:
        name  = row.get("name", "")[:34]
        score = row.get("score", "N/A")
        gate  = row.get("gate_label", "")
        status_raw = row.get("status", "UNKNOWN")
        status_str = icons.get(status_raw, status_raw)
        lines.append(
            f"║ {name:<34} {score:>8}  {gate:>8}   {status_str:<12}║"
        )
    lines.extend([
        f"╠{'═'*W}╣",
        f"║ {'OVERALL':<34} {'':>8}  {'':>8}   {icons.get(overall, overall):<12}║",
        f"╚{'═'*W}╝",
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'═'*62}")
    print(f"  LucidCredit Agent Eval Suite — {timestamp}")
    print(f"{'═'*62}")

    # Service health check
    print("\n[Pre-flight] Checking services...")
    healthy, issues = _check_services()
    if not healthy:
        for iss in issues:
            print(f"  ⚠  {iss}")
        print("\n  WARNING: Some services unavailable — evals may fail or be degraded.")
    else:
        print("  ✅ All services healthy.\n")

    # ── Import evals ──────────────────────────────────────────────────────────
    # Import here to avoid circular imports and keep each file self-contained
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from tests.agent_evals.eval_task_success       import TaskSuccessEval
    from tests.agent_evals.eval_trajectory         import TrajectoryEval
    from tests.agent_evals.eval_tool_selection     import ToolSelectionEval
    from tests.agent_evals.eval_self_aware_failure import SelfAwareFailureEval
    from tests.agent_evals.eval_rag                import RAGEval
    from tests.agent_evals.eval_compliance_security import ComplianceSecurityEval
    from tests.agent_evals.eval_operational        import OperationalEval

    # ── Run all evals ─────────────────────────────────────────────────────────
    r1, e1 = _run_eval("Eval 1 — Task Success Rate",     TaskSuccessEval().run)
    r2, e2 = _run_eval("Eval 2 — Agent Trajectory",      TrajectoryEval().run)
    r3, e3 = _run_eval("Eval 3 — Tool Selection",        ToolSelectionEval().run)
    r4, e4 = _run_eval("Eval 4 — Self-Aware Failure",    SelfAwareFailureEval().run)
    r5, e5 = _run_eval("Eval 5 — RAG Evals",             RAGEval().run)
    r6, e6 = _run_eval("Eval 6 — Compliance & Security", ComplianceSecurityEval().run)
    r7, e7 = _run_eval("Eval 7 — Operational",           OperationalEval().run)

    # ── Build scorecard rows ──────────────────────────────────────────────────
    def _gate_or_crash(report, error, *path):
        if error:
            return "CRASH"
        return _gate(report, *path)

    scorecard_rows: list[dict] = [
        {
            "name":      "1. Task Success Rate",
            "score":     _pct(r1, "task_success_rate") if r1 else "N/A",
            "gate_label": "≥85%",
            "status":    _gate_or_crash(r1, e1, "gate_status"),
        },
        {
            "name":      "2. Agent Trajectory Score",
            "score":     _score(r2, "mean_score") if r2 else "N/A",
            "gate_label": "≥0.90",
            "status":    _gate_or_crash(r2, e2, "gate_status"),
        },
        {
            "name":      "3. Tool Selection F1",
            "score":     _score(r3, "macro_f1") if r3 else "N/A",
            "gate_label": "≥0.85",
            "status":    _gate_or_crash(r3, e3, "gate_status"),
        },
        {
            "name":      "4. Self-Aware Failure Rate",
            "score":     _pct(r4, "self_aware_rate") if r4 else "N/A",
            "gate_label": "≥90%",
            "status":    _gate_or_crash(r4, e4, "gate_status"),
        },
        {
            "name":      "5a. Faithfulness Rate",
            "score":     _pct(r5, "faithfulness_rate") if r5 else "N/A",
            "gate_label": "≥95%",
            "status":    _gate_or_crash(r5, e5, "faithfulness_gate"),
        },
        {
            "name":      "5b. Context Relevance",
            "score":     _pct(r5, "mean_context_relevance") if r5 else "N/A",
            "gate_label": "≥70%",
            "status":    _gate_or_crash(r5, e5, "context_relevance_gate"),
        },
        {
            "name":      "5c. MRR / Precision@5",
            "score":     (f"{r5.mean_mrr:.2f}/{r5.mean_precision_at5:.2f}" if r5 else "N/A"),
            "gate_label": "≥0.70",
            "status":    _gate_or_crash(r5, e5, "ir_gate"),
        },
        {
            "name":      "6a. PII Detection Rate",
            "score":     _pct(r6, "pii_detection_rate") if r6 else "N/A",
            "gate_label": "=100%",
            "status":    _gate_or_crash(r6, e6, "pii_gate"),
        },
        {
            "name":      "6b. Policy Adherence",
            "score":     _pct(r6, "policy_adherence_rate") if r6 else "N/A",
            "gate_label": "=100%",
            "status":    _gate_or_crash(r6, e6, "policy_gate"),
        },
        {
            "name":      "6c. Audit Traceability",
            "score":     _pct(r6, "traceability_rate") if r6 else "N/A",
            "gate_label": "=100%",
            "status":    _gate_or_crash(r6, e6, "traceability_gate"),
        },
        {
            "name":      "6d. Injection Resistance",
            "score":     _pct(r6, "injection_resistance_rate") if r6 else "N/A",
            "gate_label": "=100%",
            "status":    _gate_or_crash(r6, e6, "injection_gate"),
        },
        {
            "name":      "7a. Latency p95 (all tiers)",
            "score":     (
                "/".join(f"T{t.tier}:{t.p95:.1f}s" for t in r7.tier_results)
                if r7 and r7.tier_results else "N/A"
            ),
            "gate_label": "SLO",
            "status":    _gate_or_crash(r7, e7, "latency_gate"),
        },
        {
            "name":      "7b. Mean Cost/Task",
            "score":     (f"${r7.cost_result.mean_cost_usd:.4f}" if r7 and r7.cost_result else "N/A"),
            "gate_label": "≤$0.04",
            "status":    _gate_or_crash(r7, e7, "cost_gate"),
        },
    ]

    # ── Determine overall gate ────────────────────────────────────────────────
    # Rule: any Eval 6 FAIL = overall FAIL
    compliance_statuses = [row["status"] for row in scorecard_rows if row["name"].startswith("6")]
    all_statuses = [row["status"] for row in scorecard_rows]

    if any(s == "FAIL" for s in compliance_statuses):
        overall = "FAIL"
    elif any(s == "CRASH" for s in all_statuses):
        overall = "FAIL"
    elif any(s == "FAIL" for s in all_statuses):
        overall = "FAIL"
    elif any(s == "WARN" for s in all_statuses):
        overall = "WARN"
    else:
        overall = "PASS"

    # ── Print scorecard ───────────────────────────────────────────────────────
    print(f"\n\n{_render_scorecard(scorecard_rows, overall)}\n")

    # ── Write JSON report ─────────────────────────────────────────────────────
    report_path = REPORTS_DIR / f"eval_{timestamp}.json"

    def _serialise_report(r, error: Optional[str]) -> dict:
        if error:
            return {"error": error, "status": "CRASH"}
        if r is None:
            return {"status": "SKIPPED"}
        try:
            # Best-effort: convert dataclass to dict
            import dataclasses
            if dataclasses.is_dataclass(r):
                return dataclasses.asdict(r)
            return {"raw": str(r)}
        except Exception:
            return {"raw": str(r)}

    full_report = {
        "timestamp": timestamp,
        "overall_gate": overall,
        "scorecard": scorecard_rows,
        "details": {
            "eval_1_task_success":        _serialise_report(r1, e1),
            "eval_2_trajectory":          _serialise_report(r2, e2),
            "eval_3_tool_selection":      _serialise_report(r3, e3),
            "eval_4_self_aware":          _serialise_report(r4, e4),
            "eval_5_rag":                 _serialise_report(r5, e5),
            "eval_6_compliance_security": _serialise_report(r6, e6),
            "eval_7_operational":         _serialise_report(r7, e7),
        },
    }

    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"JSON report written → {report_path}")

    # ── Write EVAL_HISTORY.md entry ───────────────────────────────────────────
    icon_map = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    icon = icon_map.get(overall, "⚪")

    # Build one-line summary of regressions / improvements
    fails  = [r["name"] for r in scorecard_rows if r["status"] == "FAIL"]
    warns  = [r["name"] for r in scorecard_rows if r["status"] == "WARN"]
    passes = [r["name"] for r in scorecard_rows if r["status"] == "PASS"]

    if fails:
        summary_line = f"FAILURES in: {', '.join(fails[:4])}"
    elif warns:
        summary_line = f"Warnings in: {', '.join(warns[:4])}" + ("; all others PASS" if passes else "")
    else:
        summary_line = f"All {len(passes)} evals PASS — no regressions."

    history_entry = (
        f"\n## {timestamp} — {icon} {overall}\n"
        f"{summary_line}\n"
        f"Report: {report_path.name}\n"
    )

    with open(EVAL_HISTORY_PATH, "a") as f:
        f.write(history_entry)
    print(f"History entry appended → {EVAL_HISTORY_PATH}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
