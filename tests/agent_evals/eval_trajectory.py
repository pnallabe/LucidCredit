"""
tests/agent_evals/eval_trajectory.py
=======================================
Eval 2 — Agent Trajectory Scoring

Scores the sequence of nodes the agent traversed for correctness and efficiency.
The reasoning_trace.retrieval_method field and citations together reconstruct
the actual trajectory for comparison against expected paths.

Run:
    python tests/agent_evals/eval_trajectory.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

BASE_URL = "http://localhost:8090"
ENDPOINT = f"{BASE_URL}/v1/query/analyst"
TIMEOUT = 90.0

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ─────────────────────────────────────────────────────────────────────────────
# Expected trajectory fixture (loaded from YAML)
# ─────────────────────────────────────────────────────────────────────────────

def _load_expected_trajectories() -> dict:
    path = FIXTURES_DIR / "expected_trajectories.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Test cases — one per intent type, enough to score all four paths
# ─────────────────────────────────────────────────────────────────────────────

TRAJECTORY_TEST_CASES: list[dict] = [
    # analyst_query — expects vector + api tools
    {"intent": "analyst_query", "query": "What is the total outstanding loan balance and delinquency rate?"},
    {"intent": "analyst_query", "query": "Show the charge-off rate by quarter."},
    {"intent": "analyst_query", "query": "What is the approval rate by FICO tier?"},
    {"intent": "analyst_query", "query": "How many loans were originated per month in 2023?"},
    {"intent": "analyst_query", "query": "What is the average FICO score at origination by product type?"},

    # applicant_comms — expects compliance_check to fire
    {
        "intent": "applicant_comms",
        "query": (
            "Draft an adverse action notice explaining that a DTI of 55% exceeds "
            "our 45% maximum threshold for personal loans."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Notify the applicant that their credit score of 575 is below our minimum "
            "requirement and describe their right to a free consumer report copy."
        ),
    },
    {
        "intent": "applicant_comms",
        "query": (
            "Write a decline notice citing insufficient employment history as the reason "
            "for the adverse credit decision."
        ),
    },

    # portfolio_brief — expects api tool primarily
    {"intent": "portfolio_brief", "query": "Provide a brief on delinquency trends for the board meeting."},
    {"intent": "portfolio_brief", "query": "Summarize the overall portfolio health as of the latest period."},

    # explain_decision — expects vector + db tools
    {"intent": "explain_decision", "query": "Explain why a high debt-to-income ratio leads to loan denial."},
    {"intent": "explain_decision", "query": "What are the ECOA adverse action notice requirements?"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory inference from response payload
# ─────────────────────────────────────────────────────────────────────────────

# Node name constants matching the LangGraph topology
_NODE_PARSE_INTENT      = "parse_intent"
_NODE_RETRIEVE          = "retrieve"
_NODE_GRADE             = "grade_documents"
_NODE_REASON            = "reason"
_NODE_CITATION_ENFORCER = "citation_enforcer"
_NODE_CONFIDENCE        = "confidence_score"
_NODE_COMPLIANCE        = "compliance_check"
_NODE_FORMAT            = "format_output"
_NODE_PERSIST           = "persist_session"
_NODE_ERROR             = "error"

# Tool source_type → human-readable label
_SOURCE_TYPE_LABELS = {
    "api": "api",
    "vector_doc": "vector_doc",
    "db": "db",
    "thinfile": "thinfile",
    "domain_knowledge": "domain_knowledge",
    "clarification": "clarification",
}


def _infer_trajectory(response_body: dict, intent: str) -> dict:
    """
    Reconstruct what nodes fired based on the response payload fields.

    Returns:
        {
          "nodes": list[str],        # inferred node sequence
          "tools": list[str],        # source_types actually used
          "compliance_fired": bool,
          "suppression_fired": bool,
          "confidence_fired": bool,
        }
    """
    rt = response_body.get("reasoning_trace") or {}
    citations = response_body.get("citations") or []
    suppressed = rt.get("suppressed_claims") or []
    confidence_score = response_body.get("confidence_score", 0.0)
    retrieval_method = rt.get("retrieval_method", "")
    retrieved_ctx = rt.get("retrieved_context") or []
    answer = response_body.get("answer", "")

    # Infer tools used from citations + retrieved_context source_type fields
    tool_set: set[str] = set()
    for c in citations:
        st = c.get("source_type", "")
        if st and st in _SOURCE_TYPE_LABELS:
            tool_set.add(st)
    for item in retrieved_ctx:
        st = item.get("source_type", "")
        if st and st in _SOURCE_TYPE_LABELS:
            tool_set.add(st)

    # Additional inference from retrieval_method string
    if "vector" in retrieval_method.lower():
        tool_set.add("vector_doc")
    if "api" in retrieval_method.lower() or "nl2sql" in retrieval_method.lower() or "bigquery" in retrieval_method.lower():
        tool_set.add("api")
    if "sql" in retrieval_method.lower() and "nl2sql" not in retrieval_method.lower():
        tool_set.add("db")

    # Infer node sequence from observable evidence
    nodes = [_NODE_PARSE_INTENT]

    if retrieval_method or retrieved_ctx or citations:
        nodes.append(_NODE_RETRIEVE)
        nodes.append(_NODE_GRADE)

    # reason fired if raw_analysis is populated or answer is non-empty
    raw_analysis = rt.get("raw_analysis", "")
    if raw_analysis or answer:
        nodes.append(_NODE_REASON)

    # citation_enforcer fired if suppressed_claims is present (even empty list) and
    # answer is a grounded narrative (citations present or suppression happened)
    if citations or suppressed:
        nodes.append(_NODE_CITATION_ENFORCER)

    # confidence_score node fired if confidence_score value is present
    if confidence_score is not None and confidence_score > 0:
        nodes.append(_NODE_CONFIDENCE)

    # compliance_check fired for applicant-audience intents
    # Infer from intent type (applicant_comms / explain_decision) + non-empty answer
    audience_intents = {"applicant_comms", "explain_decision"}
    compliance_fired = intent in audience_intents and bool(answer)
    if compliance_fired:
        nodes.append(_NODE_COMPLIANCE)

    if answer:
        nodes.append(_NODE_FORMAT)
        nodes.append(_NODE_PERSIST)

    return {
        "nodes": nodes,
        "tools": sorted(tool_set),
        "compliance_fired": compliance_fired,
        "suppression_fired": bool(suppressed),
        "confidence_fired": confidence_score is not None and confidence_score > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edit-distance scoring
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    """Levenshtein edit distance between two sequences of node names."""
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def _score_trajectory(
    actual_nodes: list[str],
    expected_nodes: list[str],
    actual_tools: list[str],
    expected_tools: list[str],
) -> tuple[float, list[str]]:
    """
    Compute trajectory_score and collect penalty reasons.

    Base formula: 1.0 - (edit_distance / len(expected))
    Penalize skipped mandatory nodes: -0.2 each
    Penalize unexpected extra tool calls: -0.1 each
    """
    reasons: list[str] = []
    if not expected_nodes:
        return 1.0, reasons

    dist = _edit_distance(actual_nodes, expected_nodes)
    base_score = 1.0 - (dist / len(expected_nodes))
    base_score = max(0.0, base_score)

    penalty = 0.0

    # Penalise skipped mandatory nodes
    core_nodes = {_NODE_RETRIEVE, _NODE_GRADE, _NODE_REASON, _NODE_CITATION_ENFORCER, _NODE_CONFIDENCE}
    for node in core_nodes:
        if node in expected_nodes and node not in actual_nodes:
            penalty += 0.2
            reasons.append(f"Skipped mandatory node: {node} (-0.20)")

    # Penalise unexpected extra tool calls
    expected_tool_set = set(expected_tools)
    actual_tool_set = set(actual_tools)
    extra_tools = actual_tool_set - expected_tool_set - {"domain_knowledge"}  # domain_knowledge is always ok
    for tool in extra_tools:
        penalty += 0.1
        reasons.append(f"Unexpected extra tool: {tool} (-0.10)")

    score = max(0.0, min(1.0, base_score - penalty))
    return score, reasons


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajTurnResult:
    intent: str
    query: str
    http_status: int
    trajectory_score: float
    expected_nodes: list[str]
    actual_nodes: list[str]
    expected_tools: list[str]
    actual_tools: list[str]
    penalties: list[str]
    latency_s: float


@dataclass
class TrajectoryReport:
    mean_score: float
    gate_status: str  # PASS | WARN | FAIL
    turn_results: list[TrajTurnResult]

    def print_summary(self) -> None:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[self.gate_status]
        print(f"\n{'─'*60}")
        print(f"Eval 2 — Agent Trajectory Score: {self.mean_score:.3f}  {icon} {self.gate_status}")
        print(f"  Gate: ≥0.90 PASS | 0.75-0.89 WARN | <0.75 FAIL\n")
        print(f"  {'Intent':20s}  {'Score':6s}  {'Expected path':35s}  Actual path")
        for t in self.turn_results:
            exp = " → ".join(t.expected_nodes[-5:])  # last 5 nodes for readability
            act = " → ".join(t.actual_nodes[-5:])
            score_s = f"{t.trajectory_score:.2f}"
            q_short = t.query[:40]
            print(f"  {t.intent:20s}  {score_s:6s}  {exp[:35]:35s}  {act[:35]}")
            for p in t.penalties:
                print(f"           ⚠  {p}")
        print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEval:
    test_cases: list[dict] = TRAJECTORY_TEST_CASES

    def run(self) -> TrajectoryReport:
        expected_trajectories = _load_expected_trajectories()
        print(f"[Eval 2] Agent Trajectory Scoring — {len(self.test_cases)} test cases")
        print(f"  Endpoint : {ENDPOINT}")
        print(f"  Gate     : ≥0.90 PASS | 0.75-0.89 WARN | <0.75 FAIL\n")

        turn_results: list[TrajTurnResult] = []

        with httpx.Client(timeout=TIMEOUT) as client:
            for i, case in enumerate(self.test_cases, 1):
                intent = case["intent"]
                query  = case["query"]
                print(f"  [{i:02d}/{len(self.test_cases)}] {intent:20s}  {query[:60]}...", end=" ", flush=True)

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
                    body = {}
                    status = 0
                latency = time.perf_counter() - t0

                # Get expected trajectory for this intent
                traj_spec = expected_trajectories.get(intent, {})
                expected_nodes: list[str] = traj_spec.get("nodes", [])
                expected_tools: list[str] = traj_spec.get("tools", [])

                if status != 200 or not body.get("answer"):
                    # Non-success: score 0 for trajectory
                    score = 0.0
                    penalties = [f"HTTP {status} — no answer to score trajectory"]
                    actual = {"nodes": [], "tools": []}
                else:
                    actual = _infer_trajectory(body, intent)
                    score, penalties = _score_trajectory(
                        actual["nodes"], expected_nodes,
                        actual["tools"], expected_tools,
                    )

                print(f"score={score:.3f}  {latency:.1f}s")
                for p in penalties[:2]:
                    print(f"         ⚠  {p}")

                turn_results.append(TrajTurnResult(
                    intent=intent,
                    query=query,
                    http_status=status,
                    trajectory_score=score,
                    expected_nodes=expected_nodes,
                    actual_nodes=actual.get("nodes", []),
                    expected_tools=expected_tools,
                    actual_tools=actual.get("tools", []),
                    penalties=penalties,
                    latency_s=latency,
                ))

        mean_score = (
            sum(t.trajectory_score for t in turn_results) / len(turn_results)
            if turn_results else 0.0
        )
        if mean_score >= 0.90:
            gate = "PASS"
        elif mean_score >= 0.75:
            gate = "WARN"
        else:
            gate = "FAIL"

        report = TrajectoryReport(
            mean_score=mean_score,
            gate_status=gate,
            turn_results=turn_results,
        )
        report.print_summary()
        return report


if __name__ == "__main__":
    eval_ = TrajectoryEval()
    report = eval_.run()
    sys.exit(0 if report.gate_status == "PASS" else 1)
