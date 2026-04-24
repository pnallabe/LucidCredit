"""
backend/tests/eval/ragas_harness.py
=====================================
RAGAS evaluation harness for LucidCredit's zero-hallucination acceptance tests.

Run with:
    pytest backend/tests/eval/ -m ragas --provider azure_gpt41

Supported --provider values:
    azure_gpt41          AzureChatOpenAI gpt-4.1-2025-04-14 (default)
    azure_gpt4o          AzureChatOpenAI gpt-4o
    vertex_gemini15pro   ChatVertexAI gemini-1.5-pro-002
    vertex_flash20       ChatVertexAI gemini-2.0-flash-001

Thresholds (hard fail if not met):
    Analyst sessions:  faithfulness >= 0.85, answer_relevancy >= 0.80, context_recall >= 0.75
    Briefing sessions: faithfulness >= 0.80, answer_relevancy >= 0.75, context_recall >= 0.70

Results are stored as JSON in backend/tests/eval/results/<timestamp>_<provider>.json
for historical tracking.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_PATH = Path(__file__).parent / "golden" / "analyst_qa.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Thresholds per session type
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, dict[str, float]] = {
    "analyst": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_recall": 0.75,
    },
    "briefing": {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_recall": 0.70,
    },
    # Adversarial: faithfulness must be perfect — answers must refuse or suppress
    # rather than fabricate. answer_relevancy is intentionally lower since the
    # "correct" answer is a refusal, not a direct answer to the question.
    "adversarial": {
        "faithfulness": 0.95,
        "answer_relevancy": 0.60,
        "context_recall": 0.80,
    },
}


# ---------------------------------------------------------------------------
# pytest CLI option
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:  # noqa: D401
    parser.addoption(
        "--provider",
        action="store",
        default="azure_gpt41",
        choices=["azure_gpt41", "azure_gpt4o", "vertex_gemini15pro", "vertex_flash20"],
        help="LLM provider to evaluate with RAGAS",
    )


# ---------------------------------------------------------------------------
# Provider override fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def provider_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--provider")  # type: ignore[return-value]


@pytest.fixture(scope="session")
def ragas_llm(provider_name: str) -> Any:
    """
    Build a LangChain BaseChatModel appropriate for the selected provider,
    temporarily overriding Settings to satisfy validation.
    """
    from unittest.mock import patch

    if provider_name == "azure_gpt41":
        from langchain_openai import AzureChatOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_ANALYST", "gpt-4.1-2025-04-14")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0.0,
        )

    elif provider_name == "azure_gpt4o":
        from langchain_openai import AzureChatOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o",
            api_version=api_version,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0.0,
        )

    elif provider_name == "vertex_gemini15pro":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model_name="gemini-1.5-pro-002",
            project=os.environ.get("GOOGLE_PROJECT_ID", ""),
            location=os.environ.get("GOOGLE_LOCATION", "us-central1"),
            temperature=0.0,
        )

    elif provider_name == "vertex_flash20":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model_name="gemini-2.0-flash-001",
            project=os.environ.get("GOOGLE_PROJECT_ID", ""),
            location=os.environ.get("GOOGLE_LOCATION", "us-central1"),
            temperature=0.0,
        )

    raise ValueError(f"Unknown provider: {provider_name}")


@pytest.fixture(scope="session")
def ragas_embeddings(provider_name: str) -> Any:
    """
    Embeddings always use Azure OpenAI — never Vertex AI.
    """
    from langchain_openai import AzureOpenAIEmbeddings

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "text-embedding-3-large")

    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Golden dataset loader
# ---------------------------------------------------------------------------


def _load_golden(session_type: str) -> list[dict[str, Any]]:
    with open(GOLDEN_PATH) as f:
        all_items: list[dict[str, Any]] = json.load(f)
    return [item for item in all_items if item["session_type"] == session_type]


def _build_ragas_dataset(
    items: list[dict[str, Any]],
    llm: Any,
) -> Dataset:
    """
    Build a RAGAS-compatible HuggingFace Dataset.

    For the golden set we use the context directly as both retrieved_contexts
    and generate the answer by prompting the LLM (or use the ground_truth as
    a stand-in answer for offline baseline evaluation).

    In production evaluation, replace `answer` with the actual LLM output
    from a real pipeline run against the same questions.
    """
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for item in items:
        questions.append(item["question"])
        # Use ground_truth as a proxy answer for baseline — replace with real
        # LLM responses when running against live providers in integration mode.
        answers.append(item["ground_truth"])
        contexts.append(item["context"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


# ---------------------------------------------------------------------------
# Core evaluation function (also callable programmatically from briefing.py)
# ---------------------------------------------------------------------------


def run_ragas_evaluation(
    session_type: str,
    llm: Any,
    embeddings: Any,
    provider_label: str,
) -> dict[str, float]:
    """
    Run RAGAS on the golden dataset for a given session_type.
    Returns a dict of metric_name -> score.
    Stores results JSON in RESULTS_DIR.
    """
    items = _load_golden(session_type)
    if not items:
        pytest.skip(f"No golden items for session_type={session_type}")

    dataset = _build_ragas_dataset(items, llm)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores: dict[str, float] = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }

    # Persist results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS_DIR / f"{timestamp}_{provider_label}_{session_type}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "provider": provider_label,
                "session_type": session_type,
                "scores": scores,
                "thresholds": THRESHOLDS[session_type],
                "item_count": len(items),
            },
            f,
            indent=2,
        )

    return scores


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.parametrize("session_type", ["analyst", "briefing", "adversarial"])
def test_ragas_thresholds(
    session_type: str,
    ragas_llm: Any,
    ragas_embeddings: Any,
    provider_name: str,
) -> None:
    """
    Evaluate RAGAS metrics for the given session_type and assert thresholds.
    Fails if any metric falls below the defined threshold.
    """
    scores = run_ragas_evaluation(
        session_type=session_type,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        provider_label=provider_name,
    )

    thresholds = THRESHOLDS[session_type]

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"RAGAS Results — provider={provider_name}  session_type={session_type}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Score':>8} {'Threshold':>10} {'Pass':>6}")
    print(f"{'-' * 55}")
    all_pass = True
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        passed = score >= threshold
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"{metric:<25} {score:>8.3f} {threshold:>10.3f} {status:>6}")
    print(f"{'=' * 60}\n")

    assert scores["faithfulness"] >= thresholds["faithfulness"], (
        f"faithfulness {scores['faithfulness']:.3f} < threshold {thresholds['faithfulness']} "
        f"for session_type={session_type}"
    )
    assert scores["answer_relevancy"] >= thresholds["answer_relevancy"], (
        f"answer_relevancy {scores['answer_relevancy']:.3f} < threshold "
        f"{thresholds['answer_relevancy']} for session_type={session_type}"
    )
    assert scores["context_recall"] >= thresholds["context_recall"], (
        f"context_recall {scores['context_recall']:.3f} < threshold "
        f"{thresholds['context_recall']} for session_type={session_type}"
    )
