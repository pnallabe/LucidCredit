# LucidCredit — LLM Provider Migration & Implementation Plan

> **Based on:** LLM Provider Architecture Decision (April 2026)
> **Scope:** Migrate from direct OpenAI API → Azure OpenAI primary + Vertex AI Gemini fallback + cost-optimized hybrid routing
> **Status:** Pre-Sprint 1 (execute before any Sprint 2 agent code)

---

## Architecture Decision Summary

| Role | Provider | Model | Endpoints |
|---|---|---|---|
| **Primary** | Azure OpenAI (US East 2) | `gpt-4.1` (dated pin) | All session types |
| **Applicant fallback** | None — return HTTP 503 | — | `/v1/applicant/communication` never falls back |
| **Analyst/briefing fallback** | Vertex AI Gemini 1.5 Pro | `gemini-1.5-pro-002` | `/v1/query/analyst`, `/v1/briefing/generate` |
| **Cost-optimized batch** | Vertex AI Gemini 2.0 Flash | `gemini-2.0-flash-001` | `/v1/briefing/generate` (after A/B gate) |
| **Embeddings** | Azure OpenAI only | `text-embedding-3-large` | Always — never mix embedding providers |

---

## Prompt 1 — Azure OpenAI: Configuration Layer

**File:** `backend/app/config.py`
**Depends on:** Nothing — execute first
**Acceptance criteria:** `Settings` loads cleanly from `.env`; `pytest -k test_settings` passes; no raw `openai_api_key` used in non-dev environments.

```
You are implementing the configuration layer for LucidCredit, a regulated credit decision copilot.

Rewrite `backend/app/config.py` (currently using pydantic-settings BaseSettings) to support three LLM provider configurations:
1. Azure OpenAI (primary — required in staging/production)
2. Direct OpenAI API (development/local only — gated by `environment == "development"`)
3. Vertex AI / Google Generative AI (fallback — optional at startup, required in production)

REQUIREMENTS:
- Keep the existing `Settings` class but add the following new fields:
  - `azure_openai_endpoint: str = ""` — full Azure resource URL, e.g. https://<resource>.openai.azure.com/
  - `azure_openai_api_key: str = ""` — Azure OpenAI key (separate from direct OpenAI key)
  - `azure_openai_api_version: str = "2024-10-21"` — use this default exactly
  - `azure_openai_deployment_analyst: str = "gpt-4.1-2025-04-14"` — dated deployment name for analyst sessions
  - `azure_openai_deployment_applicant: str = "gpt-4.1-2025-04-14"` — dated deployment name for applicant sessions
  - `azure_openai_deployment_embedding: str = "text-embedding-3-large"` — embedding deployment name
  - `google_project_id: str = ""` — GCP project ID for Vertex AI
  - `google_location: str = "us-central1"` — Vertex AI region
  - `vertex_model_analyst_fallback: str = "gemini-1.5-pro-002"` — Vertex fallback model for analyst
  - `vertex_model_batch_briefing: str = "gemini-2.0-flash-001"` — Vertex model for batch briefings
  - `llm_provider_mode: Literal["azure", "openai_direct"] = "azure"` — controls which client factory is used; "openai_direct" is only valid when `environment == "development"`
  - `analyst_fallback_enabled: bool = True` — enables Vertex fallback for analyst/briefing sessions
  - `briefing_use_flash: bool = False` — A/B gate for Gemini 2.0 Flash on /v1/briefing/generate (disabled until RAGAS gate passes)

- Add a `@model_validator(mode="after")` that raises a `ValueError` at startup if:
  - `llm_provider_mode == "azure"` AND `azure_openai_endpoint == ""`
  - `environment != "development"` AND `llm_provider_mode == "openai_direct"` (direct API forbidden outside dev)
  - `analyst_fallback_enabled == True` AND `google_project_id == ""` (fallback configured but Vertex unconfigured)

- Add a `model_version_hash` property that returns a deterministic string like
  `"azure:gpt-4.1-2025-04-14"` or `"vertex:gemini-1.5-pro-002"` for a given
  (provider, deployment) pair. This will be stored in every `copilot_sessions` row.

- Retain `openai_api_key: str = ""` but mark it deprecated in a comment — it is used
  ONLY when `llm_provider_mode == "openai_direct"`.

- Keep `min_citation_similarity: float = 0.85` and `min_confidence_score: float = 0.75` unchanged.

- The `get_settings()` lru_cache function stays unchanged.

After editing config.py, also update `.env.example` to add the new Azure and Vertex variables
with empty defaults and inline comments explaining each variable's purpose. Keep all existing
variables. Add a section header comment `# === Azure OpenAI (Primary — required in staging/prod) ===`
and `# === Vertex AI / Gemini (Fallback) ===`.
```

---

## Prompt 2 — Azure OpenAI: Client Factory

**File:** `backend/app/llm/provider.py` (new file — create directory `backend/app/llm/`)
**Depends on:** Prompt 1 (Settings must have Azure fields)
**Acceptance criteria:** `get_chat_client("analyst")` and `get_chat_client("applicant")` return the correct `langchain_openai.AzureChatOpenAI` instance; `get_fallback_client("analyst")` returns `langchain_google_vertexai.ChatVertexAI`; unit test mocks both.

```
You are building the LLM provider abstraction layer for LucidCredit.

Create `backend/app/llm/__init__.py` (empty) and `backend/app/llm/provider.py`.

The module must provide a single public interface — three functions:

1. `get_chat_client(session_type: Literal["analyst", "applicant", "briefing"]) -> BaseChatModel`
   - Returns an AzureChatOpenAI instance pointed at the correct dated deployment from Settings
   - Uses `session_type` to select `azure_openai_deployment_analyst` or `azure_openai_deployment_applicant`
   - "briefing" maps to the analyst deployment unless `settings.briefing_use_flash` is True
   - If `settings.llm_provider_mode == "openai_direct"` (dev only), returns ChatOpenAI instead
   - Streaming is enabled by default (streaming=True) — callers may override

2. `get_fallback_client(session_type: Literal["analyst", "briefing"]) -> BaseChatModel`
   - Returns a ChatVertexAI instance using `vertex_model_analyst_fallback`
   - If `session_type == "briefing"` AND `settings.briefing_use_flash` is True, uses `vertex_model_batch_briefing`
   - Raises `FallbackNotAvailableError` (custom exception defined in this module) if
     `settings.analyst_fallback_enabled` is False or `settings.google_project_id == ""`
   - IMPORTANT: `session_type == "applicant"` is explicitly forbidden — raise `ValueError`
     with message "Applicant sessions must not fall back to Vertex AI. Return HTTP 503 instead."

3. `get_embedding_client() -> AsyncOpenAI`
   - Returns AsyncOpenAI (Azure) for embeddings only
   - Always uses Azure regardless of `llm_provider_mode`
   - Embeddings must NEVER route to Vertex AI — raise NotImplementedError if called in a context
     where azure_openai_endpoint is empty

IMPLEMENTATION NOTES:
- Use `functools.lru_cache` on private factory functions keyed by (endpoint, deployment, api_version)
  so clients are reused across requests without being singletons in the module namespace
- AzureChatOpenAI should be constructed with:
    - azure_endpoint=settings.azure_openai_endpoint
    - azure_deployment=<deployment name>
    - api_version=settings.azure_openai_api_version
    - api_key=settings.azure_openai_api_key  (use SecretStr via pydantic)
    - temperature=0.0 (zero temperature for grounded factual output — no randomness)
    - max_retries=3
    - request_timeout=60.0
- ChatVertexAI should be constructed with:
    - model_name=<model name>
    - project=settings.google_project_id
    - location=settings.google_location
    - temperature=0.0
    - max_retries=2
    - streaming=True

EXCEPTIONS:
- Define `class FallbackNotAvailableError(RuntimeError): pass` in this module
- This is the only exception class needed here; all others bubble up naturally

DEPENDENCIES to add to requirements.txt:
- langchain-openai>=0.3.0
- langchain-google-vertexai>=2.0.0
- langchain-core>=0.3.0
- google-cloud-aiplatform>=1.60.0

Do NOT add langchain (the meta-package) — pin to the specific integration packages only.
```

---

## Prompt 3 — Azure OpenAI: Embedder Migration

**File:** `backend/app/rag/embedder.py` (modify existing)
**Depends on:** Prompt 2 (provider.py must exist with `get_embedding_client()`)
**Acceptance criteria:** `embed_text` and `batch_embed` return identical-dimension vectors (3072) as before; the `AsyncOpenAI` client now uses Azure endpoint; existing pgvector index is unaffected.

```
You are migrating the LucidCredit embedder from direct OpenAI API to Azure OpenAI.

Modify `backend/app/rag/embedder.py` to replace the internal `_client()` factory with a call to
`app.llm.provider.get_embedding_client()` from the new provider module.

REQUIREMENTS:
- Replace the private `_client()` function that constructs `AsyncOpenAI(api_key=...)` with a
  direct import of `get_embedding_client` from `app.llm.provider`
- The `embed_text` and `batch_embed` function signatures must NOT change — downstream callers
  (policy_doc_ingestor.py and future retriever.py) must require zero changes
- The model name still comes from `settings.openai_embedding_model` (now interpreted as the
  Azure deployment name, value "text-embedding-3-large")
- Add a module-level constant `EMBEDDING_DIM: int = 3072` — this is the dimension of
  text-embedding-3-large and must match the pgvector column definition exactly
- Add a log warning at DEBUG level in `batch_embed` if the returned vector length != EMBEDDING_DIM,
  which would indicate a model mismatch (do not raise, just warn — the DB insert will fail with
  a clear error from pgvector anyway)
- Keep all existing docstrings and the `batch_size: int = 64` default unchanged

IMPORTANT: Do NOT change the embedding model or the vector dimensions. The pgvector index
already exists (or will be created in Sprint 1 with 3072 dimensions). If the embedding provider
or model ever changes, a full corpus re-ingestion is required — document this in a comment above
the EMBEDDING_DIM constant.
```

---

## Prompt 4 — LangGraph: Provider-Aware Reason Node

**File:** `backend/app/agent/nodes.py` (new file — Sprint 2)
**Depends on:** Prompts 1, 2 (provider layer must exist)
**Acceptance criteria:** The `reason` node uses `get_chat_client`; on `RateLimitError` or `APIStatusError` (5xx) it retries once then calls `get_fallback_client`; applicant sessions never call `get_fallback_client`.

```
You are building the LangGraph agent nodes for LucidCredit's corrective RAG pipeline.

Create `backend/app/agent/nodes.py`. This module defines all LangGraph node functions.
Each node is a pure async function: `async def node_name(state: AgentState) -> dict`.

First, define the AgentState TypedDict in `backend/app/agent/state.py` (separate file):

```python
# backend/app/agent/state.py
from __future__ import annotations
from typing import TypedDict, Literal, Optional
from uuid import UUID

class RetrievedChunk(TypedDict):
    chunk_id: str
    source_type: Literal["vector_doc", "api", "db"]
    source_ref: str
    content: str
    relevance: Literal["RELEVANT", "IRRELEVANT", "AMBIGUOUS"]
    similarity_score: float

class AgentState(TypedDict):
    # Input
    session_id: UUID
    query: str
    intent: Literal["explain_decision", "analyst_query", "applicant_comms", "portfolio_brief"]
    audience: Literal["analyst", "applicant"]
    context_payload: dict          # raw API inputs (decision_id, application_id, etc.)

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    graded_chunks: list[RetrievedChunk]  # RELEVANT only after grading
    retrieval_sufficient: bool

    # Generation
    rendered_prompt: str
    raw_llm_output: str
    provider_used: str             # e.g. "azure:gpt-4.1-2025-04-14" — stored in session

    # Grounding
    grounded_narrative: str
    citations: list[dict]
    suppressed_claims: list[dict]
    confidence_score: float
    grounding_passed: bool         # True if confidence_score >= settings.min_confidence_score

    # Compliance (applicant audience only)
    compliance_flags: list[str]
    compliance_passed: bool

    # Output
    final_output: dict
    error: Optional[str]
```

Now define the following nodes in `backend/app/agent/nodes.py`:

NODE: `reason_node`
- This is the LLM call node. It is the ONLY place in the codebase that invokes an LLM.
- Takes `state["graded_chunks"]` and `state["rendered_prompt"]` as input
- Determines `session_type` from `state["audience"]` and `state["intent"]`:
  - `audience == "applicant"` → session_type = "applicant"
  - `intent == "portfolio_brief"` → session_type = "briefing"
  - else → session_type = "analyst"
- Calls `get_chat_client(session_type)` to get the primary LLM
- Invokes the LLM with the rendered prompt
- On success: sets `state["raw_llm_output"]` and `state["provider_used"]` from settings.model_version_hash
- On `openai.RateLimitError` or `openai.APIStatusError` (status >= 500):
  - If `session_type == "applicant"`: do NOT fall back — set `state["error"] = "PRIMARY_UNAVAILABLE"`
    and return immediately. The graph will route to a 503 response node.
  - If `session_type in ("analyst", "briefing")`: call `get_fallback_client(session_type)`,
    retry the LLM call once, update `state["provider_used"]` with the Vertex model identifier
  - If fallback also fails: set `state["error"] = "ALL_PROVIDERS_UNAVAILABLE"` and return
- Log the provider used at INFO level on every invocation for SR 11-7 audit trail purposes

NODE: `parse_intent_node`
- Classifies `state["query"]` into one of the four intent values
- Use a lightweight prompt (no tools) against the primary LLM
- Intent classification does NOT trigger fallback logic — if the primary LLM is down, return
  `state["error"] = "PRIMARY_UNAVAILABLE"` immediately

NODE: `grade_documents_node`
- Iterates `state["retrieved_chunks"]`
- For each chunk, sets `relevance` to RELEVANT / IRRELEVANT / AMBIGUOUS based on a short
  grading prompt (one LLM call for all chunks batched together, not one per chunk)
- Sets `state["retrieval_sufficient"] = True` if len(RELEVANT chunks) >= 3

NODE: `error_node`
- Called when `state["error"]` is set
- Formats a structured error response: {"error": state["error"], "session_id": ..., "retry_after": 30}
- Sets `state["final_output"]` to this dict
- Does NOT call any LLM

IMPORTANT CONSTRAINTS:
- All nodes are pure async functions — no side effects except setting state keys
- No node writes to the database — persistence happens in a separate `persist_session_node`
  called at the END of the graph after all logic is complete
- Import `get_chat_client` and `get_fallback_client` from `app.llm.provider` only
- Do not import langchain_openai or langchain_google_vertexai directly in nodes.py
```

---

## Prompt 5 — LangGraph: Graph Definition with Conditional Fallback Routing

**File:** `backend/app/agent/graph.py` (new file — Sprint 2)
**Depends on:** Prompt 4 (nodes.py must exist)
**Acceptance criteria:** `compile_graph()` returns a runnable LangGraph `CompiledGraph`; `applicant` sessions never route to `error_node` via fallback (they fail fast); graph compiles without error.

```
You are building the LangGraph state machine for LucidCredit's corrective RAG agent.

Create `backend/app/agent/graph.py` that defines and compiles the full agent graph.

Import the following from `app.agent.nodes`:
  parse_intent_node, retrieve_node (stub — see below), grade_documents_node,
  reason_node, citation_enforcer_node (stub), confidence_score_node (stub),
  compliance_check_node (stub), format_output_node (stub),
  persist_session_node (stub), error_node

Import AgentState from `app.agent.state`.

For nodes not yet implemented (marked stub), create async stub functions in nodes.py
that accept state and return `{}` (no-op). They will be replaced in later sprints.

GRAPH STRUCTURE — implement this exact topology:

```
START
  → parse_intent_node
  → retrieve_node          (parallel fan-out: vector + sql + api tools)
  → grade_documents_node
  → [conditional: retrieval_sufficient?]
       YES → reason_node
       NO  → error_node ("INSUFFICIENT_RETRIEVAL")
  → [conditional: state["error"] set after reason_node?]
       YES → error_node
       NO  → citation_enforcer_node
  → confidence_score_node
  → [conditional: grounding_passed?]
       YES and audience==applicant → compliance_check_node
       YES and audience==analyst   → format_output_node
       NO  → error_node ("INSUFFICIENT_GROUNDING")
  → [conditional after compliance_check: compliance_passed?]
       YES → format_output_node
       NO  → error_node ("COMPLIANCE_FAILED")
  → persist_session_node
  → END
```

REQUIREMENTS:
- Use `StateGraph(AgentState)` from langgraph
- Add each node with `graph.add_node(name, fn)` where name matches the function name without "_node" suffix
  (e.g., node fn `reason_node` → graph node name `"reason"`)
- Use `graph.add_conditional_edges` for all conditional branches — do NOT use `add_edge` for branching
- The error_node must be reachable from multiple points — use a shared node named `"error"`
- Compile with a Redis checkpointer for session continuity:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    checkpointer = AsyncRedisSaver.from_conn_string(settings.redis_url)
    compiled = graph.compile(checkpointer=checkpointer)
- Wrap the compile in an async factory `async def compile_graph() -> CompiledGraph` that
  initializes the checkpointer and returns the compiled graph
- The compiled graph should be cached as a module-level variable after first call:
    _graph: CompiledGraph | None = None
    async def get_graph() -> CompiledGraph: ...
- Add `langgraph>=0.2.0` and `langgraph-checkpoint-redis>=0.1.0` to requirements.txt
```

---

## Prompt 6 — Compliance Hard Gate: Applicant Session 503 Handler

**File:** `backend/app/api/v1/applicant.py` (new file — Sprint 3)
**Depends on:** Prompts 4, 5
**Acceptance criteria:** When `state["error"] == "PRIMARY_UNAVAILABLE"` and `audience == "applicant"`, the endpoint returns HTTP 503 with `Retry-After: 30`; no fallback LLM is called; test asserts this explicitly.

```
You are implementing the applicant communications API endpoint for LucidCredit.

Create `backend/app/api/v1/applicant.py` — the FastAPI router for POST /v1/applicant/communication.

REQUIREMENTS:
- Request schema: ApplicantCommunicationRequest (Pydantic v2 BaseModel)
    application_id: UUID
    source: Literal["thinfile", "credit_risk_platform"]
    communication_type: Literal["decline", "approve", "counteroffer", "incomplete"]
    channel: Literal["email", "sms", "portal", "letter"]
    language: str = "en"

- Response schema: ApplicantCommunicationResponse
    session_id: UUID
    subject_line: str
    body: str
    adverse_action_notice: dict  # structured AAN per ECOA/FCRA
    compliance_validated: bool
    citations: list[dict]
    confidence_score: float

- The endpoint calls `get_graph()` and invokes the compiled LangGraph graph with the request
  payload as the initial AgentState

- CRITICAL: Implement explicit provider failure handling:
    if final_state.get("error") in ("PRIMARY_UNAVAILABLE", "ALL_PROVIDERS_UNAVAILABLE"):
        raise HTTPException(
            status_code=503,
            detail={"error": "inference_unavailable", "message": "Primary inference provider unavailable. Applicant communication cannot be generated without GPT-4.1. Please retry.", "retry_after": 30},
            headers={"Retry-After": "30"},
        )

  Do NOT return a 200 with degraded output. Do NOT call a fallback model.
  This is a regulatory requirement: non-compliant ECOA adverse action notices must
  never be delivered to applicants.

- If compliance_passed is False in the final state:
    raise HTTPException(
        status_code=422,
        detail={"error": "compliance_failure", "flags": final_state["compliance_flags"], "message": "Generated communication failed ECOA/FCRA validation. Review compliance flags."}
    )

- All exceptions should be logged with structlog at ERROR level with session_id in the log context

- Add the router to `backend/app/main.py` under the prefix "/v1/applicant"
```

---

## Prompt 7 — RAGAS Evaluation Harness (Front-Load to Sprint 1)

**File:** `backend/tests/eval/ragas_harness.py` (new file)
**Depends on:** Prompts 1, 2, 3 (provider layer and embedder must work)
**Acceptance criteria:** Running `pytest backend/tests/eval/ -m ragas` produces a RAGAS report with faithfulness, answer_relevancy, and context_recall scores; fails the test if faithfulness < 0.85 for analyst sessions or < 0.80 for briefing sessions.

```
You are building the RAGAS evaluation harness for LucidCredit's zero-hallucination acceptance tests.

IMPORTANT: This harness runs in Sprint 1, not Sprint 5. It establishes the baseline before
any agent code is written so that provider comparisons are data-driven, not assumed.

Create `backend/tests/eval/ragas_harness.py` and a golden dataset fixture at
`backend/tests/eval/golden/analyst_qa.json`.

RAGAS HARNESS REQUIREMENTS:
- Use the `ragas` Python package (add `ragas>=0.2.0` to requirements.txt)
- Evaluate three metrics:
    1. `faithfulness` — claims in the answer are grounded in the retrieved context
    2. `answer_relevancy` — the answer addresses the question
    3. `context_recall` — retrieved context contains the information needed to answer

- The harness accepts a `--provider` CLI flag: "azure_gpt41", "azure_gpt4o", "vertex_gemini15pro", "vertex_flash20"
  This selects the provider to evaluate by temporarily overriding Settings fields

- Thresholds (hard fail if not met):
    - Analyst sessions: faithfulness >= 0.85, answer_relevancy >= 0.80, context_recall >= 0.75
    - Briefing sessions: faithfulness >= 0.80, answer_relevancy >= 0.75, context_recall >= 0.70

- Golden dataset format for `analyst_qa.json` (provide 10 seed examples):
```json
[
  {
    "id": "qa_001",
    "session_type": "analyst",
    "question": "What are the primary SHAP drivers for decision ID dec-001?",
    "ground_truth": "The primary adverse factors are DTI ratio of 0.48 (weight: -0.31) and derogatory payment history in the last 12 months (weight: -0.22), per the SHAP explanation in the CRP API response.",
    "context": [
      "SHAP explanation: feature=dti_ratio value=0.48 shap_weight=-0.31",
      "SHAP explanation: feature=derog_12m value=1 shap_weight=-0.22",
      "Decision ID: dec-001, outcome: DECLINE, score: 0.34"
    ]
  },
  {
    "id": "qa_002",
    "session_type": "analyst",
    "question": "What is the required adverse action notice timeline under Reg B?",
    "ground_truth": "Under Regulation B (12 CFR Part 1002), a creditor must notify the applicant of adverse action within 30 days of receiving a completed application.",
    "context": [
      "Regulation B, 12 CFR § 1002.9(a)(1): A creditor shall notify an applicant of action taken within 30 days after receiving a completed application."
    ]
  },
  {
    "id": "qa_003",
    "session_type": "analyst",
    "question": "What counterfactual change would most improve the applicant's score for dec-001?",
    "ground_truth": "Reducing the DTI ratio from 0.48 to below 0.43 is projected to improve the credit score by approximately 12 points, per the counterfactual analysis.",
    "context": [
      "Counterfactual: primary_lever=dti_ratio, current=0.48, target=0.43, estimated_improvement=+12 points"
    ]
  },
  {
    "id": "qa_004",
    "session_type": "analyst",
    "question": "Does SR 11-7 require model performance monitoring for third-party vendor models?",
    "ground_truth": "Yes. SR 11-7 (Board of Governors, 2011) specifies that model risk management expectations apply to models used by a banking organization regardless of whether they are developed internally or by a vendor.",
    "context": [
      "SR 11-7, Section V: Model risk management expectations apply to vendor models as well as internally developed models."
    ]
  },
  {
    "id": "qa_005",
    "session_type": "analyst",
    "question": "What is the current 90-day delinquency rate for the gig_worker segment?",
    "ground_truth": "[REQUIRES_DB_RETRIEVAL] — this question cannot be answered from the regulatory corpus alone.",
    "context": [
      "No delinquency rate data was retrieved for the gig_worker segment in this context window."
    ]
  },
  {
    "id": "qa_006",
    "session_type": "briefing",
    "question": "Summarize the key risk drivers for DECLINE decisions in Q1 2026.",
    "ground_truth": "Based on the portfolio data, the leading adverse factors in Q1 2026 DECLINE decisions were elevated DTI ratios (present in 68% of declines), insufficient credit history (42%), and recent derogatory marks (31%).",
    "context": [
      "Portfolio summary Q1 2026: top_decline_factor_1=dti_ratio pct=0.68, top_decline_factor_2=thin_file pct=0.42, top_decline_factor_3=derog_marks pct=0.31"
    ]
  },
  {
    "id": "qa_007",
    "session_type": "analyst",
    "question": "What FCRA disclosure is required when adverse action is based on a credit report?",
    "ground_truth": "Under FCRA § 615(a), when adverse action is taken based wholly or partly on information in a consumer report, the creditor must provide the applicant with the name and address of the consumer reporting agency and notice of their right to obtain a free copy of the report.",
    "context": [
      "FCRA § 615(a): Any person who takes any adverse action with respect to any consumer that is based in whole or in part on any information contained in a consumer report shall provide the consumer with: (1) notice of the adverse action; (2) name and address of the CRA; (3) notice of the right to obtain a free copy."
    ]
  },
  {
    "id": "qa_008",
    "session_type": "analyst",
    "question": "What was the funded loan volume for the auto segment in March 2026?",
    "ground_truth": "[REQUIRES_DB_RETRIEVAL] — no funded loan volume data for March 2026 auto segment was retrieved.",
    "context": [
      "No funded loan volume data was found in the retrieved context for the auto segment in March 2026."
    ]
  },
  {
    "id": "qa_009",
    "session_type": "analyst",
    "question": "What does the model card for the ThinFile Engine say about protected class performance?",
    "ground_truth": "The ThinFile Engine model card reports that the model was evaluated for disparate impact on protected classes defined under ECOA. The Equal Opportunity Difference metric is within the ±0.10 threshold on gender and race proxies in the validation dataset.",
    "context": [
      "ThinFile Engine Model Card, Fairness Section: Equal Opportunity Difference by gender proxy: 0.04. By race proxy: -0.07. Threshold: ±0.10."
    ]
  },
  {
    "id": "qa_010",
    "session_type": "analyst",
    "question": "Explain the ECOA notice requirements for incomplete applications.",
    "ground_truth": "Under Reg B § 1002.9(c), if an application is incomplete, the creditor must notify the applicant of the information needed to complete the application within 30 days and provide the specific information required.",
    "context": [
      "Regulation B, 12 CFR § 1002.9(c)(2): Within 30 days after receiving an incomplete application, the creditor shall provide notice of incompleteness identifying the information required."
    ]
  }
]
```

PYTEST STRUCTURE:
- Use `@pytest.mark.ragas` marker (register in pytest.ini)
- The test parametrizes over session_types: ["analyst", "briefing"]
- For each session_type, filter the golden dataset, run RAGAS evaluation, and assert thresholds
- Print a summary table with metric scores per session_type on test completion
- Store evaluation results as JSON in `backend/tests/eval/results/<timestamp>_<provider>.json`
  for historical tracking

Add to `pytest.ini`:
```
markers =
    ragas: RAGAS evaluation tests (slow — run separately with pytest -m ragas)
```
```

---

## Prompt 8 — Model Version Pinning in Session Persistence

**File:** `backend/app/models/session.py` (new file — Sprint 2)
**Depends on:** Prompt 1 (`model_version_hash` property on Settings)
**Acceptance criteria:** Every `copilot_sessions` row stores the exact model identifier; querying sessions by model version is possible; Alembic migration file is generated.

```
You are implementing the session persistence model for LucidCredit.

Create `backend/app/models/session.py` with the SQLAlchemy 2.0 async ORM model for
the `copilot_sessions` table. This satisfies SR 11-7 model risk audit requirements.

REQUIREMENTS:
- Use SQLAlchemy 2.0 Mapped / mapped_column syntax (not legacy Column syntax)
- Table name: `copilot_sessions`
- Columns (match the PLAN.md schema exactly, adding the following):
    - All columns from PLAN.md's copilot_sessions table
    - `provider_model` VARCHAR(100) NOT NULL — stores the model_version_hash value
      e.g. "azure:gpt-4.1-2025-04-14" or "vertex:gemini-1.5-pro-002"
    - `provider_fallback_used` BOOLEAN DEFAULT FALSE — True if Vertex fallback was invoked
    - `context_token_count` INTEGER — approximate token count of the rendered prompt
    - `output_token_count` INTEGER — approximate token count of raw_llm_output

- Add indexes on:
    - `session_id` (primary key — auto-indexed)
    - `user_id` (queries by user for audit access API)
    - `created_at` (range queries for audit log export)
    - `provider_model` (model risk: query sessions by model version — required for SR 11-7 impact analysis)
    - `confidence_score` (filtering sessions below threshold for review)

- Also create `backend/app/models/citation.py` for the `citations` table (match PLAN.md schema)
  and `backend/app/models/feedback.py` for the `user_feedback` table

- Create `backend/app/models/__init__.py` that exports all three models and a `Base` (DeclarativeBase)

- After creating the models, generate an Alembic migration:
    alembic revision --autogenerate -m "add_copilot_sessions_citations_feedback"
  
  The alembic.ini is at `backend/alembic.ini`. Verify the migration file is created under
  `backend/alembic/versions/`. Do not run the migration — just generate it.

- Add a `Base.metadata.create_all` call to `backend/app/db/session.py` that runs on startup
  in development only (gated by `settings.environment == "development"`). In production,
  Alembic manages schema exclusively.
```

---

## Prompt 9 — Circuit Breaker & Provider Health Monitoring

**File:** `backend/app/llm/circuit_breaker.py` (new file)
**Depends on:** Prompt 2 (provider.py must exist)
**Acceptance criteria:** After 3 consecutive Azure OpenAI failures in a 60s window, `circuit_breaker.is_open("azure")` returns True; analyst sessions auto-route to Vertex; applicant sessions return 503; circuit resets after 120s.

```
You are implementing a lightweight circuit breaker for LucidCredit's LLM provider routing.

Create `backend/app/llm/circuit_breaker.py`.

REQUIREMENTS:
- Implement a simple in-process circuit breaker (no external dependency beyond Redis for
  distributed state in production — see below)
- State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
- Per-provider breaker keyed by provider name: "azure", "vertex"

CONFIGURATION (hardcoded constants — no settings fields needed):
    FAILURE_THRESHOLD = 3          # consecutive failures to trip OPEN
    RESET_TIMEOUT_SECONDS = 120    # seconds in OPEN before trying HALF_OPEN
    HALF_OPEN_MAX_CALLS = 1        # only 1 call allowed in HALF_OPEN state

PUBLIC API:
    async def record_success(provider: str) -> None
    async def record_failure(provider: str) -> None
    async def is_open(provider: str) -> bool  — True means DO NOT USE this provider
    async def get_state(provider: str) -> Literal["CLOSED", "OPEN", "HALF_OPEN"]

REDIS DISTRIBUTION:
- In production (settings.environment == "production"), use Redis (settings.redis_url) to
  share circuit state across multiple FastAPI worker processes
- In development, use a module-level dict (in-process state is fine for single-worker dev)
- Redis key pattern: `lucidcredit:circuit:{provider}:failures` (counter, TTL = RESET_TIMEOUT_SECONDS)
  and `lucidcredit:circuit:{provider}:state` (string: CLOSED/OPEN/HALF_OPEN)

INTEGRATION:
- Update `backend/app/llm/provider.py` to call `await record_failure(provider)` on exception
  and `await record_success(provider)` on successful response in `get_chat_client` and `get_fallback_client`
- Update `reason_node` in `nodes.py` to call `await is_open("azure")` BEFORE calling the
  primary client — if open, skip directly to fallback logic for analyst sessions or 503 for applicant sessions

OBSERVABILITY:
- Expose a `GET /v1/health/providers` endpoint that returns the circuit state for each provider:
    {"azure": {"state": "CLOSED", "failures": 0}, "vertex": {"state": "CLOSED", "failures": 0}}
  This is consumed by the ops team during incidents and by the /v1/health check in PLAN.md
```

---

## Prompt 10 — Gemini 2.0 Flash A/B Gate for Briefings

**File:** `backend/app/api/v1/briefing.py` (new file — Sprint 3)
**Depends on:** Prompts 2, 4, 5, 7 (RAGAS gate must pass before activating Flash)
**Acceptance criteria:** When `settings.briefing_use_flash = False`, requests use GPT-4.1 via Azure; when True, Gemini 2.0 Flash is used; a separate endpoint `POST /v1/briefing/evaluate-flash` runs the RAGAS gate and sets the flag if thresholds pass.

```
You are implementing the analyst briefing endpoint with a cost-optimization A/B gate for
Gemini 2.0 Flash routing in LucidCredit.

Create `backend/app/api/v1/briefing.py`.

REQUIREMENTS:
- Request schema: BriefingRequest (matches PLAN.md spec)
    scope: Literal["segment", "portfolio", "cohort", "product"]
    filters: dict  # date_range, decision, employment_status, etc.
    sections: list[Literal["executive_summary", "risk_distribution", "key_drivers", "fairness_indicators", "recommendations"]]
    audience_role: Literal["CRO", "analyst", "compliance_officer", "board"]

- Response schema: BriefingResponse
    session_id: UUID
    sections: dict[str, str]    # section_name → narrative
    citations: list[dict]
    confidence_score: float
    provider_used: str           # exposed for ops visibility
    sql_queries_executed: list[str]

- Main endpoint: POST /v1/briefing/generate
  - Calls get_graph() and invokes the LangGraph graph with intent="portfolio_brief", audience="analyst"
  - The provider selection happens inside reason_node based on settings.briefing_use_flash

- Admin endpoint: POST /v1/briefing/evaluate-flash (protected by API key — analyst admin only)
  This endpoint:
  1. Runs the RAGAS harness against the 10-item golden set using the vertex_flash20 provider
  2. Checks: faithfulness >= 0.80 AND answer_relevancy >= 0.75 AND context_recall >= 0.70
  3. If ALL pass: sets settings.briefing_use_flash = True (in-memory for the running process)
     AND writes FLASH_GATE_PASSED=true to a Redis key `lucidcredit:feature_flags:briefing_flash`
     so it persists across restarts
  4. Returns the RAGAS scores and whether the gate opened:
     {"gate_opened": true, "scores": {"faithfulness": 0.83, "answer_relevancy": 0.78, "context_recall": 0.72}}
  5. If ANY threshold fails: returns gate_opened=false with scores — settings.briefing_use_flash stays False

- On startup, `backend/app/main.py` should check Redis for `lucidcredit:feature_flags:briefing_flash`
  and set `settings.briefing_use_flash` accordingly, so the Flash gate persists across restarts

NOTE: The gate can only be opened by the evaluate-flash endpoint and can be closed by setting
the Redis key to "false". There is no UI toggle — this is intentional to ensure RAGAS validation
always precedes Flash activation.
```

---

## Prompt 11 — Integration Test: End-to-End Provider Routing

**File:** `backend/tests/integration/test_provider_routing.py` (new file)
**Depends on:** All prior prompts
**Acceptance criteria:** All 6 test scenarios pass; no real LLM calls are made (all mocked); provider routing logic is verified by inspecting which client was instantiated.

```
You are writing integration tests for LucidCredit's provider routing logic.

Create `backend/tests/integration/test_provider_routing.py`.

Test the following 6 scenarios using pytest-asyncio and unittest.mock:

SCENARIO 1: Analyst session — Azure primary succeeds
  - Mock AzureChatOpenAI to return a valid grounded response
  - Assert state["provider_used"] == "azure:gpt-4.1-2025-04-14"
  - Assert state["error"] is None

SCENARIO 2: Analyst session — Azure fails, Vertex fallback succeeds
  - Mock AzureChatOpenAI to raise openai.APIStatusError(status_code=503)
  - Mock ChatVertexAI to return a valid grounded response
  - Assert state["provider_used"] == "vertex:gemini-1.5-pro-002"
  - Assert state["error"] is None

SCENARIO 3: Applicant session — Azure primary succeeds
  - Mock AzureChatOpenAI to return a valid grounded response
  - Assert state["provider_used"] == "azure:gpt-4.1-2025-04-14"
  - Assert state["error"] is None

SCENARIO 4: Applicant session — Azure fails (CRITICAL: must NOT fall back)
  - Mock AzureChatOpenAI to raise openai.RateLimitError
  - Assert get_fallback_client was NEVER called (use Mock.assert_not_called())
  - Assert state["error"] == "PRIMARY_UNAVAILABLE"
  - Assert the API endpoint returns HTTP 503 with Retry-After header

SCENARIO 5: Analyst session — Azure AND Vertex both fail
  - Mock both clients to raise exceptions
  - Assert state["error"] == "ALL_PROVIDERS_UNAVAILABLE"
  - Assert the API endpoint returns HTTP 503

SCENARIO 6: Briefing session with Flash gate active
  - Set settings.briefing_use_flash = True
  - Mock ChatVertexAI (gemini-2.0-flash-001) to return a valid response
  - Assert state["provider_used"] == "vertex:gemini-2.0-flash-001"
  - Assert AzureChatOpenAI was NOT called for the LLM step

Use pytest.fixture to inject a test Settings instance with:
  - llm_provider_mode="azure"
  - azure_openai_endpoint="https://test.openai.azure.com/"
  - google_project_id="test-project"
  - analyst_fallback_enabled=True
  - environment="test"

All six scenarios must pass before any production deployment of the provider routing layer.
```

---

## Prompt 12 — .env.example and Docker Compose Update

**File:** `.env.example`, `docker-compose.yml`
**Depends on:** Prompt 1 (new Settings fields)
**Acceptance criteria:** `docker-compose up` starts all services; Azure and Vertex credentials are read from host environment; no credentials are hardcoded in docker-compose.yml.

```
You are updating LucidCredit's environment configuration for the multi-provider deployment.

TASK 1 — Update `.env.example` (already partially done in Prompt 1):
Ensure the following sections exist with the correct variable names and comments.
Do not remove any existing variables.

Add section after existing OpenAI variables:
```
# === Azure OpenAI (Primary — required in staging/prod) ===
# Replace direct OpenAI API with Azure OpenAI for data residency compliance (BAA/DPA)
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_ANALYST=gpt-4.1-2025-04-14
AZURE_OPENAI_DEPLOYMENT_APPLICANT=gpt-4.1-2025-04-14
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-large

# === Provider Mode ===
# "azure" = use Azure OpenAI (required in staging/production)
# "openai_direct" = use direct OpenAI API (development only)
LLM_PROVIDER_MODE=azure

# === Vertex AI / Gemini (Fallback for analyst sessions) ===
# Application Default Credentials (ADC) are preferred over an API key in production.
# For local dev with a service account: set GOOGLE_APPLICATION_CREDENTIALS to the JSON key path.
GOOGLE_PROJECT_ID=
GOOGLE_LOCATION=us-central1
VERTEX_MODEL_ANALYST_FALLBACK=gemini-1.5-pro-002
VERTEX_MODEL_BATCH_BRIEFING=gemini-2.0-flash-001
ANALYST_FALLBACK_ENABLED=true
BRIEFING_USE_FLASH=false
```

TASK 2 — Update `docker-compose.yml`:
- The backend service should read the new env vars from the host environment (use env_file: .env)
- Add a comment block above the backend service explaining the provider configuration:
```yaml
# LLM Provider Configuration
# Primary: Azure OpenAI (set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)
# Fallback: Vertex AI (set GOOGLE_PROJECT_ID; use ADC or GOOGLE_APPLICATION_CREDENTIALS)
# See .env.example for full variable list.
# NEVER commit .env with real API keys.
```
- Add a volume mount for Google Application Default Credentials (ADC) if the env var
  GOOGLE_APPLICATION_CREDENTIALS is set on the host:
    volumes:
      - ${GOOGLE_APPLICATION_CREDENTIALS:-/dev/null}:/app/credentials/gcp-key.json:ro
  and set environment variable:
    GOOGLE_APPLICATION_CREDENTIALS: /app/credentials/gcp-key.json

- Do not change any existing service definitions (postgres, redis, etc.)
```

---

## Execution Order & Sprint Mapping

| Prompt | Sprint | Blocking? | Description |
|---|---|---|---|
| **1** — Config Layer | Sprint 1, Day 1 | **YES** — blocks all others | Azure/Vertex settings in BaseSettings |
| **2** — Client Factory | Sprint 1, Day 2 | **YES** — blocks 3, 4, 9 | LLM provider abstraction |
| **3** — Embedder Migration | Sprint 1, Day 2 | **YES** — blocks RAG pipeline | Azure OpenAI for embeddings |
| **7** — RAGAS Harness | Sprint 1, Day 3 | **YES** — blocks Gemini A/B | Golden set + evaluation framework |
| **12** — Env / Docker | Sprint 1, Day 3 | No | Environment configuration |
| **8** — Session Models | Sprint 2, Day 1 | No | ORM + Alembic migration |
| **4** — Reason Node | Sprint 2, Day 2 | **YES** — blocks 5, 6, 10 | LangGraph nodes with provider routing |
| **5** — Graph Definition | Sprint 2, Day 3 | **YES** — blocks 6, 10, 11 | Full LangGraph state machine |
| **9** — Circuit Breaker | Sprint 2, Day 4 | No | Provider health + auto-failover |
| **6** — Applicant API | Sprint 3, Day 1 | No | 503 hard gate for applicant sessions |
| **10** — Briefing Flash Gate | Sprint 3, Day 2 | No | Cost optimization A/B |
| **11** — Integration Tests | Sprint 3, Day 3 | **YES for deploy** | Provider routing test suite |

---

*LucidCredit — because every credit decision deserves a clear explanation.*
