# LucidCredit — System Architecture White Paper

**Version:** 1.0 · **Date:** May 23, 2026  
**Authors:** LucidCredit Engineering  
**Status:** Internal Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
   - 2.1 [High-Level Overview](#21-high-level-overview)
   - 2.2 [Frontend Layer](#22-frontend-layer)
   - 2.3 [API Gateway Layer](#23-api-gateway-layer)
   - 2.4 [LangGraph Agent Graph](#24-langgraph-agent-graph)
   - 2.5 [Retrieval-Augmented Generation (RAG) Pipeline](#25-retrieval-augmented-generation-rag-pipeline)
   - 2.6 [Zero-Hallucination Enforcement Layer](#26-zero-hallucination-enforcement-layer)
   - 2.7 [Compliance Layer](#27-compliance-layer)
   - 2.8 [Persistence and Session Layer](#28-persistence-and-session-layer)
   - 2.9 [LLM Provider Abstraction](#29-llm-provider-abstraction)
   - 2.10 [Tool Ecosystem](#210-tool-ecosystem)
   - 2.11 [Integration Architecture](#211-integration-architecture)
3. [Data Flow: End-to-End Request Lifecycle](#3-data-flow-end-to-end-request-lifecycle)
4. [Advantages](#4-advantages)
5. [Drawbacks](#5-drawbacks)
6. [Limitations Users Must Be Aware Of](#6-limitations-users-must-be-aware-of)
7. [Improvement Roadmap](#7-improvement-roadmap)
8. [Glossary](#8-glossary)

---

## 1. Executive Summary

LucidCredit is a production-grade, zero-hallucination AI analytical copilot for credit decisioning. It provides natural language explanations of credit decisions grounded entirely in retrieved data — policy documents, API payloads, and database records — never from parametric (model-memorised) knowledge alone.

The system serves two distinct audiences through separate reasoning pipelines:

- **Risk Analysts and CROs** — SHAP-grounded briefings, PD distributions, portfolio stress summaries, and NL→SQL analytics against BigQuery
- **Applicants** — Plain English ECOA/FCRA-compliant adverse action notices, approval summaries, and counterfactual guidance

Every generated claim is enforced against retrieved context at inference time via the `CitationEnforcer`. Claims that cannot be grounded above a configurable cosine similarity threshold are **stripped before the response reaches the caller** and appended to an immutable audit trail. This makes the hallucination boundary a hard technical constraint, not a prompt instruction.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLIENT TIER                                                           │
│                                                                        │
│   Next.js 15 UI (port 3090 dev / 3010 Docker)                         │
│   ├── Analyst page       (query builder, SHAP viz)                    │
│   ├── Applicant view     (adverse action, approval notices)           │
│   └── Audit trail page   (citation inspector)                         │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │  REST / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  API GATEWAY TIER — FastAPI (port 8090)                                │
│                                                                        │
│   /api/analyst/query          /api/analyst/briefing                   │
│   /api/applicant/narrative    /api/applicant/adverse-action           │
│   /api/audit/{id}             /api/audit/{id}/citations               │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  AGENT TIER — LangGraph State Machine                                  │
│                                                                        │
│  START → parse_intent → retrieve → grade_documents                    │
│        → [retrieval_sufficient?]                                       │
│            YES → reason (LLM call)                                    │
│            NO  → error(INSUFFICIENT_RETRIEVAL)                        │
│        → citation_enforcer                                            │
│        → confidence_score                                             │
│        → [grounding_passed?]                                          │
│            applicant  → compliance_check → format_output             │
│            analyst    → format_output                                 │
│        → persist_session → END                                        │
│                                                                        │
│  State checkpointed in Redis; sessions survive server restarts        │
└───────┬────────────────────────────────────────────┬──────────────────┘
        │                                            │
        ▼                                            ▼
┌───────────────────────┐               ┌────────────────────────────┐
│  RETRIEVAL TIER        │               │  LLM PROVIDER TIER         │
│                        │               │                            │
│  pgvector (pg16)       │               │  Primary: Azure OpenAI     │
│  Hybrid retriever:     │               │  GPT-4.1-2025-04-14        │
│  ├── Dense (65%)       │               │  text-embedding-3-large    │
│  └── BM25 re-rank(35%) │               │  (3072-dim)                │
│                        │               │                            │
│  Policy doc corpus     │               │  Fallback: Vertex AI       │
│  Reg B, FCRA, SR 11-7  │               │  Gemini 1.5 Pro            │
│  Model cards           │               │  (analyst only)            │
└───────────────────────┘               └────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL DATA TIER                                                    │
│                                                                        │
│  credit-risk-platform (port 8001)   ThinFile Engine (port 8000)       │
│  ├── Decision API                   ├── Thin-file scores              │
│  ├── SHAP values                    └── Adverse-action codes          │
│  ├── PD scores                                                        │
│  └── BigQuery analytics (NL→SQL)                                      │
│                                                                        │
│  PostgreSQL + pgvector (port 5440)  Redis (port 6380)                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Frontend Layer

**Technology:** Next.js 15 (TypeScript), Tailwind CSS  
**Ports:** 3090 (local dev), 3010 (Docker)

The frontend is a thin API consumer — all business logic lives in the backend. It proxies all `/api/*` requests to the FastAPI backend via `next.config.ts` rewrites, eliminating CORS issues in development. Four primary pages:

| Page | Audience | Key Features |
|------|----------|--------------|
| Analyst | Risk teams | NL query input, SHAP visualisation, portfolio briefing |
| Query Builder | Internal | Structured query construction, intent selector |
| Applicant | Borrowers | Decision explanation, adverse action notice, counterfactual guide |
| Audit Trail | Compliance/Governance | Citation inspector, confidence scores, suppression log |

### 2.3 API Gateway Layer

**Technology:** FastAPI (Python 3.11+), async/await throughout

Six versioned REST endpoints under `/api/v1/`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyst/query` | POST | Analyst Q&A with SHAP grounding |
| `/analyst/briefing` | POST | Portfolio briefing narrative |
| `/applicant/narrative` | POST | Applicant-facing decision explanation |
| `/applicant/adverse-action` | POST | ECOA-compliant adverse action notice |
| `/audit/{session_id}` | GET | Full audit trail for a session |
| `/audit/{session_id}/citations` | GET | All citations with similarity scores |

All POST endpoints return structured JSON with an inline `confidence` field and `citations` array. The API is fully documented via Swagger UI at `/docs`.

### 2.4 LangGraph Agent Graph

**Technology:** LangGraph (StateGraph), Redis checkpointing

The core reasoning pipeline is a compiled directed acyclic graph (DAG) of async node functions. The agent state (`AgentState`) is a typed `TypedDict` that flows through each node — no node communicates with another except via state mutations.

**Graph topology:**

```
START
  └─► parse_intent ──(UNSUPPORTED_QUERY shortcut)──► format_output
          │
          ▼
       retrieve  (vector + API + SQL tools in parallel)
          │
          ▼
    grade_documents  (LLM-as-judge: RELEVANT / IRRELEVANT / AMBIGUOUS)
          │
     ┌────┴─────────────────────┐
  sufficient?                not sufficient
     │                           │
     ▼                           ▼
   reason  (single LLM call)   error(INSUFFICIENT_RETRIEVAL)
     │
     ▼
  citation_enforcer  (embed each sentence, score vs chunks)
     │
     ▼
  confidence_score  (0.30×recall + 0.50×citation_rate + 0.20×unverified_penalty)
     │
     ├──(score < threshold)──► error(INSUFFICIENT_GROUNDING)
     │
     ├──(applicant audience)──► compliance_check ──(fail)──► error(COMPLIANCE_FAILED)
     │                                 │
     └──(analyst audience)─────────────┘
                                       │
                                       ▼
                                  format_output  (SR 11-7 injection for analyst)
                                       │
                                       ▼
                                  persist_session
                                       │
                                       ▼
                                      END
```

**Key design decisions:**
- Only `reason_node` makes an LLM call — all other nodes are deterministic
- `parse_intent_node` can short-circuit the entire graph for unsupported queries without burning tokens
- A `fast_path` flag allows `reason_node` to bypass the LLM entirely for Tier-1 latency queries, using the top retrieved chunk directly
- Session continuity across requests is provided by the Redis checkpointer — multi-turn conversations maintain history without the client resending it

### 2.5 Retrieval-Augmented Generation (RAG) Pipeline

**Technology:** pgvector (PostgreSQL 16), SQLAlchemy 2.0 async, `rank_bm25` (optional)

The retriever implements a hybrid two-stage pipeline:

**Stage 1 — Dense retrieval**
- Query is embedded using `text-embedding-3-large` (3072 dimensions)
- pgvector performs cosine similarity search, returning `top_k × 3` (over-fetch by 3× to give BM25 a sufficient candidate pool)
- Embedding dimension: 3072 (must match pgvector column definition; re-ingestion required on model change)

**Stage 2 — BM25 re-ranking**
- BM25 scores are computed over the dense-retrieved candidates
- Final score: `0.65 × dense_score + 0.35 × bm25_norm`
- Falls back to a built-in TF-IDF scorer if `rank_bm25` is not installed

**Corpus:**
- Regulatory documents: Regulation B (ECOA), FCRA § 615, SR 11-7 / OCC 2011-12
- Model cards: probability of default model, SHAP feature definitions
- Ingested via `policy_doc_ingestor.py` using `ingest_corpus.py` script

### 2.6 Zero-Hallucination Enforcement Layer

This is the most architecturally distinctive component and the system's primary differentiator. It operates in two sequential passes after generation.

**CitationEnforcer**

```
narrative
    │
    ▼
split_into_sentences()
    │
    ▼
batch_embed(sentences)   ←── text-embedding-3-large
    │
    ▼
for each sentence:
    cosine_similarity(sentence_embedding, chunk_embeddings)
    │
    ├── similarity ≥ min_citation_similarity (default 0.85) ──► kept + cited
    └── similarity < threshold OR [UNVERIFIED] tag ──────────► suppressed + logged
    │
    ▼
grounded_narrative = join(kept sentences)
```

The `[UNVERIFIED]` marker is an LLM self-flagging convention — the system prompt instructs the model to tag any claim it cannot support from context. Both self-flagged sentences and low-similarity sentences are suppressed.

**ConfidenceScorer**

Produces a calibrated scalar score in [0.0, 1.0] from three signals:

$$\text{confidence} = 0.30 \times \text{retrieval\_recall} + 0.50 \times \text{citation\_rate} + 0.20 \times (1 - \text{unverified\_rate})$$

Where:
- `retrieval_recall` = fraction of retrieved chunks graded RELEVANT
- `citation_rate` = fraction of narrative sentences that passed grounding
- `unverified_rate` = fraction of sentences LLM self-marked as [UNVERIFIED]

Responses below `min_confidence_score` (default 0.75) are rejected before reaching the caller. A lower threshold (configurable, ~0.30–0.45) applies to advisory "why/explain/how" analyst queries where synthesised reasoning from domain knowledge is expected.

### 2.7 Compliance Layer

**ECOA/FCRA Validator (`ecoa_validator.py`)**

A deterministic rule-based gate for all applicant-facing outputs. No LLM judgment is used — rules are pure regex patterns. Enforces:

1. **ECOA-001** — Discriminatory attribution detection: flags any text attributing a credit decision to a protected characteristic (race, sex, age, national origin, marital status, etc.). Critically, the patterns require a credit outcome verb in proximity — standard equal-treatment disclaimers that mention protected characteristics do not trigger the rule.
2. **ECOA-002** — Reg B § 202.9 adverse action reason requirement: at least one specific reason must be present
3. **FCRA-001** — § 615 rights disclosure: required when a consumer report was used
4. **ECOA-003** — Discouraged language patterns: evasive or vague adverse action language
5. **ECOA-004** — Rights waiver language detection

**SR 11-7 Disclosure Injector (`sr117_disclosures.py`)**

Appends a structured model risk disclosure footer to all analyst-facing and briefing outputs. Contents include:
- Session ID and provider/model identity
- Known model limitations (retrieval freshness, out-of-sample risk, heuristic confidence scores)
- Validation status: "INTERNAL USE — Pending formal model validation"
- Performance degradation warning for macroeconomic regime changes

Applicant-facing outputs are explicitly excluded — SR 11-7 language is not appropriate for consumer disclosures.

### 2.8 Persistence and Session Layer

**PostgreSQL (port 5440)**
- `policy_docs` — RAG corpus (chunks, embeddings as pgvector column)
- `copilot_sessions` — immutable session records (query, intent, audience, retrieved chunks, rendered prompt, raw LLM output, grounded narrative, confidence score, feedback signal)
- `citations` — per-session citation records with similarity scores

**Redis (port 6380)**
- LangGraph `AsyncRedisSaver` checkpointer — persists `AgentState` between requests in the same session
- Enables multi-turn conversations where follow-up questions can reference prior answers without the client resending full history
- Falls back to an in-memory checkpointer for local dev without Redis

### 2.9 LLM Provider Abstraction

**Technology:** LangChain `BaseChatModel`, `lru_cache` instance pooling

```
Session type          Primary provider              Fallback provider
──────────────────────────────────────────────────────────────────────
analyst               Azure GPT-4.1-2025-04-14      Vertex Gemini 1.5 Pro
applicant             Azure GPT-4.1-2025-04-14      *** FORBIDDEN ***
briefing              Azure GPT-4.1-2025-04-14      Gemini 2.0 Flash (A/B gate)
embedding (all)       Azure text-embedding-3-large  none
```

Applicant sessions are deliberately forbidden from using the Vertex fallback. The rationale: ECOA/FCRA compliance of Gemini outputs has not been validated, and regulatory exposure from a non-compliant adverse action notice is too high.

Provider clients are cached by `(endpoint, deployment, api_version, api_key)` tuple — instances are reused across requests without being module-level singletons that break test patching.

### 2.10 Tool Ecosystem

The agent has five tools it can invoke during the `retrieve_node` phase:

| Tool | Source | What it retrieves |
|------|--------|-------------------|
| `vector_tool` | pgvector | Policy documents, regulatory corpus, model cards |
| `crp_api_tool` | credit-risk-platform REST API | Decision explanations, SHAP values, audit records, portfolio metrics |
| `analytics_api_tool` | CRP Analytics semantic layer (NL→SQL→BigQuery) | Live portfolio metrics, delinquency rates, charge-offs, vintage performance |
| `thinfile_tool` | ThinFile Engine REST API | Thin-file model scores, alternative data signals |
| `sql_tool` | Direct PostgreSQL (read-only) | Loan applications, features, funded loans, payment history, audit logs |

**SQL tool security model:**
- Only `SELECT` statements accepted (checked by token parsing, not just `startswith` — prevents `WITH...SELECT` injection)
- Table allow-list enforced: queries touching tables outside the list raise `SqlSecurityError` before reaching the database
- Bind parameters always used — no string interpolation
- Database user has `SELECT`-only grants (defence-in-depth)
- Hard row cap: 200 rows per query to prevent OOM

**Analytics API tool hallucination guard:**
When the analytics API returns rows, `_validate_rows_for_question()` checks whether the returned column names are plausibly related to the question's subject. If the columns appear unrelated (e.g., LTV columns returned for a "prepayment rate" question), the tool returns a "no_matching_field" chunk instead of passing irrelevant data to the LLM.

### 2.11 Integration Architecture

LucidCredit is a **read-only consumer** of upstream systems:

```
credit-risk-platform ──► LucidCredit ◄── ThinFile Engine
(decisions, SHAP, BQ)                    (thin-file scores)
```

- **Never writes to or modifies upstream systems**
- Integrates exclusively via versioned REST APIs with API-key authentication
- No shared database schemas — integration is purely over the network boundary
- `CRP_API_BASE_URL` and `THINFILE_API_BASE_URL` are configurable per environment

---

## 3. Data Flow: End-to-End Request Lifecycle

A single analyst query traverses the following path:

```
1. POST /api/analyst/query
   {"query": "What drove the denial for application A-123?",
    "context_payload": {"application_id": "A-123"},
    "session_id": "uuid"}
   
2. parse_intent_node
   → intent = "explain_decision"
   → audience = "analyst"
   → Checks broad query decomposition patterns (regex)
   → fast_path = False

3. retrieve_node  (parallel tool calls)
   → crp_api_tool.fetch_decision_context("A-123")
     ← [explanation_chunk, audit_chunk]
   → vector_tool.retrieve_policy_docs("denial reasons credit decision")
     ← [reg_b_chunk, model_card_chunk]
   → 4 RetrievedChunks total

4. grade_documents_node
   → LLM grades each chunk: RELEVANT / IRRELEVANT / AMBIGUOUS
   → graded_chunks = [relevant_chunks only]
   → retrieval_sufficient = True (≥ 1 RELEVANT chunk)

5. reason_node  (ONLY LLM call in the pipeline)
   → Renders prompt from analyst_system.md template + graded_chunks
   → Calls Azure GPT-4.1
   → raw_llm_output = "The application was denied primarily due to [UNVERIFIED] a high DTI..."
   → provider_used = "azure:gpt-4.1-2025-04-14"

6. citation_enforcer_node
   → Splits raw output into sentences
   → batch_embed(sentences) + batch_embed(chunk_content)
   → Sentence "The application was denied primarily due to..." → similarity 0.91 ✓
   → Sentence "a high DTI..." self-flagged [UNVERIFIED] → SUPPRESSED
   → grounded_narrative = (sentences that passed)
   → suppressed_claims = [{"claim_text": "a high DTI...", "reason": "self_flagged"}]

7. confidence_score_node
   → retrieval_recall = 1.0  (all chunks were RELEVANT)
   → citation_rate = 0.75   (3 of 4 sentences grounded)
   → unverified_rate = 0.25
   → confidence_score = 0.30×1.0 + 0.50×0.75 + 0.20×0.75 = 0.825
   → grounding_passed = True  (0.825 ≥ 0.75 threshold)

8. format_output_node  (analyst path — skips compliance check)
   → Injects SR 11-7 disclosure footer
   → Appends conversation_history entry
   → final_output = {"answer": ..., "citations": [...], "confidence": 0.825}

9. persist_session_node
   → Writes copilot_sessions row
   → Writes citations rows

10. Response returned to caller
    {"answer": "...", "citations": [...], "confidence": 0.825,
     "suppressed_claims": 1, "provider": "azure:gpt-4.1-2025-04-14"}
```

---

## 4. Advantages

### 4.1 Hard Hallucination Boundary

The most significant architectural advantage is that hallucination prevention is a **hard technical constraint, not a soft prompt instruction**. The `CitationEnforcer` operates post-generation at the embedding similarity level — the model cannot include uncited claims even if it tries. This is fundamentally more reliable than "don't hallucinate" instructions in the system prompt, which LLMs can and do violate under adversarial inputs or distribution shift.

### 4.2 Deterministic, Auditable Compliance

The ECOA/FCRA validator is entirely regex-based — no LLM judgment. This means compliance behaviour is:
- **Deterministic** — same input always produces same compliance result
- **Testable** — rule coverage can be measured exactly with unit tests
- **Auditable** — regulators can inspect the exact rule that fired
- **Fast** — nanoseconds, not LLM latency

This is the correct design for a regulatory gate. LLM-based compliance checking introduces non-determinism into a domain where determinism is legally required.

### 4.3 Immutable, Queryable Audit Trail

Every session persists the full chain: input query → retrieved chunks → graded chunks → rendered prompt → raw LLM output → grounded narrative → confidence score → citations → suppressed claims → compliance flags. This provides:
- SR 11-7 compliance evidence for model governance examinations
- ECOA adverse action audit trail for regulatory review
- Full reproducibility: any session can be replayed from stored inputs
- `GET /api/audit/{session_id}/citations` with per-citation similarity scores

### 4.4 Audience-Aware Routing

Analyst and applicant sessions use different system prompts, different compliance paths (SR 11-7 vs ECOA/FCRA), different confidence thresholds (advisory queries get a lower threshold), and different fallback policies (Vertex fallback forbidden for applicant sessions). This prevents the system from accidentally delivering SHAP values and probability-of-default bands to applicants, or diluting regulatory-compliant applicant language with technical register.

### 4.5 Hybrid Retrieval Quality

The BM25 re-ranking layer addresses a known weakness of pure dense retrieval: queries with domain-specific terminology (FICO, DTI, SHAP, Reg B, § 615) often have poor cosine similarity to general-language embeddings but strong BM25 scores. The 65/35 dense/BM25 blend provides better recall on regulatory and model-specific terminology without sacrificing semantic search quality for natural language queries.

### 4.6 Defense-in-Depth Security

Multiple independent security layers for data access:
1. SQL `SELECT`-only enforcement at the Python layer (token parsing)
2. Table allow-list at the Python layer
3. Database user with `SELECT`-only grants at the PostgreSQL layer
4. Row cap (200 rows) to prevent OOM
5. API-key authentication on all external tool calls
6. Read-only consumer pattern — no write path to any upstream system

### 4.7 Provider Resilience

The Vertex AI fallback for analyst sessions provides resilience against Azure OpenAI outages. The provider abstraction (`BaseChatModel`) means swapping providers requires only configuration changes, not code changes. The `lru_cache` client pooling avoids connection churn without creating untestable module-level singletons.

### 4.8 Multi-Turn Session Continuity

Redis checkpointing of the full `AgentState` means follow-up questions can reference prior answers without the client resending conversation history. For portfolio briefings where an analyst asks sequential drill-down questions, this dramatically reduces latency and payload size.

### 4.9 Narrow Surface Area

LucidCredit's scope is deliberately narrow: explain and brief, never decide. It cannot modify credit decisions, it cannot write to upstream systems, and it has no administrative interface. A compromised LucidCredit instance is a read-only data exposure risk, not a credit decision manipulation risk.

---

## 5. Drawbacks

### 5.1 Embedding Latency at Citation Enforcement

The `CitationEnforcer` embeds every generated sentence and every relevant chunk content on every request. For a narrative with 10 sentences and 8 relevant chunks, this is 18 embedding API calls (batched, but still). At Azure's typical 200–400ms per batch, citation enforcement adds 400–800ms to every request. This makes the end-to-end P50 latency for analyst queries ~4–9 seconds on uncached requests — well above the 2.5-second Tier-1 target.

### 5.2 Vendor Lock-in at the Embedding Layer

The entire retrieval and citation enforcement pipeline is coupled to `text-embedding-3-large` (3072 dimensions). Changing the embedding model or provider requires a **full corpus re-ingestion** because existing vectors are incompatible with vectors from a different model or dimension. There is no embedding version column in the `policy_docs` table, making incremental migration impossible without a schema change.

### 5.3 Single LLM Reasoning Step

The `reason_node` makes a single LLM call with all retrieved context in the prompt. For complex analytical queries requiring multi-step reasoning (e.g., "compare the delinquency trend against macro conditions and explain the divergence"), a single pass is insufficient. The current architecture cannot chain reasoning steps — it retrieves everything upfront and reasons once.

### 5.4 Regex Fragility in the ECOA Validator

The discriminatory attribution patterns are sophisticated multi-clause regexes covering eight distinct grammatical structures. This breadth was necessary to achieve adequate coverage, but the patterns are brittle: minor paraphrasing (e.g., passive vs active voice, gerunds, ellipsis) can evade detection. There are known gap categories (e.g., implication through statistically protected proxies — ZIP codes correlated with race — which pass all current patterns).

### 5.5 No Streaming Response

The architecture does not support streaming output to the frontend. The `CitationEnforcer` requires the **complete** generated narrative before it can begin embedding sentences. Streaming and post-generation citation enforcement are architecturally incompatible — the entire generation must be buffered before enforcement begins. This contributes to the perceived latency problem.

### 5.6 Stateless Query Decomposition

Broad query decomposition (e.g., "portfolio health" → 5 sub-queries) is handled by a hard-coded regex dispatch table in `nodes.py`. Adding new query decompositions requires code changes and redeployment. There is no dynamic or learned decomposition.

### 5.7 Context Window Pressure at Scale

The `reason_node` renders all graded chunks into a single prompt. For a complex BigQuery analytics query that returns large result sets across multiple sub-questions (e.g., "run all BigQuery analysis" → 18 sub-queries), the aggregated context can approach the model's context window limit, causing truncation or refusals.

### 5.8 Limited Feedback Loop

The `copilot_sessions` table has a `feedback_signal` column, but there is no automated loop from feedback to retrieval tuning, threshold adjustment, or prompt improvement. The RAGAS evaluation is run manually and offline — there is no continuous quality monitoring in production.

---

## 6. Limitations Users Must Be Aware Of

### 6.1 Responses Are Bounded by What Was Retrieved

**The system can only tell you what it found.** If the relevant policy document is not in the corpus, or the CRP API does not have the decision record, the system will return an `INSUFFICIENT_RETRIEVAL` error rather than fabricate an answer. This is by design — but users must understand that a refusal to answer is not always a system failure. It may mean the information simply is not available in the connected data sources.

### 6.2 Confidence Scores Are Heuristic, Not Statistical

The confidence score (0.0–1.0) is a weighted combination of retrieval recall, citation rate, and unverified rate. It is a useful proxy for answer quality but is **not a statistically calibrated probability**. A confidence score of 0.85 does not mean there is an 85% probability the answer is correct. Users should not treat the confidence score as a substitute for human review on high-stakes decisions.

### 6.3 Policy Corpus Freshness

Retrieval quality depends on the freshness of ingested documents. If Regulation B is amended or the SR 11-7 guidance is updated, the corpus must be manually re-ingested. There is no automatic corpus refresh mechanism. Users relying on the system for regulatory guidance should verify that the corpus is current before acting on answers.

### 6.4 The SR 11-7 Footer Is Not a Formal Validation

The SR 11-7 disclosure injected into analyst outputs explicitly states: **"INTERNAL USE — Pending formal model validation. This model is approved for analytical assistance only. It is NOT approved as a sole-input credit decision tool."** This system must not be used as the sole basis for credit decisions. Human review by a qualified credit analyst is required.

### 6.5 Applicant Fallback Is Prohibited

Applicant-facing sessions do not have a fallback provider. If Azure OpenAI is unavailable, applicant communications cannot be generated. This is intentional — ECOA/FCRA compliance of any alternative provider has not been validated — but it means planned maintenance windows will interrupt applicant-facing functionality.

### 6.6 The SQL Tool Is Scoped to a Fixed Table Allow-List

The `sql_tool` only executes queries against a specific set of pre-approved tables. Analysts cannot use natural language to query arbitrary tables. If a new data source needs to be queried, the allow-list in `sql_tool.py` must be updated by an engineer and the service redeployed.

### 6.7 Suppressed Claims Are Silently Removed

When the `CitationEnforcer` strips a sentence, the caller receives a shorter, grounded narrative without explicit notification that suppression occurred (beyond the `suppressed_claims` count in the response). Callers who do not inspect the `suppressed_claims` field may not realise the answer is incomplete. The audit trail records every suppression, but the primary response surface does not prominently surface this.

### 6.8 Multi-Turn Context Has No Expiry

Conversation history is persisted in Redis indefinitely (until TTL or explicit deletion). For long sessions with many turns, the accumulated conversation history grows unbounded and is included in every subsequent prompt. Very long sessions will eventually exceed the context window, causing silent truncation of older turns.

### 6.9 BigQuery Analytics Require the credit-risk-platform to Be Running

The NL→SQL→BigQuery pipeline is a pass-through to the credit-risk-platform analytics API. If that service is unavailable, all portfolio analytics queries will fail with `INSUFFICIENT_RETRIEVAL`. LucidCredit has no caching layer for BigQuery results — every analytics query hits the upstream service.

### 6.10 ECOA Proxy Discrimination Is Not Fully Detected

The `EcoaValidator` detects direct discriminatory attribution (e.g., "denied because of your race"). It does **not** detect proxy discrimination — decisions correlating with ZIP code, surname, or other features that serve as statistical proxies for protected characteristics. Detecting proxy discrimination requires analysis of the model's feature importances, which is the responsibility of the upstream credit-risk-platform, not LucidCredit.

---

## 7. Improvement Roadmap

### 7.1 Priority 1 — Security and Safety (P0)

**PII Output Scrubbing**  
The `EcoaValidator._check_pii_leak()` method detects PII in inputs, but no mechanism strips PII from generated narratives before they are returned. The LLM can echo identifiers (email addresses, SSNs) from the query context verbatim into its answer. A `scrub_pii_from_output()` function should be added and called in `format_output_node` on all audience types. Eval 6a PII detection is currently at 40% against a 100% gate.

**Prompt Injection Hardening**  
Current injection resistance is 75% against a 100% gate (Eval 6d). Retrieval chunks from external APIs can contain adversarial instructions. The grading system prompt should include explicit injection-awareness instructions, and retrieved chunk content should be wrapped in a structured delimiter (e.g., `<context>\n...\n</context>`) to prevent instruction leakage.

**Policy Adherence**  
Eval 6b policy adherence is at 30% against a 100% gate. The system prompt for the reason node should be audited against the full policy corpus to identify gaps, and the ECOA/FCRA regex patterns should be extended with the grammatical structures currently evading detection.

### 7.2 Priority 2 — Correctness (P1)

**Faithfulness Improvement**  
RAGAS faithfulness is at 0.742 against a 0.95 gate for some session types. The primary lever is retrieval quality: expanding the corpus to cover more domain-specific topics will reduce the fraction of claims the LLM generates from parametric memory. A secondary lever is reducing the `min_citation_similarity` threshold for well-defined domain-knowledge queries while raising it for open-ended queries.

**Self-Aware Failure / Refusal Rate**  
Eval 4 refusal rate is 0.00 against a 0.90 gate. The `parse_intent_node` should be extended to detect a broader range of unsupported query patterns and short-circuit with a structured refusal before retrieval begins. The current regex list covers portfolio and risk query patterns but lacks coverage for out-of-scope queries (personal financial advice, legal counsel, etc.).

### 7.3 Priority 3 — Performance (P2)

**Tier-1 Latency: 9.5s → ≤2.5s (Eval 7a)**  
The current P50 latency of 9.5 seconds is dominated by three serial operations: embedding the query (200ms), pgvector search (100ms), grading documents (LLM call, 1–2s), reasoning (LLM call, 2–4s), and citation enforcement (embedding 10+ sentences, 400–800ms). To reach 2.5s:

1. **Pre-cache embeddings** — embed and cache the top-N corpus chunks at startup; skip re-embedding during citation enforcement for cached chunks
2. **Parallel tool calls** — the retrieve phase already parallelises tools; document grading could be parallelised with a structured output LLM call on batched chunks rather than sequential grading
3. **Streaming with deferred citation enforcement** — stream the narrative to the frontend, run citation enforcement in background, and send a correction/suppression event if sentences are removed (SSE pattern)
4. **Fast-path expansion** — the existing `fast_path` flag for simple queries should be broadened; metrics queries from the analytics API that return structured rows do not need sentence-level citation enforcement

**Analytics API Resiliency**  
Tool Selection F1 is 0.379 against 0.85 primarily because the analytics API was unavailable during eval runs. A circuit-breaker pattern and short-TTL result cache (Redis, 60-second TTL for idempotent metrics queries) would allow LucidCredit to serve the most recent result instead of failing with `INSUFFICIENT_RETRIEVAL`.

### 7.4 Priority 4 — Quality and Coverage (P3)

**Embedding Model Versioning**  
Add an `embedding_model_version` column to the `policy_docs` table. This enables:
- Incremental corpus migration when the embedding model changes
- A/B testing of different embedding models against retrieval quality metrics
- Elimination of the current all-or-nothing re-ingestion requirement

**Continuous Confidence Calibration**  
The confidence score formula (0.30/0.50/0.20 weights) was determined heuristically. A calibration dataset derived from the 50 golden Q&A pairs in the RAGAS harness should be used to learn optimal weights via logistic regression. Confidence scores should be validated as probability estimates (Brier score, reliability diagram).

**Dynamic Query Decomposition**  
Replace the hard-coded regex decomposition table with an LLM-based intent parser that dynamically generates sub-questions from broad queries. This removes the need to maintain and redeploy a growing regex list and enables decomposition of novel query patterns not in the original dataset.

**Conversation History Windowing**  
Implement a sliding window over `conversation_history` — retain only the last N turns (configurable, default 10) in the prompt. Summarise older turns into a condensed context block using a separate LLM call. This prevents unbounded context growth without losing conversational context.

**Corpus Auto-Refresh**  
Add a scheduled task (cron or Cloud Scheduler) that polls regulatory source URLs for document changes and triggers incremental re-ingestion. Changes to Regulation B, FCRA, or SR 11-7 guidance should be detected within 24 hours of publication.

**Production Monitoring**  
Integrate Prometheus metrics for:
- Per-endpoint P50/P95/P99 latency
- Citation enforcement suppression rate (leading indicator of retrieval degradation)
- Confidence score distribution (population shift = model drift signal)
- ECOA validator violation rate by rule code (compliance health)

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| **BM25** | Best Match 25 — probabilistic term-frequency ranking algorithm used for lexical re-ranking of dense retrieval results |
| **ECOA** | Equal Credit Opportunity Act — US federal law prohibiting credit discrimination based on protected characteristics |
| **FCRA** | Fair Credit Reporting Act — US federal law governing use of consumer credit reports in credit decisions |
| **FastAPI** | Python async web framework used for the LucidCredit API gateway |
| **LangGraph** | Graph-based LLM orchestration framework; extends LangChain with a typed state machine model |
| **pgvector** | PostgreSQL extension providing vector similarity search; used as the primary vector store |
| **RAG** | Retrieval-Augmented Generation — architecture that grounds LLM outputs in retrieved documents rather than parametric memory |
| **RAGAS** | RAG Assessment framework — automated evaluation of faithfulness, answer relevancy, and context recall against golden Q&A sets |
| **Reg B** | Regulation B — Federal Reserve's implementing regulation for ECOA |
| **SHAP** | SHapley Additive exPlanations — model-agnostic method for explaining individual predictions from ML models |
| **SR 11-7** | Federal Reserve / OCC guidance on model risk management — requires documentation, validation, and disclosure of model limitations |
| **text-embedding-3-large** | Azure OpenAI embedding model (3072 dimensions) used for all dense retrieval and citation enforcement |

---

*This white paper reflects the LucidCredit architecture as of Sprint 5 (May 2026). For implementation details, see the codebase under `backend/app/`. For operational procedures, consult the runbook under `docs/OPERATIONAL_RUNBOOK.md`.*
