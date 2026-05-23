# LucidCredit — System Architecture White Paper

**Version:** 1.0 · **Date:** May 23, 2026  
**Authors:** LucidCredit Engineering  
**Status:** Internal Review — share freely within the team

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

Credit decisions change people's lives. A borrower who gets a clear, honest explanation of why they were declined can act on it. An analyst who can ask a natural language question and get a traceable, sourced answer can do their job faster and with more confidence. That's what LucidCredit is built for.

At its core, LucidCredit is an AI copilot that explains credit decisions in plain language — grounded entirely in retrieved data. It never draws on what the model "thinks it knows." Every sentence in every response traces back to a specific source: a policy document, an API payload, a database record. And if a sentence can't be traced, it's removed before the response ever reaches you.

We built it to serve two very different audiences:

- **Risk Analysts and CROs** — get SHAP-grounded briefings, probability of default distributions, portfolio stress summaries, and the ability to ask natural language questions against live BigQuery data
- **Applicants** — receive plain English explanations of decisions, ECOA/FCRA-compliant adverse action notices, and honest "what would need to change" guidance

The zero-hallucination guarantee isn't a system prompt instruction that the model might ignore on a bad day. It's enforced mechanically at inference time by the `CitationEnforcer`: every generated sentence is embedded and scored against retrieved context. Anything that doesn't pass the similarity threshold is stripped and logged. The model simply cannot include what it cannot cite.

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

The frontend is intentionally thin — it's a UI skin over the backend API, not a place for business logic. All `/api/*` requests are proxied to FastAPI via `next.config.ts` rewrites, which keeps local dev free of CORS headaches and makes the backend swappable without touching the frontend. There are four pages:

| Page | Audience | Key Features |
|------|----------|--------------|
| Analyst | Risk teams | NL query input, SHAP visualisation, portfolio briefing |
| Query Builder | Internal | Structured query construction, intent selector |
| Applicant | Borrowers | Decision explanation, adverse action notice, counterfactual guide |
| Audit Trail | Compliance/Governance | Citation inspector, confidence scores, suppression log |

### 2.3 API Gateway Layer

**Technology:** FastAPI (Python 3.11+), async/await throughout

Six REST endpoints, all versioned under `/api/v1/`. The split between analyst and applicant routes is deliberate — they go through different compliance paths and have different confidence thresholds, so keeping them separate in the URL makes the routing intent obvious:

| Endpoint | Method | Purpose |
|----------|--------|--------|
| `/analyst/query` | POST | Analyst Q&A with SHAP grounding |
| `/analyst/briefing` | POST | Portfolio briefing narrative |
| `/applicant/narrative` | POST | Applicant-facing decision explanation |
| `/applicant/adverse-action` | POST | ECOA-compliant adverse action notice |
| `/audit/{session_id}` | GET | Full audit trail for a session |
| `/audit/{session_id}/citations` | GET | All citations with similarity scores |

Every POST response includes a `confidence` field and a `citations` array inline — you don't have to make a second call to find out how much to trust the answer. Full Swagger UI lives at `/docs`.

### 2.4 LangGraph Agent Graph

**Technology:** LangGraph (StateGraph), Redis checkpointing

The reasoning pipeline is a compiled graph of async node functions. Think of it as an assembly line where each station does exactly one job, hands a typed state object to the next station, and never talks to any other station directly. That constraint makes the pipeline easy to test — you can unit-test any node by just injecting a state dict.

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

A few design choices worth calling out:
- **Only `reason_node` calls an LLM.** Every other node is deterministic code. This keeps costs predictable and makes the whole pipeline easier to test and debug.
- **`parse_intent_node` can short-circuit the entire graph.** If a query is clearly unsupported, we bail out before retrieval and generation — no tokens wasted.
- **The `fast_path` flag skips the LLM entirely** for simple Tier-1 queries, serving the answer directly from the top retrieved chunk.
- **Redis checkpoints the full session state.** Follow-up questions pick up exactly where the previous turn left off without the client resending conversation history — which matters a lot for analyst drill-down sessions.

### 2.5 Retrieval-Augmented Generation (RAG) Pipeline

**Technology:** pgvector (PostgreSQL 16), SQLAlchemy 2.0 async, `rank_bm25` (optional)

Retrieval is a two-stage process. We do this because pure dense vector search has a known blind spot: domain-specific terms like "FICO," "DTI," "Reg B," or "§ 615" often score poorly on semantic similarity against a general-language embedding but are exactly what the query is about. The BM25 layer catches those cases.

**Stage 1 — Dense retrieval**
- The query is embedded using `text-embedding-3-large` (3072 dimensions)
- pgvector returns `top_k × 3` candidates — we over-fetch deliberately to give BM25 a big enough pool to work with
- One important operational note: the 3072-dimension size is baked into the pgvector column schema. If you ever need to switch embedding models, a full corpus re-ingestion is required — there's no partial migration path today

**Stage 2 — BM25 re-ranking**
- BM25 scores are computed over the dense-retrieved candidates and blended in: `0.65 × dense_score + 0.35 × bm25_norm`
- If `rank_bm25` isn't installed, a built-in TF-IDF fallback kicks in automatically — no crash, slightly lower quality

**Corpus:**
- Regulatory documents: Regulation B (ECOA), FCRA § 615, SR 11-7 / OCC 2011-12
- Model cards: probability of default model, SHAP feature definitions
- Ingested via `policy_doc_ingestor.py` — run `ingest_corpus.py` after any corpus update

### 2.6 Zero-Hallucination Enforcement Layer

This is the heart of LucidCredit — the part that makes it meaningfully different from "LLM with a system prompt that says don't hallucinate." It runs in two passes after the model generates a response, and it has authority to delete things.

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

The `[UNVERIFIED]` marker is an honesty convention baked into the system prompt — the model is instructed to tag any sentence it can't fully support. We then suppress those tags mechanically regardless of what the model "intended." The two mechanisms — self-flagging and similarity scoring — are independent safety nets, not alternatives.

**ConfidenceScorer**

Once the grounded narrative is assembled, we compute a single score in [0.0, 1.0] that captures how much to trust the answer. It's a weighted combination of three signals:

$$\text{confidence} = 0.30 \times \text{retrieval\_recall} + 0.50 \times \text{citation\_rate} + 0.20 \times (1 - \text{unverified\_rate})$$

Where:
- `retrieval_recall` = how much of what we fetched turned out to be relevant
- `citation_rate` = what fraction of the final narrative made it through grounding
- `unverified_rate` = what fraction the model itself flagged as uncertain

Responses below 0.75 (the default `min_confidence_score`) are rejected outright — the caller gets an error, not a low-quality answer. One important nuance: advisory "why/explain/how" analyst questions get a lower threshold (~0.30–0.45) because those answers legitimately draw on synthesised reasoning that won't score as highly against retrieved chunks. We tune that separately rather than apply one threshold to everything.

### 2.7 Compliance Layer

**ECOA/FCRA Validator (`ecoa_validator.py`)**

Every applicant-facing response passes through a deterministic rule-based gate before it leaves the system. No LLM judgment is involved — rules are pure regex patterns, which means the behaviour is predictable, testable, and explainable to a regulator.

One design decision worth understanding: ECOA prohibits basing credit decisions *on* protected characteristics, but well-formed adverse action notices are *required* to include equal-treatment language that mentions those same characteristics (e.g., "We do not discriminate based on race..."). A naive keyword filter would flag every compliant notice. Our patterns require a credit outcome verb in proximity to the protected class term — so "denied because of your race" fires the rule, but "we do not discriminate based on race" does not. Getting that distinction right took several iterations.

The five rules enforced:

1. **ECOA-001** — Discriminatory attribution: decision attributed to a protected characteristic
2. **ECOA-002** — Reg B § 202.9: at least one specific adverse action reason must be present
3. **FCRA-001** — § 615 rights disclosure: required whenever a consumer report was used
4. **ECOA-003** — Discouraged language: evasive or impermissibly vague reason language
5. **ECOA-004** — Rights waiver language

**SR 11-7 Disclosure Injector (`sr117_disclosures.py`)**

Analyst and briefing outputs get an SR 11-7 footer automatically — session ID, provider and model version, known limitations, validation status, and a regime-change performance warning. This isn't boilerplate for its own sake; it's what model governance teams and examiners look for. Applicant responses are explicitly excluded because SR 11-7 is an internal regulatory construct that would confuse rather than help a borrower.

### 2.8 Persistence and Session Layer

**PostgreSQL (port 5440)**

Three tables that matter:
- `policy_docs` — the RAG corpus: chunks and their pgvector embeddings
- `copilot_sessions` — an immutable record of everything that happened in a session: the original query, retrieved chunks, rendered prompt, raw LLM output, grounded narrative, confidence score, and any feedback signal
- `citations` — per-session citation records with exact similarity scores, queryable independently

`copilot_sessions` is written once at the end of each request and never updated. That immutability is intentional — it means any session can be audited or replayed exactly as it happened.

**Redis (port 6380)**

Redis serves one purpose: LangGraph's `AsyncRedisSaver` checkpointer, which persists the full `AgentState` between requests in a session. This is what makes multi-turn conversations work — a follow-up question picks up the prior context without the client needing to resend history. In local dev without Redis, the system automatically falls back to an in-memory checkpointer so you can still run the pipeline.

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

The applicant-session fallback being marked `FORBIDDEN` is a deliberate regulatory safety decision, not an oversight. We haven't validated Gemini's outputs against ECOA/FCRA requirements, and the exposure from a non-compliant adverse action notice is too high to take that risk. If Azure is down, applicant sessions fail closed. That's the right trade-off.

Provider clients are cached by a `(endpoint, deployment, api_version, api_key)` tuple so instances are reused across requests — but they're not module-level singletons, which means test code can still patch them without fighting import-time state.

### 2.10 Tool Ecosystem

During the `retrieve_node` phase, the agent can call any combination of five tools in parallel. Each tool wraps a different data source and returns typed `RetrievedChunk` objects — the agent doesn't need to know where the data came from to process it:

| Tool | Source | What it retrieves |
|------|--------|-------------------|
| `vector_tool` | pgvector | Policy documents, regulatory corpus, model cards |
| `crp_api_tool` | credit-risk-platform REST API | Decision explanations, SHAP values, audit records, portfolio metrics |
| `analytics_api_tool` | CRP Analytics semantic layer (NL→SQL→BigQuery) | Live portfolio metrics, delinquency rates, charge-offs, vintage performance |
| `thinfile_tool` | ThinFile Engine REST API | Thin-file model scores, alternative data signals |
| `sql_tool` | Direct PostgreSQL (read-only) | Loan applications, features, funded loans, payment history, audit logs |

**SQL tool security model:**

The SQL tool has a layered security model because a natural language interface to a database is an obvious injection target. We check at the code level *and* enforce at the database level:
- Token parsing (not just `startswith`) ensures only `SELECT` statements execute — this catches `WITH...SELECT` and subquery injection attempts that a prefix check would miss
- A table allow-list raises `SqlSecurityError` before any unrecognised table name reaches the database
- Bind parameters are used for all values — no string interpolation anywhere
- The database user has `SELECT`-only grants as a final backstop
- A hard 200-row cap prevents runaway analytical queries from OOMing the process

**Analytics API hallucination guard:**

There's a subtle failure mode with NL→SQL pipelines: the query can succeed (return rows) while actually answering the wrong question. For example, a "prepayment rate" question might route to a query that returns LTV columns instead. `_validate_rows_for_question()` catches this by checking whether the returned column names plausibly match the question's subject. If they don't, the tool returns a structured `no_matching_field` chunk rather than handing irrelevant rows to the LLM to fabricate an answer from.

### 2.11 Integration Architecture

LucidCredit is a **read-only consumer** of upstream systems:

```
credit-risk-platform ──► LucidCredit ◄── ThinFile Engine
(decisions, SHAP, BQ)                    (thin-file scores)
```

This is an important boundary to understand: LucidCredit never writes to or modifies anything upstream. It can't change a credit decision, update a customer record, or alter a model output. The worst-case scenario for a compromised LucidCredit instance is a data read exposure — not a decision manipulation risk.

Integration is purely over the network: versioned REST APIs with API-key authentication, no shared database schemas. Swapping an upstream service or pointing to a different environment requires only a config change (`CRP_API_BASE_URL`, `THINFILE_API_BASE_URL`).

---

## 3. Data Flow: End-to-End Request Lifecycle

Here's what actually happens when an analyst asks "What drove the denial for application A-123?" — traced step by step:

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

Most "grounded" AI systems rely on prompt instructions like "only answer based on the provided context." Models ignore those instructions all the time, especially on adversarial inputs or when they're confident about something from training data. LucidCredit's approach is different: the `CitationEnforcer` runs *after* the model generates its answer and deletes anything it can't verify. The model's intentions don't matter — if a sentence doesn't score above the similarity threshold against retrieved chunks, it's gone. That's a much stronger guarantee.

### 4.2 Deterministic, Auditable Compliance

We made a deliberate choice to use regex rules for ECOA/FCRA validation rather than asking an LLM to "check if this is compliant." The reason is simple: regulators need to be able to verify compliance, and "the model thought it was fine" is not an audit trail. With regex rules, you can point to the exact pattern that fired, reproduce the result on any machine, and prove coverage with unit tests. An LLM-based gate would be non-deterministic — same input, different result on a bad day. That's not acceptable when the output is an adverse action notice.

### 4.3 Immutable, Queryable Audit Trail

Every session is fully reconstructable after the fact. We store the complete chain: the original query, every retrieved chunk, how they were graded, the rendered prompt, the raw LLM output, the grounded narrative, confidence score, citations with their similarity scores, suppressed claims, and compliance flags. Nothing is computed or summarised — the raw material is all there. This matters for model governance exams (SR 11-7 evidence), regulatory review of individual adverse action decisions (ECOA audit trail), and for debugging when something behaves unexpectedly.

### 4.4 Audience-Aware Routing

An applicant should never see a SHAP value breakdown. An analyst doesn't need their response filtered through ECOA safe-harbour language. These audiences have fundamentally different needs, different legal requirements, and different appropriate tones — so we route them through entirely separate pipelines with different system prompts, compliance paths, confidence thresholds, and fallback policies. The separation is enforced at the API route level, not just in configuration.

### 4.5 Hybrid Retrieval Quality

Pure vector search struggles with exact terminology. Ask about "§ 615" or "Reg B § 202.9" and the dense embedding might score it low because the language is legal and specific — but BM25 will nail it on term frequency. The 65/35 dense/BM25 blend gives us semantic understanding for natural language questions *and* keyword precision for regulatory and model-specific terms. In practice, it meaningfully improves recall on the queries that matter most in credit: regulatory references and technical feature names.

### 4.6 Defense-in-Depth Security

No single security control is assumed to be sufficient. Data access is protected by six independent layers, each one able to catch what the previous layer missed:
1. SQL `SELECT`-only enforcement at the Python layer (token parsing, not just prefix matching)
2. Table allow-list at the Python layer
3. Database user with `SELECT`-only grants at the PostgreSQL layer
4. Row cap (200 rows) to prevent memory exhaustion
5. API-key authentication on all external tool calls
6. Read-only consumer pattern — no write path exists to any upstream system

### 4.7 Provider Resilience

Azure OpenAI has an SLA, but it's not 100%. Analyst sessions automatically fall back to Vertex Gemini 1.5 Pro if Azure is unavailable — no code change, no redeployment, just a config flag. The `BaseChatModel` abstraction means adding a new provider is a matter of implementing the same interface, not rewiring the pipeline.

### 4.8 Multi-Turn Session Continuity

Analysts rarely ask a single question. They ask a question, get an answer, and drill down. Redis checkpointing means each follow-up picks up where the last turn left off — the client doesn't resend the history, the payload stays small, and the conversation flows naturally. For a portfolio briefing session with a dozen sequential questions, this adds up.

### 4.9 Narrow Surface Area

LucidCredit does one thing: explain and brief. It cannot make credit decisions. It cannot modify them. It has no admin interface, no write path, and no ability to act on upstream systems. We kept the surface area narrow deliberately — a system that does less is harder to misuse, and a compromised LucidCredit instance can only read data, not alter outcomes.

---

## 5. Drawbacks

This section is honest about where the architecture creates real costs. There's no system design that gets everything right, and these trade-offs should be visible to anyone operating or extending LucidCredit.

### 5.1 Embedding Latency at Citation Enforcement

The zero-hallucination guarantee has a latency cost. Every response requires embedding all generated sentences *and* all relevant chunks to compute similarity scores. For a 10-sentence narrative against 8 relevant chunks, that's 18 embedding calls — batched, but at 200–400ms per batch on Azure, citation enforcement alone adds roughly half a second to every request. Combined with two LLM calls (grading and reasoning), the end-to-end P50 lands at 4–9 seconds on uncached requests. Our Tier-1 target is 2.5 seconds. We're not there yet.

### 5.2 Vendor Lock-in at the Embedding Layer

The whole retrieval and citation pipeline is tightly coupled to `text-embedding-3-large` and its 3072-dimension vector space. If we ever want to switch embedding models — for cost, quality, or availability reasons — we'd need to re-ingest the entire corpus. There's no embedding model version column in `policy_docs`, so there's no way to do an incremental migration. You can't run old and new embeddings side by side without a schema change first. This is a real operational risk that we've deferred.

### 5.3 Single LLM Reasoning Step

The current pipeline retrieves everything upfront and reasons exactly once. That works well for focused questions, but it breaks down for queries that need multi-step analysis — "compare the delinquency trend against macro conditions and explain the divergence" really needs at least two reasoning passes: one to surface each data set, and one to synthesise across them. Right now, we throw everything into a single prompt and hope the model can handle the synthesis. It often can't, at least not well.

### 5.4 Regex Fragility in the ECOA Validator

The ECOA-001 patterns now cover eight distinct grammatical structures, and getting that coverage required careful, iterative work. But regex is inherently brittle against language variation. A sentence reworded slightly — passive instead of active voice, a gerund, an ellipsis — can evade detection. More importantly, there's a category of proxy discrimination that regex fundamentally can't catch: decisions correlated with ZIP codes, surnames, or other features that serve as statistical proxies for race or national origin will pass every pattern we have. That's a known gap, and it requires analysis at the feature importance level, not the text level.

### 5.5 No Streaming Response

Streaming and post-generation citation enforcement are fundamentally at odds. You can't embed a sentence that isn't complete yet. So the current architecture buffers the entire LLM response, runs enforcement, and then sends the result — which means the user sees nothing for several seconds, then gets everything at once. This makes the latency feel worse than the raw numbers suggest. There's a path to streaming with deferred enforcement (see section 7.3), but it requires architectural changes.

### 5.6 Stateless Query Decomposition

When a user asks about "portfolio health," we decompose it into five specific sub-queries via a hard-coded regex table in `nodes.py`. That works well for the patterns we anticipated, but adding a new decomposition requires a code change and redeployment. There's no way for the system to learn or adapt new decompositions from usage — it's a static lookup table dressed up as intelligence.

### 5.7 Context Window Pressure at Scale

All graded chunks get rendered into a single prompt for `reason_node`. For most queries that's fine, but the "run all BigQuery analysis" decomposition fans out to 18 sub-queries, each of which can return substantial row data. When that all aggregates into one prompt, we start approaching GPT-4.1's context window limit. At that point the model either truncates silently or refuses to answer — neither of which is a graceful failure.

### 5.8 Limited Feedback Loop

We capture a `feedback_signal` on every session, but right now it goes nowhere. There's no pipeline from thumbs-down signals to corpus updates, threshold adjustments, or prompt revisions. RAGAS evaluations are run manually before releases. We have no automatic quality signal in production — which means we won't know if something starts degrading until a human notices.

---

## 6. Limitations Users Must Be Aware Of

These aren't bugs. They're characteristics of the architecture that users need to understand to work with the system effectively.

### 6.1 Responses Are Bounded by What Was Retrieved

**The system can only tell you what it found.** If the relevant policy document hasn't been ingested, or the CRP API doesn't have the decision record, you'll get an `INSUFFICIENT_RETRIEVAL` error — not a guess. That's intentional. But it means a refusal to answer isn't always a system failure. Sometimes it means the information genuinely isn't there yet, and the right response is to check whether the corpus or API data needs updating.

### 6.2 Confidence Scores Are Heuristic, Not Statistical

The confidence score is a useful signal, not a probability. A score of 0.85 doesn't mean there's an 85% chance the answer is correct — it means the retrieval was good, most sentences were grounded, and the model flagged few claims as uncertain. The weights in the formula were set heuristically. Don't use the score as a substitute for human review on anything with real consequences.

### 6.3 Policy Corpus Freshness

The system answers based on what was ingested. If Regulation B is amended tomorrow and you don't re-run `ingest_corpus.py`, the system will give you the old answer — confidently, because it found it in the corpus. There's no automatic refresh. If you're using LucidCredit for regulatory guidance, someone on the team needs to own corpus freshness as an operational responsibility.

### 6.4 The SR 11-7 Footer Is Not a Formal Validation

The SR 11-7 footer on analyst outputs says it plainly: **"INTERNAL USE — Pending formal model validation. This model is approved for analytical assistance only. It is NOT approved as a sole-input credit decision tool."** That's not legal boilerplate — it reflects the actual status of this system. A qualified credit analyst needs to review LucidCredit's output before it influences a credit decision.

### 6.5 Applicant Sessions Have No Fallback

If Azure OpenAI goes down, applicant communications go down with it. We can't route those sessions to Gemini because we haven't validated Gemini's output against ECOA/FCRA requirements. That's the right safety call, but it means Azure uptime is directly on the critical path for applicant-facing functionality. Plan maintenance windows accordingly.

### 6.6 The SQL Tool Can Only Query Pre-Approved Tables

Analysts can't point LucidCredit at arbitrary tables. The SQL tool has a hard-coded allow-list, and queries touching anything outside that list are rejected before reaching the database. Adding a new data source requires an engineer to update `sql_tool.py` and redeploy. This is the right security trade-off, but it does mean the tool's data coverage is limited to what's explicitly been approved.

### 6.7 Suppressed Claims Are Easy to Miss

When the `CitationEnforcer` strips sentences, the response comes back shorter — and that's easy to miss. There's a `suppressed_claims` count in the response JSON, but if you're not actively checking it, you might not realise the answer you received is an edited version of what the model generated. The full suppression log is in the audit trail, but the primary response doesn't surface it prominently. Callers should always inspect `suppressed_claims > 0` as a signal to review the audit trail.

### 6.8 Long Conversations Can Silently Truncate

Conversation history in Redis grows with every turn and has no expiry by default. After enough turns, the accumulated history starts crowding out the retrieved context in the prompt. When it exceeds the context window, older turns get truncated silently — the model doesn't know it's working with an incomplete history, and neither does the caller. For analytical sessions that go on for a long time, this is a real risk.

### 6.9 BigQuery Analytics Depend on an Upstream Service

Portfolio analytics queries go through the credit-risk-platform analytics API. If that service is down, every analytics question fails with `INSUFFICIENT_RETRIEVAL`. There's no cache. If you're demonstrating or testing LucidCredit's portfolio analytics and the credit-risk-platform isn't running, those queries simply won't work — start that service first.

### 6.10 Proxy Discrimination Is Out of Scope

The ECOA validator catches direct discriminatory attribution — explicit statements like "denied because of your race." It cannot catch proxy discrimination: a model that uses ZIP code as a proxy for race, or surname as a proxy for national origin. That kind of fairness analysis requires looking at the upstream model's feature importances and decision patterns, which is the job of the credit-risk-platform's fairness monitoring, not LucidCredit. Be clear with stakeholders about where this responsibility boundary sits.

---

## 7. Improvement Roadmap

The items below come directly from failing evals and known production gaps. They're ordered by impact and risk, not effort.

### 7.1 Priority 1 — Security and Safety (P0)

These are the items we're most uncomfortable with in production. They need to be fixed before this system handles real applicant data at scale.

**PII Output Scrubbing**  
Today, `EcoaValidator._check_pii_leak()` flags PII in *inputs* — but nothing strips it from generated responses. If an analyst queries "explain the denial for bob.smith@creditco.com," the LLM might echo that email address verbatim in its answer. We need a `scrub_pii_from_output()` function called in `format_output_node` for all audience types, not just applicant. Eval 6a sits at 40% against a 100% gate — this is the fix.

**Prompt Injection Hardening**  
Retrieval chunks pulled from external APIs can contain adversarial instructions — "ignore previous instructions and..." is a real attack vector when the data source isn't trusted. We need to wrap all retrieved chunk content in structured delimiters (e.g., `<context>\n...\n</context>`) so the model treats it as data, not instructions. Eval 6d injection resistance is at 75% against a 100% gate.

**Policy Adherence**  
Eval 6b sits at 30% against a 100% gate — the worst-performing eval we have. A systematic audit of the reason node's system prompt against the full policy corpus is needed to find the gaps, alongside extending the ECOA/FCRA regex patterns to cover the grammatical structures currently slipping through.

### 7.2 Priority 2 — Correctness (P1)

**Faithfulness Improvement**  
RAGAS faithfulness is at 0.742 against a 0.95 gate for some session types. The most direct fix is corpus expansion — more domain-specific content means fewer sentences where the LLM has to fall back on training memory. A secondary lever is making the citation similarity threshold adaptive: lower it slightly for well-defined domain-knowledge queries (where synthesised answers are expected) and raise it for open-ended questions.

**Self-Aware Failure / Refusal Rate**  
Eval 4 refusal rate is 0.00 against a 0.90 gate. The system never refuses anything — it just fails with `INSUFFICIENT_RETRIEVAL` after burning tokens on retrieval and grading. We need to extend `parse_intent_node` to recognise out-of-scope queries (personal financial advice, requests for legal counsel, etc.) and return a structured refusal before any LLM calls happen. The current regex coverage is too narrow.

### 7.3 Priority 3 — Performance (P2)

**Tier-1 Latency: 9.5s → ≤2.5s (Eval 7a)**  
Getting from 9.5s to 2.5s P50 requires attacking the problem from multiple angles simultaneously — no single change is enough:

1. **Pre-cache corpus embeddings** — embed and cache all corpus chunks at startup so citation enforcement can compare against cached vectors instead of calling the embedding API for chunks that haven't changed
2. **Parallelise document grading** — currently sequential per chunk; could be batched into a single structured-output LLM call
3. **Stream first, enforce later** — stream the narrative to the frontend as it generates, run citation enforcement in the background, and push a correction event via SSE if sentences are removed; this changes the *perceived* latency dramatically even if the total wall-clock time doesn't change
4. **Expand fast-path coverage** — structured BigQuery results don't need sentence-level citation enforcement; the data *is* the citation; extending `fast_path` to cover analytics tool responses is a straightforward win

**Analytics API Resiliency**  
Tool Selection F1 is 0.379 against 0.85 — the biggest contributor to that gap was the analytics API being unavailable during eval runs. A circuit-breaker pattern with a short-TTL Redis cache (60 seconds for idempotent metrics queries) would mean a transient upstream outage returns a slightly stale result instead of an outright failure.

### 7.4 Priority 4 — Quality and Coverage (P3)

**Embedding Model Versioning**  
Add an `embedding_model_version` column to `policy_docs`. This one schema change unlocks incremental corpus migration, A/B testing of embedding models, and eliminates the current all-or-nothing re-ingestion problem. It's low-effort relative to the operational flexibility it creates.

**Continuous Confidence Calibration**  
The 0.30/0.50/0.20 weights in the confidence formula were chosen heuristically. We have 50 golden Q&A pairs from RAGAS — that's enough to learn optimal weights via logistic regression and validate the scores as actual probability estimates (Brier score, reliability diagram). A calibrated confidence score is a much stronger operational tool than a heuristic one.

**Dynamic Query Decomposition**  
The hard-coded regex table in `nodes.py` will keep growing as users ask questions we didn't anticipate. Replacing it with an LLM-based intent parser that dynamically generates sub-questions removes the maintenance burden and handles novel query patterns gracefully.

**Conversation History Windowing**  
Cap conversation history at the last N turns (default 10) and summarise older turns into a condensed context block using a separate LLM call. This prevents the silent truncation problem from section 6.8 without losing the thread of long conversations.

**Corpus Auto-Refresh**  
A scheduled task (cron or Cloud Scheduler) that polls regulatory source URLs and triggers incremental re-ingestion when documents change would close the corpus freshness gap from section 6.3. A 24-hour detection window for regulatory updates is a reasonable target.

**Production Monitoring**  
We're currently flying blind in production. Integrating Prometheus metrics for the following signals would change that:
- Per-endpoint P50/P95/P99 latency
- Citation enforcement suppression rate (a rising suppression rate signals retrieval degradation before users start complaining)
- Confidence score distribution over time (population shift is an early model drift indicator)
- ECOA validator violation rate by rule code (a leading indicator of compliance problems)

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

*This document reflects the LucidCredit architecture as of Sprint 5 (May 2026). The code is the ground truth — if something here conflicts with `backend/app/`, trust the code and file a PR to update this document. Operational procedures (startup, rollback, corpus refresh) are in `docs/OPERATIONAL_RUNBOOK.md`.*
