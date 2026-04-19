# LucidCredit — AI Analytical Copilot for Credit Decisions
> **Zero-hallucination. Grounded in data. Clear to everyone.**

## Project Identity

| Field | Value |
|---|---|
| **Project Name** | LucidCredit |
| **Status** | Greenfield — standalone project |
| **Stack** | Python 3.11+, FastAPI, LangGraph, OpenAI GPT-4o, PostgreSQL, pgvector, Redis, Next.js 14 (TypeScript) |
| **Created** | April 19, 2026 |

## Vision

LucidCredit is a **zero-hallucination analytical copilot** that provides natural language explanations of credit decisions for two distinct audiences:

- **Analyst Briefings** — internal credit analysts, risk officers, compliance teams
- **Applicant Communications** — borrower-facing adverse action notices, approval summaries, counterfactual guidance

It is a **standalone service** that:
- Queries live and historical data from credit databases, loan ledgers, and third-party systems via a **RAG-based retrieval architecture** — it never generates facts from parametric memory alone.
- Integrates with `credit-risk-platform` exclusively via versioned REST APIs.
- Is architecturally independent from AgentHiveHQ, though its output format is compatible with AgentHiveHQ's `CreditCopilotOutput` Zod schema for optional future embedding.
- Provides a verifiable **grounding chain** — every claim in a generated narrative is traceable to a source document, API response, or database record.

## Core Principles

| Principle | Implementation |
|---|---|
| **Zero Hallucination** | Every sentence carries a source_ref linking to a retrieved document, API payload, or database row. Claims without grounding are suppressed. Confidence scorer rejects outputs below threshold. |
| **Audience-Aware Tone** | Analyst briefings use technical register (SHAP values, PD bands, DTI ratios). Applicant communications use Plain English, ECOA/FCRA-compliant language. |
| **Immutable Audit Trail** | Every copilot session is persisted: input query, retrieved context chunks, rendered prompt, raw LLM output, grounded narrative, confidence score, and feedback signal. |
| **Read-Only Data Access** | The SQL tool is hard-scoped to SELECT queries. No tool has write access to any upstream system. |
| **Regulatory Compliance** | All applicant-facing outputs are validated against an ECOA/FCRA rule engine before delivery. Analyst outputs carry SR 11-7 model risk disclosures. |

## Directory Structure

```
LucidCredit/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/v1/
│   │   │   ├── explain.py
│   │   │   ├── query.py
│   │   │   ├── briefing.py
│   │   │   ├── applicant.py
│   │   │   ├── audit.py
│   │   │   └── health.py
│   │   ├── agent/
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   ├── grounding.py
│   │   │   ├── tools/
│   │   │   │   ├── crp_api_tool.py
│   │   │   │   ├── thinfile_tool.py
│   │   │   │   ├── sql_tool.py
│   │   │   │   ├── vector_tool.py
│   │   │   │   └── ledger_tool.py
│   │   │   └── prompts/
│   │   │       ├── analyst_system.md
│   │   │       ├── applicant_system.md
│   │   │       └── grading_system.md
│   │   ├── rag/
│   │   │   ├── ingestor.py
│   │   │   ├── chunker.py
│   │   │   ├── embedder.py
│   │   │   └── retriever.py
│   │   ├── compliance/
│   │   │   ├── ecoa_validator.py
│   │   │   ├── adverse_action.py
│   │   │   └── sr117_disclosures.py
│   │   ├── models/
│   │   │   ├── session.py
│   │   │   ├── citation.py
│   │   │   └── feedback.py
│   │   ├── schemas/
│   │   │   ├── explain.py
│   │   │   ├── query.py
│   │   │   ├── briefing.py
│   │   │   └── applicant.py
│   │   └── db/
│   │       └── session.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── analyst/
│   │   │   ├── applicant/
│   │   │   ├── query/
│   │   │   └── audit/
│   │   ├── components/
│   │   │   ├── NarrativeCard.tsx
│   │   │   ├── CitationDrawer.tsx
│   │   │   ├── ConfidenceBadge.tsx
│   │   │   ├── FeatureBreakdown.tsx
│   │   │   └── CounterfactualPanel.tsx
│   │   └── lib/
│   │       └── copilot-client.ts
│   ├── package.json
│   └── next.config.ts
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── INTEGRATION_MANUAL.md
├── PLAN.md
├── README.md
├── .env.example
├── .gitignore
└── docker-compose.yml
```

## Data Sources & Integration Contracts

### Source 1 — credit-risk-platform (Primary)

Base URL: `http://credit-risk-platform-decision-api:8081`

| Used Endpoint | Purpose |
|---|---|
| `GET /v1/decisions/{id}/explanation` | Fetch SHAP values, counterfactuals, NLG summary as grounding context |
| `GET /v1/decisions/{id}/audit` | Full immutable audit record — source of truth for decision narratives |
| `POST /v1/decisions` | On-demand re-scoring for counterfactual "what-if" queries |
| `GET /v1/metrics` | Portfolio-level context for analyst briefings |
| `POST /v1/stress-test/run` | Scenario analysis queries |

The copilot **does not replicate** explainability logic. It **consumes** SHAP/counterfactual outputs from the platform API and grounds its narratives in them.

### Source 2 — ThinFile_Credit_Underwriting_Engine

Base URL: `http://thinfile-api:8000`

| Used Endpoint | Purpose |
|---|---|
| `POST /score` | Live scoring for thin-file applicant explanations |
| `GET /score/{id}/adverse-action` | ECOA adverse action codes for decline communications |
| `GET /audit/logs/{id}` | Feature snapshot for explanation grounding |

### Source 3 — Vector Document Store (pgvector)

| Corpus | Content | Update Cadence |
|---|---|---|
| `regulatory_docs` | ECOA, FCRA, SR 11-7, Reg B, CFPB guidance, FFIEC handbooks | Monthly |
| `policy_docs` | Internal credit policies, underwriting guidelines, pricing matrices | On change |
| `model_cards` | Model cards from ThinFile Engine and credit-risk-platform | On model release |
| `decision_archive` | Anonymized historical decision audit records (last 24 months) | Daily delta |
| `product_docs` | Loan product sheets, T&Cs, APR schedules | On change |

### Source 4 — Read-Only SQL Access

| Database | Host | Schema Access |
|---|---|---|
| `credit_risk_loans` (replica) | crp-analytics-db:5432 | `loan_applications`, `features`, `funded_loans` |
| `credit_risk_transactions` (replica) | crp-analytics-db:5433 | `bank_accounts`, `payment_history` |
| `thinfile_audit` (replica) | thinfile-db:5432 | `audit_logs`, `feature_snapshots` |

## LangGraph Agent Design (Corrective RAG)

```
[START]
   |
   v
[parse_intent]    — classify intent: explain_decision | analyst_query | applicant_comms | portfolio_brief
   |
   v
[retrieve]        — parallel fan-out: vector_tool + sql_tool + crp_api_tool
   |
   v
[grade_documents] — grade each chunk: RELEVANT | IRRELEVANT | AMBIGUOUS
   |
   +-- insufficient coverage? --> [web_search_fallback] (regulatory docs only)
   |
   v
[reason]          — GPT-4o chain-of-thought over grounded context only
   |
   v
[citation_enforcer] — map every claim to source_ref; reject uncited claims
   |
   v
[confidence_score]  — per-claim Bayesian confidence; aggregate session confidence
   |
   +-- confidence < 0.75? --> [clarification_request]
   |
   v
[compliance_check]  — ECOA/FCRA validator (applicant audience only)
   |
   v
[format_output]     — render audience-appropriate narrative
   |
   v
[persist_session]   — write session + citations + confidence to PostgreSQL
   |
   v
[END]
```

## Zero-Hallucination Enforcement

### Layer 1 — Prompt Engineering
The system prompt forbids the model from generating any numerical figure, date, name, regulatory reference, or risk classification not present in retrieved context. The `grading_system.md` prompt instructs the model to self-check each sentence and mark uncertain claims with `[UNVERIFIED]`.

### Layer 2 — Citation Enforcer (grounding.py)
Post-generation parse: every sentence is split into atomic claims. Each claim is matched against retrieved context using embedding similarity (threshold >= 0.85). Claims that fail are:
- (a) flagged
- (b) removed from the final narrative
- (c) logged to audit table with `suppression_reason`

### Layer 3 — Confidence Scorer (grounding.py)
Calibrated `confidence_score in [0.0, 1.0]` per response aggregating:
- Retrieval recall (relevant chunks / total chunks retrieved)
- Claim citation rate (cited claims / total claims)
- Model self-assessed uncertainty

Responses with score < 0.75 return a structured `InsufficientGrounding` error.

## API Specification

### POST /v1/explain/decision
Request:
```json
{
  "decision_id": "uuid",
  "source": "credit-risk-platform",
  "audience": "analyst",
  "language": "en",
  "include_counterfactual": true,
  "include_shap_narrative": true
}
```
Response:
```json
{
  "session_id": "uuid",
  "narrative": "string",
  "citations": [{"claim": "string", "source_type": "api", "source_ref": "string", "confidence": 0.94}],
  "confidence_score": 0.91,
  "counterfactual": {"primary_lever": "Reduce DTI from 0.48 to below 0.43", "estimated_score_improvement": "+12 points"},
  "adverse_action_codes": ["AA-007"],
  "compliance_flags": [],
  "audience": "analyst"
}
```

### POST /v1/query/analyst
Request:
```json
{"query": "string", "data_scope": ["credit_risk_loans"], "session_id": null}
```
Response:
```json
{"session_id": "uuid", "answer": "string", "citations": [], "sql_queries_executed": ["SELECT ..."], "confidence_score": 0.88, "follow_up_suggestions": []}
```

### POST /v1/applicant/communication
Request:
```json
{"application_id": "uuid", "source": "thinfile", "communication_type": "decline", "channel": "email", "language": "en"}
```
Response:
```json
{"session_id": "uuid", "subject_line": "string", "body": "string", "adverse_action_notice": {}, "compliance_validated": true, "citations": [], "confidence_score": 0.96}
```

### POST /v1/briefing/generate
Request:
```json
{
  "scope": "segment",
  "filters": {"date_range": ["2026-01-01", "2026-03-31"], "decision": "DECLINE", "employment_status": "gig_worker"},
  "sections": ["executive_summary", "risk_distribution", "key_drivers", "fairness_indicators", "recommendations"],
  "audience_role": "CRO"
}
```

## Data Models (PostgreSQL — lucidcredit DB)

### copilot_sessions
| Column | Type | Notes |
|---|---|---|
| session_id | UUID PK | |
| query_text | TEXT | Original user query |
| intent | VARCHAR(50) | Classified intent |
| audience | VARCHAR(20) | analyst / applicant |
| retrieved_chunks | JSONB | All chunks with source refs |
| rendered_prompt | TEXT | Full prompt sent to LLM |
| raw_llm_output | TEXT | Unmodified LLM response |
| grounded_narrative | TEXT | Post-citation-enforcement narrative |
| confidence_score | FLOAT | Aggregate session confidence |
| suppressed_claims | JSONB | Claims removed by enforcer |
| compliance_flags | JSONB | |
| source_system | VARCHAR(50) | credit-risk-platform / thinfile |
| created_at | TIMESTAMPTZ | |
| user_id | VARCHAR(100) | |

### citations
| Column | Type | Notes |
|---|---|---|
| citation_id | UUID PK | |
| session_id | UUID FK | |
| claim_text | TEXT | |
| source_type | VARCHAR(20) | api / db / vector_doc |
| source_ref | TEXT | URL, table.row, or doc chunk ID |
| similarity_score | FLOAT | Embedding match score |
| confidence | FLOAT | Per-claim confidence |

### user_feedback
| Column | Type | Notes |
|---|---|---|
| feedback_id | UUID PK | |
| session_id | UUID FK | |
| rating | SMALLINT | 1-5 |
| useful | BOOLEAN | |
| correction | TEXT | User-provided correction |
| created_at | TIMESTAMPTZ | |

## Coding Standards

- **Python**: Pydantic v2, async/await throughout, SQLAlchemy 2.0 async ORM, Alembic, pytest + pytest-asyncio, ruff + mypy strict
- **API**: FastAPI versioned routers (/v1/), structured error responses, OpenAPI auto-generated
- **LangGraph**: Tool definitions typed with Pydantic; State typed with TypedDict; all nodes are pure functions; graph compiled with Redis checkpointer
- **Frontend**: Next.js 14 App Router, TypeScript strict, Tailwind CSS, shadcn/ui
- **Security**: API key auth middleware; JWT for applicant portal; credentials in env vars only
- **Observability**: structlog JSON logging, OpenTelemetry traces, Prometheus /metrics

## Sprint Plan

### Sprint 1 — Core RAG & Retrieval
- Project scaffolding, Docker Compose, PostgreSQL + pgvector setup
- Document ingestor, chunker, embedder (text-embedding-3-large)
- Hybrid retriever (dense + BM25 re-rank)
- crp_api_tool.py — typed client for credit-risk-platform
- thinfile_tool.py — typed client for ThinFile Engine
- sql_tool.py — read-only query executor with allow-list validation
- Unit tests: retrieval precision@5, SQL injection resistance

### Sprint 2 — LangGraph Agent & Zero-Hallucination Layer
- LangGraph graph definition: all nodes, state schema, edges, conditional routing
- grounding.py: CitationEnforcer + ConfidenceScorer
- System prompts: analyst_system.md, applicant_system.md, grading_system.md
- Session persistence (copilot_sessions, citations tables)
- Integration test: citation rate > 95% on explain_decision flow

### Sprint 3 — API Endpoints & Compliance Layer
- All FastAPI routes: /v1/explain, /v1/query, /v1/applicant, /v1/briefing, /v1/audit
- ecoa_validator.py — ECOA/FCRA output compliance check
- adverse_action.py — adverse action notice generator
- sr117_disclosures.py — SR 11-7 model risk language injector
- Load test: p99 < 8s single explain, < 30s briefing

### Sprint 4 — Frontend & Analyst UX
- Next.js app scaffold with App Router
- NarrativeCard with inline citation hover-cards (CitationDrawer)
- ConfidenceBadge component (color-coded by score band)
- FeatureBreakdown — SHAP waterfall from API response
- CounterfactualPanel — interactive "what would change this decision?"
- Analyst query console with streaming SSE

### Sprint 5 — Hardening, Evaluation & Deployment
- RAG evaluation: faithfulness, answer relevancy, context recall (RAGAS)
- Hallucination regression: 50 golden Q&A pairs with known-good grounded answers
- Adversarial tests: out-of-context numerics, invented regulations
- Docker Compose multi-service deployment
- CI/CD: GitHub Actions — lint, test, build, push
- Full README with local dev quickstart

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over bare LangChain | Deterministic state machine with conditional routing for CRAG pattern |
| pgvector over Pinecone/Weaviate | PostgreSQL-native, consistent with sibling projects, no external SaaS |
| Hybrid retrieval (dense + BM25) | BM25 re-rank improves precision on regulatory text with exact codes |
| Read-only data access | Separation of read/write concerns; simplifies compliance scope |
| Standalone (not in AgentHiveHQ) | Own persistence, compliance layer, deployment unit |
| API-only integration with credit-risk-platform | Avoids shared DB coupling; consumes outputs not logic |
| RAGAS evaluation | Industry-standard RAG evaluation is primary zero-hallucination acceptance criterion |

---

*LucidCredit — because every credit decision deserves a clear explanation.*
