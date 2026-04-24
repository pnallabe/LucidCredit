# LucidCredit

> **Zero-hallucination AI analytical copilot for credit decisions.**

[![CI](https://github.com/swarnabale/lucidcredit/actions/workflows/ci.yml/badge.svg)](https://github.com/swarnabale/lucidcredit/actions/workflows/ci.yml)
[![Docker](https://github.com/swarnabale/lucidcredit/actions/workflows/docker.yml/badge.svg)](https://github.com/swarnabale/lucidcredit/actions/workflows/docker.yml)

LucidCredit provides natural language explanations of credit decisions grounded entirely in retrieved data — no parametric hallucination. Every claim traces back to a source document, API response, or database record, enforced at inference time by `CitationEnforcer` and `ConfidenceScorer`.

---

## Contents

- [Audiences](#audiences)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start — Docker](#quick-start--docker)
- [Quick Start — Local Dev](#quick-start--local-dev)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [RAGAS Evaluation](#ragas-evaluation)
- [API Reference](#api-reference)
- [Sprint History](#sprint-history)

---

## Audiences

| Role | Use Case |
|---|---|
| **Risk Analyst / CRO** | SHAP-grounded briefings, PD distributions, portfolio stress summaries |
| **Credit Officer** | Decision narratives, counterfactual "what-if" analysis |
| **Applicant** | Plain English ECOA/FCRA-compliant adverse action and approval notices |
| **Model Governance** | SR 11-7 audit trail, provider routing logs, confidence scores |

---

## Architecture

```
┌──────────────────┐   REST/JSON   ┌──────────────────────────────────────────┐
│  Next.js 15 UI   │◄─────────────►│  FastAPI  (port 8090)                    │
│  (port 3090 dev) │               │  ┌─────────────────────────────────────┐ │
│  Analyst page    │               │  │  LangGraph agent graph              │ │
│  Query builder   │               │  │  ├── retrieve_node  (pgvector)      │ │
│  Applicant view  │               │  │  ├── citation_enforcer              │ │
│  Audit trail     │               │  │  ├── confidence_scorer              │ │
└──────────────────┘               │  │  └── compliance_node                │ │
                                   │  │       ├── ECOA validator            │ │
                                   │  │       └── SR 11-7 disclosures       │ │
                                   │  └─────────────────────────────────────┘ │
                                   │                                          │
                                   │  PostgreSQL + pgvector (port 5440)       │
                                   │  Redis (LangGraph checkpointer, 6380)    │
                                   └──────────────────────────────────────────┘
                                            ▲              ▲
                                   credit-risk-platform   ThinFile Engine
                                   (decisions, SHAP)      (thin-file scores)
```

**Zero-hallucination guarantee**: Every generated claim is matched against retrieved chunks via embedding similarity. Claims below the confidence threshold are stripped and logged to the audit trail. The model is forbidden from providing information it cannot cite.

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Docker + Docker Compose | 25.0 |
| Python | 3.11 |
| Node.js | 20 LTS |
| PostgreSQL (pgvector) | pg16 (via Docker) |

---

## Quick Start — Docker

The fastest path to a running stack:

```bash
# 1. Clone and enter the project
git clone https://github.com/swarnabale/lucidcredit.git
cd lucidcredit

# 2. Copy the environment template and fill in your Azure OpenAI keys
cp .env.example .env
# Edit .env — at minimum set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY

# 3. Start all services (db, redis, copilot-api, copilot-ui)
docker compose up -d

# 4. Apply database migrations
docker compose exec copilot-api alembic upgrade head

# 5. Ingest the regulatory + model card corpus
docker compose exec copilot-api python scripts/ingest_corpus.py

# 6. Open the UI
open http://localhost:3010
# API docs available at http://localhost:8090/docs
```

To stop all services: `docker compose down`

---

## Quick Start — Local Dev

### Backend

```bash
cd backend

# Create venv
python3.11 -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Set environment (copy from root .env)
set -a && source ../.env && set +a

# Apply migrations
alembic upgrade head

# Start the API server (hot-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
# → http://localhost:8090/docs
```

### Frontend

```bash
cd frontend

npm install

# Start the dev server
npm run dev
# → http://localhost:3090
```

The frontend proxies all `/api/*` requests to `http://localhost:8090` via `next.config.ts` rewrites.

---

## Environment Variables

Create `.env` at the project root (copy from `.env.example`):

```dotenv
# ── Azure OpenAI ─────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-2025-04-14
AZURE_OPENAI_API_VERSION=2024-10-21

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6380

# ── Security ─────────────────────────────────────────────────────────────────
JWT_SECRET_KEY=<generate-with-openssl-rand-hex-32>

# ── Provider fallback (optional — requires GCP) ───────────────────────────────
ANALYST_FALLBACK_ENABLED=false
GOOGLE_PROJECT_ID=          # required only when ANALYST_FALLBACK_ENABLED=true
```

---

## Running Tests

### Unit + Integration (100 tests, no live LLM)

```bash
cd backend
source .venv/bin/activate
set -a && source ../.env && set +a

# All non-eval tests
pytest tests/unit/ tests/integration/ -v

# Breakdown by sprint
pytest tests/unit/test_sprint1.py -v        # 33 tests — RAG & retrieval
pytest tests/integration/test_sprint2.py -v # 27 tests — LangGraph agent
pytest tests/integration/test_sprint3.py -v # 40 tests — API & compliance
```

### Hallucination Regression (no live LLM)

```bash
pytest tests/eval/test_hallucination_regression.py -v
# Tests: citation enforcement, confidence scoring, ECOA compliance, adversarial patterns
```

---

## RAGAS Evaluation

Full RAGAS evaluation against 50 golden Q&A pairs across three session types.  
**Requires live Azure OpenAI credentials.**

```bash
# Run against Azure GPT-4.1 (default)
pytest tests/eval/ -m ragas --provider azure_gpt41 -v

# Other providers
pytest tests/eval/ -m ragas --provider azure_gpt4o -v
pytest tests/eval/ -m ragas --provider vertex_gemini15pro -v

# Results are written to
#   tests/eval/results/<timestamp>_<provider>.json
```

**Threshold targets per session type:**

| Session type | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| analyst | ≥ 0.90 | ≥ 0.80 | ≥ 0.85 |
| briefing | ≥ 0.88 | ≥ 0.78 | ≥ 0.82 |
| adversarial | ≥ 0.95 | ≥ 0.60 | ≥ 0.80 |

The adversarial session validates that the model refuses to fabricate answers to trick questions (invented regulations, out-of-context numerics, protected-class proxies).

---

## API Reference

Full interactive docs at `http://localhost:8090/docs` (Swagger UI).

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/api/analyst/query` | POST | Analyst Q&A with SHAP grounding |
| `/api/analyst/briefing` | POST | Portfolio briefing narrative |
| `/api/applicant/narrative` | POST | Applicant-facing decision explanation |
| `/api/applicant/adverse-action` | POST | ECOA-compliant adverse action notice |
| `/api/audit/{session_id}` | GET | Full audit trail for a session |
| `/api/audit/{session_id}/citations` | GET | All citations with similarity scores |

---

## Sprint History

| Sprint | Deliverable | Tests |
|---|---|---|
| Sprint 1 | Core RAG: pgvector ingestion, retriever, chunk ranking | 33 |
| Sprint 2 | LangGraph agent, zero-hallucination layer (CitationEnforcer, ConfidenceScorer) | 27 |
| Sprint 3 | FastAPI endpoints, ECOA validator, SR 11-7 disclosures, audit trail | 40 |
| Sprint 4 | Next.js 15 frontend: Analyst, Query, Applicant, Audit pages | — |
| Sprint 5 | Hardening: 50 golden Q&As, RAGAS harness, hallucination regression, Docker, CI/CD | + |

---

## Integration

- **`credit-risk-platform`** — Consumes decision and explainability APIs; reads SHAP values, PD scores, feature importances
- **`ThinFile_Credit_Underwriting_Engine`** — Consumes thin-file model scores and adverse-action reason codes
- Both are read-only integrations; LucidCredit never writes to upstream systems

---

## License

MIT — see [LICENSE](LICENSE)
