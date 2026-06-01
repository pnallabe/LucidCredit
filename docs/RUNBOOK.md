# LucidCredit — Production Runbook

## Services

| Service | Port | Description |
|---------|------|-------------|
| `copilot-api` | 8090 | FastAPI backend (LangGraph agent) |
| `copilot-ui` | 3010 | Next.js frontend |
| `db` | 5440 | PostgreSQL 16 + pgvector |
| `redis` | 6381 | Redis 7 (caching + circuit breaker) |

---

## Start / Stop

```bash
# Start all services (detached)
cd /path/to/LucidCredit
docker-compose up -d

# Start only backend + dependencies
docker-compose up -d db redis copilot-api

# Stop all
docker-compose down

# Stop and wipe all data volumes (destructive)
docker-compose down -v
```

---

## Health Checks

```bash
# Backend
curl http://localhost:8090/v1/health

# Frontend (dev)
curl http://localhost:3010

# Database
docker-compose exec db pg_isready -U lucidcredit

# Redis
docker-compose exec redis redis-cli ping
```

Expected healthy response from backend:
```json
{"status": "ok", "version": "..."}
```

---

## Logs

```bash
# Tail backend logs
docker-compose logs -f copilot-api

# Tail all services
docker-compose logs -f

# Last 100 lines from backend
docker-compose logs --tail=100 copilot-api
```

---

## Database Migrations

```bash
# Apply all pending migrations
cd backend
alembic upgrade head

# Check current migration state
alembic current

# Downgrade one step (reversible rollback)
alembic downgrade -1

# Downgrade to a specific revision
alembic downgrade <revision_id>
```

---

## Rollback Procedure

### 1. Application rollback (Docker image)

```bash
# Re-deploy a previous image tag
docker-compose down copilot-api
IMAGE_TAG=<previous-tag> docker-compose up -d copilot-api
```

### 2. Database rollback

```bash
# Identify the target revision
cd backend && alembic history

# Downgrade to the revision before the bad migration
alembic downgrade <target-revision>
```

> **Warning**: Downgrading migrations that drop columns or tables is destructive.
> Always take a DB snapshot before running alembic downgrade in production.

### 3. DB snapshot (before risky migrations)

```bash
docker-compose exec db pg_dump -U lucidcredit lucidcredit \
  > /tmp/lucidcredit_backup_$(date +%Y%m%d_%H%M%S).sql
```

---

## Running Tests (CI-equivalent)

```bash
cd backend

# Set required env vars (copy from .env or CI secrets)
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_KEY="..."
export DATABASE_URL="postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit"
export REDIS_URL="redis://localhost:6381"
export JWT_SECRET_KEY="..."
export GOOGLE_PROJECT_ID="..."
export ENVIRONMENT="test"

# Full CI suite (no live services needed — all mock-based)
pytest tests/unit/ tests/integration/ tests/eval/test_hallucination_regression.py -v

# Live integration tests (requires analytics API on :8001 + backend on :8090)
# Run from repo root after starting services
python3 tests/e2e_chain_audit.py
python3 tests/chatbot_chain_test.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | Embedding model deployment name |
| `AZURE_OPENAI_API_VERSION` | Yes | API version (e.g. `2024-10-21`) |
| `DATABASE_URL` | Yes | Async PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `JWT_SECRET_KEY` | Yes | Secret for JWT signing |
| `GOOGLE_PROJECT_ID` | No | GCP project for Vertex AI fallback |
| `ANALYST_FALLBACK_ENABLED` | No | `true` to enable Vertex AI fallback for analyst sessions (default: `false`) |
| `ENVIRONMENT` | No | `production` / `test` / `development` |

---

## Incident Response

### Backend returns 503 for applicant sessions
- **Cause**: Azure OpenAI primary is down. This is the designed safe failure mode — applicant sessions never fall back to a secondary provider.
- **Action**: Check Azure OpenAI service status. Resume when primary is healthy. Retry-After: 30s is sent in the 503 response.

### Backend returns 503 for analyst/briefing sessions
- **Cause**: Both Azure OpenAI and Vertex AI (if configured) are unavailable.
- **Action**: Check both providers. Enable `ANALYST_FALLBACK_ENABLED=true` and set Vertex credentials if Azure is down.

### Confidence scores consistently low (< 0.75)
- **Cause**: RAG retrieval not returning relevant chunks — likely vector store issue or bad embeddings.
- **Check**: Query the `copilot_sessions` table for `confidence_score` distribution. Look at `retrieved_chunks` JSONB for empty arrays.

### High rate of suppressed claims in responses
- **Cause**: LLM generating statements that can't be grounded to retrieved context.
- **Check**: `copilot_sessions.suppressed_claims` column. If consistently > 30%, review prompt templates and chunk quality.

### DB connection pool exhaustion
```bash
# Check active connections
docker-compose exec db psql -U lucidcredit -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections older than 10 minutes
docker-compose exec db psql -U lucidcredit -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
   WHERE state = 'idle' AND query_start < now() - interval '10 minutes';"
```

### Redis circuit breaker stuck open
```bash
# Flush circuit breaker keys
docker-compose exec redis redis-cli keys "circuit_breaker:*" | xargs redis-cli del

# Or flush all (use with caution in production)
docker-compose exec redis redis-cli flushdb
```

---

## Audit Trail Queries

```sql
-- Recent sessions with low confidence
SELECT session_id, query_text, confidence_score, created_at
FROM copilot_sessions
WHERE confidence_score < 0.5
ORDER BY created_at DESC
LIMIT 20;

-- Compliance flags in the last 24 hours
SELECT session_id, query_text, compliance_flags, created_at
FROM copilot_sessions
WHERE compliance_flags != '[]'
  AND created_at > now() - interval '24 hours'
ORDER BY created_at DESC;

-- Provider fallback usage rate
SELECT
  provider_fallback_used,
  count(*) as session_count,
  round(count(*) * 100.0 / sum(count(*)) over (), 1) as pct
FROM copilot_sessions
GROUP BY provider_fallback_used;
```
