# LucidCredit

> **Zero-hallucination AI analytical copilot for credit decisions.**

LucidCredit provides natural language explanations of credit decisions grounded entirely in retrieved data — no parametric hallucination. Every claim traces back to a source document, API response, or database record.

## Audiences
- **Analyst Briefings** — SHAP narratives, PD distributions, portfolio stress summaries for risk officers and CROs
- **Applicant Communications** — Plain English ECOA/FCRA-compliant adverse action notices and approval summaries

## Integration
- Consumes `credit-risk-platform` decision and explainability APIs
- Consumes `ThinFile_Credit_Underwriting_Engine` score and adverse-action APIs
- Read-only SQL access to analytics replicas
- pgvector document store for regulatory, policy, and model card corpora

## Stack
Python 3.11 · FastAPI · LangGraph · GPT-4o · pgvector · Redis · Next.js 14
