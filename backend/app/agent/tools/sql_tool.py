"""
backend/app/agent/tools/sql_tool.py
=====================================
Tool: read-only SQL executor with allow-list validation.

Public API
----------
    execute_analyst_query(sql, params) -> SqlQueryResult
        Validates the statement is a SELECT-only query, checks all referenced
        tables are in the allow-list, executes it, and returns rows as JSON.

Security model
--------------
- Only SELECT statements are accepted (checked by token parsing, not just
  startswith — catches WITH...SELECT and subquery injection attempts).
- Table allow-list is enforced: any table name outside the list triggers a
  SqlSecurityError before the query reaches the database.
- Bind parameters (not string interpolation) are always used for values.
- The database user has SELECT-only grants; this code adds a defence-in-depth
  layer on top of that.

The executed SQL text is returned with the result for grounding/audit.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.agent.state import RetrievedChunk
from app.db.session import AsyncSessionLocal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table allow-list  (read-replicas only — see PLAN.md Source 4)
# ---------------------------------------------------------------------------
_ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        # credit_risk_loans
        "loan_applications",
        "features",
        "funded_loans",
        # credit_risk_transactions
        "bank_accounts",
        "payment_history",
        # thinfile_audit
        "audit_logs",
        "feature_snapshots",
        # lucidcredit own tables (read)
        "policy_docs",
        "copilot_sessions",
        "citations",
    }
)

# Hard row cap — prevents runaway analytical queries from OOMing the server
_MAX_ROWS = 200


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SqlSecurityError(ValueError):
    """Raised when the SQL statement fails security validation."""


class SqlExecutionError(RuntimeError):
    """Raised when the database returns an error."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SqlQueryResult:
    sql: str
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Strip SQL comments before validation
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
# Table names after FROM / JOIN  (handles schema-qualified names too)
_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def _validate_select_only(sql: str) -> None:
    """Raise SqlSecurityError if *sql* contains any DML/DDL keyword."""
    clean = _strip_comments(sql).upper()
    forbidden = (
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "CALL", "COPY",
    )
    for kw in forbidden:
        # Word-boundary check avoids false positives in column names
        if re.search(rf"\b{kw}\b", clean):
            raise SqlSecurityError(
                f"Statement contains forbidden keyword '{kw}'. "
                "Only SELECT queries are permitted."
            )

    first_token = clean.lstrip().split()[0] if clean.strip() else ""
    if first_token not in ("SELECT", "WITH", "EXPLAIN"):
        raise SqlSecurityError(
            f"Statement must begin with SELECT / WITH / EXPLAIN, got '{first_token}'."
        )


def _validate_tables(sql: str) -> None:
    """Raise SqlSecurityError if any referenced table is not in the allow-list."""
    clean = _strip_comments(sql)

    # Collect CTE names defined by WITH cte_name AS (...)
    # so we don't flag them as unknown tables when they appear in FROM clauses.
    _CTE_NAME_RE = re.compile(
        r"\bWITH\b.*?\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    cte_names: set[str] = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(clean)}

    referenced: list[str] = []
    for match in _TABLE_RE.finditer(clean):
        # Strip optional schema prefix (e.g. public.loan_applications → loan_applications)
        table = match.group(1).split(".")[-1].lower()
        if table not in cte_names:
            referenced.append(table)

    disallowed = [t for t in referenced if t not in _ALLOWED_TABLES]
    if disallowed:
        raise SqlSecurityError(
            f"Table(s) not in allow-list: {', '.join(disallowed)}. "
            f"Permitted tables: {', '.join(sorted(_ALLOWED_TABLES))}."
        )


def validate_sql(sql: str) -> None:
    """Run all security checks. Raises SqlSecurityError on failure."""
    _validate_select_only(sql)
    _validate_tables(sql)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

async def execute_analyst_query(
    sql: str,
    params: dict[str, Any] | None = None,
) -> SqlQueryResult:
    """
    Validate *sql*, execute it read-only, and return a :class:`SqlQueryResult`.

    All rows are JSON-serializable. At most ``_MAX_ROWS`` rows are returned;
    if the query would return more, ``truncated=True`` is set.

    Raises :class:`SqlSecurityError` on policy violations.
    Raises :class:`SqlExecutionError` on database errors.
    """
    try:
        validate_sql(sql)
    except SqlSecurityError:
        raise  # re-raise to caller

    # Inject a LIMIT if not already present to cap runaway queries
    capped_sql = sql
    if "LIMIT" not in sql.upper():
        capped_sql = f"{sql.rstrip().rstrip(';')} LIMIT {_MAX_ROWS + 1}"

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(capped_sql), params or {})
            keys = list(result.keys())
            all_rows = result.fetchall()
    except Exception as exc:
        raise SqlExecutionError(str(exc)) from exc

    truncated = len(all_rows) > _MAX_ROWS
    rows = all_rows[:_MAX_ROWS]

    rows_as_dicts: list[dict[str, Any]] = []
    for row in rows:
        rows_as_dicts.append(
            {k: (v if not hasattr(v, "isoformat") else v.isoformat()) for k, v in zip(keys, row)}
        )

    return SqlQueryResult(
        sql=sql,
        rows=rows_as_dicts,
        row_count=len(rows_as_dicts),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Convert SQL result to RetrievedChunk for the agent
# ---------------------------------------------------------------------------

def sql_result_to_chunk(result: SqlQueryResult) -> RetrievedChunk:
    """Wrap a SqlQueryResult as a RetrievedChunk for use in the agent graph."""
    content = json.dumps(
        {
            "sql": result.sql,
            "rows": result.rows[:50],  # include first 50 rows in chunk content
            "row_count": result.row_count,
            "truncated": result.truncated,
        },
        default=str,
    )
    return RetrievedChunk(
        chunk_id=f"sql:{hash(result.sql) & 0xFFFFFFFF:08x}",
        source_type="db",
        source_ref=f"sql_query:{result.sql[:120]}",
        content=content,
        relevance="AMBIGUOUS",
        similarity_score=1.0,
    )
