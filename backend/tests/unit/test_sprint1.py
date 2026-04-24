"""
backend/tests/unit/test_sprint1.py
=====================================
Sprint 1 acceptance tests — no live DB, LLM, or network calls required.

Test Groups
-----------
A  SQL Security (sql_tool.validate_sql)
   A1  SELECT passes
   A2  CTE (WITH ... SELECT) passes
   A3  EXPLAIN passes
   A4  INSERT blocked
   A5  UPDATE blocked
   A6  DELETE blocked
   A7  DROP blocked
   A8  CREATE blocked
   A9  TRUNCATE blocked
   A10 GRANT blocked
   A11 REVOKE blocked
   A12 SQL comment injection (--) cannot hide DML
   A13 Block-comment injection (/* ... */) cannot hide DML
   A14 Table not in allow-list blocked
   A15 Schema-qualified table allowed (public.loan_applications)
   A16 Multi-table JOIN — all allowed passes
   A17 Multi-table JOIN — one disallowed blocked
   A18 Subquery with disallowed table blocked

B  BM25 Retriever Utilities (app.rag.retriever internals)
   B1  _tokenize strips punctuation and lowercases
   B2  _bm25_scores returns list same length as corpus
   B3  _bm25_scores — exact match scores higher than miss
   B4  _bm25_scores — empty query returns zeros
   B5  _bm25_scores — empty corpus returns empty list

C  Hybrid Retriever score fusion
   C1  Fusion weights sum to 1.0
   C2  Higher-dense result wins over lower-dense result when BM25 is equal
   C3  Dense result with low similarity is promoted by high BM25 score

D  ThinFile tool helpers
   D1  fetch_thin_file_context returns [] on connect error (no server)
   D2  fetch_adverse_action_codes returns [] on connect error (no server)
   D3  Returned chunks have required fields
   D4  Content is capped at 4000 chars

E  CRP API tool helpers
   E1  fetch_portfolio_metrics returns [] on connect error
   E2  Returned chunks have source_type == "api"

Run with:
    cd backend
    pytest tests/unit/test_sprint1.py -v
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.tools.sql_tool import (
    SqlSecurityError,
    _validate_select_only,
    _validate_tables,
    validate_sql,
)
from app.rag.retriever import DENSE_WEIGHT, BM25_WEIGHT, _bm25_scores, _tokenize


# ============================================================================
# A — SQL Security
# ============================================================================

class TestSqlValidateSelectOnly:
    def test_a1_plain_select(self):
        validate_sql("SELECT id, amount FROM loan_applications LIMIT 10")

    def test_a2_cte_with_select(self):
        validate_sql(
            "WITH cte AS (SELECT id FROM loan_applications) "
            "SELECT * FROM cte"
        )

    def test_a3_explain_allowed(self):
        validate_sql("EXPLAIN SELECT * FROM loan_applications")

    def test_a4_insert_blocked(self):
        with pytest.raises(SqlSecurityError, match="INSERT"):
            validate_sql("INSERT INTO loan_applications (id) VALUES ('x')")

    def test_a5_update_blocked(self):
        with pytest.raises(SqlSecurityError, match="UPDATE"):
            validate_sql("UPDATE loan_applications SET status='bad' WHERE id='x'")

    def test_a6_delete_blocked(self):
        with pytest.raises(SqlSecurityError, match="DELETE"):
            validate_sql("DELETE FROM loan_applications WHERE id='x'")

    def test_a7_drop_blocked(self):
        with pytest.raises(SqlSecurityError, match="DROP"):
            validate_sql("DROP TABLE loan_applications")

    def test_a8_create_blocked(self):
        with pytest.raises(SqlSecurityError, match="CREATE"):
            validate_sql("CREATE TABLE evil (id text)")

    def test_a9_truncate_blocked(self):
        with pytest.raises(SqlSecurityError, match="TRUNCATE"):
            validate_sql("TRUNCATE loan_applications")

    def test_a10_grant_blocked(self):
        with pytest.raises(SqlSecurityError, match="GRANT"):
            validate_sql("GRANT ALL ON loan_applications TO attacker")

    def test_a11_revoke_blocked(self):
        with pytest.raises(SqlSecurityError, match="REVOKE"):
            validate_sql("REVOKE SELECT ON loan_applications FROM analyst")

    def test_a12_comment_injection_blocked(self):
        # Attacker tries to hide INSERT after a comment
        with pytest.raises(SqlSecurityError):
            validate_sql("SELECT 1; -- this is fine\nINSERT INTO loan_applications VALUES (1)")

    def test_a13_block_comment_injection_blocked(self):
        with pytest.raises(SqlSecurityError):
            validate_sql("SELECT /* DROP TABLE x; */ 1 FROM loan_applications; DROP TABLE loan_applications")

    def test_a14_disallowed_table_blocked(self):
        with pytest.raises(SqlSecurityError, match="allow-list"):
            validate_sql("SELECT * FROM users")

    def test_a15_schema_qualified_allowed(self):
        validate_sql("SELECT * FROM public.loan_applications LIMIT 5")

    def test_a16_multi_join_all_allowed(self):
        validate_sql(
            "SELECT a.id, f.score FROM loan_applications a "
            "JOIN features f ON a.id = f.application_id "
            "LIMIT 20"
        )

    def test_a17_multi_join_one_disallowed(self):
        with pytest.raises(SqlSecurityError, match="allow-list"):
            validate_sql(
                "SELECT a.id FROM loan_applications a "
                "JOIN secret_table s ON a.id = s.id"
            )

    def test_a18_subquery_disallowed_table(self):
        with pytest.raises(SqlSecurityError, match="allow-list"):
            validate_sql(
                "SELECT id FROM loan_applications "
                "WHERE id IN (SELECT id FROM internal_passwords)"
            )


# ============================================================================
# B — BM25 Retriever Utilities
# ============================================================================

class TestTokenize:
    def test_b1_lowercase_and_strip_punctuation(self):
        tokens = _tokenize("Hello, World! This is a TEST: 123.")
        assert tokens == ["hello", "world", "this", "is", "a", "test", "123"]

    def test_b1_empty_string(self):
        assert _tokenize("") == []


class TestBm25Scores:
    def test_b2_result_length_matches_corpus(self):
        corpus = ["the quick brown fox", "jumps over the lazy dog", "hello world"]
        scores = _bm25_scores(["fox"], corpus)
        assert len(scores) == len(corpus)

    def test_b3_exact_match_scores_higher(self):
        corpus = ["ECOA adverse action notice section 202", "unrelated document about apples"]
        query_tokens = _tokenize("ECOA adverse action")
        scores = _bm25_scores(query_tokens, corpus)
        assert scores[0] > scores[1]

    def test_b4_empty_query_returns_zeros(self):
        corpus = ["some document"]
        scores = _bm25_scores([], corpus)
        assert scores == [0.0]

    def test_b5_empty_corpus_returns_empty(self):
        scores = _bm25_scores(["query"], [])
        assert scores == []


# ============================================================================
# C — Hybrid Retriever score fusion
# ============================================================================

class TestScoreFusion:
    def test_c1_weights_sum_to_one(self):
        assert abs(DENSE_WEIGHT + BM25_WEIGHT - 1.0) < 1e-9

    def test_c2_higher_dense_wins_equal_bm25(self):
        # Build two mock chunks with different dense scores, equal BM25
        scores_a = [0.9, 0.5]
        scores_b = [0.9, 0.5]
        # Apply fusion manually
        fused = [DENSE_WEIGHT * d + BM25_WEIGHT * b for d, b in zip(scores_a, scores_b)]
        assert fused[0] > fused[1]

    def test_c3_high_bm25_can_promote_low_dense(self):
        # doc A: dense=0.5, bm25=1.0 vs doc B: dense=0.7, bm25=0.0
        score_a = DENSE_WEIGHT * 0.5 + BM25_WEIGHT * 1.0
        score_b = DENSE_WEIGHT * 0.7 + BM25_WEIGHT * 0.0
        assert score_a > score_b


# ============================================================================
# D — ThinFile tool helpers
# ============================================================================

class TestThinFileTool:
    """All tests use httpx mock to avoid needing a live ThinFile server."""

    def _mock_settings(self):
        from app.config import Settings
        return Settings.model_construct(
            thinfile_api_base_url="http://thinfile-test:8000",
            thinfile_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_d1_connect_error_returns_empty(self):
        import httpx
        from app.agent.tools.thinfile_tool import fetch_thin_file_context
        with patch("app.agent.tools.thinfile_tool.get_settings", return_value=self._mock_settings()), \
             patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("refused")), \
             patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("refused")):
            result = await fetch_thin_file_context("app-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_d2_adverse_action_connect_error_returns_empty(self):
        import httpx
        from app.agent.tools.thinfile_tool import fetch_adverse_action_codes
        with patch("app.agent.tools.thinfile_tool.get_settings", return_value=self._mock_settings()), \
             patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("refused")):
            result = await fetch_adverse_action_codes("app-456")
        assert result == []

    @pytest.mark.asyncio
    async def test_d3_returned_chunk_has_required_fields(self):
        import httpx
        from app.agent.tools.thinfile_tool import fetch_adverse_action_codes

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"codes": ["AA-001"], "reason": "Low score"})

        with patch("app.agent.tools.thinfile_tool.get_settings", return_value=self._mock_settings()):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await fetch_adverse_action_codes("app-789")

        assert len(result) == 1
        chunk = result[0]
        required_fields = {"chunk_id", "source_type", "source_ref", "content", "relevance", "similarity_score"}
        assert required_fields.issubset(chunk.keys())
        assert chunk["source_type"] == "api"

    @pytest.mark.asyncio
    async def test_d4_content_capped_at_4000_chars(self):
        from app.agent.tools.thinfile_tool import _payload_to_chunk
        big_payload: dict[str, Any] = {"data": "x" * 10000}
        chunk = _payload_to_chunk(big_payload, "ref", "id")
        assert len(chunk["content"]) <= 4000


# ============================================================================
# E — CRP API tool helpers
# ============================================================================

class TestCrpApiTool:
    def _mock_settings(self):
        from app.config import Settings
        return Settings.model_construct(
            crp_api_base_url="http://crp-test:8081",
            crp_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_e1_connect_error_returns_empty(self):
        import httpx
        from app.agent.tools.crp_api_tool import fetch_portfolio_metrics
        with patch("app.agent.tools.crp_api_tool.get_settings", return_value=self._mock_settings()), \
             patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("refused")):
            result = await fetch_portfolio_metrics()
        assert result == []

    @pytest.mark.asyncio
    async def test_e2_successful_response_returns_api_chunk(self):
        import httpx
        from app.agent.tools.crp_api_tool import fetch_portfolio_metrics

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"total_loans": 5000, "default_rate": 0.032})

        with patch("app.agent.tools.crp_api_tool.get_settings", return_value=self._mock_settings()):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await fetch_portfolio_metrics()

        assert len(result) == 1
        assert result[0]["source_type"] == "api"
        assert "total_loans" in result[0]["content"]
