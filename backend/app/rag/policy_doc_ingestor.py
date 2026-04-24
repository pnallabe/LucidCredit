"""
backend/app/rag/policy_doc_ingestor.py
========================================
Phase 6 — LucidCredit RAG ingestion pipeline for governance Markdown documents.

Scans ``docs/governance/**/*.md`` from the credit-risk-platform repository,
parses YAML front matter, chunks the body text, embeds chunks via OpenAI
text-embedding-3-large, and upserts into the ``policy_docs`` pgvector table.

CLI
---
    # Dry run (no DB writes, no embeddings):
    python backend/app/rag/policy_doc_ingestor.py \\
        --governance-root ../credit-risk-platform/docs/governance/ \\
        --dry-run

    # Full ingest:
    python backend/app/rag/policy_doc_ingestor.py \\
        --governance-root ../credit-risk-platform/docs/governance/ \\
        --db-url postgresql+asyncpg://lucidcredit:password@localhost:5440/lucidcredit

    # Single section only:
    python backend/app/rag/policy_doc_ingestor.py \\
        --governance-root ../credit-risk-platform/docs/governance/ \\
        --section policy_manuals

Public API
----------
    scan_governance_docs(root_path)     → list[Path]
    extract_front_matter(md_path)       → dict
    chunk_document(text, ...)           → list[str]
    upsert_chunks(chunks, db_session)   → int
    ingest_all(governance_root, db_url) → dict
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Bootstrap path so the module can be run directly from the backend/ root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parent.parent.parent  # .../LucidCredit/backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Default governance root relative to this file:
# LucidCredit/backend/app/rag/  →  ../../../../credit-risk-platform/docs/governance/
_DEFAULT_GOVERNANCE_ROOT = _HERE.parents[4] / "credit-risk-platform" / "docs" / "governance"

# ---------------------------------------------------------------------------
# Lazy imports from app package (after sys.path fix)
# ---------------------------------------------------------------------------
from app.config import get_settings  # noqa: E402  (after path fix)
from app.rag.embedder import batch_embed  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting helper (using tiktoken if available, else word-count approx)
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def _token_count(text: str) -> int:
        return len(_ENCODER.encode(text))

    def _token_slice(text: str, max_tokens: int) -> str:
        tokens = _ENCODER.encode(text)
        return _ENCODER.decode(tokens[:max_tokens])

except ImportError:  # pragma: no cover
    log.warning("tiktoken not installed — using word-count approximation (words × 1.33)")

    def _token_count(text: str) -> int:  # type: ignore[misc]
        return max(1, int(len(text.split()) * 1.33))

    def _token_slice(text: str, max_tokens: int) -> str:  # type: ignore[misc]
        words = text.split()
        limit = max(1, int(max_tokens / 1.33))
        return " ".join(words[:limit])


# ===========================================================================
# 1. Document scanning
# ===========================================================================

def scan_governance_docs(root_path: Path) -> list[Path]:
    """
    Recursively find all ``*.md`` files under *root_path*.

    Returns paths sorted deterministically (lexicographic).
    Raises :class:`FileNotFoundError` if *root_path* does not exist.
    """
    root_path = Path(root_path).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Governance root not found: {root_path}")
    paths = sorted(root_path.rglob("*.md"))
    log.info("Found %d governance documents under %s", len(paths), root_path)
    return paths


# ===========================================================================
# 2. Front-matter extraction
# ===========================================================================

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def extract_front_matter(md_path: Path) -> dict[str, Any]:
    """
    Parse YAML front matter delimited by ``---`` at the top of *md_path*.

    Returns the parsed dict, or ``{}`` if no front matter is present.
    Unknown keys are passed through unchanged.
    """
    content = md_path.read_text(encoding="utf-8")
    m = _FRONT_MATTER_RE.match(content)
    if not m:
        return {}
    try:
        fm: dict = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        log.warning("YAML parse error in %s: %s", md_path, exc)
        return {}
    return fm


def _body_text(md_path: Path) -> str:
    """Return the Markdown body with YAML front matter stripped."""
    content = md_path.read_text(encoding="utf-8")
    return _FRONT_MATTER_RE.sub("", content, count=1)


def _infer_effective_to(fm: dict[str, Any]) -> date | None:
    """
    Infer ``effective_to`` from ``report_year`` (model_validation / fair_lending docs)
    when the key is absent.
    """
    if "effective_to" in fm and fm["effective_to"]:
        val = fm["effective_to"]
        if isinstance(val, date):
            return val
        try:
            return datetime.strptime(str(val), "%Y-%m-%d").date()
        except ValueError:
            return None
    if "report_year" in fm:
        try:
            return date(int(fm["report_year"]), 12, 31)
        except (ValueError, TypeError):
            pass
    if "quarter" in fm:
        # e.g. "2020-Q3"
        try:
            yr, qn = str(fm["quarter"]).split("-Q")
            q_end_month = int(qn) * 3
            q_end_day = {3: 31, 6: 30, 9: 30, 12: 31}[q_end_month]
            return date(int(yr), q_end_month, q_end_day)
        except (ValueError, KeyError):
            pass
    return None


def _coerce_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val), "%Y-%m-%d").date()
    except ValueError:
        return None


# ===========================================================================
# 3. Chunking
# ===========================================================================

def chunk_document(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Paragraph-level chunking with sliding-window token overlap.

    Strategy
    --------
    1. Split body on double-newline paragraph boundaries.
    2. Merge paragraphs shorter than 50 tokens into the following paragraph.
    3. For paragraphs exceeding *max_tokens*, split at sentence boundaries until
       each sub-paragraph fits.
    4. Accumulate paragraphs into a chunk until adding the next would exceed
       *max_tokens*.  When a chunk is full, save it and start the next chunk
       pre-seeded with the last *overlap_tokens* tokens from the previous chunk.
    """
    # Step 1 — split on paragraph boundaries
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Step 2 — merge short paragraphs (<50 tokens) forward
    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            combined = buffer + "\n\n" + para
            if _token_count(buffer) < 50:
                buffer = combined
                continue
            else:
                merged.append(buffer)
                buffer = para
        else:
            buffer = para
    if buffer:
        merged.append(buffer)

    # Step 3 — split oversized paragraphs at sentence boundaries
    normalized: list[str] = []
    for para in merged:
        if _token_count(para) <= max_tokens:
            normalized.append(para)
        else:
            # Split at sentence boundaries (. ! ? followed by whitespace)
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                candidate = (current + " " + sent).strip() if current else sent
                if _token_count(candidate) <= max_tokens:
                    current = candidate
                else:
                    if current:
                        normalized.append(current)
                    # If a single sentence exceeds max_tokens, hard-slice it
                    if _token_count(sent) > max_tokens:
                        normalized.append(_token_slice(sent, max_tokens))
                    else:
                        current = sent
            if current:
                normalized.append(current)

    # Step 4 — accumulate into final chunks with overlap
    chunks: list[str] = []
    current_chunk = ""
    overlap_prefix = ""

    for para in normalized:
        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if _token_count(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Build overlap prefix from end of current_chunk
                overlap_prefix = _token_slice_tail(current_chunk, overlap_tokens)
            current_chunk = (overlap_prefix + "\n\n" + para).strip() if overlap_prefix else para
            overlap_prefix = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:2000]]  # Fallback: return raw text head


def _token_slice_tail(text: str, n_tokens: int) -> str:
    """Return the last *n_tokens* tokens of *text* as a string."""
    try:
        tokens = _ENCODER.encode(text)
        tail_tokens = tokens[-n_tokens:]
        return _ENCODER.decode(tail_tokens)
    except NameError:
        words = text.split()
        limit = max(1, int(n_tokens / 1.33))
        return " ".join(words[-limit:])


# ===========================================================================
# 4. Upsert to pgvector
# ===========================================================================

_UPSERT_SQL = text("""
INSERT INTO policy_docs (
    id, source_path, chunk_index, content, embedding,
    source_type, product_type, effective_from, effective_to,
    version_tag, doc_type, metadata_json
) VALUES (
    gen_random_uuid(), :source_path, :chunk_index, :content, CAST(:embedding AS vector),
    :source_type, :product_type, :effective_from, :effective_to,
    :version_tag, :doc_type, :metadata_json
)
ON CONFLICT (source_path, chunk_index)
DO UPDATE SET
    content        = EXCLUDED.content,
    embedding      = EXCLUDED.embedding,
    source_type    = EXCLUDED.source_type,
    product_type   = EXCLUDED.product_type,
    effective_from = EXCLUDED.effective_from,
    effective_to   = EXCLUDED.effective_to,
    version_tag    = EXCLUDED.version_tag,
    doc_type       = EXCLUDED.doc_type,
    metadata_json  = EXCLUDED.metadata_json
""")


async def upsert_chunks(
    chunks: list[dict[str, Any]],
    db_session: AsyncSession,
) -> int:
    """
    INSERT … ON CONFLICT (source_path, chunk_index) DO UPDATE.

    Each item in *chunks* must have keys:
        text, embedding, source_path, chunk_index,
        source_type, product_type, effective_from, effective_to,
        version_tag, doc_type, metadata (dict)

    Returns the number of rows upserted.
    """
    if not chunks:
        return 0

    for chunk in chunks:
        embedding_val = chunk.get("embedding")
        if embedding_val is not None:
            # Format as pgvector literal: '[0.1,0.2,...]'
            embedding_str = "[" + ",".join(f"{v:.8f}" for v in embedding_val) + "]"
        else:
            embedding_str = None

        await db_session.execute(
            _UPSERT_SQL,
            {
                "source_path":   chunk["source_path"],
                "chunk_index":   chunk["chunk_index"],
                "content":       chunk["text"],
                "embedding":     embedding_str,
                "source_type":   chunk.get("source_type"),
                "product_type":  chunk.get("product_type"),
                "effective_from": chunk.get("effective_from"),
                "effective_to":   chunk.get("effective_to"),
                "version_tag":   chunk.get("version_tag"),
                "doc_type":      chunk.get("doc_type"),
                "metadata_json": json.dumps(chunk.get("metadata") or {}),
            },
        )

    await db_session.commit()
    return len(chunks)


# ===========================================================================
# 5. Main ingestion pipeline
# ===========================================================================

async def ingest_all(
    governance_root: Path,
    db_url: str,
    dry_run: bool = False,
    section: str | None = None,
    embedding_batch_size: int = 64,
) -> dict[str, Any]:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    governance_root:
        Path to ``docs/governance/`` from credit-risk-platform.
    db_url:
        asyncpg connection string, e.g. ``postgresql+asyncpg://...``.
    dry_run:
        If True, scan + chunk but skip embedding calls and DB writes.
    section:
        Restrict ingestion to a single subdirectory name under *governance_root*
        (e.g. ``"policy_manuals"``).  None means ingest all sections.
    embedding_batch_size:
        Number of chunks per OpenAI embedding call.

    Returns
    -------
    dict with keys: docs_found, chunks_created, rows_upserted, errors
    """
    governance_root = Path(governance_root).resolve()

    # ---- scan ---------------------------------------------------------------
    all_paths = scan_governance_docs(governance_root)
    if section:
        all_paths = [p for p in all_paths if p.parent.name == section]
        log.info("Section filter '%s' → %d documents", section, len(all_paths))

    summary: dict[str, Any] = {
        "docs_found": len(all_paths),
        "chunks_created": 0,
        "rows_upserted": 0,
        "errors": [],
    }

    if not all_paths:
        log.warning("No documents found — check governance_root: %s", governance_root)
        return summary

    # ---- DB setup -----------------------------------------------------------
    engine = None
    SessionFactory: async_sessionmaker | None = None
    if not dry_run:
        engine = create_async_engine(db_url, echo=False)
        SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
        # Bootstrap pgvector extension + table
        async with engine.begin() as conn:
            from app.db.session import _INIT_SQL  # noqa: PLC0415
            for _stmt in _INIT_SQL.split(";"):
                _stmt = _stmt.strip()
                if _stmt:
                    await conn.exec_driver_sql(_stmt)

    # ---- process each document ----------------------------------------------
    all_chunk_records: list[dict[str, Any]] = []

    for doc_path in all_paths:
        try:
            fm = extract_front_matter(doc_path)
            body = _body_text(doc_path)
            text_chunks = chunk_document(body)

            effective_from = _coerce_date(
                fm.get("effective_from") or fm.get("report_year") and f"{fm['report_year']}-01-01"
            )
            effective_to   = _infer_effective_to(fm)
            doc_type       = fm.get("doc_type")
            product_type   = fm.get("product_type")
            version_tag    = fm.get("version_tag")

            # Use relative path as the stable source_path key
            try:
                rel_path = str(doc_path.relative_to(governance_root))
            except ValueError:
                rel_path = str(doc_path)

            for idx, chunk_text in enumerate(text_chunks):
                all_chunk_records.append({
                    "source_path":   rel_path,
                    "chunk_index":   idx,
                    "text":          chunk_text,
                    "embedding":     None,  # filled below
                    "source_type":   doc_type or doc_path.parent.name,
                    "product_type":  product_type,
                    "effective_from": effective_from,
                    "effective_to":   effective_to,
                    "version_tag":   version_tag,
                    "doc_type":      doc_type,
                    "metadata":      {
                        k: str(v) for k, v in fm.items()
                        if k not in ("effective_from", "effective_to", "doc_type",
                                     "product_type", "version_tag")
                    },
                })

            summary["chunks_created"] += len(text_chunks)
            log.debug("%-60s  %2d chunks", rel_path, len(text_chunks))

        except Exception as exc:
            msg = f"{doc_path}: {exc}"
            log.error("Failed to process %s: %s", doc_path, exc)
            summary["errors"].append(msg)

    if dry_run:
        log.info(
            "DRY RUN complete — %d docs, %d chunks (no embeddings or DB writes)",
            summary["docs_found"], summary["chunks_created"],
        )
        return summary

    # ---- embed --------------------------------------------------------------
    log.info("Embedding %d chunks in batches of %d …", len(all_chunk_records), embedding_batch_size)
    texts = [r["text"] for r in all_chunk_records]
    try:
        vectors = await batch_embed(texts, batch_size=embedding_batch_size)
        for record, vec in zip(all_chunk_records, vectors):
            record["embedding"] = vec
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        summary["errors"].append(f"Embedding: {exc}")
        return summary

    # ---- upsert (batched by document for memory efficiency) -----------------
    log.info("Upserting %d chunk rows to policy_docs …", len(all_chunk_records))
    async with SessionFactory() as session:  # type: ignore[union-attr]
        rows = await upsert_chunks(all_chunk_records, session)
        summary["rows_upserted"] = rows

    log.info(
        "Ingestion complete — docs=%d  chunks=%d  upserted=%d  errors=%d",
        summary["docs_found"], summary["chunks_created"],
        summary["rows_upserted"], len(summary["errors"]),
    )
    return summary


# ===========================================================================
# Verification helper  (PRD §8 check 7)
# ===========================================================================

async def verify_ingestion(db_url: str) -> dict[str, Any]:
    """
    Query policy_docs and assert minimum ingestion quality.

    Returns a dict with keys: total_rows, distinct_product_types,
    distinct_doc_types, policy_manual_rows_with_product_type,
    rows_with_effective_from, passed (bool), failures (list[str]).
    """
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        row = (await conn.execute(text(
            "SELECT COUNT(*), "
            "COUNT(DISTINCT product_type), "
            "COUNT(DISTINCT doc_type), "
            "SUM(CASE WHEN doc_type='policy_manual' AND product_type IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN effective_from IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM policy_docs"
        ))).one()

    total, dp_types, d_types, pm_with_pt, rows_with_ef = row

    failures: list[str] = []
    if (total or 0) < 9:
        failures.append(f"total_rows={total} < 9 (expected ≥9)")
    if (pm_with_pt or 0) == 0:
        failures.append("policy_manual rows missing product_type")
    if (rows_with_ef or 0) == 0:
        failures.append("no rows have effective_from set")

    result = {
        "total_rows":                       total,
        "distinct_product_types":           dp_types,
        "distinct_doc_types":               d_types,
        "policy_manual_rows_with_product_type": pm_with_pt,
        "rows_with_effective_from":         rows_with_ef,
        "passed":                           len(failures) == 0,
        "failures":                         failures,
    }
    return result


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6 — LucidCredit governance doc ingestor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--governance-root",
        default=str(_DEFAULT_GOVERNANCE_ROOT),
        help="Path to docs/governance/ from credit-risk-platform",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help="asyncpg DB URL (defaults to DATABASE_URL env var / .env)",
    )
    p.add_argument(
        "--section",
        choices=["policy_manuals", "committee_minutes", "model_validation", "fair_lending"],
        default=None,
        help="Ingest only this section (default: all)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and chunk but skip embedding + DB writes",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After ingestion, run PRD §8 check 7 verification query",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per OpenAI embedding batch",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    db_url = args.db_url or get_settings().database_url

    summary = await ingest_all(
        governance_root   = Path(args.governance_root),
        db_url            = db_url,
        dry_run           = args.dry_run,
        section           = args.section,
        embedding_batch_size = args.batch_size,
    )

    print("\n" + "=" * 60)
    print("PHASE 6 INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Governance root : {args.governance_root}")
    print(f"  Dry run         : {args.dry_run}")
    print(f"  Docs found      : {summary['docs_found']}")
    print(f"  Chunks created  : {summary['chunks_created']}")
    print(f"  Rows upserted   : {summary['rows_upserted']}")
    print(f"  Errors          : {len(summary['errors'])}")
    if summary["errors"]:
        for err in summary["errors"]:
            print(f"    ✗ {err}")
    print("=" * 60)

    if args.verify and not args.dry_run:
        print("\nRunning PRD §8 check 7 verification …")
        v = await verify_ingestion(db_url)
        print(f"  total_rows               : {v['total_rows']}")
        print(f"  distinct_product_types   : {v['distinct_product_types']}")
        print(f"  distinct_doc_types       : {v['distinct_doc_types']}")
        print(f"  pm_rows_with_product_type: {v['policy_manual_rows_with_product_type']}")
        print(f"  rows_with_effective_from : {v['rows_with_effective_from']}")
        status = "PASS" if v["passed"] else "FAIL"
        print(f"  Status: {status}")
        if v["failures"]:
            for f in v["failures"]:
                print(f"    ✗ {f}")
        return 0 if v["passed"] else 1

    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
