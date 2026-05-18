"""
scripts/populate_rag_ground_truth.py
=====================================
Run against a live LucidCredit backend to populate rag_ground_truth.json
with real chunk IDs from the vector retriever.

Replaces all PLACEHOLDER chunk IDs with the actual source_refs returned by
the live system for each eval query, enabling Eval 5c (IR Precision) to score
properly.

Usage:
    cd "/Users/swarnabale/Documents/My Projects/LucidCredit"
    python3 scripts/populate_rag_ground_truth.py [--backend-url http://localhost:8090]

Requirements:
    - LucidCredit backend running on BACKEND_URL (default: http://localhost:8090)
    - httpx installed (ships with the backend requirements)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "agent_evals" / "fixtures" / "rag_ground_truth.json"
DEFAULT_BACKEND_URL = "http://localhost:8090"


async def populate(backend_url: str, dry_run: bool) -> None:
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)

    if not FIXTURE_PATH.exists():
        print(f"ERROR: Fixture not found: {FIXTURE_PATH}", file=sys.stderr)
        sys.exit(1)

    fixture: list[dict] = json.loads(FIXTURE_PATH.read_text())
    print(f"Loaded {len(fixture)} entries from {FIXTURE_PATH}")
    print(f"Backend URL: {backend_url}")
    if dry_run:
        print("[DRY RUN] No writes will be performed.")
    print()

    updated = 0
    skipped = 0

    async with httpx.AsyncClient(base_url=backend_url, timeout=30.0) as client:
        for entry in fixture:
            eval_id = entry.get("eval_id", "?")
            query = entry.get("query", "")
            if not query:
                print(f"WARN [{eval_id}] no query field — skipping")
                skipped += 1
                continue

            try:
                resp = await client.post(
                    "/v1/query/analyst",
                    json={
                        "query": query,
                        "audience": "analyst",
                        "intent": "analyst_query",
                    },
                )
            except httpx.ConnectError:
                print(f"ERROR [{eval_id}] Cannot connect to {backend_url} — is the backend running?")
                sys.exit(1)

            if resp.status_code != 200:
                print(f"ERR  [{eval_id}] HTTP {resp.status_code} — {resp.text[:120]}")
                skipped += 1
                continue

            data = resp.json()
            retrieved = (
                data.get("reasoning_trace", {}).get("retrieved_context", [])
            )
            chunk_ids = [
                r.get("source_ref", "UNKNOWN")
                for r in retrieved
                if r.get("relevance") in ("RELEVANT", "AMBIGUOUS")
            ]

            if chunk_ids:
                entry["expected_chunk_ids"] = chunk_ids
                updated += 1
                print(f"OK   [{eval_id}] {len(chunk_ids)} chunk(s): {chunk_ids[:2]}")
            else:
                print(f"WARN [{eval_id}] no RELEVANT/AMBIGUOUS chunks returned — query: {query[:60]}")
                skipped += 1

    print()
    print(f"Results: {updated} updated, {skipped} skipped out of {len(fixture)} total")

    if dry_run:
        print("[DRY RUN] Skipping write.")
        return

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {len(fixture)} entries to {FIXTURE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate rag_ground_truth.json with real chunk IDs")
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help=f"LucidCredit backend base URL (default: {DEFAULT_BACKEND_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch real chunk IDs but do not write to the fixture file",
    )
    args = parser.parse_args()
    asyncio.run(populate(args.backend_url, args.dry_run))


if __name__ == "__main__":
    main()
