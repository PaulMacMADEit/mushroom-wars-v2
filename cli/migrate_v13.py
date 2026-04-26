"""One-shot Supabase migration for the sim-v1.3 / curriculum schema.

Adds:
  - runs.elo_score float DEFAULT 1200
  - runs.elo_n_matches int DEFAULT 0
  - kv table (key text PK, value text, updated_at timestamptz default now())

Idempotent: ALTER TABLE ADD COLUMN IF NOT EXISTS, CREATE TABLE IF NOT EXISTS.
Safe to re-run.

Usage:
    python cli/migrate_v13.py
    python cli/migrate_v13.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import connect


MIGRATIONS = [
    # 1. Elo score columns on runs.
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS elo_score FLOAT DEFAULT 1200",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS elo_n_matches INT DEFAULT 0",
    # 2. kv table for global curriculum state (e.g. current phase).
    """
    CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 3. Index for fast Elo champion lookup (top-k by elo_score, scoped to a project).
    "CREATE INDEX IF NOT EXISTS idx_runs_elo_score ON runs (project, elo_score DESC) WHERE status='done'",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[migrate_v13] {'dry-run' if args.dry_run else 'applying'} migrations:")
    for sql in MIGRATIONS:
        compact = " ".join(sql.split())
        print(f"  - {compact[:120]}")

    if args.dry_run:
        print("[migrate_v13] dry-run; nothing applied")
        return

    with connect() as conn:
        with conn.cursor() as cur:
            for sql in MIGRATIONS:
                cur.execute(sql)
        conn.commit()

    print("[migrate_v13] done")


if __name__ == "__main__":
    main()
