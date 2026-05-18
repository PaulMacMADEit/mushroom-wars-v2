"""Apply all infra/*.sql files to the Supabase Postgres in dependency order.

Idempotent — every file uses CREATE ... IF NOT EXISTS / DROP ... IF EXISTS.
Used for bringing up a fresh Supabase project.

Run:
    .venv/bin/python scripts/apply_infra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg
from cli.db import db_url


ORDER = [
    "schema.sql",
    "worker_state.sql",
    "jobs.sql",
    "attribution_jobs.sql",
    "run_feature_importance.sql",
    "rpc.sql",
    "rpc_interactive_play.sql",
    "rls.sql",
]


def main() -> None:
    infra = ROOT / "infra"
    # autocommit=True because each SQL file has its own BEGIN/COMMIT
    with psycopg.connect(db_url(), autocommit=True) as conn:
        conn.prepare_threshold = None
        for name in ORDER:
            path = infra / name
            if not path.exists():
                print(f"  SKIP {name} (missing)")
                continue
            sql = path.read_text()
            print(f"  APPLY {name} ({len(sql):,} chars)...", end=" ", flush=True)
            with conn.cursor() as cur:
                cur.execute(sql)
            print("✓")
    print("\nAll infra applied.")


if __name__ == "__main__":
    main()
