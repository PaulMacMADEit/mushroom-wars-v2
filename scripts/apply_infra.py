"""Apply all infra/*.sql files + cli/migrate_*.py migrations in dependency order.

Idempotent — every file uses CREATE ... IF NOT EXISTS / DROP ... IF EXISTS.
Used for bringing up a fresh Supabase project end-to-end.

Run:
    .venv/bin/python scripts/apply_infra.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg
from cli.db import db_url


SQL_ORDER = [
    "schema.sql",
    "worker_state.sql",
    "jobs.sql",
    "attribution_jobs.sql",
    "run_feature_importance.sql",
    "rpc.sql",
    "rpc_interactive_play.sql",
    "rls.sql",
]

# Order matters: migrate_v13 creates kv + elo_n_matches that the others need.
MIGRATIONS = [
    "cli/migrate_v13.py",
    "cli/migrate_champion_archive.py",
    "cli/migrate_elo_anchor_1000.py",
    "cli/migrate_elo_status.py",
    "cli/migrate_play_rpc.py",
]


def main() -> None:
    infra = ROOT / "infra"
    with psycopg.connect(db_url(), autocommit=True) as conn:
        conn.prepare_threshold = None
        for name in SQL_ORDER:
            path = infra / name
            if not path.exists():
                print(f"  SKIP {name} (missing)")
                continue
            sql = path.read_text()
            print(f"  APPLY {name} ({len(sql):,} chars)...", end=" ", flush=True)
            with conn.cursor() as cur:
                cur.execute(sql)
            print("✓")

    print()
    for script in MIGRATIONS:
        print(f"  RUN   {script}...", end=" ", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / script)],
                           capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            print("✗")
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            sys.exit(1)
        print("✓")

    print("\nAll infra + migrations applied.")


if __name__ == "__main__":
    main()
