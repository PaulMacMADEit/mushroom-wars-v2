"""Add `elo_status` column to runs.

Values:
  - 'unrated'         : default for new rows / runs that haven't been auto-rated yet
  - 'rated'           : passed the auto-rate gate, has a meaningful elo_score
  - 'did_not_perform' : failed the gate (couldn't beat random_legal at >=70% over 30 games)

Idempotent — uses ADD COLUMN IF NOT EXISTS.

Usage:
    python cli/migrate_elo_status.py
    python cli/migrate_elo_status.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sql_add_col = (
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS elo_status TEXT NOT NULL DEFAULT 'unrated'"
    )
    sql_backfill = (
        "UPDATE runs SET elo_status = 'rated' "
        "WHERE elo_status = 'unrated' AND elo_n_matches >= 1"
    )

    with connect() as conn:
        with conn.cursor() as cur:
            if args.dry_run:
                print(f"[migrate] [dry-run] would: {sql_add_col}")
                print(f"[migrate] [dry-run] would: {sql_backfill}")
                return

            cur.execute(sql_add_col)
            print("[migrate] ensured column runs.elo_status (default 'unrated')")

            cur.execute(sql_backfill)
            print(f"[migrate] backfilled {cur.rowcount} pre-existing rated runs to 'rated'")

            cur.execute("SELECT elo_status, COUNT(*) FROM runs GROUP BY elo_status ORDER BY 1")
            for status, n in cur.fetchall():
                print(f"  {status}: {n}")
        conn.commit()

    print("[migrate] done")


if __name__ == "__main__":
    main()
