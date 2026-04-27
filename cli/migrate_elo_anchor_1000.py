"""Anchor Elo at 1000.

Migration:
  1. Subtract 200 from every existing elo_score (1200 → 1000 baseline).
  2. Change column default from 1200 to 1000.

Idempotent in the sense that re-running is safe — but the SHIFT is NOT
idempotent (running twice would push everything to 800). We track this
via the kv table:

    kv['elo_anchor_1000_applied'] = '1'

If this row exists we skip the shift. The column-default change is
always re-applied since ALTER ... SET DEFAULT is itself idempotent.

Usage:
    python cli/migrate_elo_anchor_1000.py
    python cli/migrate_elo_anchor_1000.py --dry-run
    python cli/migrate_elo_anchor_1000.py --force   # apply shift even if marker is set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import connect


MARKER_KEY = "elo_anchor_1000_applied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="apply the elo shift even if the kv marker is set")
    args = ap.parse_args()

    with connect() as conn:
        with conn.cursor() as cur:
            # Check if shift was applied already.
            cur.execute("SELECT value FROM kv WHERE key = %s", (MARKER_KEY,))
            marker = cur.fetchone()
            already = marker is not None and marker[0] == '1'

            if already and not args.force:
                print(f"[migrate] elo shift already applied (kv['{MARKER_KEY}']='1'); "
                      f"only re-applying column default")
            else:
                cur.execute("SELECT COUNT(*), MIN(elo_score), MAX(elo_score), AVG(elo_score) "
                            "FROM runs WHERE elo_score IS NOT NULL")
                n, lo, hi, avg = cur.fetchone()
                print(f"[migrate] before shift: {n} rows, "
                      f"min={lo and float(lo):.1f}, max={hi and float(hi):.1f}, "
                      f"avg={avg and float(avg):.1f}")
                if args.dry_run:
                    print("[migrate] [dry-run] would: UPDATE runs SET elo_score = elo_score - 200")
                else:
                    cur.execute("UPDATE runs SET elo_score = elo_score - 200 "
                                "WHERE elo_score IS NOT NULL")
                    print(f"[migrate] shifted {cur.rowcount} rows by -200")
                    cur.execute(
                        "INSERT INTO kv (key, value, updated_at) VALUES (%s, '1', NOW()) "
                        "ON CONFLICT (key) DO UPDATE SET value = '1', updated_at = NOW()",
                        (MARKER_KEY,),
                    )

            # Always re-set the column default — idempotent.
            if args.dry_run:
                print("[migrate] [dry-run] would: ALTER TABLE runs ALTER COLUMN elo_score SET DEFAULT 1000")
            else:
                cur.execute("ALTER TABLE runs ALTER COLUMN elo_score SET DEFAULT 1000")
                print("[migrate] column default set to 1000")

            cur.execute("SELECT COUNT(*), MIN(elo_score), MAX(elo_score), AVG(elo_score) "
                        "FROM runs WHERE elo_score IS NOT NULL")
            n, lo, hi, avg = cur.fetchone()
            print(f"[migrate] after:  {n} rows, "
                  f"min={lo and float(lo):.1f}, max={hi and float(hi):.1f}, "
                  f"avg={avg and float(avg):.1f}")

        if not args.dry_run:
            conn.commit()

    print("[migrate] done")


if __name__ == "__main__":
    main()
