"""Add champion archive table and runs.bench_vector column.

champions table: frozen past model checkpoints used as permanent evaluation
benchmarks. Entries are never deleted within a run — they're the reference pool.

runs.bench_vector: JSON object keyed by champion id -> win_rate (0.0-1.0)
against that champion. Written by bench_eval after each evaluation sweep.
Archive cap is 20 rows; oldest within the same arch_era are pruned first when
the cap is exceeded.

Usage:
    python cli/migrate_champion_archive.py          # apply
    python cli/migrate_champion_archive.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.db import connect


DDL = [
    # Champion archive — frozen past checkpoints
    """
    CREATE TABLE IF NOT EXISTS champions (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_run_id   UUID NOT NULL REFERENCES runs(id),
        archived_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        arch_era        TEXT NOT NULL DEFAULT 'unknown',
        weights_url     TEXT NOT NULL,
        obs_norm_url    TEXT,
        label           TEXT NOT NULL,
        notes           TEXT
    )
    """,

    # Index for era-based pruning (oldest-in-era eviction)
    "CREATE INDEX IF NOT EXISTS champions_era_archived ON champions(arch_era, archived_at)",

    # bench_vector: per-run win-rate vector keyed by champion.id
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS bench_vector JSONB",

    # PFSP sampling weight (harmonic mean of win-rate proximity to 0.5)
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS pfsp_weight FLOAT",

    # Mid-run weight snapshots (10-min timer during training)
    """
    CREATE TABLE IF NOT EXISTS run_snapshots (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id          UUID NOT NULL REFERENCES runs(id),
        snap_n          INT NOT NULL,
        snapped_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        weights_url     TEXT NOT NULL,
        obs_norm_url    TEXT,
        UNIQUE (run_id, snap_n)
    )
    """,

    "CREATE INDEX IF NOT EXISTS run_snapshots_run_id ON run_snapshots(run_id, snap_n)",
]


def apply(dry_run: bool = False) -> None:
    print(f"[migrate] {'DRY RUN — ' if dry_run else ''}applying champion archive migration")
    with connect() as c:
        with c.cursor() as cur:
            for stmt in DDL:
                stmt = stmt.strip()
                preview = stmt[:80].replace("\n", " ")
                print(f"  SQL: {preview}...")
                if not dry_run:
                    cur.execute(stmt)
        if not dry_run:
            c.commit()
            print("[migrate] committed.")
        else:
            print("[migrate] dry-run: rolled back.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
