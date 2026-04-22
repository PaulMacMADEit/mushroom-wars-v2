"""One-time: register the `random_legal` baseline as a pseudo-run.

This lets the match runner treat "vs random_legal" as a normal queued match
between two run IDs. The pseudo-row has no weights/optimizer/obs_norm
artifacts; `match_runner._load_agent` detects this and falls back to
`random_legal_opponent` for the side(s) that are the baseline.

Usage:
    python cli/register_baseline.py
    python cli/register_baseline.py --id random-legal-baseline --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


BASELINE_ID = "00000000-0000-0000-0000-000000000001"  # fixed UUID for "random_legal"
BASELINE_LABEL = "baseline-random-legal"
BASELINE_MODEL_ID = "baseline"  # placeholder model row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=BASELINE_ID)
    ap.add_argument("--label", default=BASELINE_LABEL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        with conn.cursor() as cur:
            # Ensure a `baseline` model row exists so the runs.model_id FK is satisfied.
            cur.execute("""
                INSERT INTO models (id, project, name, what_changed,
                                    obs_size, num_actions, layers)
                VALUES (%s, %s, 'Random-legal baseline',
                        'Synthetic model row so the random-legal opponent can be referenced as a run.',
                        0, 0, '{"kind":"random_legal"}'::jsonb)
                ON CONFLICT (id) DO NOTHING
            """, (BASELINE_MODEL_ID, PROJECT))

            # Ensure a `baseline-sim` simulator row exists — matches.simulator_id FK.
            cur.execute("""
                INSERT INTO simulators (id, project, name, what_changed)
                VALUES ('baseline-sim', %s, 'Baseline', 'Placeholder sim row for baseline opponent')
                ON CONFLICT (id) DO NOTHING
            """, (PROJECT,))

            # Now the pseudo-run itself.
            launch = int(time.time() * 1000)
            if args.force:
                cur.execute("DELETE FROM runs WHERE id = %s", (args.id,))
            cur.execute("""
                INSERT INTO runs (id, model_id, simulator_id, project, label, description,
                                  status, budget_ms, seed, hyperparams, machine, launch_at)
                VALUES (%s, %s, 'baseline-sim', %s, %s,
                        'Random-legal opponent. No weights; match_runner detects and uses random_legal_opponent.',
                        'done', 0, 'baseline', '{}'::jsonb, 'n/a', %s)
                ON CONFLICT (id) DO NOTHING
            """, (args.id, BASELINE_MODEL_ID, PROJECT, args.label, launch))
            affected = cur.rowcount
        conn.commit()

    if affected == 0 and not args.force:
        print(f"baseline run {args.id!r} already exists (use --force to re-create).")
    else:
        print(f"registered baseline pseudo-run {args.id!r} ({args.label!r})")


if __name__ == "__main__":
    main()
