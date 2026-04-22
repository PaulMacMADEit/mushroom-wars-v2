"""Queue new runs in Supabase.

One call can queue multiple seeds against the same (model, sim, config).
Each queued row is a status='queued' runs row that any worker can claim.

Usage:
    python cli/push_experiments.py --model v9.0-smoke --sim sim-v1.0 \\
        --budget 120 --label smoke-120s --seeds a,b,c \\
        --config '{"lr":0.0003,"entropy_coef":0.01}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_id (e.g. v9.0-smoke)")
    ap.add_argument("--sim", required=True, help="simulator_id (e.g. sim-v1.0)")
    ap.add_argument("--label", required=True, help="short human tag")
    ap.add_argument("--budget", type=int, required=True, help="wall-clock budget (seconds)")
    ap.add_argument("--seeds", default="a", help="comma-separated seeds (e.g. a,b,c)")
    ap.add_argument("--config", default="{}", help="JSON hyperparams dict")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    hyperparams = json.loads(args.config)
    budget_ms = args.budget * 1000
    launch_at = int(time.time() * 1000)

    rows = []
    for seed in seeds:
        rows.append((
            args.model,
            args.sim,
            PROJECT,
            args.label,
            args.description,
            "queued",
            budget_ms,
            seed,
            json.dumps(hyperparams),
            "unassigned",   # machine filled on claim
            launch_at,
        ))

    with connect() as conn:
        with conn.cursor() as cur:
            inserted_ids = []
            for row in rows:
                cur.execute("""
                    INSERT INTO runs (model_id, simulator_id, project, label, description,
                                      status, budget_ms, seed, hyperparams, machine, launch_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING id
                """, row)
                inserted_ids.append(cur.fetchone()[0])
        conn.commit()

    print(f"queued {len(inserted_ids)} runs under label {args.label!r} "
          f"(model={args.model}, sim={args.sim}, budget={args.budget}s):")
    for seed, run_id in zip(seeds, inserted_ids):
        print(f"  seed={seed}  id={run_id}")


if __name__ == "__main__":
    main()
