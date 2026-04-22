"""Queue a continuation of an existing run.

ARCHITECTURE §10.4: any completed run can be extended. The worker
detects `parent_run_id` on the claimed row, downloads the parent's
weights / optimizer / obs_norm from Storage, initializes the trainer
from that state, and continues training for the new budget.

Usage:
    python cli/continue_training.py --parent <run_id> --budget 600 \\
        --label selfplay-10min-ext

Parent's hyperparams are inherited by default; override with --config.
Seed defaults to <parent_seed>-c1 / c2 / … so child runs are
distinguishable on the dashboard.
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
    ap.add_argument("--parent", required=True, help="parent run UUID")
    ap.add_argument("--budget", type=int, required=True, help="new wall-clock budget (seconds)")
    ap.add_argument("--label", required=True, help="short label for the continuation")
    ap.add_argument("--seed", default=None, help="override seed (default: <parent-seed>-c1)")
    ap.add_argument("--config", default=None, help="JSON hyperparams override (default: inherit parent's)")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_id, simulator_id, seed, hyperparams::text,
                       cumulative_budget_ms, budget_ms, status
                FROM runs WHERE id = %s
            """, (args.parent,))
            row = cur.fetchone()
            if not row:
                raise SystemExit(f"parent run {args.parent!r} not found")
            model_id, sim_id, parent_seed, parent_hp_text, parent_cum_ms, parent_budget_ms, parent_status = row

            if parent_status != "done":
                raise SystemExit(
                    f"parent status is {parent_status!r}; can only continue from 'done' runs"
                )

            # Inherit hyperparams unless overridden. Overlay any --config keys on top.
            hp = json.loads(parent_hp_text) if parent_hp_text else {}
            if args.config:
                hp.update(json.loads(args.config))

            seed = args.seed or f"{parent_seed or 'a'}-c1"
            budget_ms = args.budget * 1000
            cumulative = (parent_cum_ms or parent_budget_ms or 0) + budget_ms
            launch_at = int(time.time() * 1000)

            cur.execute("""
                INSERT INTO runs (
                    model_id, simulator_id, project, label, description,
                    status, budget_ms, seed, hyperparams, machine, launch_at,
                    parent_run_id, is_continuation, cumulative_budget_ms
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s::jsonb, 'unassigned', %s,
                        %s, true, %s)
                RETURNING id
            """, (
                model_id, sim_id, PROJECT, args.label, args.description,
                budget_ms, seed, json.dumps(hp), launch_at,
                args.parent, cumulative,
            ))
            new_id = cur.fetchone()[0]
        conn.commit()

    print(f"queued continuation {new_id}")
    print(f"  parent      = {args.parent}")
    print(f"  label       = {args.label}")
    print(f"  model       = {model_id}")
    print(f"  sim         = {sim_id}")
    print(f"  seed        = {seed}")
    print(f"  budget      = {args.budget}s")
    print(f"  cumulative  = {cumulative // 1000}s  (parent + this)")


if __name__ == "__main__":
    main()
