"""
Queue the b6 experiment batch — retry b5's extended-time question (b5 was discarded).

Background: b4 (2026-04-26) showed 60-min default cfg vs 0385b326 (b3-endurance,
strongest neural-trained checkpoint) plateaus at ~0.345 (mean of 4 seeds,
range 0.326–0.357). b5 tried 90–120 min to break through but was discarded
before running. b6 retries with 4 × 90-min seeds for better statistical power.

b6 asks: does 90 min break through the 0.34 plateau vs 0385b326?

Layout (6h total, 4 runs):
  4 × 90m  default K=4 cfg, seeds {1,2,3,42}  — variance + breakthrough probe

All vs `0385b326-dd1c-4397-a5b8-e941215f67f5` (b3-endurance, rate 0.672 @ 20
updates — the strongest neural-trained checkpoint).

Default cfg = lr=3e-4 entropy=0.01 gamma=0.99 clip=0.2. Level random_8_16.
sim-v1.3 (upgraded from b5's sim-v1.2). No cfg variants.

Usage:
  python scripts/queue_b6.py --dry-run     # decide and log, no inserts
  python scripts/queue_b6.py               # actually queue
  python scripts/queue_b6.py --opponent <run-id>   # override opponent
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect


DEFAULT_CFG = {
    "n_envs": 1024,
    "rollout_steps": 64,
    "fused_rollout": True,
    "action_repeat": 4,
    "vec_mode": "sync",
    "sim_backend": "jax",
    "level_name": "random_8_16",
}
MODEL_ID = "v9.0-1024"
SIM_ID   = "sim-v1.3"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


B3_ENDURANCE_OPPONENT = "0385b326-dd1c-4397-a5b8-e941215f67f5"


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # Breakthrough probe: 4 seeds × 90 min default cfg.
    for seed in [1, 2, 3, 42]:
        runs.append({
            "label":   f"b6-{epoch}-default90-s{seed}",
            "minutes": 90,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b6 default K=4 90min vs b3-endurance; retry b5 (discarded) — does 90 min break through the 0.34 plateau?",
        })

    return runs


def _push(conn, runs: list[dict], dry_run: bool) -> list[str]:
    inserted = []
    launch_at = int(time.time() * 1000)
    with conn.cursor() as cur:
        for r in runs:
            hp = dict(r["config"])
            hp["opponent_name"]   = r["opponent_name"]
            hp["opponent_kwargs"] = r["opponent_kwargs"]
            row = (
                MODEL_ID, SIM_ID, PROJECT,
                r["label"], r.get("notes"),
                "queued", r["minutes"] * 60 * 1000,
                r["seed"], json.dumps(hp),
                "unassigned", launch_at,
            )
            if dry_run:
                print(f"  [dry-run] would queue: label={r['label']} budget={r['minutes']}m seed={r['seed']}")
                continue
            cur.execute(
                """
                INSERT INTO runs (model_id, simulator_id, project, label, description,
                                  status, budget_ms, seed, hyperparams, machine, launch_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                row,
            )
            inserted.append(str(cur.fetchone()[0]))
    if not dry_run:
        conn.commit()
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--opponent", default=None,
                    help="UUID of opponent run; defaults to b3-endurance (strongest neural-trained checkpoint)")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            opp_id = B3_ENDURANCE_OPPONENT
            print(f"using b3-endurance opponent: {opp_id[:8]} (rate 0.672 @ 20 updates, neural-trained)")

        runs = _build_batch(opp_id)
        total_min = sum(r["minutes"] for r in runs)
        print(f"b6 batch: {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

        ids = _push(conn, runs, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<40} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
