"""
Queue the b12 experiment batch — v13.0 time-scaling consistency check.

Background: b11 proved v13.0 can reach 0.719 rate in 60min vs the PFSP
champion pool (initial opponent 6bcccd34). Only 1 of 5 b11 runs completed
(s1, 145 updates). b12 promotes the b11 winner (1e385310) as the new
opponent and re-runs the same time-scaling layout to answer:

  1. Can multiple seeds consistently beat the 0.719-rated champion?
  2. Does 90min training push the ceiling beyond 0.719?

Layout (5.5h total, 5 runs):
  4 × 60m  karp baseline cfg, seeds {1,2,3,4}  — consistency
  1 × 90m  karp baseline cfg, seed 7            — ceiling probe

All runs use PFSP-rotated opponents with initial opponent = b11-s1 winner
(1e385310, rate 0.719, 145 updates vs neural).

v13 baseline cfg (from karpathy_loop.yaml):
  lr=3e-3, entropy=0.01, gamma=0.99, clip=0.2, K=2 (action_repeat),
  n_envs=1800, rollout=8, update_epochs=4, minibatch_size=512,
  gae_lambda=0.95, max_grad_norm=0.5, reward_version=5 (v1.7 pure terminal),
  level phase1_full_mix_4_8 (50/50 random_close_4_8 + random_4_8).

Usage:
  python scripts/queue_b12.py --dry-run     # preview, no inserts
  python scripts/queue_b12.py               # actually queue
  python scripts/queue_b12.py --opponent <run-id>   # override initial opponent
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
    "n_envs": 1800,
    "rollout_steps": 8,
    "fused_rollout": True,
    "action_repeat": 2,
    "vec_mode": "sync",
    "sim_backend": "jax",
    "level_name": "phase1_full_mix_4_8",
    "level_mix": {"random_close_4_8": 1.0, "random_4_8": 1.0},
    "update_epochs": 4,
    "minibatch_size": 512,
    "lr": 3e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "reward_v13": True,
    "reward_version": 5,
    "normalize_obs": True,
    "obs_clip": 10.0,
    "replay_per_update": True,
    "replay_games_per_update": 2,
    "self_play": False,
    "snapshot_every": 10,
    "pool_max_size": 20,
    "latest_bias": 0.8,
    "initial_opponent": "random_legal",
    "opponent_pool_mode": "rotate_per_update",
    "leaderboard_bias": 1.0,
    "leaderboard_recency_decay": 0.5,
    "leaderboard_source": "pfsp",
    "leaderboard_top_k": 10,
}
MODEL_ID = "v13.0"
SIM_ID   = "sim-v1.4"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# b11 winner: rate 0.719, 145 updates, 60min vs PFSP champions
LATEST_CHAMPION_RUN = "1e385310-2b41-477e-bc94-157788c696c2"


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # 4 × 60-min consistency seeds
    for seed in [1, 2, 3, 4]:
        runs.append({
            "label":   f"b12-{epoch}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cpu"},
            "notes":   "b12 v13.0 time-scaling: 60min consistency vs b11 winner (0.719)",
        })

    # 1 × 90-min ceiling probe
    runs.append({
        "label":   f"b12-{epoch}-ceiling90-s7",
        "minutes": 90,
        "seed":    "7",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cpu"},
        "notes":   "b12 v13.0 time-scaling: 90min ceiling probe vs b11 winner (0.719)",
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
                    help="UUID of initial opponent run; defaults to b11 winner (1e385310)")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            opp_id = LATEST_CHAMPION_RUN
            print(f"using b11 winner: {opp_id[:8]} (rate 0.719, 145 updates)")

        runs = _build_batch(opp_id)
        total_min = sum(r["minutes"] for r in runs)
        print(f"b12 batch: {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

        ids = _push(conn, runs, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<40} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
