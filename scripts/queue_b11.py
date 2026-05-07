"""
Queue the b11 experiment batch — v13.0 time-scaling debut.

Background: b5-b10 all failed or discarded — the b-batch system was stuck on
model_id v10.1 (obs=1008, actions=4097) while the codebase moved to v12→v13
(obs=192, actions=129). b11 is a fresh start on v13.0.

v13.0 karp sweep (fires 1-24) ran 10-min cells. Best done runs cluster at
0.95-0.96 rate vs PFSP-rotated champion pool. The open question is whether
60-90 min training continues to improve, as b3 showed for v10.1 (coin-flip
at 30min → 0.672 at 60min).

b11 asks: does training v13.0 for 60-90 min, with the current karp baseline
cfg and PFSP-rotated opponents, yield win-rates above the 10-min karp ceiling?

Layout (5.5h total, 5 runs):
  4 × 60m  karp baseline cfg, seeds {1,2,3,4}  — consistency
  1 × 90m  karp baseline cfg, seed 7            — ceiling probe

All runs start with initial opponent = latest v13 champion (6bcccd34,
v13.1.24-Continue-update_epochs-hi) and rotate opponents per PPO update
from the PFSP champion archive.

v13 baseline cfg (from karpathy_loop.yaml as of 2026-05-07):
  lr=3e-3, entropy=0.01, gamma=0.99, clip=0.2, K=2 (action_repeat),
  n_envs=1800, rollout=8, update_epochs=4, minibatch_size=512,
  gae_lambda=0.95, max_grad_norm=0.5, reward_version=5 (v1.7 pure terminal),
  level phase1_full_mix_4_8 (50/50 random_close_4_8 + random_4_8).

Usage:
  python scripts/queue_b11.py --dry-run     # preview, no inserts
  python scripts/queue_b11.py               # actually queue
  python scripts/queue_b11.py --opponent <run-id>   # override initial opponent
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


LATEST_CHAMPION_RUN = "6bcccd34-b352-4bda-935c-7a830f408f93"


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # 4 × 60-min consistency seeds
    for seed in [1, 2, 3, 4]:
        runs.append({
            "label":   f"b11-{epoch}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cpu"},
            "notes":   "b11 v13.0 time-scaling: 60min karp baseline vs PFSP-rotated champions",
        })

    # 1 × 90-min ceiling probe
    runs.append({
        "label":   f"b11-{epoch}-ceiling90-s7",
        "minutes": 90,
        "seed":    "7",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cpu"},
        "notes":   "b11 v13.0 time-scaling: 90min ceiling probe vs PFSP-rotated champions",
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
                    help="UUID of initial opponent run; defaults to latest v13 champion (6bcccd34)")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            opp_id = LATEST_CHAMPION_RUN
            print(f"using latest v13 champion: {opp_id[:8]} (v13.1.24-Continue-update_epochs-hi)")

        runs = _build_batch(opp_id)
        total_min = sum(r["minutes"] for r in runs)
        print(f"b11 batch: {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

        ids = _push(conn, runs, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<40} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
