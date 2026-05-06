"""
Queue the b10 experiment batch — v10.1 time-scaling continuation.

Background: b8+b9 queued 10 runs (8×60m + 2×90m) vs the self-play champion
(b7c5b2d4, rate 0.822). Queue drained with 0 results in last-24h summary
(2026-05-06) — no new opponent to promote. b10 continues the same design
with fresh seeds to accumulate statistical power.

b10 asks: same as b8/b9 — does 60-90 min training under v10.1 default cfg
reliably beat the strongest self-play chain agent?

Layout (5.5h total, 5 runs):
  4 × 60m  default v10.1 cfg, seeds {10,11,14,17}  — fresh consistency seeds
  1 × 90m  default v10.1 cfg, seed 21               — ceiling probe

All vs `b7c5b2d4` (v10.3.02-SelfPlay-Base-06, rate 0.822 vs neural — strongest
neural-trained agent from current self-play chain).

v10.1 baseline cfg (from karpathy_loop.yaml as of 2026-04-30):
  lr=1e-3, entropy=0.01, gamma=0.97, clip=0.2, K=2, n_envs=1024, rollout=8,
  gae_lambda=0.95, update_epochs=4, minibatch_size=512, max_grad_norm=0.5,
  reward_version=2 (v1.4), level random_close_4_5.

Usage:
  python scripts/queue_b10.py --dry-run     # preview, no inserts
  python scripts/queue_b10.py               # actually queue
  python scripts/queue_b10.py --opponent <run-id>   # override opponent
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
    "rollout_steps": 8,
    "fused_rollout": True,
    "action_repeat": 2,
    "vec_mode": "sync",
    "sim_backend": "jax",
    "level_name": "random_close_4_5",
    "update_epochs": 4,
    "minibatch_size": 512,
    "lr": 1e-3,
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "reward_v13": True,
    "reward_version": 2,
    "normalize_obs": True,
    "obs_clip": 10.0,
}
MODEL_ID = "v10.1"
SIM_ID   = "sim-v1.3"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


SELFPLAY_CHAMPION = "b7c5b2d4-08bf-49b5-a1bf-ad66a6a7f983"


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # 4 × 60-min consistency seeds (fresh seeds, no overlap with b8/b9)
    for seed in [10, 11, 14, 17]:
        runs.append({
            "label":   f"b10-{epoch}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b10 v10.1 time-scaling: 60min default cfg vs self-play champion (Base-06, 0.822)",
        })

    # 1 × 90-min ceiling probe
    runs.append({
        "label":   f"b10-{epoch}-ceiling90-s21",
        "minutes": 90,
        "seed":    "21",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
        "notes":   "b10 v10.1 time-scaling: 90min ceiling probe vs self-play champion (Base-06, 0.822)",
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
                    help="UUID of opponent run; defaults to self-play champion (Base-06, rate 0.822)")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            opp_id = SELFPLAY_CHAMPION
            print(f"using self-play champion: {opp_id[:8]} (v10.3.02-SelfPlay-Base-06, rate 0.822)")

        runs = _build_batch(opp_id)
        total_min = sum(r["minutes"] for r in runs)
        print(f"b10 batch: {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

        ids = _push(conn, runs, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<40} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
