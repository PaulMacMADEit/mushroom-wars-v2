"""
Queue the b10 experiment batch — v12 long-cell endurance test.

Background: the karpathy loop has been running 20-min cells in a bootstrap
chain (v12.0.21–v12.0.34, then v12.1.01–v12.1.02 continuation). The
strongest checkpoint is v12.0.31-Bootstrap-entropy_coef-mid (rate 0.926,
69 updates, vs neural 0a18601b). All prior b-series batches (b3–b9) used
v10.1 config — b10 is the first b-series batch on v12 config.

b10 asks: do 60–90 min training cells at v12 baseline cfg meaningfully
beat the 0.926 champion? The karp loop caps at 20-min cells (~35–70
updates); 60 min should yield ~200+ updates — testing whether the plateau
is time-bound or structural.

Layout (5.5h total, 5 runs):
  4 × 60m  v12 baseline cfg, seeds {1,2,3,42}   — consistency seeds
  1 × 90m  v12 baseline cfg, seed 7              — ceiling probe

All vs `79250233` (v12.0.31-Bootstrap-entropy_coef-mid, rate 0.926 vs
neural — strongest bootstrap chain agent).

v12 baseline cfg (from karpathy_loop.yaml as of 2026-05-03):
  lr=3e-3, entropy=0.01, gamma=0.99, clip=0.2, K=4, n_envs=1800,
  rollout=8, gae_lambda=0.95, update_epochs=4, minibatch_size=512,
  max_grad_norm=0.5, reward_version=5 (v1.7 pure terminal),
  level phase1_full_mix_4_8 (50/50 close+ranged 4-8 building maps),
  action_repeat=2, fused_rollout=true, sim_backend=jax.

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
}
MODEL_ID = "v12.0"
SIM_ID   = "sim-v1.4"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


BOOTSTRAP_CHAMPION = "79250233-8822-45f6-9eb2-52c1b4ef3993"


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # 4 × 60-min consistency seeds
    for seed in [1, 2, 3, 42]:
        runs.append({
            "label":   f"b10-{epoch}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b10 v12 long-cell: 60min default cfg vs bootstrap champion (entropy_coef-mid, 0.926)",
        })

    # 1 × 90-min ceiling probe
    runs.append({
        "label":   f"b10-{epoch}-ceiling90-s7",
        "minutes": 90,
        "seed":    "7",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
        "notes":   "b10 v12 long-cell: 90min ceiling probe vs bootstrap champion (entropy_coef-mid, 0.926)",
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
                    help="UUID of opponent run; defaults to bootstrap champion (entropy_coef-mid, rate 0.926)")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            opp_id = BOOTSTRAP_CHAMPION
            print(f"using bootstrap champion: {opp_id[:8]} (v12.0.31-Bootstrap-entropy_coef-mid, rate 0.926)")

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
