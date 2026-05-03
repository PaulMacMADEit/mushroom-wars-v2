"""
Queue a v12 long-cell endurance batch — 60–90 min cells, multiple seeds.

Renamed 2026-05-03 from queue_b10.py per Paul: cryptic `b10-...` label
format replaced with `v12.<step>-Endurance-default60-s<seed>` so the
dashboard reader can tell what's being tested without grepping scripts.
The b3–b9 series used legacy v10.1 config; everything from this script
onward is v12 + sim-v1.4 + continuation-only.

What this asks: do 60–90 min training cells meaningfully beat what the
karp loop's 20-min cells produce from the same parent? The karp loop
caps at 20-min cells (~35–70 PPO updates); 60 min should yield ~200+
updates — testing whether the plateau is time-bound or structural.

Layout (5.5h total, 5 runs):
  4 × 60m  default cfg, seeds {1,2,3,42}   — variance estimation
  1 × 90m  default cfg, seed 7              — ceiling probe

All cells warm-start (continuation) from the same parent run. Default
parent picker = strongest recent sim-v1.4 done run by training rate
(same logic as scripts/karp_backstop.py:_pick_continuation_parent).
Pass --from-run-id to override.

NEVER queues Bootstrap (Paul rule, 2026-05-02 23:40 PT). If no
compatible parent is found, exits non-zero.

Usage:
  python scripts/queue_v12_endurance.py --dry-run     # preview, no inserts
  python scripts/queue_v12_endurance.py               # auto-pick parent
  python scripts/queue_v12_endurance.py --from-run-id <run-id>
  python scripts/queue_v12_endurance.py --opponent <run-id>   # neural fixed-opp override
  python scripts/queue_v12_endurance.py --version-prefix v12.3 --major-change Endurance
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


def _pick_continuation_parent() -> tuple[str, str, float] | None:
    """Strongest recent sim-v1.4 done run by training rate (last 12, floor 0.70)."""
    with connect() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id, label, (result->>'rate')::numeric AS rate
              FROM runs
             WHERE project=%s
               AND simulator_id='sim-v1.4'
               AND status='done'
               AND result IS NOT NULL
               AND result->>'rate' IS NOT NULL
               AND (result->>'rate')::numeric >= 0.70
             ORDER BY finished_at DESC
             LIMIT 12
        """, (PROJECT,))
        rows = cur.fetchall()
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r[2]))
    return (str(best[0]), best[1], float(best[2]))


def _build_batch(version_prefix: str, major_change: str,
                 opponent_run_id: str) -> list[dict]:
    """Build the 5-run layout. Labels: {prefix}-{major_change}-default60-s{seed}."""
    runs = []
    for seed in [1, 2, 3, 42]:
        runs.append({
            "label":   f"{version_prefix}-{major_change}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   f"v12 endurance: 60min default cfg vs fixed neural opp {opponent_run_id[:8]}",
        })
    runs.append({
        "label":   f"{version_prefix}-{major_change}-ceiling90-s7",
        "minutes": 90,
        "seed":    "7",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
        "notes":   f"v12 endurance: 90min ceiling probe vs fixed neural opp {opponent_run_id[:8]}",
    })
    return runs


def _push(conn, runs: list[dict], parent_run_id: str, dry_run: bool) -> list[str]:
    inserted = []
    launch_at = int(time.time() * 1000)
    with conn.cursor() as cur:
        for r in runs:
            hp = dict(r["config"])
            hp["opponent_name"]   = r["opponent_name"]
            hp["opponent_kwargs"] = r["opponent_kwargs"]
            if dry_run:
                print(f"  [dry-run] would queue: label={r['label']} budget={r['minutes']}m "
                      f"seed={r['seed']} parent={parent_run_id[:8]}")
                continue
            cur.execute(
                """
                INSERT INTO runs (model_id, simulator_id, project, label, description,
                                  status, budget_ms, seed, hyperparams, machine, launch_at,
                                  parent_run_id, is_continuation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, true)
                RETURNING id
                """,
                (
                    MODEL_ID, SIM_ID, PROJECT,
                    r["label"], r.get("notes"),
                    "queued", r["minutes"] * 60 * 1000,
                    r["seed"], json.dumps(hp),
                    "unassigned", launch_at,
                    parent_run_id,
                ),
            )
            inserted.append(str(cur.fetchone()[0]))
    if not dry_run:
        conn.commit()
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-run-id", default=None,
                    help="UUID of parent run to warm-start from. Default: strongest "
                         "recent sim-v1.4 done run by training rate. Refuses to "
                         "queue Bootstrap.")
    ap.add_argument("--opponent", default=None,
                    help="UUID of fixed neural opponent. Default: same as --from-run-id "
                         "(self-play vs parent). For PFSP rotation, use the karp loop "
                         "(scripts/queue_karp_sweep.py) instead — this script is for "
                         "fixed-opp variance/endurance probes.")
    ap.add_argument("--version-prefix", default="v12.2",
                    help="Label prefix. Default v12.2 (next available step after v12.1 "
                         "Continue sweeps).")
    ap.add_argument("--major-change", default="Endurance",
                    help="Major-change descriptor in label. Default: Endurance.")
    args = ap.parse_args()

    # Pick parent
    if args.from_run_id:
        parent_id = args.from_run_id
        parent_label = "(override)"
        parent_rate = float("nan")
    else:
        picked = _pick_continuation_parent()
        if picked is None:
            print("[v12-endurance] no sim-v1.4 done run with rate>=0.70 — refusing to "
                  "queue Bootstrap. Train a base via the karp loop first.")
            sys.exit(2)
        parent_id, parent_label, parent_rate = picked
    print(f"[v12-endurance] continuation parent: {parent_label} ({parent_id[:8]}, "
          f"rate={parent_rate:.3f})" if parent_label != "(override)"
          else f"[v12-endurance] continuation parent (override): {parent_id[:8]}")

    # Default opponent = parent (self-play vs same agent). Override with --opponent.
    opp_id = args.opponent or parent_id
    if opp_id == parent_id:
        print(f"[v12-endurance] training opponent = parent (self-play vs {opp_id[:8]})")
    else:
        print(f"[v12-endurance] training opponent (override): {opp_id[:8]}")

    runs = _build_batch(args.version_prefix, args.major_change, opp_id)
    total_min = sum(r["minutes"] for r in runs)
    print(f"[v12-endurance] {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

    with connect() as conn:
        ids = _push(conn, runs, parent_id, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs (continuation, parent={parent_id[:8]})")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<48} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
