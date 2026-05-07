"""
Queue the b3 experiment batch — directed neural-opponent sweep.

Background: b1+b2 (2026-04-25) saturated random_legal — every K=4 cfg solves
it in ~22-24 updates. b3 asks: does default K=4 cfg still converge against a
*real* opponent? How does seed variance look? Do entropy / lr variants help?

The opponent is whichever recent done run has the highest update count and
win-rate >= 0.6 — same selection rule the cron agent uses. Pass --opponent
<run-id> to override.

Layout (~6h total, 11 runs):
  6 × 30m  default K=4 cfg, seeds {1,2,3,4,5,6}      — variance core
  2 × 30m  default + entropy_coef=0.005, seeds {1,2} — sharper-policy variant
  2 × 30m  default + lr=1e-4, seeds {1,2}            — slower-lr variant
  1 × 60m  default cfg, seed=42                       — endurance

All against the same neural opponent. Level random_8_16 (matches b2's primary
training level). Default cfg = lr=3e-4 entropy=0.01 gamma=0.99 clip=0.2.

Usage:
  python scripts/queue_b3.py --dry-run     # decide and log, no inserts
  python scripts/queue_b3.py               # actually queue
  python scripts/queue_b3.py --opponent <run-id>   # override opponent

The cron timer (`mushroom-cron.timer`) should be stopped before this runs;
otherwise the next cron fire will cancel any unstarted b3 runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
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
SIM_ID   = "sim-v1.2"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _strongest_done_run(conn) -> dict | None:
    since = _utc_now() - timedelta(hours=24)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, hyperparams::text, result::text, weights_url
              FROM runs
             WHERE project = %s
               AND status = 'done'
               AND launch_at >= %s
               AND weights_url IS NOT NULL
            """,
            (PROJECT, int(since.timestamp() * 1000)),
        )
        rows = cur.fetchall()
    candidates = []
    for row in rows:
        result = json.loads(row[3]) if row[3] else None
        if not result:
            continue
        rate = result.get("rate") or 0
        updates = result.get("updates") or 0
        if rate >= 0.6 and updates > 0:
            candidates.append({
                "id":       str(row[0]),
                "label":    row[1],
                "rate":     rate,
                "updates":  updates,
            })
    if not candidates:
        return None
    candidates.sort(key=lambda r: -r["updates"])
    return candidates[0]


def _cancel_queued_backlog(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE runs SET status='discarded', finished_at=NOW()
             WHERE status='queued' AND project=%s
            RETURNING id
            """,
            (PROJECT,),
        )
        ids = cur.fetchall()
    conn.commit()
    return len(ids)


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # Variance core: 6 seeds at default cfg vs neural opponent.
    for i, seed in enumerate([1, 2, 3, 4, 5, 6]):
        runs.append({
            "label":   f"b3-{epoch}-default-s{seed}",
            "minutes": 30,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b3 default K=4 vs neural; variance estimate",
        })

    # Sharper-policy variant: entropy=0.005 ×2 seeds.
    for seed in [1, 2]:
        cfg = dict(DEFAULT_CFG)
        cfg["entropy_coef"] = 0.005
        runs.append({
            "label":   f"b3-{epoch}-ent005-s{seed}",
            "minutes": 30,
            "seed":    str(seed),
            "config":  cfg,
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b3 entropy_coef=0.005 vs neural; does sharper policy beat a real opponent faster",
        })

    # Slower-lr variant: lr=1e-4 ×2 seeds.
    for seed in [1, 2]:
        cfg = dict(DEFAULT_CFG)
        cfg["lr"] = 1e-4
        runs.append({
            "label":   f"b3-{epoch}-lr1e4-s{seed}",
            "minutes": 30,
            "seed":    str(seed),
            "config":  cfg,
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b3 lr=1e-4 vs neural; b2 said 1e-4 hurts vs random_legal — does same hold vs real opp?",
        })

    # Endurance: 60m at default to see ceiling.
    runs.append({
        "label":   f"b3-{epoch}-endurance",
        "minutes": 60,
        "seed":    "42",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
        "notes":   "b3 endurance — ceiling probe at default cfg",
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
                    help="UUID of opponent run; defaults to strongest recent")
    ap.add_argument("--no-cancel-backlog", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            strongest = _strongest_done_run(conn)
            if not strongest:
                print("ERROR: no done run found in last 24h with rate>=0.6 + weights_url")
                sys.exit(2)
            opp_id = strongest["id"]
            print(f"strongest opponent: {strongest['id'][:8]} "
                  f"(updates={strongest['updates']}, rate={strongest['rate']:.3f})")

        if not args.no_cancel_backlog and not args.dry_run:
            n = _cancel_queued_backlog(conn)
            print(f"discarded {n} previously queued runs")

        runs = _build_batch(opp_id)
        total_min = sum(r["minutes"] for r in runs)
        print(f"b3 batch: {len(runs)} runs, total budget = {total_min} min ({total_min/60:.1f}h)")

        ids = _push(conn, runs, args.dry_run)
    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"queued {len(ids)} new runs")
        for rid, r in zip(ids, runs):
            print(f"  {rid[:8]}  {r['label']:<40} budget={r['minutes']}m")


if __name__ == "__main__":
    main()
