"""
Queue the b4 experiment batch — 60-min consistency sweep + 90-min ceiling probe.

Background: b3 (2026-04-25) confirmed cfg variants don't separate at 30 min vs
the strong neural opponent — all 10 runs clustered at coin-flip parity (mean
0.498, range 4.7pp). Only the 60-min endurance broke through (rate 0.672 @ 20
updates). **Time, not config, is the lever.**

b4 asks: (1) is 0.672 in 60 min reproducible across seeds against the b3
endurance checkpoint? (2) does 90 min lift the ceiling notably?

Layout (5.5h total, 5 runs):
  4 × 60m  default K=4 cfg, seeds {1,2,3,4}  — consistency probe
  1 × 90m  default K=4 cfg, seed=42           — ceiling probe

All vs `0385b326-dd1c-4397-a5b8-e941215f67f5` (b3-endurance, 0.672 @ 20 updates,
the strongest neural-trained checkpoint we have).

Default cfg = lr=3e-4 entropy=0.01 gamma=0.99 clip=0.2. Level random_8_16. No
cfg variants — b3 ruled them out at this run length.

Usage:
  python scripts/queue_b4.py --dry-run     # decide and log, no inserts
  python scripts/queue_b4.py               # actually queue
  python scripts/queue_b4.py --opponent <run-id>   # override opponent

Note: this script is auto-run by PaulLinux's mushroom-pull-batch.timer at
11:30 Pacific each day if a new queue_b{N}.py exists in main. The state
file ~/.local/state/mushroom-wars/last_run_batch tracks the last-run N.
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


B3_ENDURANCE_OPPONENT = "0385b326-dd1c-4397-a5b8-e941215f67f5"  # b3-endurance, rate 0.672 @ 20 updates


def _build_batch(opponent_run_id: str) -> list[dict]:
    """Layout per the module docstring."""
    epoch = _utc_now().strftime("%y%m%d-%H%M")
    runs = []

    # Consistency probe: 4 seeds × 60 min default cfg.
    for seed in [1, 2, 3, 4]:
        runs.append({
            "label":   f"b4-{epoch}-default60-s{seed}",
            "minutes": 60,
            "seed":    str(seed),
            "config":  dict(DEFAULT_CFG),
            "opponent_name":   "neural",
            "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
            "notes":   "b4 default K=4 60min vs b3-endurance; consistency probe — is 0.67 reproducible?",
        })

    # Ceiling probe: 1 × 90 min default cfg.
    runs.append({
        "label":   f"b4-{epoch}-ceiling90",
        "minutes": 90,
        "seed":    "42",
        "config":  dict(DEFAULT_CFG),
        "opponent_name":   "neural",
        "opponent_kwargs": {"opponent_run_id": opponent_run_id, "device": "cuda"},
        "notes":   "b4 ceiling probe — does 90 min lift win-rate notably above 60-min 0.67?",
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
                    help="UUID of opponent run; defaults to b3-endurance (the strongest neural-trained checkpoint)")
    ap.add_argument("--no-cancel-backlog", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        if args.opponent:
            opp_id = args.opponent
            print(f"using opponent run id (override): {opp_id}")
        else:
            # Hardcoded to b3-endurance. _strongest_done_run would pick
            # da2205e1 (67 updates @ 0.951) which trained vs random_legal
            # only — wrong choice for measuring vs a real opponent.
            opp_id = B3_ENDURANCE_OPPONENT
            print(f"using b3-endurance opponent: {opp_id[:8]} (rate 0.672 @ 20 updates, neural-trained)")

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
