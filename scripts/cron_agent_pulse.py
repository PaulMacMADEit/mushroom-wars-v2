"""
Cron-fired training scheduler.

Runs on a 3-hour cadence (via systemd timer on PaulLinux). Each fire:

  1. Reads last batch's results from Supabase (`runs` rows from the last 6h).
  2. Cancels any queued backlog (status='queued') so we always queue the
     next batch fresh — runs that are 'running' are NOT touched.
  3. Decides next batch using a simple curriculum:
       - Always include 1-2 baseline (random_legal) probes for sanity.
       - Curriculum-graduate based on best leaderboard win-rate so far.
       - Throw in some self-play vs the strongest checkpoint we have.
  4. Pushes new runs into Supabase via INSERT.
  5. Logs a summary to stdout (captured by systemd journal).

The curriculum logic is hardcoded for now (no LLM). It can be swapped for
a Claude-driven decider later by replacing `_decide_next_batch`.

Run lengths target 10–60 min per Paul's preference (more experiments,
shorter runs). Total batch budget aims for ~6h of training (so even if
the cron skips a fire, we have a backlog).

Usage:
    python scripts/cron_agent_pulse.py            # full run (default)
    python scripts/cron_agent_pulse.py --dry-run  # decide but don't push
    python scripts/cron_agent_pulse.py --no-cancel-backlog
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

# Default training cfg. Anything not overridden per-run uses this.
DEFAULT_CFG = {
    "n_envs": 1024,
    "rollout_steps": 64,
    "fused_rollout": True,
    "action_repeat": 4,
    "vec_mode": "sync",
    "sim_backend": "jax",
}

# Models we're allowed to register runs against. Pick the latest known.
DEFAULT_MODEL_ID = "v9.0-1024"
DEFAULT_SIM_ID   = "sim-v1.2"

# Curriculum stages. Triggered by total wall-time accumulated against
# random_legal across all completed runs.
CURRICULUM = [
    # name, p(level), small/medium/large mix
    ("phase1_small",   {"random_4_8":  0.7, "random_6_10": 0.3}),
    ("phase2_mixed",   {"random_4_8":  0.4, "random_6_10": 0.4, "random_8_16": 0.2}),
    ("phase3_full",    {"random_4_8":  0.2, "random_6_10": 0.2, "random_8_16": 0.4, "random_16_24": 0.2}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_recent_runs(conn, since: datetime) -> list[dict]:
    """Return runs created since `since` with their status + result."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, label, hyperparams::text, result::text,
                   weights_url, obs_norm_url,
                   launch_at, started_at, finished_at
              FROM runs
             WHERE launch_at >= %s
               AND project = %s
             ORDER BY launch_at DESC
            """,
            (int(since.timestamp() * 1000), PROJECT),
        )
        rows = cur.fetchall()
    out = []
    for row in rows:
        out.append({
            "id":           str(row[0]),
            "status":       row[1],
            "label":        row[2],
            "hyperparams":  json.loads(row[3]) if row[3] else {},
            "result":       json.loads(row[4]) if row[4] else None,
            "weights_url":  row[5],
            "obs_norm_url": row[6],
            "launch_at":    row[7],
            "started_at":   row[8],
            "finished_at":  row[9],
        })
    return out


def _cancel_queued_backlog(conn, dry_run: bool) -> int:
    """Mark all currently queued runs as cancelled. Running runs are
    untouched. Returns the number of rows affected."""
    with conn.cursor() as cur:
        if dry_run:
            cur.execute(
                "SELECT COUNT(*) FROM runs WHERE status='queued' AND project=%s",
                (PROJECT,),
            )
            n = cur.fetchone()[0]
            print(f"[dry-run] would cancel {n} queued runs")
            return n
        cur.execute(
            """
            UPDATE runs
               SET status = 'cancelled',
                   finished_at = NOW()
             WHERE status = 'queued'
               AND project = %s
            RETURNING id
            """,
            (PROJECT,),
        )
        ids = cur.fetchall()
    conn.commit()
    print(f"  cancelled {len(ids)} queued runs from previous batch")
    return len(ids)


def _strongest_checkpoint(runs: list[dict]) -> dict | None:
    """Pick the run with the most updates and a non-trivial win rate. Used
    as the self-play opponent in the next batch."""
    candidates = [
        r for r in runs
        if r["status"] == "done"
        and r.get("result")
        and r.get("weights_url")
        and (r["result"].get("rate") or 0) >= 0.6
    ]
    if not candidates:
        return None
    # Most updates wins.
    candidates.sort(key=lambda r: -(r["result"].get("updates") or 0))
    return candidates[0]


def _summarise_recent_runs(runs: list[dict]) -> dict:
    done = [r for r in runs if r["status"] == "done"]
    failed = [r for r in runs if r["status"] == "failed"]
    running = [r for r in runs if r["status"] == "running"]
    queued = [r for r in runs if r["status"] == "queued"]

    win_rates = [
        (r["result"].get("rate") or 0)
        for r in done
        if r.get("result")
    ]
    return {
        "total":    len(runs),
        "done":     len(done),
        "failed":   len(failed),
        "running":  len(running),
        "queued":   len(queued),
        "max_win":  max(win_rates) if win_rates else None,
        "mean_win": (sum(win_rates) / len(win_rates)) if win_rates else None,
    }


def _decide_curriculum_phase(recent_runs: list[dict]) -> str:
    """Use the highest win rate seen against the *best* opponent so far to
    pick a curriculum phase. Phase 1 if we don't have strong policies yet."""
    done_with_neural_opp = [
        r for r in recent_runs
        if r["status"] == "done"
        and r["hyperparams"].get("opponent_name") == "neural"
        and r.get("result")
        and (r["result"].get("rate") or 0) >= 0.55
    ]
    if len(done_with_neural_opp) >= 3:
        return "phase3_full"
    done_with_random = [
        r for r in recent_runs
        if r["status"] == "done"
        and r["hyperparams"].get("opponent_name", "random_legal") == "random_legal"
        and r.get("result")
        and (r["result"].get("rate") or 0) >= 0.95
    ]
    if len(done_with_random) >= 3:
        return "phase2_mixed"
    return "phase1_small"


def _pick_level(phase: str, rng: random.Random) -> str:
    weights = dict(CURRICULUM)[phase]
    levels = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(levels, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Batch design
# ---------------------------------------------------------------------------

def _decide_next_batch(
    recent_runs: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Decide the next batch's specs. Returns a list of run dicts ready for
    INSERT. Targets ~6h total wall budget across short (10–60 min) runs."""
    phase = _decide_curriculum_phase(recent_runs)
    strongest = _strongest_checkpoint(recent_runs)
    print(f"  curriculum phase: {phase}")
    if strongest:
        sup = strongest["result"].get("updates", "?")
        print(f"  strongest checkpoint: {strongest['id'][:8]}  (updates={sup}, "
              f"rate={strongest['result'].get('rate'):.3f})")
    else:
        print("  no strong checkpoint yet — staying on random_legal opponent")

    runs = []
    epoch = _utc_now().strftime("%y%m%d-%H%M")

    # 6 short runs (15 min each = 90 min) at the current curriculum phase.
    for i in range(6):
        level = _pick_level(phase, rng)
        runs.append({
            "label": f"cron-{epoch}-{phase}-{i:02d}",
            "minutes": 15,
            "seed": str(rng.randint(0, 2**31 - 1)),
            "config": {
                **DEFAULT_CFG,
                "level_name": level,
            },
            "opponent_name": "random_legal",
            "notes": f"phase={phase} short probe vs random_legal",
        })

    # 4 medium runs (30 min each = 120 min) at the current phase.
    for i in range(4):
        level = _pick_level(phase, rng)
        runs.append({
            "label": f"cron-{epoch}-{phase}-med-{i:02d}",
            "minutes": 30,
            "seed": str(rng.randint(0, 2**31 - 1)),
            "config": {
                **DEFAULT_CFG,
                "level_name": level,
            },
            "opponent_name": "random_legal",
            "notes": f"phase={phase} medium probe vs random_legal",
        })

    # If we have a strong checkpoint, do 3 self-play runs (30 min each = 90 min).
    if strongest is not None:
        for i in range(3):
            level = _pick_level(phase, rng)
            runs.append({
                "label": f"cron-{epoch}-{phase}-selfplay-{i:02d}",
                "minutes": 30,
                "seed": str(rng.randint(0, 2**31 - 1)),
                "config": {
                    **DEFAULT_CFG,
                    "level_name": level,
                },
                "opponent_name": "neural",
                "opponent_kwargs": {
                    "opponent_run_id": strongest["id"],
                    "device": "cuda",
                },
                "notes": f"phase={phase} self-play vs strongest checkpoint",
            })

    # 1 long endurance run (60 min) at the most-mixed level we use this phase.
    longest_level = max(dict(CURRICULUM)[phase].items(), key=lambda kv: kv[1])[0]
    runs.append({
        "label": f"cron-{epoch}-{phase}-endurance",
        "minutes": 60,
        "seed": str(rng.randint(0, 2**31 - 1)),
        "config": {
            **DEFAULT_CFG,
            "level_name": longest_level,
        },
        "opponent_name": "random_legal",
        "notes": f"phase={phase} 1h endurance for reward-curve trajectory",
    })

    total_min = sum(r["minutes"] for r in runs)
    print(f"  decided {len(runs)} runs, total wall budget = {total_min} min "
          f"({total_min/60:.1f}h)")
    return runs


# ---------------------------------------------------------------------------
# Push runs to Supabase
# ---------------------------------------------------------------------------

def _push_runs(conn, runs: list[dict], model_id: str, sim_id: str, dry_run: bool) -> list[str]:
    """INSERT new runs as queued. Returns inserted ids."""
    inserted = []
    launch_at = int(time.time() * 1000)
    with conn.cursor() as cur:
        for r in runs:
            hp = dict(r["config"])
            if "opponent_name" in r:
                hp["opponent_name"] = r["opponent_name"]
            if "opponent_kwargs" in r:
                hp["opponent_kwargs"] = r["opponent_kwargs"]
            row = (
                model_id, sim_id, PROJECT,
                r["label"], r.get("notes"),
                "queued", r["minutes"] * 60 * 1000,
                r["seed"], json.dumps(hp),
                "unassigned", launch_at,
            )
            if dry_run:
                print(f"  [dry-run] would queue: label={r['label']} budget={r['minutes']}m")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log without writing to Supabase")
    ap.add_argument("--no-cancel-backlog", action="store_true",
                    help="don't cancel pre-existing queued runs (default: cancel)")
    ap.add_argument("--review-window-hours", type=int, default=6,
                    help="how far back to look at recent runs (default: 6h)")
    ap.add_argument("--model-id",     default=DEFAULT_MODEL_ID)
    ap.add_argument("--sim-id",       default=DEFAULT_SIM_ID)
    ap.add_argument("--seed",         type=int, default=None,
                    help="seed for run-level randomness; default: time-based")
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    now = _utc_now()
    print(f"[{now.isoformat()}] cron_agent_pulse fire")
    print(f"  model={args.model_id} sim={args.sim_id} seed={args.seed}")

    since = now - timedelta(hours=args.review_window_hours)
    with connect() as conn:
        recent = _read_recent_runs(conn, since)
        summary = _summarise_recent_runs(recent)
        print(f"  reviewed {summary['total']} runs since {since.isoformat()}: "
              f"done={summary['done']} running={summary['running']} "
              f"failed={summary['failed']} queued={summary['queued']}")
        if summary["max_win"] is not None:
            print(f"  max win_rate observed: {summary['max_win']:.3f}  "
                  f"(mean over done runs: {summary['mean_win']:.3f})")

        # Cancel pre-existing queued backlog (the running run is preserved).
        if not args.no_cancel_backlog:
            _cancel_queued_backlog(conn, args.dry_run)

        # Decide what to queue next.
        next_runs = _decide_next_batch(recent, rng)

        # Push.
        ids = _push_runs(conn, next_runs, args.model_id, args.sim_id, args.dry_run)

    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"  queued {len(ids)} new runs")
        for i, rid in enumerate(ids[:5]):
            print(f"    {i:02d} id={rid}  label={next_runs[i]['label']}")
        if len(ids) > 5:
            print(f"    ... +{len(ids)-5} more")
    print(f"[{_utc_now().isoformat()}] done")


if __name__ == "__main__":
    sys.exit(main())
