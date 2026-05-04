"""Bulk Elo rating for unrated runs.

Goes through every done run that has < ELO_MIN_MATCHES matches and runs
N matches per run vs the current top-Elo set + random_legal, writing the
Elo deltas back via scripts/tournament.update_elo_from_match.

Useful for:
  - Backfilling a fleet of cron-queued runs that didn't get rated yet.
  - Re-rating after a model bump where old Elo numbers are stale.

Usage:
    python scripts/rate_all_runs.py                    # rate all unrated done runs
    python scripts/rate_all_runs.py --matches 3        # 3 matches per run (default 3)
    python scripts/rate_all_runs.py --since-hours 48   # only recent runs
    python scripts/rate_all_runs.py --sim sim-v1.3     # restrict by sim
    python scripts/rate_all_runs.py --dry-run          # show plan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.18")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SIM_BACKEND", "jax")

from cli.db import PROJECT, connect


ELO_MIN_MATCHES = 3   # mirrors cron_agent_pulse.ELO_MIN_MATCHES


def _read_unrated(conn, sim_id: str | None, since_hours: int | None) -> list[dict]:
    where = ["project=%s", "status='done'", "weights_url IS NOT NULL",
             "(elo_n_matches IS NULL OR elo_n_matches < %s)"]
    params: list = [PROJECT, ELO_MIN_MATCHES]
    if sim_id:
        where.append("simulator_id=%s")
        params.append(sim_id)
    if since_hours:
        cutoff = int((datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp() * 1000)
        where.append("launch_at >= %s")
        params.append(cutoff)
    sql = (
        "SELECT id, label, elo_score, elo_n_matches, finished_at "
        "FROM runs WHERE " + " AND ".join(where) +
        " ORDER BY finished_at DESC NULLS LAST"
    )
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [
        {"id": str(r[0]), "label": r[1],
         "elo_score": float(r[2]) if r[2] is not None else 1000.0,
         "elo_n_matches": int(r[3]) if r[3] is not None else 0,
         "finished_at": r[4]}
        for r in rows
    ]


def _read_top_elo(conn, limit: int = 5) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, elo_score, elo_n_matches FROM runs
             WHERE project=%s AND status='done' AND weights_url IS NOT NULL
               AND elo_n_matches >= %s
             ORDER BY elo_score DESC LIMIT %s
            """,
            (PROJECT, ELO_MIN_MATCHES, limit),
        )
        return [{"id": str(r[0]), "label": r[1],
                 "elo_score": float(r[2]), "elo_n_matches": int(r[3])}
                for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=3,
                    help="matches per run (default 3 — enough to clear ELO_MIN_MATCHES)")
    ap.add_argument("--games", type=int, default=64,
                    help="games per match (default 64)")
    ap.add_argument("--level", default="random_8_16")
    ap.add_argument("--since-hours", type=int, default=None)
    ap.add_argument("--sim", default=None, help="restrict to simulator_id (e.g. sim-v1.3)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="cap how many runs to process (None = all)")
    ap.add_argument("--include-rated", action="store_true",
                    help="also re-rate already-rated runs (default: skip them)")
    args = ap.parse_args()

    # Imports that need JAX init.
    import torch  # noqa: F401
    from scripts.tournament import run_match, update_elo_from_match

    with connect() as conn:
        unrated = _read_unrated(conn, args.sim, args.since_hours)
        if args.include_rated:
            # Replace filter with simply "all done with weights".
            with conn.cursor() as cur:
                where = ["project=%s", "status='done'", "weights_url IS NOT NULL"]
                params: list = [PROJECT]
                if args.sim:
                    where.append("simulator_id=%s")
                    params.append(args.sim)
                if args.since_hours:
                    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=args.since_hours)).timestamp() * 1000)
                    where.append("launch_at >= %s")
                    params.append(cutoff)
                cur.execute(
                    "SELECT id, label, elo_score, elo_n_matches FROM runs WHERE " +
                    " AND ".join(where) + " ORDER BY finished_at DESC NULLS LAST",
                    tuple(params),
                )
                unrated = [{"id": str(r[0]), "label": r[1],
                            "elo_score": float(r[2]) if r[2] is not None else 1000.0,
                            "elo_n_matches": int(r[3]) if r[3] is not None else 0,
                            "finished_at": None}
                           for r in cur.fetchall()]

        if args.max_runs:
            unrated = unrated[:args.max_runs]

        top = _read_top_elo(conn, limit=5)

    print(f"[rate_all] {len(unrated)} runs to rate")
    print(f"[rate_all] top-Elo set: {[t['label'][:40] for t in top]}")

    if not unrated:
        print("[rate_all] nothing to do")
        return

    # Build opponent pool — top-Elo runs + random_legal.
    opponents = [t["id"] for t in top] + ["random_legal"]
    opponents = list(dict.fromkeys(opponents))

    if args.dry_run:
        n_matches = sum(args.matches for _ in unrated)
        eta_min = n_matches * (args.games / 64) * 6 / 60   # ~6s per 64-game match on GPU
        print(f"[rate_all] [dry-run] would run {n_matches} matches "
              f"(~{eta_min:.1f}min)")
        for u in unrated[:10]:
            print(f"    {u['label'][:50]:50s}  Elo={u['elo_score']:.0f} n={u['elo_n_matches']}")
        if len(unrated) > 10:
            print(f"    ... +{len(unrated)-10} more")
        return

    t0 = time.time()
    n_done = 0
    n_failed = 0
    for u in unrated:
        # Pick `args.matches` opponents (cycling through the pool).
        for i in range(args.matches):
            opp = opponents[i % len(opponents)]
            short_opp = opp[:8] if len(opp) == 36 else opp
            try:
                with connect() as c:
                    res = run_match(
                        p1=u["id"], p2=opp,
                        games=args.games, level=args.level,
                        seed=i, verbose=False,
                    )
                    update_elo_from_match(
                        c, p1_run_id=u["id"],
                        p2_run_id=opp if opp != "random_legal" else None,
                        result=res, k=32,
                    )
                    elapsed = time.time() - t0
                    rate = res["p1_wins"] / max(res["total"], 1)
                    print(f"  [{n_done+1}/{len(unrated)*args.matches}] {u['label'][:42]:42s} "
                          f"vs {short_opp:14s} rate={rate:.3f}  "
                          f"({elapsed/60:.1f}min elapsed)", flush=True)
                    n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"  [fail] {u['label'][:42]} vs {short_opp}: {e}")

    wall = time.time() - t0
    print(f"\n[rate_all] done: {n_done} matches in {wall/60:.1f}min, {n_failed} failed")


if __name__ == "__main__":
    main()
