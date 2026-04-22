"""Rebench matches to refresh the Elo graph.

Two modes:

  --all-vs-top       every completed run (not yet in top-K by Elo) plays the
                     current top-K. Use this after re-architecting or after
                     a long idle to bridge fresh runs into the graph.

  --run-vs-top <id>  a single run plays the current top-K. Use when a
                     previously-failed match left a gap.

Baseline opponent (`random_legal` pseudo-run) is always included unless
--no-baseline is passed, so absolute "vs random" numbers stay fresh.

Usage:
    python cli/rebench.py --all-vs-top --games 10
    python cli/rebench.py --run-vs-top <uuid> --games 20 --level random_8_12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


BASELINE_RUN_ID = "00000000-0000-0000-0000-000000000001"


def _current_top(conn, k: int) -> list[str]:
    """Replay Elo over all done games, return top-k run IDs (excluding baseline)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT g.player_1_run_id::text, g.player_2_run_id::text, g.winner::text
              FROM games g JOIN matches m ON g.match_id = m.id
             WHERE m.project = %s AND m.status = 'done'
             ORDER BY m.created_at, g.game_index
        """, (PROJECT,))
        games = cur.fetchall()
    K, INIT = 32, 1200
    elo: dict[str, float] = {}
    for p1, p2, winner in games:
        ra = elo.get(p1, INIT); rb = elo.get(p2, INIT)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = 1.0 if winner == p1 else (0.0 if winner == p2 else 0.5)
        elo[p1] = ra + K * (sa - ea)
        elo[p2] = rb + K * ((1 - sa) - (1 - ea))
    ranked = [rid for rid, _ in sorted(elo.items(), key=lambda x: -x[1])
              if rid != BASELINE_RUN_ID]
    return ranked[:k]


def _queue_match(conn, a: str, b: str, games: int, level_name: str, description: str) -> str:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO matches (project, description, model_a_run_id, model_b_run_id,
                                 simulator_id, games_planned, status, summary)
            VALUES (%s, %s, %s, %s,
                    (SELECT simulator_id FROM runs WHERE id = %s),
                    %s, 'queued', %s::jsonb)
            RETURNING id
        """, (PROJECT, description, a, b, a, games,
              json.dumps({"level_name": level_name})))
        return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-vs-top",  action="store_true",
                    help="every eligible run plays the current top-K")
    ap.add_argument("--run-vs-top",  help="single run UUID plays the top-K")
    ap.add_argument("--top-k",       type=int, default=5)
    ap.add_argument("--games",       type=int, default=10)
    ap.add_argument("--level",       default="random_8_12")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the baseline (vs random_legal) match")
    args = ap.parse_args()

    if not args.all_vs_top and not args.run_vs_top:
        raise SystemExit("need --all-vs-top or --run-vs-top <uuid>")

    with connect() as conn:
        top = _current_top(conn, args.top_k)
        if not top:
            raise SystemExit("no existing matches to compute top-K from — run an initial round-robin first")

        # Pick the set of runs that will play the top-K.
        if args.all_vs_top:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id::text FROM runs
                     WHERE project = %s AND status = 'done'
                       AND id::text != %s
                     ORDER BY queued_at
                """, (PROJECT, BASELINE_RUN_ID))
                candidates = [r[0] for r in cur.fetchall()]
        else:
            candidates = [args.run_vs_top]

        queued = 0
        for rid in candidates:
            for opp in top:
                if opp == rid:
                    continue
                _queue_match(conn, rid, opp, args.games, args.level,
                             "rebench-vs-top")
                queued += 1
            if not args.no_baseline and rid != BASELINE_RUN_ID:
                _queue_match(conn, rid, BASELINE_RUN_ID, args.games, args.level,
                             "rebench-vs-baseline")
                queued += 1
        conn.commit()

    print(f"queued {queued} match{'es' if queued != 1 else ''} "
          f"against top-{args.top_k} + baseline, level={args.level!r}, games={args.games}")


if __name__ == "__main__":
    main()
