"""Queue a head-to-head evaluation match between two trained runs.

The worker picks it up just like a training run, plays `--games` games
(alternating P1/P2 each game so side-advantage cancels), writes each
game's result into `games`, and updates `matches.summary` with the
aggregate rate.

Usage:
    python cli/queue_eval.py --run-a <uuid> --run-b <uuid> --games 20 \\
        --level random_8_12 --description "after 10-min run"

    # Round-robin over a handful of runs at a given level:
    python cli/queue_eval.py --round-robin id1,id2,id3,id4 --games 10 --level crossroads_6
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


def _queue_one(conn, run_a, run_b, games, level_name, description):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO matches (
                project, description, model_a_run_id, model_b_run_id,
                simulator_id, games_planned, status, summary
            )
            VALUES (%s, %s, %s, %s,
                    (SELECT simulator_id FROM runs WHERE id = %s),
                    %s, 'queued', %s::jsonb)
            RETURNING id
        """, (PROJECT, description, run_a, run_b, run_a,
              games, json.dumps({"level_name": level_name})))
        return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", help="UUID of run A")
    ap.add_argument("--run-b", help="UUID of run B")
    ap.add_argument("--round-robin", help="comma-separated UUIDs — queue every A-vs-B pair")
    ap.add_argument("--games", type=int, required=True, help="games per match")
    ap.add_argument("--level", default="crossroads_6",
                    help="level_name (static or dynamic like random_8_12)")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    if not args.round_robin and not (args.run_a and args.run_b):
        raise SystemExit("must supply --run-a + --run-b, or --round-robin")

    pairs: list[tuple[str, str]] = []
    if args.round_robin:
        ids = [s.strip() for s in args.round_robin.split(",") if s.strip()]
        if len(ids) < 2:
            raise SystemExit("round-robin needs ≥2 runs")
        pairs = list(itertools.combinations(ids, 2))
    else:
        pairs = [(args.run_a, args.run_b)]

    queued = []
    with connect() as conn:
        for a, b in pairs:
            match_id = _queue_one(conn, a, b, args.games, args.level, args.description)
            queued.append((match_id, a, b))
        conn.commit()

    print(f"queued {len(queued)} match{'es' if len(queued) != 1 else ''} "
          f"on level={args.level!r} ({args.games} games each):")
    for mid, a, b in queued:
        print(f"  {mid}  A={a[:8]}… vs B={b[:8]}…")


if __name__ == "__main__":
    main()
