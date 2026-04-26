"""Quick post-run summary of all sim-v1.3 runs.

Reads Supabase, prints a per-run line with key metrics, and reports the
champion + finish-speed if available.

Usage:
    python scripts/v13_summary.py
    python scripts/v13_summary.py --since-hours 6
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=24)
    ap.add_argument("--sim-id", default="sim-v1.3")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, label, status, simulator_id, hyperparams::text, result::text,
                       elo_score, elo_n_matches, started_at, finished_at
                  FROM runs
                 WHERE project=%s AND simulator_id=%s
                   AND launch_at >= %s
                 ORDER BY launch_at DESC
                """,
                (PROJECT, args.sim_id, int(since.timestamp() * 1000)),
            )
            rows = cur.fetchall()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT key, value, updated_at FROM kv ORDER BY key"
            )
            kv = cur.fetchall()

    print(f"=== sim-v1.3 runs (last {args.since_hours}h) ===")
    print(f"{'status':10s} {'label':50s} {'level':22s} {'K':>2s} "
          f"{'rate':>6s} {'updates':>7s} {'elo':>6s} {'matches':>7s}")
    for r in rows:
        rid, label, status, sim_id, hp_s, res_s, elo_score, elo_matches, _start, _end = r
        hp  = json.loads(hp_s)  if hp_s  else {}
        res = json.loads(res_s) if res_s else {}
        rate    = res.get("rate")
        updates = res.get("updates")
        level   = hp.get("level_name", "?")
        K       = hp.get("action_repeat", "?")
        rate_s    = f"{rate:.3f}" if rate is not None else "-"
        updates_s = str(updates) if updates is not None else "-"
        print(f"{status:10s} {label[:50]:50s} {level[:22]:22s} {K!s:>2s} "
              f"{rate_s:>6s} {updates_s:>7s} {(elo_score or 1200):>6.0f} {elo_matches or 0:>7d}")

    print(f"\n=== kv table ===")
    for k, v, when in kv:
        print(f"  {k} = {v}   (updated {when})")


if __name__ == "__main__":
    main()
