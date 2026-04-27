"""Re-rate historical runs through the new 4-step Elo gate.

Selective by default — rebenching every run is wasteful since old weak runs
that scored ~1000 vs random_legal will just score ~1000 again. Picks runs
that meet ANY of:
  - elo_score >= --min-elo (default 1050: above-anchor, the interesting tier)
  - finished_at >= --since-days ago (default 14: recent regardless of Elo)

Resets the picked rows to a clean slate before re-running the gate:
  - elo_score    = 1000 (default)
  - elo_n_matches = 0
  - elo_status   = 'unrated'

Then calls workers.elo_gate.run_elo_gate(run_id, label) for each.

Usage:
    python cli/rebench_with_gate.py --dry-run            # show what would be rebenched
    python cli/rebench_with_gate.py                       # run with defaults
    python cli/rebench_with_gate.py --min-elo 1100        # tighter
    python cli/rebench_with_gate.py --since-days 30 --min-elo 1000   # everything in last 30d
    python cli/rebench_with_gate.py --ids <uuid1>,<uuid2> # explicit list
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import connect, PROJECT


def _select_run_ids(min_elo: float, since_days: int, explicit_ids: list[str] | None) -> list[tuple[str, str, float | None]]:
    if explicit_ids:
        with connect() as c:
            with c.cursor() as cur:
                cur.execute("""
                    SELECT id, label, elo_score
                      FROM runs
                     WHERE id = ANY(%s) AND status='done' AND weights_url IS NOT NULL
                     ORDER BY finished_at DESC NULLS LAST
                """, (explicit_ids,))
                return [(str(r[0]), r[1], float(r[2]) if r[2] is not None else None) for r in cur.fetchall()]

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT id, label, elo_score
                  FROM runs
                 WHERE project = %s
                   AND status = 'done'
                   AND weights_url IS NOT NULL
                   AND ((elo_score IS NOT NULL AND elo_score >= %s)
                        OR finished_at >= %s)
                 ORDER BY finished_at DESC NULLS LAST
            """, (PROJECT, min_elo, cutoff))
            return [(str(r[0]), r[1], float(r[2]) if r[2] is not None else None) for r in cur.fetchall()]


def _reset_row(run_id: str) -> None:
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                UPDATE runs
                   SET elo_score = 1000,
                       elo_n_matches = 0,
                       elo_status = 'unrated'
                 WHERE id = %s
            """, (run_id,))
        c.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-elo", type=float, default=1050.0,
                    help="rebench any run with current Elo >= this (default 1050)")
    ap.add_argument("--since-days", type=int, default=14,
                    help="rebench any run finished within this many days regardless of Elo (default 14)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated run UUIDs; bypasses min-elo / since-days selection")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selected runs and exit, no rating")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of runs to rebench (debug)")
    args = ap.parse_args()

    explicit = [s.strip() for s in args.ids.split(",")] if args.ids else None
    selected = _select_run_ids(args.min_elo, args.since_days, explicit)
    if args.limit:
        selected = selected[: args.limit]

    print(f"[rebench] selected {len(selected)} runs "
          f"(min_elo={args.min_elo}, since_days={args.since_days}, "
          f"explicit_ids={'yes' if explicit else 'no'})")
    for rid, label, elo in selected:
        print(f"  {rid[:8]}  Elo={elo!s:>8}  {label}")

    if args.dry_run:
        print("[rebench] dry-run: no changes made")
        return

    if not selected:
        print("[rebench] nothing to do")
        return

    # Import lazily — pulls in JAX/torch only when actually rating.
    from workers.elo_gate import run_elo_gate

    t0 = time.perf_counter()
    ok = 0
    fail = 0
    for i, (rid, label, _elo) in enumerate(selected, 1):
        print(f"\n[rebench] [{i}/{len(selected)}] {label}  ({rid[:8]})")
        try:
            _reset_row(rid)
            run_elo_gate(rid, label)
            ok += 1
        except Exception:
            traceback.print_exc()
            fail += 1
            print(f"[rebench] [{i}/{len(selected)}] FAILED — continuing")

    wall = time.perf_counter() - t0
    print(f"\n[rebench] done. ok={ok} fail={fail} wall={wall/60:.1f}min")


if __name__ == "__main__":
    main()
