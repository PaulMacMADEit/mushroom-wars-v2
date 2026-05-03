"""Fast-fail watcher for a single training run.

Purpose: when a run is queued/running, poll its DB row and log significant
state transitions. Exit immediately when the run hits a terminal status
(done/failed/cancelled/discarded). Designed to be launched as a background
task right after queueing so any failure surfaces inside ~30s instead of
"next time someone happens to check."

Usage:
    python scripts/watch_run.py <run_id>
    python scripts/watch_run.py <run_id> --interval 30 --max-wait 7200

Output schema (tab-separated, parseable by other scripts):
    [HH:MM:SS] state=<status> wall=<sec> updates=<n> latest_win=<rate>

On terminal:
    [HH:MM:SS] TERMINAL state=<status> wall=<sec> exit_reason=<error[:200]>

Why this script exists vs ad-hoc bash polling: a typed Python helper means
we can centralise the fast-fail criteria (e.g. "exit early if run failed
within first 5 min — critical to surface infra bugs"). Easier to extend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect


TERMINAL_STATES = {"done", "failed", "cancelled", "discarded"}


def _fmt(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _query_run(run_id: str) -> dict | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT label, status, started_at, finished_at, wall_ms, error, "
                "       result::text "
                "FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    label, status, started_at, finished_at, wall_ms, error, result_json = row
    parsed = {
        "label":       label,
        "status":      status,
        "started_at":  started_at,
        "finished_at": finished_at,
        "wall_s":      (wall_ms or 0) / 1000.0,
        "error":       error,
    }
    if result_json:
        try:
            res = json.loads(result_json)
            parsed["updates"]    = res.get("updates")
            parsed["rate"]       = res.get("rate")
            fm                   = res.get("final_metrics") or {}
            parsed["final_win"]  = fm.get("win_rate")
        except Exception:
            pass
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", help="UUID of the run row")
    ap.add_argument("--interval", type=int, default=30,
                    help="poll interval seconds (default 30)")
    ap.add_argument("--max-wait", type=int, default=7200,
                    help="give up after N seconds (default 2h)")
    ap.add_argument("--early-fail-window", type=int, default=300,
                    help="if run fails within this many seconds of starting, "
                         "flag as early-fail (likely infra bug). default 300.")
    args = ap.parse_args()

    t0 = time.time()
    last_status: str | None = None
    print(f"[{_fmt(time.time())}] WATCH start run_id={args.run_id} "
          f"interval={args.interval}s", flush=True)

    while True:
        if time.time() - t0 > args.max_wait:
            print(f"[{_fmt(time.time())}] GIVE_UP exceeded max-wait={args.max_wait}s",
                  flush=True)
            return 2

        info = _query_run(args.run_id)
        if info is None:
            print(f"[{_fmt(time.time())}] run row not found", flush=True)
            return 3

        status = info["status"]
        if status != last_status:
            extras = []
            if info.get("rate") is not None:
                extras.append(f"rate={info['rate']:.3f}")
            if info.get("updates") is not None:
                extras.append(f"updates={info['updates']}")
            if info.get("final_win") is not None:
                extras.append(f"final_win={info['final_win']:.3f}")
            extra_str = " " + " ".join(extras) if extras else ""
            print(f"[{_fmt(time.time())}] state={status} "
                  f"wall={info['wall_s']:.0f}s{extra_str}", flush=True)
            last_status = status

        if status in TERMINAL_STATES:
            wall = info["wall_s"]
            err = info.get("error") or ""
            err_excerpt = err.replace("\n", " | ")[:300] if err else ""
            print(f"[{_fmt(time.time())}] TERMINAL state={status} "
                  f"wall={wall:.0f}s {('exit_reason='+err_excerpt) if err_excerpt else ''}",
                  flush=True)

            if status == "failed" and wall < args.early_fail_window:
                # Fast-fail flag: failures within the first N sec of starting
                # are almost always infra/config bugs (wrong sim backend,
                # bad weights, model_id mismatch). Surface explicitly.
                print(f"[{_fmt(time.time())}] EARLY_FAIL run died in {wall:.0f}s "
                      f"(<{args.early_fail_window}s window) — likely an infra/config bug. "
                      f"Re-check hyperparams + worker code before re-queueing.",
                      flush=True)
                return 4
            if status == "failed":
                return 1
            if status in ("cancelled", "discarded"):
                return 5
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
