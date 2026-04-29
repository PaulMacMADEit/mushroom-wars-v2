"""Rerate job handler: subprocess-wraps `scripts/rate_all_runs.py`.

Streams stdout into runs.result.log every ~5s so the dashboard can tail it.
On success, parses the trailing summary line and writes a structured result.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from cli.db import connect

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FLUSH_INTERVAL_S = 5.0
LOG_TAIL_BYTES = 8000  # cap stored log to keep result jsonb small


def handle(job: dict, mark_done_fn, mark_failed_fn) -> None:
    hp = job.get("hyperparams") or {}
    matches  = int(hp.get("matches",  4))
    games    = int(hp.get("games",    64))
    max_runs = int(hp.get("max_runs", 30))
    level    = str(hp.get("level",    "random_8_16"))

    cmd = [
        sys.executable, str(ROOT / "scripts" / "rate_all_runs.py"),
        "--include-rated",
        "--matches",  str(matches),
        "--games",    str(games),
        "--max-runs", str(max_runs),
        "--level",    level,
    ]
    print(f"[job:rerate] {job['id']} cmd={' '.join(cmd)}", flush=True)

    t0 = time.time()
    log_lines: list[str] = []
    last_flush = t0
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            log_lines.append(line)
            now = time.time()
            if now - last_flush >= LOG_FLUSH_INTERVAL_S:
                _flush_log(job["id"], _tail("\n".join(log_lines)),
                           int((now - t0) * 1000))
                last_flush = now
    finally:
        proc.wait()

    wall_ms = int((time.time() - t0) * 1000)
    log = "\n".join(log_lines)

    if proc.returncode != 0:
        with connect() as conn:
            mark_failed_fn(
                conn, job["id"],
                f"rate_all_runs exited {proc.returncode}\n\n{_tail(log)}",
                wall_ms,
            )
        return

    matches_done, matches_failed = _parse_summary(log)
    result = {
        "kind": "rerate",
        "params": {"matches": matches, "games": games,
                   "max_runs": max_runs, "level": level},
        "matches_done": matches_done,
        "matches_failed": matches_failed,
        "log": _tail(log),
        "wall_s": round(wall_ms / 1000, 1),
    }
    with connect() as conn:
        mark_done_fn(
            conn, job["id"], result,
            games_played=matches_done * games,
            wall_ms=wall_ms,
        )


def _tail(s: str) -> str:
    return s if len(s) <= LOG_TAIL_BYTES else "…(truncated)…\n" + s[-LOG_TAIL_BYTES:]


def _flush_log(run_id, log: str, wall_ms: int) -> None:
    """Write incremental log so the dashboard can show progress.

    Failure is non-fatal: if the DB write hiccups, we keep the in-memory log
    and try again next interval.
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                   SET result = jsonb_build_object(
                                  'kind', 'rerate',
                                  'log', %s::text,
                                  'wall_s', %s::float,
                                  'in_progress', true),
                       wall_ms = %s
                 WHERE id = %s
                """,
                (log, wall_ms / 1000.0, wall_ms, run_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[job:rerate] log flush failed (non-fatal): {e}", flush=True)


def _parse_summary(log: str) -> tuple[int, int]:
    """Parse the trailing '[rate_all] done: N matches in M.Mmin, K failed'."""
    for line in reversed(log.splitlines()):
        m = re.search(r"\[rate_all\] done:\s+(\d+)\s+matches.*?(\d+)\s+failed",
                      line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0
