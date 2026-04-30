"""Full rerank: clear the leaderboard, then rebuild it.

Phase 1 — Reset. Sets elo_score=1000, elo_n_matches=0, elo_status='unrated',
bench_vector=NULL, pfsp_weight=NULL on every rated run. Champion rows in
the `champions` archive are NOT touched (those are frozen checkpoints,
not leaderboard entries).

Phase 2 — Rebuild. Subprocess-runs scripts/rate_all_runs.py once per pass.
Each pass plays each newly-unrated run vs the current top-Elo set.

Defaults err on the side of "fast and useful" rather than "exhaustive":
max_runs=30, matches=4, games=64, passes=1 → 120 matches at ~5s each on
Mac MPS = ~10 min. Crank max_runs / passes for a deeper sweep.

Concurrency note: PaulLinux training + auto-admission can fire while
this job runs. New runs land freshly rated in the rebuilt leaderboard
just like during normal operation. No locking needed.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

from cli.db import connect, PROJECT

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FLUSH_INTERVAL_S = 5.0
LOG_TAIL_BYTES = 12000


def handle(job: dict, mark_done_fn, mark_failed_fn) -> None:
    hp = job.get("hyperparams") or {}
    max_runs = int(hp.get("max_runs", 30))
    matches  = int(hp.get("matches",  4))
    games    = int(hp.get("games",    64))
    level    = str(hp.get("level",    "random_close_4_5"))
    passes   = int(hp.get("passes",   1))

    print(f"[job:rerate_full] max_runs={max_runs} matches={matches} "
          f"games={games} level={level} passes={passes}", flush=True)

    log_lines: list[str] = []
    t0 = time.time()
    last_flush = t0

    def _say(line: str):
        nonlocal last_flush
        log_lines.append(line)
        print(line, flush=True)
        now = time.time()
        if now - last_flush >= LOG_FLUSH_INTERVAL_S:
            _flush_log(job["id"], _tail("\n".join(log_lines)),
                       int((now - t0) * 1000))
            last_flush = now

    # ---------------------------------------------------------------
    # Phase 1: reset
    # ---------------------------------------------------------------
    _say("[phase 1] resetting leaderboard…")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE runs
               SET elo_score      = 1000,
                   elo_n_matches  = 0,
                   elo_status     = 'unrated',
                   bench_vector   = NULL,
                   pfsp_weight    = NULL
             WHERE project = %s
               AND simulator_id <> 'admin'
               AND elo_n_matches IS NOT NULL
               AND elo_n_matches >= 1
            """,
            (PROJECT,),
        )
        n_reset = cur.rowcount
        conn.commit()
    _say(f"[phase 1] reset {n_reset} runs to Elo 1000 (unrated)")

    # ---------------------------------------------------------------
    # Phase 2: rebuild via rate_all_runs.py — one pass per iteration.
    # rate_all_runs picks opponents from the current top-Elo set, so
    # a single pass would compare new to runs that just got rated.
    # Multiple passes let the leaderboard self-stabilize.
    # ---------------------------------------------------------------
    pass_summaries = []
    for p in range(1, passes + 1):
        _say("")
        _say(f"[phase 2.{p}/{passes}] rate_all_runs --include-rated "
             f"--max-runs {max_runs} --matches {matches} --games {games} "
             f"--level {level}")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "rate_all_runs.py"),
            "--include-rated",
            "--max-runs", str(max_runs),
            "--matches",  str(matches),
            "--games",    str(games),
            "--level",    level,
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:
                _say(line.rstrip("\n"))
        finally:
            proc.wait()
        if proc.returncode != 0:
            err = f"pass {p} failed: rate_all_runs exited {proc.returncode}"
            _say(err)
            with connect() as conn:
                mark_failed_fn(
                    conn, job["id"],
                    err + "\n\n" + _tail("\n".join(log_lines)),
                    int((time.time() - t0) * 1000),
                )
            return

        md, mf = _parse_summary(log_lines)
        pass_summaries.append({"pass": p, "matches_done": md, "matches_failed": mf})

    wall_ms = int((time.time() - t0) * 1000)

    # Snapshot the rebuilt top-of-leaderboard for the result row.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT label, elo_score, elo_n_matches FROM runs
            WHERE project = %s AND simulator_id <> 'admin'
              AND elo_n_matches >= 1
            ORDER BY elo_score DESC LIMIT 10
            """,
            (PROJECT,),
        )
        top10 = [{"label": r[0], "elo": r[1], "n": r[2]} for r in cur.fetchall()]

    _say("")
    _say(f"[done] reset {n_reset} runs, ran {passes} pass(es), "
         f"wall {wall_ms/60000:.1f} min")
    _say("[done] new top 5:")
    for row in top10[:5]:
        _say(f"  {row['label'][:50]:50s} Elo {row['elo']:.0f}  (n={row['n']})")

    result = {
        "kind": "rerate_full",
        "params": {"max_runs": max_runs, "matches": matches,
                   "games": games, "level": level, "passes": passes},
        "n_reset": n_reset,
        "passes": pass_summaries,
        "top10_after": top10,
        "log": _tail("\n".join(log_lines)),
        "wall_s": round(wall_ms / 1000, 1),
    }

    matches_total = sum(p["matches_done"] for p in pass_summaries)
    with connect() as conn:
        mark_done_fn(
            conn, job["id"], result,
            games_played=matches_total * games,
            wall_ms=wall_ms,
        )


def _parse_summary(log_lines: list[str]) -> tuple[int, int]:
    """Pull the most recent `[rate_all] done: …` line."""
    for line in reversed(log_lines):
        m = re.search(r"\[rate_all\] done:\s+(\d+)\s+matches.*?(\d+)\s+failed",
                      line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def _tail(s: str) -> str:
    return s if len(s) <= LOG_TAIL_BYTES else "…(truncated)…\n" + s[-LOG_TAIL_BYTES:]


def _flush_log(run_id, log: str, wall_ms: int) -> None:
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                   SET result = jsonb_build_object(
                                  'kind', 'rerate_full',
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
        print(f"[job:rerate_full] log flush failed (non-fatal): {e}", flush=True)
