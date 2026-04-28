"""Karpathy-loop backstop: server-side timer that keeps the loop running
even when Claude's session is asleep.

Runs on a 30-min systemd timer on PaulLinux. Each fire:
  1. Checks if there's already an active karp- run (queued or running) OR
     a karp- run that finished in the last 25 min.
  2. If yes → no-op (Claude is presumably keeping up).
  3. If no  → queue the next karp sweep via queue_karp_sweep.py.
  4. Also clears stale 'running' rows + any non-karp queued junk, since
     Claude may not have been awake to do this either.

This is a SAFETY NET. The expected primary driver is Claude calling
ScheduleWakeup hourly. If Claude's chain breaks (auth lapse, fire gets
skipped, prompt times out), the backstop keeps karp runs flowing.

Idempotent: safe to fire every 30 min indefinitely.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect, PROJECT


# Window in which the loop is considered "active". If a karp- run is
# queued, running, or finished within this many minutes, the backstop
# stays out of the way.
ACTIVE_WINDOW_MIN = 25


def _karp_is_active() -> tuple[bool, str]:
    """True if Claude has been queueing recently. Returns (is_active, reason)."""
    with connect() as c, c.cursor() as cur:
        # Any queued/running karp = active
        cur.execute("""
            SELECT count(*) FROM runs
             WHERE project=%s AND label LIKE 'karp-%%'
               AND status IN ('queued','running')
        """, (PROJECT,))
        n_live = cur.fetchone()[0]
        if n_live > 0:
            return True, f"{n_live} karp- run(s) queued/running"

        # Any karp- finished in the active window = active
        cur.execute("""
            SELECT count(*) FROM runs
             WHERE project=%s AND label LIKE 'karp-%%'
               AND finished_at >= now() - (%s * interval '1 minute')
        """, (PROJECT, ACTIVE_WINDOW_MIN))
        n_recent = cur.fetchone()[0]
        if n_recent > 0:
            return True, f"{n_recent} karp- run(s) finished in last {ACTIVE_WINDOW_MIN}min"

    return False, "no karp- activity in window"


def _clear_clutter() -> None:
    """Mirror what Claude does: stale running → failed; non-karp queued → discarded."""
    with connect() as c, c.cursor() as cur:
        cur.execute("""
            UPDATE runs
               SET status='failed',
                   error='cleared by karp-backstop: stale',
                   finished_at=now()
             WHERE project=%s AND status='running'
               AND started_at < now() - interval '90 minutes'
        """, (PROJECT,))
        n_failed = cur.rowcount
        cur.execute("""
            UPDATE runs
               SET status='discarded',
                   error='cleared by karp-backstop',
                   finished_at=now()
             WHERE project=%s AND status='queued'
               AND label NOT LIKE 'karp-%%'
        """, (PROJECT,))
        n_discarded = cur.rowcount
        c.commit()
    if n_failed or n_discarded:
        print(f"[backstop] cleared {n_failed} stale running, {n_discarded} non-karp queued")


def main() -> None:
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n=== karp-backstop fire {t0} ===")

    active, reason = _karp_is_active()
    if active:
        print(f"[backstop] no-op: {reason}")
        return

    print(f"[backstop] {reason} — taking over")
    _clear_clutter()

    # Defer to the Claude-facing queue script; same source of truth.
    print("[backstop] calling queue_karp_sweep.py")
    py = sys.executable
    rc = subprocess.call(
        [py, str(ROOT / "scripts" / "queue_karp_sweep.py")],
        cwd=str(ROOT),
    )
    if rc == 0:
        print("[backstop] queued one sweep")
    else:
        print(f"[backstop] queue script exited {rc} — will retry next fire")
        sys.exit(rc)


if __name__ == "__main__":
    main()
