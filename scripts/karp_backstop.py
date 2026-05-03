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


# Backstop only stays out of the way when the worker is *currently* busy
# with a karp run (queued or running). The previous "recently finished"
# check (25min window) caused 15-min idle gaps when a sweep finished and
# Claude wasn't awake to queue the next one. GPU sat at 0% during those
# gaps; we'd rather over-queue and let the cap (max_karp_queue_depth)
# limit pile-up.

# Protected label families (do not discard; counted as "karp-active"):
#   - karpv2-...       legacy A
#   - karp-...         very-legacy
#   - v\d+\.\d+\....   current convention (v10.2.02-LargeMap-...)
#
# 2026-04-30: previously these checks used `LIKE 'karpv2-%'` and
# `NOT LIKE 'karp-%'`, which DISCARDED v10.1.37-lr-mid/-hi and similar
# new-format runs as "non-karp queued junk" within 30 min of being queued.
# Caught when v10.1.37-lr-{mid,hi} were marked discarded by the backstop
# while -lo was running; only the running cell survived.
_PROTECTED_SQL = (
    "(label LIKE 'karpv2-%%' "
    " OR label LIKE 'karp-%%' "
    " OR label ~ '^v[0-9]+\\.[0-9]+\\.')"
)


def _karp_is_active() -> tuple[bool, str]:
    """True iff there's a queued or running karp-style run.

    Recognises all three label families (legacy + current).
    """
    with connect() as c, c.cursor() as cur:
        cur.execute(f"""
            SELECT count(*) FROM runs
             WHERE project=%s AND {_PROTECTED_SQL}
               AND status IN ('queued','running')
        """, (PROJECT,))
        n_live = cur.fetchone()[0]
    if n_live > 0:
        return True, f"{n_live} karp-style run(s) queued/running"
    return False, "queue empty"


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
        cur.execute(f"""
            UPDATE runs
               SET status='discarded',
                   error='cleared by karp-backstop',
                   finished_at=now()
             WHERE project=%s AND status='queued'
               AND NOT {_PROTECTED_SQL}
               AND simulator_id <> 'admin'
        """, (PROJECT,))
        n_discarded = cur.rowcount
        c.commit()
    if n_failed or n_discarded:
        print(f"[backstop] cleared {n_failed} stale running, {n_discarded} non-karp queued")


def _pick_continuation_parent() -> tuple[str, str, float] | None:
    """Pick the strongest recent sim-v1.4 done run as continuation parent.

    Strategy (Paul confirmed 2026-05-02 23:40 PT — never queue Bootstrap):
      1. Among the last 12 sim-v1.4 done runs that have `result.rate`,
         pick the highest training rate (rate >= 0.70 floor).
      2. If none qualify, return None — caller MUST refuse to queue
         (exit non-zero) rather than fall back to Bootstrap. The Claude-
         side fire chain is responsible for bootstrap experiments.

    Returns (run_id, label, rate) or None.
    """
    with connect() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id, label,
                   (result->>'rate')::numeric AS rate
              FROM runs
             WHERE project=%s
               AND simulator_id='sim-v1.4'
               AND status='done'
               AND result IS NOT NULL
               AND result->>'rate' IS NOT NULL
               AND (result->>'rate')::numeric >= 0.70
             ORDER BY finished_at DESC
             LIMIT 12
        """, (PROJECT,))
        rows = cur.fetchall()
    if not rows:
        return None
    # Best of last 12 by training rate
    best = max(rows, key=lambda r: float(r[2]))
    return (str(best[0]), best[1], float(best[2]))


def main() -> None:
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n=== karp-backstop fire {t0} ===")

    active, reason = _karp_is_active()
    if active:
        print(f"[backstop] no-op: {reason}")
        return

    print(f"[backstop] {reason} — taking over")
    _clear_clutter()

    # Continuation rule (2026-05-02): never Bootstrap. If we can't find
    # a compatible sim-v1.4 parent with rate>=0.70, exit non-zero so the
    # timer retries next fire rather than starting from random.
    parent = _pick_continuation_parent()
    if parent is None:
        print("[backstop] no sim-v1.4 done run with rate>=0.70 — refusing to "
              "queue Bootstrap; will retry next fire")
        sys.exit(2)
    parent_id, parent_label, parent_rate = parent
    print(f"[backstop] continuation parent: {parent_label} "
          f"({parent_id[:8]}, rate={parent_rate:.3f})")

    print("[backstop] calling queue_karp_sweep.py --from-run-id ...")
    py = sys.executable
    rc = subprocess.call(
        [py, str(ROOT / "scripts" / "queue_karp_sweep.py"),
         "--from-run-id", parent_id],
        cwd=str(ROOT),
    )
    if rc == 0:
        print("[backstop] queued one continuation sweep")
    else:
        print(f"[backstop] queue script exited {rc} — will retry next fire")
        sys.exit(rc)


if __name__ == "__main__":
    main()
