"""Chain N continuation runs — each starts from the previous one's weights.

Queues one continuation, waits for the worker to finish it, then queues the
next with the just-completed run as its parent. Useful for "train for 2
hours total in 4 chunks so I get intermediate checkpoints for Elo".

Usage:
    python scripts/chain_train.py \\
        --from <parent_run_id> \\
        --chain 4 \\
        --budget 1800 \\
        --label-prefix selfplay-chain \\
        --config '{"level_name": "random_8_12", "n_envs": 32, "vec_mode": "async", "self_play": true, "snapshot_every": 10}'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect


def _queue_continuation(parent_id: str, budget_s: int, label: str, config: str) -> str:
    """Invoke cli/continue_training.py as a subprocess so we reuse its
    argument parsing + insert logic. Returns the new run's UUID."""
    out = subprocess.check_output([
        sys.executable, str(ROOT / "cli" / "continue_training.py"),
        "--parent", parent_id,
        "--budget", str(budget_s),
        "--label", label,
        "--config", config,
    ], text=True)
    # First line: "queued continuation <uuid>"
    first = out.strip().splitlines()[0]
    prefix = "queued continuation "
    if not first.startswith(prefix):
        raise RuntimeError(f"unexpected output from continue_training: {first}")
    return first[len(prefix):].strip()


def _wait_terminal(run_id: str, poll_s: float = 30.0) -> dict:
    """Block until the run moves out of queued/running. Returns the final row."""
    while True:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT status, games_played, wall_ms,
                           (result->>'rate')::float,
                           result->>'updates',
                           error
                    FROM runs WHERE id = %s
                """, (run_id,))
                row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"run {run_id} vanished from DB")
        status, games, wall, rate, updates, err = row
        if status == "done":
            return {"status": status, "games": games, "wall_ms": wall, "rate": rate, "updates": updates}
        if status == "failed":
            return {"status": status, "error": err}
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="parent", required=True, help="seed parent run UUID")
    ap.add_argument("--chain", type=int, required=True, help="how many continuations to queue sequentially")
    ap.add_argument("--budget", type=int, required=True, help="seconds per run")
    ap.add_argument("--label-prefix", required=True, help="label becomes '<prefix>-<i>'")
    ap.add_argument("--config", default="{}", help="JSON hyperparams passed to each continuation")
    args = ap.parse_args()

    parent = args.parent
    t0 = time.time()
    for i in range(1, args.chain + 1):
        label = f"{args.label_prefix}-{i}"
        print(f"[chain] [{i}/{args.chain}] queueing {label!r} parent={parent}")
        new_id = _queue_continuation(parent, args.budget, label, args.config)
        print(f"[chain] [{i}/{args.chain}] id={new_id} — waiting…")
        result = _wait_terminal(new_id)
        elapsed = time.time() - t0
        if result["status"] != "done":
            print(f"[chain] [{i}/{args.chain}] FAILED after {elapsed:.0f}s: {result.get('error','')[:400]}")
            raise SystemExit(2)
        print(f"[chain] [{i}/{args.chain}] done in total {elapsed:.0f}s  "
              f"rate={result['rate']:.3f} games={result['games']} updates={result['updates']}")
        parent = new_id

    print(f"[chain] finished {args.chain} links in {time.time() - t0:.0f}s, final run id={parent}")


if __name__ == "__main__":
    main()
