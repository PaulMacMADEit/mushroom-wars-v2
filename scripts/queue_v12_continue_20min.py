"""Queue a 20-min v12 continuation run that warm-starts from a parent run.

Same hyperparams as queue_v12_baseline_20min.py, but sets parent_run_id +
is_continuation=true so the worker downloads the parent's weights /
optimizer / obs_norm and resumes from there instead of fresh init.

Usage:
    .venv/bin/python scripts/queue_v12_continue_20min.py <parent_run_id>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect
from cli.loop_config import load


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: queue_v12_continue_20min.py <parent_run_id>")
        sys.exit(1)
    parent = sys.argv[1].strip()

    cfg = load()
    base_hp = dict(cfg.baseline_hyperparams)

    base_hp["level_name"] = "random_close_4_6"
    base_hp["level_mix"]  = None
    base_hp["opponent_name"] = "random_legal"
    base_hp.pop("opponent_kwargs", None)
    base_hp["archive_eval_every"]    = 999_999_999
    base_hp["archive_eval_min_pool"] = 999_999_999
    base_hp["replay_per_update"]     = True

    label = f"v12.0.continue-from-{parent[:8]}-20min"
    desc  = (f"Warm-start continuation of run {parent}: 20 min vs random_legal "
             f"on random_close_4_6. Replays on (one per PPO update).")
    budget_ms = 20 * 60 * 1000
    launch_at = int(time.time() * 1000)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runs
              (model_id, simulator_id, project, label, description,
               status, budget_ms, seed, hyperparams, machine, launch_at,
               parent_run_id, is_continuation)
            VALUES
              (%s, %s, %s, %s, %s,
               'queued', %s, %s, %s::jsonb, %s, %s,
               %s, true)
            RETURNING id
            """,
            (
                cfg.model["model_id"], cfg.model["simulator_id"], PROJECT,
                label, desc,
                budget_ms, "continue",
                json.dumps(base_hp), "unassigned", launch_at,
                parent,
            ),
        )
        rid = str(cur.fetchone()[0])
        conn.commit()

    print(f"queued continuation run {rid[:8]}  {label}")
    print(f"  parent: {parent}")
    print(f"  budget: {budget_ms // 60000} min")
    print(f"  level:  random_close_4_6")
    print(f"  opp:    random_legal")
    print(f"  replays: ON")


if __name__ == "__main__":
    main()
