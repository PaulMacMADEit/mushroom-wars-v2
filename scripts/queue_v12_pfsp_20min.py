"""Queue a 20-min v12 self-play run that rotates through the PFSP archive
each PPO update. Warm-starts from a parent run.

Mode: opponent_pool_mode=rotate_per_update + leaderboard_bias=0.3
  - Per-update opponent rotation: trainer picks a random PFSP-weighted
    archive member at the start of each PPO update and swaps the
    opponent callable in-place. ~50ms per swap.
  - Fused rollout stays on (compatible with rotate-per-update).
  - leaderboard_bias=0.3 → 30% of envs use a champion, 70% use the
    initial (random_legal) opponent. Mix gives the agent both
    "reliable wins" and "challenging fights."

Usage:
    .venv/bin/python scripts/queue_v12_pfsp_20min.py <parent_run_id>
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
        print("usage: queue_v12_pfsp_20min.py <parent_run_id>")
        sys.exit(1)
    parent = sys.argv[1].strip()

    cfg = load()
    base_hp = dict(cfg.baseline_hyperparams)

    # Level + level mix.
    base_hp["level_name"] = "random_close_4_6"
    base_hp["level_mix"]  = None

    # Self-play via per-update rotation against the PFSP-weighted archive.
    base_hp["opponent_name"]      = "random_legal"   # initial / fallback for non-pool envs
    base_hp.pop("opponent_kwargs", None)
    base_hp["opponent_pool_mode"] = "rotate_per_update"
    base_hp["leaderboard_bias"]   = 0.3              # 30% pool, 70% initial
    base_hp["leaderboard_source"] = "pfsp"
    base_hp["leaderboard_top_k"]  = 10

    # Disable archive_eval (it's been flaky on small archives + we already get
    # rate vs random_legal from the rollout itself).
    base_hp["archive_eval_every"]    = 999_999_999
    base_hp["archive_eval_min_pool"] = 999_999_999

    # Replays on so we can watch the policy adapt mid-run.
    base_hp["replay_per_update"] = True

    label = f"v12.0.pfsp-from-{parent[:8]}-20min"
    desc  = (f"PFSP self-play continuation of {parent}: 20 min, "
             f"rotate-per-update against v12 champion archive (bias=0.3), "
             f"random_close_4_6, replays on.")
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
                budget_ms, "pfsp",
                json.dumps(base_hp), "unassigned", launch_at,
                parent,
            ),
        )
        rid = str(cur.fetchone()[0])
        conn.commit()

    print(f"queued PFSP run {rid[:8]}  {label}")
    print(f"  parent:        {parent}")
    print(f"  budget:        {budget_ms // 60000} min")
    print(f"  level:         random_close_4_6")
    print(f"  pool mode:     rotate_per_update")
    print(f"  pool bias:     0.30")
    print(f"  pool source:   pfsp")
    print(f"  replays:       ON")


if __name__ == "__main__":
    main()
