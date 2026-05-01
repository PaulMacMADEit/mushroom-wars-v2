"""Queue a single 20-min v12 baseline run vs random_legal on random_close_4_6.

Reads baseline hyperparams from configs/karpathy_loop.yaml (so we stay in sync
with the karp loop). Overrides:
  - level_name = random_close_4_6 (small maps)
  - level_mix  = None              (single level, not a mix)
  - opponent   = random_legal
  - budget     = 1200000 ms (20 min)
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
    cfg = load()
    base_hp = dict(cfg.baseline_hyperparams)

    # Single-level overrides
    base_hp["level_name"] = "random_close_4_6"
    base_hp["level_mix"]  = None
    base_hp["opponent_name"] = "random_legal"
    base_hp.pop("opponent_kwargs", None)

    label = "v12.0.baseline-RandomLegal-Close4_6-20min"
    desc  = "v12.0 baseline: 20 min vs random_legal on random_close_4_6 (small maps). Single run, no sweep."
    budget_ms = 20 * 60 * 1000  # 1200000 ms = 20 min
    launch_at = int(time.time() * 1000)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runs
              (model_id, simulator_id, project, label, description,
               status, budget_ms, seed, hyperparams, machine, launch_at)
            VALUES
              (%s, %s, %s, %s, %s,
               'queued', %s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (
                cfg.model["model_id"], cfg.model["simulator_id"], PROJECT,
                label, desc,
                budget_ms, "baseline",
                json.dumps(base_hp), "unassigned", launch_at,
            ),
        )
        rid = str(cur.fetchone()[0])
        conn.commit()

    print(f"queued baseline run {rid[:8]}  {label}")
    print(f"  budget: {budget_ms // 60000} min")
    print(f"  level:  random_close_4_6")
    print(f"  opp:    random_legal")
    print(f"  model:  {cfg.model['model_id']}")
    print(f"  sim:    {cfg.model['simulator_id']}")


if __name__ == "__main__":
    main()
