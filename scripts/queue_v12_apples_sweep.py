"""Queue 3-cell entropy_coef sweep on random_close_4_6 — apples-to-apples
match for the v12.0.baseline-RandomLegal-Close4_6-20min run.

Same hyperparams as the karp config except:
  - level_name = random_close_4_6 (matches the baseline's level)
  - level_mix = None (single level)
  - opponent  = random_legal
  - archive_eval disabled (so we measure pure PPO throughput)
  - 5-min cells (matches karp cell budget)

Three cells: entropy_coef ∈ {0.003, 0.01, 0.03}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect
from cli.loop_config import load


CELLS = [
    {"label": "lo",  "value": 0.003},
    {"label": "mid", "value": 0.01},
    {"label": "hi",  "value": 0.03},
]


def main() -> None:
    cfg = load()
    base_hp = dict(cfg.baseline_hyperparams)

    # Single-level overrides + bench parity with the 20-min baseline.
    base_hp["level_name"] = "random_close_4_6"
    base_hp["level_mix"]  = None
    base_hp["opponent_name"] = "random_legal"
    base_hp.pop("opponent_kwargs", None)
    # Disable archive_eval so we isolate PPO throughput.
    base_hp["archive_eval_every"]    = 999_999_999
    base_hp["archive_eval_min_pool"] = 999_999_999

    budget_ms = 5 * 60 * 1000  # 5-min cells
    launch_at = int(time.time() * 1000)

    with connect() as conn, conn.cursor() as cur:
        for cell in CELLS:
            hp = {**base_hp, "entropy_coef": cell["value"]}
            label = f"v12.0.apples-Close4_6-entropy_coef-{cell['label']}"
            desc  = (f"v12.0 apples-to-apples sweep: 5-min entropy_coef={cell['value']} "
                     f"({cell['label']} cell), level=random_close_4_6, opp=random_legal, "
                     f"archive_eval OFF. Pairs with v12.0.baseline-RandomLegal-Close4_6-20min.")
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
                    budget_ms, cell["label"],
                    json.dumps(hp), "unassigned", launch_at,
                ),
            )
            rid = str(cur.fetchone()[0])
            print(f"queued {rid[:8]}  {label}  entropy_coef={cell['value']}")
        conn.commit()

    print(f"queued {len(CELLS)} cells, budget={budget_ms // 60000}min each")


if __name__ == "__main__":
    main()
