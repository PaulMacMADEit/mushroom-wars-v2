"""Queue a single 20-minute validation run for sim-v1.3.

Standalone helper to drop a Phase 1 (close-map, K=4, reward_v13=True) run
into the queue for the worker to pick up. Used to verify end-to-end on
PaulLinux before letting the cron-agent fire its first batch.

Usage:
    python scripts/queue_v13_validation.py
    python scripts/queue_v13_validation.py --minutes 20 --level random_close_4_6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--level", default="random_close_4_6")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label-suffix", default="v13-validate")
    ap.add_argument("--model-id", default="v9.0-1024")
    ap.add_argument("--sim-id",   default="sim-v1.3")
    args = ap.parse_args()

    epoch = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    label = f"{epoch}-{args.label_suffix}"

    hp = {
        "n_envs":         1024,
        "rollout_steps":  64,
        "fused_rollout":  True,
        "action_repeat":  4,
        "vec_mode":       "sync",
        "sim_backend":    "jax",
        "level_name":     args.level,
        "reward_v13":     True,
        "gamma":          0.97,
        "opponent_name":  "random_legal",
    }

    launch_at = int(time.time() * 1000)

    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (model_id, simulator_id, project, label, description,
                                  status, budget_ms, seed, hyperparams, machine, launch_at)
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s::jsonb, 'unassigned', %s)
                RETURNING id
                """,
                (
                    args.model_id, args.sim_id, PROJECT,
                    label,
                    f"sim-v1.3 validation: {args.level}, K=4, reward_v13=True",
                    args.minutes * 60 * 1000,
                    str(args.seed),
                    json.dumps(hp),
                    launch_at,
                ),
            )
            rid = str(cur.fetchone()[0])
        c.commit()

    print(f"queued validation run: {rid}  label={label}  minutes={args.minutes}")
    print(f"hyperparams: {json.dumps(hp, indent=2)}")


if __name__ == "__main__":
    main()
