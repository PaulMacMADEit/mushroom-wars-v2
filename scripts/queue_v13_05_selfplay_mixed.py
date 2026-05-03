"""Queue a v13 self-play run with v12+v13 cross-lineage opponent mix.

Cross-era is enabled by the adapter shipped in commit 51511f7
("v13: chain reorder + head MLPs; backward-compatible with v12"):
- same OBS_DIM=192, same env action space (129)
- training/nets registry instantiates the right ActorCritic per
  net_version stamp; checkpoints from v12 load via the v12 class
- workers/worker.py:_download_pfsp_champions is era-agnostic; it
  pulls the newest top_k champions by archived_at across all eras.
  bench_eval IS era-locked (workers/bench_eval.py), but the trainer's
  PFSP archive download is not.

Usage:
    python scripts/queue_v13_05_selfplay_mixed.py <parent_run_id> [--minutes 15]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_run_id")
    ap.add_argument("--minutes", type=int, default=15)
    ap.add_argument("--level",   type=str, default="random_close_4_8")
    args = ap.parse_args()
    parent  = args.parent_run_id.strip()
    minutes = max(1, int(args.minutes))
    level   = args.level.strip()

    hp = {
        "level_name":              level,
        "n_envs":                  32,
        "vec_mode":                "async",
        "self_play":               True,
        "sim_backend":             "numpy",     # forced by self_play=True
        "fused_rollout":           False,       # forced by self_play=True
        "latest_bias":             0.6,         # 60% latest own snapshot
        "leaderboard_bias":        0.5,         # 50% cross-lineage archive
        "leaderboard_source":      "pfsp",
        "leaderboard_top_k":       20,
        "leaderboard_recency_decay": 1.06,      # 3^(1/19), oldest ~3x newest
        "snapshot_every":          10,
        "archive_eval_every":      100_000,
    }

    label = "v13.0.5-selfplay-mixed"
    desc  = (
        f"v13 self-play with v12+v13 cross-lineage mix. parent={parent[:8]}, "
        f"{minutes} min on {level}. self_play=True, latest_bias=0.6, "
        f"leaderboard_bias=0.5, top_k=20, recency_decay=1.06."
    )

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status, weights_url, model_id FROM runs WHERE id = %s",
            (parent,),
        )
        row = cur.fetchone()
        if not row:
            sys.exit(f"parent {parent} not found")
        p_status, p_w, p_model = row
        if p_status != "done":
            sys.exit(f"parent status={p_status} (need 'done')")
        if not p_w:
            sys.exit("parent has no weights_url")

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
                p_model, "sim-v1.4", PROJECT,
                label, desc,
                minutes * 60 * 1000, "selfplay-mixed",
                json.dumps(hp), "PaulLinux", int(time.time() * 1000),
                parent,
            ),
        )
        rid = str(cur.fetchone()[0])
        conn.commit()

    print(f"queued: {rid[:8]}  {label}")
    print(f"  parent:  {parent[:8]}")
    print(f"  budget:  {minutes} min on {level}")
    print(f"  archive: top_k=20 cross-era (v12 + v13 via 51511f7 adapter)")


if __name__ == "__main__":
    main()
