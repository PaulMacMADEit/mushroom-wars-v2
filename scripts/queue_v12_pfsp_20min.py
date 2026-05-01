"""Queue a 20-min v12 self-play run that rotates through the v12 champion
archive each PPO update. Warm-starts from a parent run.

PURE self-play: no random_legal. Initial opponent = most-recent v12 champion;
trainer rotates each update to a different PFSP-weighted archive member.
With fused_rollout=true, ALL envs use the rotating opponent on every update.

Usage:
    .venv/bin/python scripts/queue_v12_selfplay_20min.py <parent_run_id>
        # back-compat alias also accepted:
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


def _resolve_initial_opponent(cur) -> tuple[str, dict] | None:
    """Pick the most-recent v12 champion as the initial opponent for the
    rotating self-play loop. Returns (opp_name, opp_kwargs) or None if the
    archive has no v12 entries yet (caller should error out — pure self-play
    requires an existing champion).
    """
    cur.execute("""
        SELECT c.source_run_id, c.label
          FROM champions c
         WHERE c.arch_era = 'v12'
         ORDER BY c.archived_at DESC
         LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        return None
    run_id = str(row[0])
    label  = row[1]
    print(f"[queue] initial opponent: {label}  ({run_id[:8]})")
    return "neural", {"device": "cuda", "opponent_run_id": run_id}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: queue_v12_selfplay_20min.py <parent_run_id>")
        sys.exit(1)
    parent = sys.argv[1].strip()

    cfg = load()
    base_hp = dict(cfg.baseline_hyperparams)

    base_hp["level_name"] = "random_close_4_6"
    base_hp["level_mix"]  = None

    with connect() as conn, conn.cursor() as cur:
        opp = _resolve_initial_opponent(cur)
    if opp is None:
        print("[queue] ERROR: no v12 champions in archive — can't seed self-play.")
        sys.exit(2)
    opp_name, opp_kwargs = opp

    base_hp["opponent_name"]      = opp_name
    base_hp["opponent_kwargs"]    = opp_kwargs
    base_hp["opponent_pool_mode"] = "rotate_per_update"
    base_hp["leaderboard_bias"]   = 1.0   # gates archive download; rotate
                                          # mode supersedes per-env mixing.
    base_hp["leaderboard_source"] = "pfsp"
    base_hp["leaderboard_top_k"]  = 10

    base_hp["archive_eval_every"]    = 999_999_999
    base_hp["archive_eval_min_pool"] = 999_999_999
    base_hp["replay_per_update"]      = True
    base_hp["replay_games_per_update"] = 3

    label = f"v12.0.selfplay-from-{parent[:8]}-20min"
    desc  = (f"Pure v12 self-play continuation of {parent}: 20 min, "
             f"rotate-per-update against v12 champion archive, no random_legal, "
             f"random_close_4_6, replays on. v1.6 rewards.")
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
                budget_ms, "selfplay",
                json.dumps(base_hp), "unassigned", launch_at,
                parent,
            ),
        )
        rid = str(cur.fetchone()[0])
        conn.commit()

    print(f"queued self-play run {rid[:8]}  {label}")
    print(f"  parent:        {parent}")
    print(f"  budget:        {budget_ms // 60000} min")
    print(f"  level:         random_close_4_6")
    print(f"  pool mode:     rotate_per_update")
    print(f"  pool source:   pfsp")
    print(f"  initial opp:   {opp_kwargs.get('opponent_run_id', '')[:8]}")
    print(f"  reward:        v1.6 (LOSE -7.5, DRAW -1.25)")
    print(f"  replays:       ON")


if __name__ == "__main__":
    main()
