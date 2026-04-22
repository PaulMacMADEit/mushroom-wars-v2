"""Register the current sim into Supabase `simulators`.

Reads git SHA + current config constants, packages a `features` + `benchmark`
blob, INSERTs with ON CONFLICT DO NOTHING.

Usage:
    python cli/register_sim.py --id sim-v1.0 --what-changed "initial Python port"
    python cli/register_sim.py --id sim-v1.0 --force  # overwrite existing row
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect
from sim import config as C


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def features_blob() -> dict:
    return {
        "max_building_slots":       C.MAX_BUILDING_SLOTS,
        "max_unit_group_slots":     C.MAX_UNIT_GROUP_SLOTS,
        "decision_interval_ticks":  C.DECISION_INTERVAL_TICKS,
        "game_timeout_ticks":       C.GAME_TIMEOUT_TICKS,
        "send_percentages":         list(C.SEND_PERCENTAGES),
        "def_bonus":                [C.DEF_BONUS_NUM, C.DEF_BONUS_DEN],
        "levels":                   ["crossroads_6"],
        "reward": {
            "capture": C.REWARD_CAPTURE,
            "loss":    C.REWARD_LOSS,
            "win":     C.REWARD_WIN,
            "lose":    C.REWARD_LOSE,
            "draw":    C.REWARD_DRAW,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="sim-v1.0")
    ap.add_argument("--name", default="Python sim v1.0 (initial port)")
    ap.add_argument("--what-changed", default="Initial Python port of the TS sim; see ARCHITECTURE §4.")
    ap.add_argument("--parent-sim", default=None)
    ap.add_argument("--benchmark", default=None,
                    help='JSON dict of benchmark results (optional)')
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing row with same id")
    args = ap.parse_args()

    sha = git_sha()
    features = features_blob()
    benchmark = json.loads(args.benchmark) if args.benchmark else None

    with connect() as conn:
        with conn.cursor() as cur:
            if args.force:
                cur.execute("""
                    INSERT INTO simulators (id, project, name, parent_sim, what_changed,
                                            git_sha, features, benchmark)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        name         = EXCLUDED.name,
                        parent_sim   = EXCLUDED.parent_sim,
                        what_changed = EXCLUDED.what_changed,
                        git_sha      = EXCLUDED.git_sha,
                        features     = EXCLUDED.features,
                        benchmark    = EXCLUDED.benchmark
                """, (args.id, PROJECT, args.name, args.parent_sim, args.what_changed,
                      sha, json.dumps(features),
                      json.dumps(benchmark) if benchmark else None))
            else:
                cur.execute("""
                    INSERT INTO simulators (id, project, name, parent_sim, what_changed,
                                            git_sha, features, benchmark)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (id) DO NOTHING
                """, (args.id, PROJECT, args.name, args.parent_sim, args.what_changed,
                      sha, json.dumps(features),
                      json.dumps(benchmark) if benchmark else None))
            affected = cur.rowcount
        conn.commit()

    if affected == 0 and not args.force:
        print(f"simulator {args.id!r} already exists (use --force to overwrite).")
    else:
        print(f"registered simulator {args.id!r} @ {sha[:8]}")


if __name__ == "__main__":
    main()
