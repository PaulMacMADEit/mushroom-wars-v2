#!/usr/bin/env python
"""Record one game to a JSON event log for the browser replay viewer.

This is the smoke-test entry point for the replay pipeline. v0 uses random
legal actions on both sides so it's self-contained — no weights, no network.
Swap in PPO agents later by following `workers/match_runner._play_one_game`.

    python scripts/record_game.py --out /tmp/game.json --seed 42
    open dashboard/game.html   # drag the JSON onto the page
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import config as C
from sim import levels
from sim.actions import Action, compute_mask, decode
from sim.engine import step_tick
from sim.envs.replay import Recorder


def _sample_legal_action(state, player: int, rng: np.random.Generator) -> Action:
    mask = compute_mask(state, player)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return Action(kind="noop")
    return decode(int(rng.choice(valid_idx)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",   required=True, help="Where to write events.json")
    ap.add_argument("--level", default="crossroads_6")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--max-ticks", type=int, default=C.GAME_TIMEOUT_TICKS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    state = levels.reset(args.level, seed=args.seed)

    recorder = Recorder(
        game_id=str(uuid.uuid4()),
        level_name=args.level,
        seed=args.seed,
    )
    recorder.capture_map(state)

    decide_every = C.DECISION_INTERVAL_TICKS
    for tick in range(args.max_ticks):
        a1 = a2 = None
        if tick % decide_every == 0:
            a1 = _sample_legal_action(state, C.OWNER_P1, rng)
            a2 = _sample_legal_action(state, C.OWNER_P2, rng)

        buf = recorder.get_tick_events_buffer()
        _, _, done = step_tick(state, a1, a2, events=buf)
        recorder.absorb_tick(state)
        if done:
            break

    out_path = recorder.write_json(args.out)
    data = recorder.to_dict()
    print(
        f"wrote {out_path} "
        f"({out_path.stat().st_size:,} bytes, "
        f"{len(data['events'])} events, "
        f"tick={state.tick}, winner={data['winner']})"
    )


if __name__ == "__main__":
    main()
