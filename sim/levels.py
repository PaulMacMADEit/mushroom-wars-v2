"""
Level definitions. v0.1: one hand-crafted 6-building symmetric map.

Design constraints:
- 180° rotational symmetry for fair self-play.
- Corner-to-corner travel = 8 ticks (the MAX_TRAVEL_TICKS cap).
- Neutral garrisons vary (1-5 real units) to give the agent interesting decisions.
- Starting bases at 10 real units. All buildings share identical stats
  (capacity, production rate) per v0.1 scope.

Coordinate space: 0-700 "map units". Renderer scales to pixels.
"""

from __future__ import annotations

import numpy as np

from sim import config as C
from sim.state import State, precompute_distances


# Each building: (owner, x, y, garrison_real, type_id)
# garrison stored in real units here for readability; apply() scales to internal.
_CROSSROADS_6 = [
    # P1 and P2 bases (opposite corners)
    (C.OWNER_P1, 100, 100, 10, C.TYPE_BASIC),
    (C.OWNER_P2, 600, 600, 10, C.TYPE_BASIC),
    # Neutrals — symmetric pairs, varied garrison
    (C.OWNER_NEUTRAL, 350, 100, 1, C.TYPE_BASIC),   # N1 top
    (C.OWNER_NEUTRAL, 350, 600, 1, C.TYPE_BASIC),   # N2 bottom (mirror of N1)
    (C.OWNER_NEUTRAL, 100, 350, 5, C.TYPE_BASIC),   # N3 left
    (C.OWNER_NEUTRAL, 600, 350, 5, C.TYPE_BASIC),   # N4 right (mirror of N3)
]


LEVELS: dict[str, list] = {
    "crossroads_6": _CROSSROADS_6,
}


def apply(state: State, level_name: str = "crossroads_6") -> None:
    """Populate `state.buildings` from a level definition. Runs precompute_distances()."""
    if level_name not in LEVELS:
        raise ValueError(f"Unknown level: {level_name}")
    level = LEVELS[level_name]
    if len(level) > C.MAX_BUILDING_SLOTS:
        raise ValueError(f"Level has {len(level)} buildings; max is {C.MAX_BUILDING_SLOTS}")

    b = state.buildings
    g = state.unit_groups
    b[:] = 0  # clear all slots
    g[:] = 0  # clear all in-flight groups

    for slot, (owner, x, y, garrison_real, type_id) in enumerate(level):
        if type_id not in C.BUILDING_STATS:
            raise ValueError(f"Unknown building type_id: {type_id}")
        stats = C.BUILDING_STATS[type_id]
        b[slot]["alive"]    = 1
        b[slot]["owner"]    = owner
        b[slot]["type_id"]  = type_id
        b[slot]["garrison"] = garrison_real * C.SCALE
        b[slot]["capacity"] = stats["capacity"]
        b[slot]["x"]        = x
        b[slot]["y"]        = y

    # Slots len(level)..MAX are left as alive=0 (empty).
    state.tick = 0
    state.phase = C.PHASE_PLAYING
    for key in state.perf:
        state.perf[key] = 0
    precompute_distances(state)


def reset(level_name: str = "crossroads_6", seed: int = 0) -> State:
    """Fresh state ready to step. seed is currently only used by callers via
    np.random; kept in the signature for callers that pass it through."""
    from sim.state import empty_state
    state = empty_state()
    apply(state, level_name)
    return state
