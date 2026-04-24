"""
Level definitions.

Two flavours:
  1. Named static levels (LEVELS dict) — reproducible, one geometry per name.
  2. Dynamic random levels — name pattern `random_<min>_<max>` picks a random
     count N ∈ [min, max] and generates a 180°-symmetric board per reset.

Design constraints (apply to both):
- 180° rotational symmetry so self-play is fair.
- Coordinate space: 0-700 "map units"; renderer scales to pixels.
- Bases at 10 real units; neutral garrisons in [1, 5].
- Minimum 80-unit separation between buildings so travel times are meaningful.
"""

from __future__ import annotations

import re

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


# ---------------------------------------------------------------------------
# Dynamic random level generator
# ---------------------------------------------------------------------------

_RANDOM_RE = re.compile(r"^random_(\d+)_(\d+)$")
_ASYM_RE   = re.compile(r"^asym_(\d+)_(\d+)$")

# Placement constraints.
_MAP_SIZE       = 700
_BORDER         = 80
_MIN_SEP        = 80       # min distance between any two buildings
_CENTER_EXCLUSION = 50      # neutrals mustn't sit on the symmetry axis (too close to center)


def generate_random_level(n_buildings: int, rng: np.random.Generator) -> list:
    """Build a 180°-symmetric level with `n_buildings` total (2 bases + rest neutral).

    Returns the same list-of-tuples shape that LEVELS uses, so `apply()` can
    consume it directly.
    """
    if n_buildings < 2:
        raise ValueError("need at least 2 buildings (one base per player)")
    if n_buildings > C.MAX_BUILDING_SLOTS:
        raise ValueError(f"n_buildings={n_buildings} > MAX_BUILDING_SLOTS={C.MAX_BUILDING_SLOTS}")

    level: list = []
    placed: list[tuple[int, int]] = []

    # Bases: slot 0 = P1, slot 1 = P2, mirror of each other. Keep them well
    # inside the corners so neutrals have room to land between.
    bx = int(rng.integers(_BORDER, _MAP_SIZE // 2 - 100))
    by = int(rng.integers(_BORDER, _MAP_SIZE // 2 - 100))
    level.append((C.OWNER_P1, bx, by, 10, C.TYPE_BASIC))
    level.append((C.OWNER_P2, _MAP_SIZE - bx, _MAP_SIZE - by, 10, C.TYPE_BASIC))
    placed.extend([(bx, by), (_MAP_SIZE - bx, _MAP_SIZE - by)])

    # Neutrals: place in mirror pairs. If N-2 is odd, add one center-ish neutral.
    n_neutrals = n_buildings - 2
    pairs = n_neutrals // 2
    has_center = (n_neutrals % 2 == 1)

    for _ in range(pairs):
        nx = ny = None
        for _attempt in range(50):
            cx = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
            cy = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
            # Keep off the symmetry axis so the mirror is a genuinely different slot.
            if (cx - _MAP_SIZE // 2) ** 2 + (cy - _MAP_SIZE // 2) ** 2 < _CENTER_EXCLUSION ** 2:
                continue
            mx, my = _MAP_SIZE - cx, _MAP_SIZE - cy
            # Minimum separation to every existing building AND its mirror.
            if not all((cx - px) ** 2 + (cy - py) ** 2 >= _MIN_SEP ** 2 for px, py in placed):
                continue
            if not all((mx - px) ** 2 + (my - py) ** 2 >= _MIN_SEP ** 2 for px, py in placed):
                continue
            nx, ny = cx, cy
            break
        if nx is None:
            # Give up placing — return whatever we've got. Caller will retry
            # with a different N or seed.
            break
        garrison = int(rng.integers(1, 6))
        level.append((C.OWNER_NEUTRAL, nx, ny, garrison, C.TYPE_BASIC))
        level.append((C.OWNER_NEUTRAL, _MAP_SIZE - nx, _MAP_SIZE - ny, garrison, C.TYPE_BASIC))
        placed.extend([(nx, ny), (_MAP_SIZE - nx, _MAP_SIZE - ny)])

    if has_center:
        garrison = int(rng.integers(2, 6))
        level.append((C.OWNER_NEUTRAL, _MAP_SIZE // 2, _MAP_SIZE // 2, garrison, C.TYPE_BASIC))

    return level


def generate_asymmetric_level(n_buildings: int, rng: np.random.Generator) -> list:
    """Build an asymmetric random level — no mirroring.

    Bases are placed in opposite halves of the map (left/right) so both
    players have *some* starting region, but distances, counts of nearby
    neutrals, and geometry are independent. May produce unfair starts —
    that's the point (use for variety / stress-testing, not fair eval).
    """
    if n_buildings < 2:
        raise ValueError("need at least 2 buildings (one base per player)")
    if n_buildings > C.MAX_BUILDING_SLOTS:
        raise ValueError(f"n_buildings={n_buildings} > MAX_BUILDING_SLOTS={C.MAX_BUILDING_SLOTS}")

    level: list = []
    placed: list[tuple[int, int]] = []

    # Bases in opposite halves. P1 gets the left half, P2 the right.
    mid = _MAP_SIZE // 2
    p1x = int(rng.integers(_BORDER, mid - 40))
    p1y = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
    p2x = int(rng.integers(mid + 40, _MAP_SIZE - _BORDER))
    p2y = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
    level.append((C.OWNER_P1, p1x, p1y, 10, C.TYPE_BASIC))
    level.append((C.OWNER_P2, p2x, p2y, 10, C.TYPE_BASIC))
    placed.extend([(p1x, p1y), (p2x, p2y)])

    # Neutrals scattered anywhere. No mirror pairs, no center exclusion.
    n_neutrals = n_buildings - 2
    for _ in range(n_neutrals):
        for _attempt in range(80):
            cx = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
            cy = int(rng.integers(_BORDER, _MAP_SIZE - _BORDER))
            if all((cx - px) ** 2 + (cy - py) ** 2 >= _MIN_SEP ** 2 for px, py in placed):
                garrison = int(rng.integers(1, 6))
                level.append((C.OWNER_NEUTRAL, cx, cy, garrison, C.TYPE_BASIC))
                placed.append((cx, cy))
                break
        # If we fail to place this neutral, skip it — caller gets a level
        # with fewer buildings than requested, which is still valid.

    return level


def _resolve_level(level_name: str, seed: int | None) -> list:
    """Look up a static name or generate a random level. Returns the level list."""
    if level_name in LEVELS:
        return LEVELS[level_name]
    m = _RANDOM_RE.match(level_name)
    if m:
        n_min, n_max = int(m.group(1)), int(m.group(2))
        if not (2 <= n_min <= n_max <= C.MAX_BUILDING_SLOTS):
            raise ValueError(f"random level bounds out of range: {level_name!r}")
        rng = np.random.default_rng(seed)
        n = int(rng.integers(n_min, n_max + 1))
        return generate_random_level(n, rng)
    m = _ASYM_RE.match(level_name)
    if m:
        n_min, n_max = int(m.group(1)), int(m.group(2))
        if not (2 <= n_min <= n_max <= C.MAX_BUILDING_SLOTS):
            raise ValueError(f"asym level bounds out of range: {level_name!r}")
        rng = np.random.default_rng(seed)
        n = int(rng.integers(n_min, n_max + 1))
        return generate_asymmetric_level(n, rng)
    raise ValueError(f"Unknown level: {level_name}")


def apply(state: State, level_name: str = "crossroads_6", seed: int | None = None) -> None:
    """Populate `state.buildings` from a level definition. Runs precompute_distances().

    For dynamic level names like `random_8_32`, the optional `seed` drives the
    per-episode PRNG. Passing None picks fresh entropy each call.
    """
    level = _resolve_level(level_name, seed)
    if len(level) > C.MAX_BUILDING_SLOTS:
        raise ValueError(f"Level has {len(level)} buildings; max is {C.MAX_BUILDING_SLOTS}")

    state.buildings_alive[:]    = 0
    state.buildings_owner[:]    = 0
    state.buildings_type[:]     = 0
    state.buildings_garrison[:] = 0
    state.buildings_capacity[:] = 0
    state.buildings_x[:]        = 0
    state.buildings_y[:]        = 0
    state.groups_alive[:]    = 0
    state.groups_owner[:]    = 0
    state.groups_src[:]      = 0
    state.groups_tgt[:]      = 0
    state.groups_count[:]    = 0
    state.groups_progress[:] = 0
    state.groups_travel[:]   = 0

    for slot, (owner, x, y, garrison_real, type_id) in enumerate(level):
        if type_id not in C.BUILDING_STATS:
            raise ValueError(f"Unknown building type_id: {type_id}")
        stats = C.BUILDING_STATS[type_id]
        state.buildings_alive[slot]    = 1
        state.buildings_owner[slot]    = owner
        state.buildings_type[slot]     = type_id
        state.buildings_garrison[slot] = garrison_real * C.SCALE
        state.buildings_capacity[slot] = stats["capacity"]
        state.buildings_x[slot]        = x
        state.buildings_y[slot]        = y

    # Slots len(level)..MAX are left as alive=0 (empty).
    state.tick = 0
    state.phase = C.PHASE_PLAYING
    for key in state.perf:
        state.perf[key] = 0
    precompute_distances(state)


def reset(level_name: str = "crossroads_6", seed: int | None = 0) -> State:
    """Fresh state ready to step. `seed` drives dynamic level generation for
    names like `random_N_M`; for static names it's ignored."""
    from sim.state import empty_state
    state = empty_state()
    apply(state, level_name, seed=seed)
    return state
