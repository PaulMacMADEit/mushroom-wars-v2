"""
State container for one game.

Storage philosophy:
- Fixed-size structured numpy arrays (not Python objects). Numba-ready,
  vec-env-friendly, fits in L1 cache.
- All quantities integer. Floats only during distance precomputation (once at reset()).
- Capacity pre-allocated at MAX_BUILDING_SLOTS / MAX_UNIT_GROUP_SLOTS so the
  neural observation shape never changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim import config as C


# ---------------------------------------------------------------------------
# Structured dtypes
# ---------------------------------------------------------------------------

# Smallest sensible types per field. See ARCHITECTURE §8.1 + discussion.
BUILDING_DTYPE = np.dtype([
    ("alive",         np.int8),    # 0 = empty slot, 1 = in-play
    ("owner",         np.int8),    # 0=neutral, 1=P1, 2=P2
    ("type_id",       np.int8),    # 0=BASIC in v0.1; reserved for more types
    ("level",         np.int8),    # 0 in v0.1; reserved for upgrades
    ("garrison",      np.int16),   # fixed-point, scale=10 (so 35 = 3.5 real)
    ("capacity",      np.int16),   # fixed-point, scale=10
    ("prod_timer",    np.int16),   # ticks since last unit produced (unused in v0.1)
    ("upgrade_timer", np.int16),   # ticks remaining on upgrade (0 in v0.1)
    ("x",             np.int16),   # map units
    ("y",             np.int16),   # map units
])

UNIT_GROUP_DTYPE = np.dtype([
    ("alive",         np.int8),    # 0 = slot free
    ("owner",         np.int8),    # 1=P1, 2=P2 (groups are never neutral)
    ("src_slot",      np.int8),    # index into buildings
    ("tgt_slot",      np.int8),
    ("count",         np.int16),   # fixed-point units in flight
    ("progress",      np.int16),   # ticks elapsed since spawn
    ("travel_ticks",  np.int16),   # ticks needed to arrive
])


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class State:
    """One game's full state. All fields are either numpy arrays or small scalars."""

    buildings:   np.ndarray    # (MAX_BUILDING_SLOTS,) BUILDING_DTYPE
    unit_groups: np.ndarray    # (MAX_UNIT_GROUP_SLOTS,) UNIT_GROUP_DTYPE

    # Precomputed once in reset(). Read-only during play.
    travel_matrix:   np.ndarray  # (MAX_BUILDING_SLOTS, MAX_BUILDING_SLOTS) int16 ticks
    distance_matrix: np.ndarray  # (MAX_BUILDING_SLOTS, MAX_BUILDING_SLOTS) float32 raw

    # Scalars
    tick:          int = 0
    phase:         int = C.PHASE_PLAYING
    seed:          int = 0

    # Lightweight per-subsystem profiling (nanoseconds accumulated this game).
    # Always on; cost ~200 ns per phase per tick. Zeroed at reset.
    perf: dict = field(default_factory=lambda: {
        "production_ns": 0,
        "movement_ns":   0,
        "combat_ns":     0,
        "actions_ns":    0,
        "victory_ns":    0,
        "total_ns":      0,
        "n_ticks":       0,
    })


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def empty_state(seed: int = 0) -> State:
    """Zero-initialized state. Use `levels.apply(state, level)` to populate."""
    buildings   = np.zeros(C.MAX_BUILDING_SLOTS,   dtype=BUILDING_DTYPE)
    unit_groups = np.zeros(C.MAX_UNIT_GROUP_SLOTS, dtype=UNIT_GROUP_DTYPE)
    travel    = np.zeros((C.MAX_BUILDING_SLOTS, C.MAX_BUILDING_SLOTS), dtype=np.int16)
    distance  = np.zeros((C.MAX_BUILDING_SLOTS, C.MAX_BUILDING_SLOTS), dtype=np.float32)
    return State(
        buildings=buildings,
        unit_groups=unit_groups,
        travel_matrix=travel,
        distance_matrix=distance,
        seed=seed,
    )


def precompute_distances(state: State) -> None:
    """Fill distance_matrix + travel_matrix for all alive building pairs.

    Called once from reset() after a level is applied. Distances never change
    during a game (buildings don't move), so this is pure setup cost.
    """
    b = state.buildings
    alive = b["alive"].astype(bool)

    xs = b["x"].astype(np.float32)
    ys = b["y"].astype(np.float32)

    # Broadcast to (N, N) distances. float32 is fine — this runs once.
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    # Travel ticks: ceil(distance / speed), clamped to [MIN, MAX].
    travel = np.ceil(dist / C.TRAVEL_SPEED).astype(np.int32)
    travel = np.clip(travel, C.MIN_TRAVEL_TICKS, C.MAX_TRAVEL_TICKS)

    # Zero out rows/cols for dead slots and self-pairs (i == j).
    mask_2d = alive[:, None] & alive[None, :]
    travel = np.where(mask_2d, travel, 0)
    np.fill_diagonal(travel, 0)
    np.fill_diagonal(dist, 0.0)

    state.travel_matrix[:] = travel.astype(np.int16)
    state.distance_matrix[:] = dist


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def count_owned_buildings(state: State, owner: int) -> int:
    b = state.buildings
    return int(np.sum((b["alive"] == 1) & (b["owner"] == owner)))


def count_owned_units(state: State, owner: int) -> int:
    """Total units (garrison + in-flight) for a player, in internal scale."""
    b = state.buildings
    g = state.unit_groups
    in_garrison = int(np.sum(np.where((b["alive"] == 1) & (b["owner"] == owner),
                                      b["garrison"], 0)))
    in_flight   = int(np.sum(np.where((g["alive"] == 1) & (g["owner"] == owner),
                                      g["count"], 0)))
    return in_garrison + in_flight


def has_in_flight(state: State, owner: int) -> bool:
    g = state.unit_groups
    return bool(np.any((g["alive"] == 1) & (g["owner"] == owner)))
