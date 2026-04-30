"""
State container for one game.

Storage philosophy:
- Parallel (struct-of-arrays) numpy ndarrays, not array-of-structs. Each field is
  a contiguous 1-D ndarray of shape `(MAX_BUILDING_SLOTS,)` or `(MAX_UNIT_GROUP_SLOTS,)`.
  This layout is what a JAX pytree wants and what XLA can vmap cleanly.
- Fixed capacity: MAX_BUILDING_SLOTS / MAX_UNIT_GROUP_SLOTS so the neural
  observation shape never changes and every batched state has the same shape.
- All quantities integer. Floats only during distance precomputation (once at reset()).
- Structured-array-style access (`state.buildings["owner"]`, `state.unit_groups["count"]`)
  is preserved via read/write proxies that forward to the underlying parallel ndarrays.
  Hot-path code should prefer `state.buildings_owner` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim import config as C


# ---------------------------------------------------------------------------
# Back-compat proxies (structured-array-style views over parallel ndarrays)
# ---------------------------------------------------------------------------

class _ArrayProxy:
    """Dict-of-ndarrays with structured-array ergonomics.

    - `proxy["owner"]` returns the owner ndarray (read+write).
    - `proxy[i]` returns a row view for one slot (scalar integer index).
    - `proxy[bool_mask]` / `proxy[slice]` returns a sub-proxy with each field
      sliced — so `len(proxy[mask])` counts selected rows.
    - `proxy[:] = 0` broadcasts the assignment to every field.
    - Equality is fieldwise: two proxies with the same fields compare equal
      iff all underlying ndarrays compare equal. This lets
      `np.array_equal(s1.buildings, s2.buildings)` work in tests.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: dict[str, np.ndarray]):
        object.__setattr__(self, "_fields", fields)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._fields[key]
        # Scalar integer → row view (one slot).
        if isinstance(key, (int, np.integer)):
            return _RowView(self._fields, int(key))
        # slice / ndarray / list → sub-proxy with each field sliced.
        return _ArrayProxy({name: arr[key] for name, arr in self._fields.items()})

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._fields[key][:] = value
            return
        # `proxy[:] = 0` — broadcast the assignment to every field.
        for arr in self._fields.values():
            arr[key] = value

    def __len__(self):
        return next(iter(self._fields.values())).shape[0]

    def __iter__(self):
        n = len(self)
        for i in range(n):
            yield _RowView(self._fields, i)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _ArrayProxy):
            return NotImplemented
        if set(self._fields.keys()) != set(other._fields.keys()):
            return False
        return all(
            np.array_equal(self._fields[k], other._fields[k])
            for k in self._fields
        )

    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None):
        """Assemble a structured ndarray from the parallel fields.

        Called by `np.array_equal(proxy1, proxy2)` and `np.asarray(proxy)`.
        Structured-array equality is fieldwise, which matches the proxy's
        semantics. Returning a copy here is fine — this path is test-only.
        """
        dt = np.dtype([(name, arr.dtype) for name, arr in self._fields.items()])
        out = np.empty(len(self), dtype=dt)
        for name, arr in self._fields.items():
            out[name] = arr
        return out if dtype is None else out.astype(dtype)


class _RowView:
    """One-row view of the parallel ndarrays. Read/write flows through."""

    __slots__ = ("_fields", "_idx")

    def __init__(self, fields: dict[str, np.ndarray], idx):
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_idx", idx)

    def __getitem__(self, name: str):
        return self._fields[name][self._idx]

    def __setitem__(self, name: str, value) -> None:
        self._fields[name][self._idx] = value


# ---------------------------------------------------------------------------
# State dataclass — parallel ndarrays for buildings + unit groups
# ---------------------------------------------------------------------------

@dataclass
class State:
    """One game's full state. Every field is a numpy ndarray or small scalar."""

    # Buildings (parallel ndarrays, all length MAX_BUILDING_SLOTS)
    buildings_alive:    np.ndarray   # int8
    buildings_owner:    np.ndarray   # int8
    buildings_type:     np.ndarray   # int8
    buildings_garrison: np.ndarray   # int16
    buildings_capacity: np.ndarray   # int16
    buildings_x:        np.ndarray   # int16
    buildings_y:        np.ndarray   # int16

    # Unit groups (parallel ndarrays, all length MAX_UNIT_GROUP_SLOTS)
    groups_alive:    np.ndarray     # int8
    groups_owner:    np.ndarray     # int8
    groups_src:      np.ndarray     # int8
    groups_tgt:      np.ndarray     # int8
    groups_count:    np.ndarray     # int16
    groups_progress: np.ndarray     # int16
    groups_travel:   np.ndarray     # int16

    # v10 decision-interval bookkeeping. arrivals_* updated by the engine
    # every tick; prev_* and last_actions_* snapshotted by the env at the
    # start of each decision interval. Encoder reads them to surface
    # "you were attacked" / "tower flipped" / "your action history" —
    # signals otherwise invisible when travel < decision interval. Mirrored
    # P1↔P2 by the opponent path, same as buildings_owner.
    arrivals_p1:           np.ndarray   # (N,)        int16 — landings per bldg, P1
    arrivals_p2:           np.ndarray   # (N,)        int16 — landings per bldg, P2
    prev_buildings_owner:  np.ndarray   # (N,)        int8  — owner snapshot at interval start
    last_actions_p1:       np.ndarray   # (HISTORY_K, 4) int8 — [kind, type_idx, src, tgt], idx 0 = newest
    last_actions_p2:       np.ndarray   # (HISTORY_K, 4) int8

    # Precomputed once in reset(). Read-only during play.
    travel_matrix:   np.ndarray  # (MAX_BUILDING_SLOTS, MAX_BUILDING_SLOTS) int16 ticks
    distance_matrix: np.ndarray  # (MAX_BUILDING_SLOTS, MAX_BUILDING_SLOTS) float32 raw

    # Scalars
    tick:          int = 0
    phase:         int = C.PHASE_PLAYING
    # Reward scheme. 0 = v1.2 (default, back-compat), 1 = v1.3 (rebalance).
    # Engine indexes into REWARD_*_BY_VERSION lookups via this field.
    reward_version: int = C.REWARD_VERSION_V12
    # v10: snapshots of total units per side at the start of the current
    # decision interval. Used by the encoder for the reward_delta feature.
    prev_p1_units_total: int = 0
    prev_p2_units_total: int = 0

    # Lightweight per-subsystem profiling (nanoseconds accumulated this game).
    perf: dict = field(default_factory=lambda: {
        "production_ns": 0,
        "movement_ns":   0,
        "combat_ns":     0,
        "actions_ns":    0,
        "victory_ns":    0,
        "total_ns":      0,
        "n_ticks":       0,
    })

    # Cached structured-style proxies — built once per State instance so
    # `state.buildings["owner"]` keeps working for test + external code.
    # Not a dataclass field (excluded from repr/equality); populated in
    # __post_init__.
    def __post_init__(self) -> None:
        self._refresh_proxies()

    def _refresh_proxies(self) -> None:
        # Structured-dtype field names are the second segment of buildings_*.
        object.__setattr__(self, "buildings", _ArrayProxy({
            "alive":    self.buildings_alive,
            "owner":    self.buildings_owner,
            "type_id":  self.buildings_type,
            "garrison": self.buildings_garrison,
            "capacity": self.buildings_capacity,
            "x":        self.buildings_x,
            "y":        self.buildings_y,
        }))
        object.__setattr__(self, "unit_groups", _ArrayProxy({
            "alive":        self.groups_alive,
            "owner":        self.groups_owner,
            "src_slot":     self.groups_src,
            "tgt_slot":     self.groups_tgt,
            "count":        self.groups_count,
            "progress":     self.groups_progress,
            "travel_ticks": self.groups_travel,
        }))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def empty_state() -> State:
    """Zero-initialized state. Use `levels.apply(state, level)` to populate."""
    N = C.MAX_BUILDING_SLOTS
    M = C.MAX_UNIT_GROUP_SLOTS
    K = C.HISTORY_K
    return State(
        buildings_alive    = np.zeros(N, dtype=np.int8),
        buildings_owner    = np.zeros(N, dtype=np.int8),
        buildings_type     = np.zeros(N, dtype=np.int8),
        buildings_garrison = np.zeros(N, dtype=np.int16),
        buildings_capacity = np.zeros(N, dtype=np.int16),
        buildings_x        = np.zeros(N, dtype=np.int16),
        buildings_y        = np.zeros(N, dtype=np.int16),
        groups_alive    = np.zeros(M, dtype=np.int8),
        groups_owner    = np.zeros(M, dtype=np.int8),
        groups_src      = np.zeros(M, dtype=np.int8),
        groups_tgt      = np.zeros(M, dtype=np.int8),
        groups_count    = np.zeros(M, dtype=np.int16),
        groups_progress = np.zeros(M, dtype=np.int16),
        groups_travel   = np.zeros(M, dtype=np.int16),
        arrivals_p1          = np.zeros(N,        dtype=np.int16),
        arrivals_p2          = np.zeros(N,        dtype=np.int16),
        prev_buildings_owner = np.zeros(N,        dtype=np.int8),
        last_actions_p1      = np.zeros((K, 4),   dtype=np.int8),
        last_actions_p2      = np.zeros((K, 4),   dtype=np.int8),
        travel_matrix   = np.zeros((N, N), dtype=np.int16),
        distance_matrix = np.zeros((N, N), dtype=np.float32),
    )


def precompute_distances(state: State) -> None:
    """Fill distance_matrix + travel_matrix for all alive building pairs.

    Called once from reset() after a level is applied. Distances never change
    during a game (buildings don't move), so this is pure setup cost.
    """
    alive = state.buildings_alive.astype(bool)

    xs = state.buildings_x.astype(np.float32)
    ys = state.buildings_y.astype(np.float32)

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    travel = np.ceil(dist / C.TRAVEL_SPEED).astype(np.int32)
    travel = np.clip(travel, C.MIN_TRAVEL_TICKS, C.MAX_TRAVEL_TICKS)

    mask_2d = alive[:, None] & alive[None, :]
    travel = np.where(mask_2d, travel, 0)
    np.fill_diagonal(travel, 0)
    np.fill_diagonal(dist, 0.0)

    state.travel_matrix[:] = travel.astype(np.int16)
    state.distance_matrix[:] = dist


# ---------------------------------------------------------------------------
# Small helpers (kept as a compatibility shim per JAX_PORT_PLAN §4 Phase 0)
# ---------------------------------------------------------------------------

def count_owned_buildings(state: State, owner: int) -> int:
    alive = state.buildings_alive == 1
    return int(np.sum(alive & (state.buildings_owner == owner)))


def count_owned_units(state: State, owner: int) -> int:
    """Total units (garrison + in-flight) for a player, in internal scale."""
    in_garrison = int(np.sum(np.where(
        (state.buildings_alive == 1) & (state.buildings_owner == owner),
        state.buildings_garrison, 0)))
    in_flight = int(np.sum(np.where(
        (state.groups_alive == 1) & (state.groups_owner == owner),
        state.groups_count, 0)))
    return in_garrison + in_flight


def has_in_flight(state: State, owner: int) -> bool:
    return bool(np.any((state.groups_alive == 1) & (state.groups_owner == owner)))
