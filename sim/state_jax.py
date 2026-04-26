"""
JAX-native state container — pytree mirror of sim/state.py.

Storage philosophy:
- Same parallel-ndarray layout as the numpy backend (see sim/state.py).
- Every field is a `jnp.ndarray`, so the whole StateJax is one pytree that
  `jax.jit` / `jax.vmap` / `jax.tree_util.tree_map` can traverse without
  special casing.
- Scalars (`tick`, `phase`) are 0-D jnp arrays so they get swept into the
  pytree the same way (no Python ints in the hot path).
- `rng_key` is carried on the state so stochastic ops (few, but reserved)
  can consume/split it without implicit global RNG.

Event emission (spawn/arrive/capture/end) is NOT part of StateJax — the JAX
hot path is event-free by design. Replay still runs against the numpy
backend.

This module is Phase 1 of JAX_PORT_PLAN.md: the container + converters, no
engine logic yet.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sim import config as C
from sim.state import State, empty_state


@struct.dataclass
class StateJax:
    """JAX-native game state. All fields are `jnp.ndarray`.

    Shape convention for a SINGLE game (Phase 1). Phase 3 will add a leading
    `n_envs` batch dimension when vmap wraps the step function.
    """

    # Buildings (parallel arrays, length MAX_BUILDING_SLOTS)
    buildings_alive:    jnp.ndarray   # int8
    buildings_owner:    jnp.ndarray   # int8
    buildings_type:     jnp.ndarray   # int8
    buildings_garrison: jnp.ndarray   # int16
    buildings_capacity: jnp.ndarray   # int16
    buildings_x:        jnp.ndarray   # int16
    buildings_y:        jnp.ndarray   # int16

    # Unit groups (parallel arrays, length MAX_UNIT_GROUP_SLOTS)
    groups_alive:    jnp.ndarray   # int8
    groups_owner:    jnp.ndarray   # int8
    groups_src:      jnp.ndarray   # int8
    groups_tgt:      jnp.ndarray   # int8
    groups_count:    jnp.ndarray   # int16
    groups_progress: jnp.ndarray   # int16
    groups_travel:   jnp.ndarray   # int16

    # Precomputed once at reset (on numpy, lifted to device in from_numpy_state).
    travel_matrix: jnp.ndarray     # (N, N) int16

    # Scalars as 0-D jnp arrays so they live inside the pytree.
    tick:  jnp.ndarray             # int32 scalar
    phase: jnp.ndarray             # int8  scalar
    # Reward scheme version (0=v1.2, 1=v1.3). Mirrors numpy `State.reward_version`.
    reward_version: jnp.ndarray    # int8  scalar

    # RNG carried on-state so stochastic ops (reserved) can split it.
    rng_key: jnp.ndarray           # uint32 shape (2,)


# ---------------------------------------------------------------------------
# Shape-defining constants
# ---------------------------------------------------------------------------

N_BLDG  = C.MAX_BUILDING_SLOTS
N_GROUP = C.MAX_UNIT_GROUP_SLOTS


# ---------------------------------------------------------------------------
# Converters — numpy State ↔ StateJax
# ---------------------------------------------------------------------------

def from_numpy_state(state: State, rng_key: jnp.ndarray | None = None) -> StateJax:
    """Copy a numpy State onto the JAX device as a StateJax pytree.

    `rng_key` defaults to a deterministic zero key; callers threading stochastic
    ops should pass a real key (e.g. `jax.random.PRNGKey(seed)`).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    # Use jnp.array (not asarray) so same-dtype numpy arrays get copied
    # onto device. asarray would zero-copy-alias, which breaks downstream
    # code that mutates the numpy state after constructing a StateJax from it.
    return StateJax(
        buildings_alive    = jnp.array(state.buildings_alive,    dtype=jnp.int8),
        buildings_owner    = jnp.array(state.buildings_owner,    dtype=jnp.int8),
        buildings_type     = jnp.array(state.buildings_type,     dtype=jnp.int8),
        buildings_garrison = jnp.array(state.buildings_garrison, dtype=jnp.int16),
        buildings_capacity = jnp.array(state.buildings_capacity, dtype=jnp.int16),
        buildings_x        = jnp.array(state.buildings_x,        dtype=jnp.int16),
        buildings_y        = jnp.array(state.buildings_y,        dtype=jnp.int16),
        groups_alive    = jnp.array(state.groups_alive,    dtype=jnp.int8),
        groups_owner    = jnp.array(state.groups_owner,    dtype=jnp.int8),
        groups_src      = jnp.array(state.groups_src,      dtype=jnp.int8),
        groups_tgt      = jnp.array(state.groups_tgt,      dtype=jnp.int8),
        groups_count    = jnp.array(state.groups_count,    dtype=jnp.int16),
        groups_progress = jnp.array(state.groups_progress, dtype=jnp.int16),
        groups_travel   = jnp.array(state.groups_travel,   dtype=jnp.int16),
        travel_matrix   = jnp.array(state.travel_matrix,   dtype=jnp.int16),
        tick            = jnp.array(state.tick,  dtype=jnp.int32),
        phase           = jnp.array(state.phase, dtype=jnp.int8),
        reward_version  = jnp.array(state.reward_version, dtype=jnp.int8),
        rng_key         = rng_key,
    )


def to_numpy_state(state_jax: StateJax) -> State:
    """Materialise a StateJax back into a numpy-backed State.

    Intended for the parity harness and replay paths; not for the hot training
    loop. The returned State has zeroed `distance_matrix` (not needed after
    reset) and a fresh `perf` dict.
    """
    out = empty_state()
    out.buildings_alive[:]    = np.asarray(state_jax.buildings_alive,    dtype=np.int8)
    out.buildings_owner[:]    = np.asarray(state_jax.buildings_owner,    dtype=np.int8)
    out.buildings_type[:]     = np.asarray(state_jax.buildings_type,     dtype=np.int8)
    out.buildings_garrison[:] = np.asarray(state_jax.buildings_garrison, dtype=np.int16)
    out.buildings_capacity[:] = np.asarray(state_jax.buildings_capacity, dtype=np.int16)
    out.buildings_x[:]        = np.asarray(state_jax.buildings_x,        dtype=np.int16)
    out.buildings_y[:]        = np.asarray(state_jax.buildings_y,        dtype=np.int16)
    out.groups_alive[:]    = np.asarray(state_jax.groups_alive,    dtype=np.int8)
    out.groups_owner[:]    = np.asarray(state_jax.groups_owner,    dtype=np.int8)
    out.groups_src[:]      = np.asarray(state_jax.groups_src,      dtype=np.int8)
    out.groups_tgt[:]      = np.asarray(state_jax.groups_tgt,      dtype=np.int8)
    out.groups_count[:]    = np.asarray(state_jax.groups_count,    dtype=np.int16)
    out.groups_progress[:] = np.asarray(state_jax.groups_progress, dtype=np.int16)
    out.groups_travel[:]   = np.asarray(state_jax.groups_travel,   dtype=np.int16)
    out.travel_matrix[:]   = np.asarray(state_jax.travel_matrix,   dtype=np.int16)
    out.tick  = int(state_jax.tick)
    out.phase = int(state_jax.phase)
    out.reward_version = int(state_jax.reward_version)
    return out


def states_equal(a: State, b: State) -> bool:
    """Byte-identity check over every gameplay-relevant field. Used by the
    parity harness. Ignores `perf` and `distance_matrix`.
    """
    checks = [
        ("buildings_alive",    a.buildings_alive,    b.buildings_alive),
        ("buildings_owner",    a.buildings_owner,    b.buildings_owner),
        ("buildings_type",     a.buildings_type,     b.buildings_type),
        ("buildings_garrison", a.buildings_garrison, b.buildings_garrison),
        ("buildings_capacity", a.buildings_capacity, b.buildings_capacity),
        ("buildings_x",        a.buildings_x,        b.buildings_x),
        ("buildings_y",        a.buildings_y,        b.buildings_y),
        ("groups_alive",    a.groups_alive,    b.groups_alive),
        ("groups_owner",    a.groups_owner,    b.groups_owner),
        ("groups_src",      a.groups_src,      b.groups_src),
        ("groups_tgt",      a.groups_tgt,      b.groups_tgt),
        ("groups_count",    a.groups_count,    b.groups_count),
        ("groups_progress", a.groups_progress, b.groups_progress),
        ("groups_travel",   a.groups_travel,   b.groups_travel),
        ("travel_matrix",   a.travel_matrix,   b.travel_matrix),
    ]
    for name, ax, bx in checks:
        if not np.array_equal(ax, bx):
            return False
    if int(a.tick) != int(b.tick):
        return False
    if int(a.phase) != int(b.phase):
        return False
    return True
