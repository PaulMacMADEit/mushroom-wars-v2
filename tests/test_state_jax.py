"""Phase 1 smoke: StateJax pytree + numpy<->jax round-trip converters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sim import config as C
from sim.levels import reset
from sim.state_jax import (
    StateJax,
    from_numpy_state,
    states_equal,
    to_numpy_state,
)


def test_from_numpy_state_preserves_field_values():
    """Every building/group/matrix field survives the numpy→jax copy."""
    src = reset(level_name="crossroads_6", seed=42)
    sj = from_numpy_state(src)

    assert isinstance(sj, StateJax)
    assert np.array_equal(np.asarray(sj.buildings_alive),    src.buildings_alive)
    assert np.array_equal(np.asarray(sj.buildings_owner),    src.buildings_owner)
    assert np.array_equal(np.asarray(sj.buildings_garrison), src.buildings_garrison)
    assert np.array_equal(np.asarray(sj.buildings_x),        src.buildings_x)
    assert np.array_equal(np.asarray(sj.buildings_y),        src.buildings_y)
    assert np.array_equal(np.asarray(sj.groups_alive),       src.groups_alive)
    assert np.array_equal(np.asarray(sj.travel_matrix),      src.travel_matrix)
    assert int(sj.tick)  == src.tick
    assert int(sj.phase) == src.phase


def test_round_trip_crossroads():
    """numpy → jax → numpy returns byte-identical gameplay state."""
    src = reset(level_name="crossroads_6", seed=0)
    sj = from_numpy_state(src)
    back = to_numpy_state(sj)
    assert states_equal(src, back)


@pytest.mark.parametrize("level", ["crossroads_6", "random_8_32", "asym_6_12"])
def test_round_trip_multiple_levels(level):
    """Round-trip is stable across static + dynamic level generators."""
    src = reset(level_name=level, seed=7)
    sj = from_numpy_state(src)
    back = to_numpy_state(sj)
    assert states_equal(src, back)


def test_statejax_is_a_pytree():
    """jax.tree_util.tree_map walks every StateJax leaf without special casing."""
    src = reset(seed=1)
    sj = from_numpy_state(src)

    # Add 0 to every leaf; output pytree should match input element-for-element.
    noop = jax.tree_util.tree_map(lambda x: x + 0, sj)
    assert isinstance(noop, StateJax)
    back = to_numpy_state(noop)
    assert states_equal(src, back)


def test_statejax_field_shapes():
    """Building/group arrays have the fixed capacity shape required for XLA."""
    sj = from_numpy_state(reset(seed=3))
    assert sj.buildings_alive.shape    == (C.MAX_BUILDING_SLOTS,)
    assert sj.buildings_garrison.shape == (C.MAX_BUILDING_SLOTS,)
    assert sj.groups_alive.shape       == (C.MAX_UNIT_GROUP_SLOTS,)
    assert sj.groups_progress.shape    == (C.MAX_UNIT_GROUP_SLOTS,)
    assert sj.travel_matrix.shape      == (C.MAX_BUILDING_SLOTS, C.MAX_BUILDING_SLOTS)
    assert sj.tick.shape  == ()
    assert sj.phase.shape == ()
    assert sj.rng_key.shape == (2,)


def test_rng_key_defaults_deterministic_and_accepts_override():
    src = reset(seed=5)
    default_key = from_numpy_state(src).rng_key
    assert np.array_equal(np.asarray(default_key), np.asarray(jax.random.PRNGKey(0)))

    custom = jax.random.PRNGKey(12345)
    sj = from_numpy_state(src, rng_key=custom)
    assert np.array_equal(np.asarray(sj.rng_key), np.asarray(custom))


def test_statejax_dtypes_match_plan():
    sj = from_numpy_state(reset(seed=2))
    assert sj.buildings_alive.dtype    == jnp.int8
    assert sj.buildings_owner.dtype    == jnp.int8
    assert sj.buildings_garrison.dtype == jnp.int16
    assert sj.buildings_capacity.dtype == jnp.int16
    assert sj.groups_alive.dtype       == jnp.int8
    assert sj.groups_count.dtype       == jnp.int16
    assert sj.travel_matrix.dtype      == jnp.int16
    assert sj.tick.dtype               == jnp.int32
    assert sj.phase.dtype              == jnp.int8
