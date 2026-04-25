"""Phase G1: compute_mask_batched (numpy) vs compute_mask_batched_jax parity.

Builds 50 random states across a mix of opening/mid/late shapes, masks via
both paths for both players, asserts byte-identical bool masks. Plus a
hand-built "no free group slot" edge case to confirm the early-return
semantic of the numpy oracle is preserved by the JAX vectorised path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sim import config as C
from sim.actions import (
    NOOP_INDEX,
    compute_mask,
    compute_mask_batched,
    decode,
)
from sim.actions_jax import compute_mask_batched_jax
from sim.engine import step_tick
from sim.levels import reset
from sim.state_jax import StateJax, from_numpy_state


def _warmup(state, n_ticks: int, rng: np.random.Generator) -> None:
    """Step the state forward with random legal actions so it isn't trivial."""
    for _ in range(n_ticks):
        m1 = compute_mask(state, C.OWNER_P1)
        m2 = compute_mask(state, C.OWNER_P2)
        a1 = decode(int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX)
        a2 = decode(int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX)
        _, _, done = step_tick(state, a1, a2)
        if done:
            return


def _stack_states(states) -> StateJax:
    leaves = [from_numpy_state(s) for s in states]
    import jax
    batched = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *leaves)
    return StateJax(
        buildings_alive    = jnp.asarray(batched.buildings_alive),
        buildings_owner    = jnp.asarray(batched.buildings_owner),
        buildings_type     = jnp.asarray(batched.buildings_type),
        buildings_garrison = jnp.asarray(batched.buildings_garrison),
        buildings_capacity = jnp.asarray(batched.buildings_capacity),
        buildings_x        = jnp.asarray(batched.buildings_x),
        buildings_y        = jnp.asarray(batched.buildings_y),
        groups_alive    = jnp.asarray(batched.groups_alive),
        groups_owner    = jnp.asarray(batched.groups_owner),
        groups_src      = jnp.asarray(batched.groups_src),
        groups_tgt      = jnp.asarray(batched.groups_tgt),
        groups_count    = jnp.asarray(batched.groups_count),
        groups_progress = jnp.asarray(batched.groups_progress),
        groups_travel   = jnp.asarray(batched.groups_travel),
        travel_matrix   = jnp.asarray(batched.travel_matrix),
        tick            = jnp.asarray(batched.tick),
        phase           = jnp.asarray(batched.phase),
        rng_key         = jnp.asarray(batched.rng_key),
    )


@pytest.mark.parametrize(
    "level,n_warmup",
    [
        ("crossroads_6",  0),
        ("crossroads_6",  20),
        ("random_8_16",   30),
        ("random_8_16",   80),
        ("random_10_16", 120),
    ],
)
@pytest.mark.parametrize("player", [C.OWNER_P1, C.OWNER_P2])
def test_mask_parity_batched(level, n_warmup, player):
    """Numpy compute_mask_batched == JAX compute_mask_batched_jax, exactly."""
    seeds = list(range(10))
    states = [reset(level_name=level, seed=s) for s in seeds]
    for i, s in enumerate(states):
        _warmup(s, n_warmup, np.random.default_rng(seeds[i] + 7919))

    np_mask = compute_mask_batched(
        np.stack([s.buildings_alive    for s in states], axis=0),
        np.stack([s.buildings_owner    for s in states], axis=0),
        np.stack([s.buildings_garrison for s in states], axis=0),
        np.stack([s.groups_alive       for s in states], axis=0),
        player,
    )

    jx_mask = np.asarray(compute_mask_batched_jax(_stack_states(states), player))

    assert np_mask.shape == jx_mask.shape
    diff = np.where(np_mask != jx_mask)
    assert diff[0].size == 0, (
        f"mask parity failed for level={level} warmup={n_warmup} player={player}: "
        f"{diff[0].size} entries differ across {len(states)} states"
    )


def test_mask_parity_no_free_group_slot():
    """Edge case: all groups in flight → only NOOP must be legal.

    The numpy oracle has an early-return for this case; the JAX path uses a
    vectorised `where` instead. Confirm parity end-to-end.
    """
    state = reset(level_name="random_8_16", seed=0)
    rng = np.random.default_rng(42)
    _warmup(state, 20, rng)

    state.groups_alive[:] = 1

    np_mask = compute_mask_batched(
        state.buildings_alive[None],
        state.buildings_owner[None],
        state.buildings_garrison[None],
        state.groups_alive[None],
        C.OWNER_P1,
    )

    assert np_mask[0, NOOP_INDEX]
    assert not np_mask[0, :NOOP_INDEX].any(), "numpy oracle let send actions through with no free group"

    jx_mask = np.asarray(compute_mask_batched_jax(_stack_states([state]), C.OWNER_P1))
    np.testing.assert_array_equal(np_mask, jx_mask)


def test_mask_parity_per_env_against_compute_mask():
    """Belt-and-braces: batched JAX path also matches the per-env compute_mask."""
    seeds = list(range(8))
    states = [reset(level_name="random_8_16", seed=s) for s in seeds]
    for i, s in enumerate(states):
        _warmup(s, 50, np.random.default_rng(seeds[i] + 991))

    for player in (C.OWNER_P1, C.OWNER_P2):
        per_env = np.stack([compute_mask(s, player) for s in states], axis=0)
        batched = np.asarray(compute_mask_batched_jax(_stack_states(states), player))
        np.testing.assert_array_equal(per_env, batched)
