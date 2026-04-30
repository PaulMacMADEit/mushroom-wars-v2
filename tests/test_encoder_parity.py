"""Phase A: encode_obs (numpy) vs encode_obs_batched (jax) parity.

Builds 50 random states across a mix of empty / mid-game / late-game shapes,
encodes via both paths, asserts max abs diff < 1e-5 (float32 epsilon-ish).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import NOOP_INDEX, compute_mask, decode
from sim.engine import step_tick
from sim.envs.mushroom_env import MushroomEnv
from sim.envs.opponents import noop_opponent
from sim.levels import reset
from sim.state_jax import from_numpy_state
from training.encoder import encode_obs
from training.encoder_jax import encode_obs_batched_jit


def _obs_dict_from_state(state):
    """Build the obs dict that `encode_obs` consumes from a numpy State."""
    return {
        "buildings_alive":    state.buildings_alive,
        "buildings_owner":    state.buildings_owner,
        "buildings_type":     state.buildings_type,
        "buildings_garrison": state.buildings_garrison,
        "buildings_capacity": state.buildings_capacity,
        "buildings_x":        state.buildings_x,
        "buildings_y":        state.buildings_y,
        "groups_alive":       state.groups_alive,
        "groups_owner":       state.groups_owner,
        "groups_src":         state.groups_src,
        "groups_tgt":         state.groups_tgt,
        "groups_count":       state.groups_count,
        "groups_progress":    state.groups_progress,
        "groups_travel":      state.groups_travel,
        "tick":               np.int32(state.tick),
        # v10 fields — encoder reads them directly.
        "arrivals_p1":          state.arrivals_p1,
        "arrivals_p2":          state.arrivals_p2,
        "prev_buildings_owner": state.prev_buildings_owner,
        "prev_p1_units_total":  np.int32(state.prev_p1_units_total),
        "prev_p2_units_total":  np.int32(state.prev_p2_units_total),
        "last_actions_p1":      state.last_actions_p1,
        "last_actions_p2":      state.last_actions_p2,
    }


def _warmup(state, n_ticks: int, rng: np.random.Generator) -> None:
    """Step the state forward with random actions so it isn't trivial."""
    for _ in range(n_ticks):
        m1 = compute_mask(state, C.OWNER_P1)
        m2 = compute_mask(state, C.OWNER_P2)
        a1 = decode(int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX)
        a2 = decode(int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX)
        _, _, done = step_tick(state, a1, a2)
        if done:
            return


@pytest.mark.parametrize(
    "level,n_warmup",
    [
        ("crossroads_6",  0),    # opening — many features near zero
        ("crossroads_6",  20),
        ("random_8_16",   30),
        ("random_8_16",   80),   # mid-game
        ("random_10_16", 120),   # near-timeout
    ],
)
def test_encoder_parity_single_states(level, n_warmup):
    """Per-state parity: numpy encode == jax encode within 1e-5."""
    seeds = list(range(10))
    states = [reset(level_name=level, seed=s) for s in seeds]
    for i, s in enumerate(states):
        _warmup(s, n_warmup, np.random.default_rng(seeds[i] + 7919))

    # Numpy reference: per-env encode.
    np_out = np.stack(
        [encode_obs(_obs_dict_from_state(s)) for s in states],
        axis=0,
    )

    # JAX: stack states into one batched StateJax via tree_map (picks up new fields automatically).
    import jax
    import jax.numpy as jnp
    leaves = [from_numpy_state(s) for s in states]
    batched_jax = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *leaves
    )
    jx_out = np.asarray(encode_obs_batched_jit(batched_jax))

    assert np_out.shape == jx_out.shape
    max_diff = float(np.abs(np_out - jx_out).max())
    assert max_diff < 1e-5, (
        f"encoder parity failed for level={level} warmup={n_warmup}: "
        f"max |np - jax| = {max_diff:.2e}"
    )


def test_encoder_parity_through_env_step():
    """End-to-end: a MushroomEnv obs dict vs the JAX-batched encoder.

    Drives a real env through 30 ticks, encodes the obs dict via the numpy
    path each tick, and compares against the JAX encoder reading the same
    state.
    """
    env = MushroomEnv(level_name="crossroads_6", opponent=noop_opponent, seed=2026)
    obs_dict, _ = env.reset(seed=2026)

    for tick in range(30):
        np_vec = encode_obs(obs_dict)
        # JAX path reads the underlying State, not the obs dict. Lift via
        # tree_map so any field added to StateJax flows through unchanged.
        import jax
        import jax.numpy as jnp
        sj = from_numpy_state(env.state)
        sj_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], sj)
        jx_vec = np.asarray(encode_obs_batched_jit(sj_b))[0]

        max_diff = float(np.abs(np_vec - jx_vec).max())
        assert max_diff < 1e-5, (
            f"end-to-end encoder parity failed at tick {tick}: max diff = {max_diff:.2e}"
        )

        # Step with NOOP so the env evolves.
        obs_dict, _, terminated, truncated, _ = env.step(int(NOOP_INDEX))
        if terminated or truncated:
            break
