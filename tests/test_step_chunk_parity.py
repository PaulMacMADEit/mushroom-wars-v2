"""Phase B: JaxVecEnv.step_chunk(actions, K) parity.

K=1: byte-identical to a single .step() call.
K=4: byte-identical to four sequential .step() calls where ticks 1..3 are NOOP.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import NOOP_INDEX, compute_mask, decode
from sim.engine_jax import (
    ACTION_DIM,
    ACTION_KIND_NOOP,
    ACTION_KIND_SEND,
    encode_action,
)
from sim.envs.jax_vec_env import JaxVecEnv
from sim.levels import reset


def _build_random_action_batch(env: JaxVecEnv, rng: np.random.Generator) -> np.ndarray:
    """Pick a legal random P1 + P2 action per env via the numpy mask oracle."""
    states = env.snapshot_numpy_states()
    a = np.zeros((env.n_envs, 2, ACTION_DIM), dtype=np.int32)
    for i, s in enumerate(states):
        m1 = compute_mask(s, C.OWNER_P1)
        m2 = compute_mask(s, C.OWNER_P2)
        a1_idx = int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX
        a2_idx = int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX
        for k, idx in enumerate((a1_idx, a2_idx)):
            ax = decode(idx)
            if ax.kind == "noop":
                a[i, k] = [ACTION_KIND_NOOP, 0, 0, 0]
            else:
                a[i, k] = [ACTION_KIND_SEND, ax.type_idx, ax.src, ax.tgt]
    return a


def _states_equal_batch(a: JaxVecEnv, b: JaxVecEnv) -> bool:
    """Compare two batched states field-by-field."""
    a_states = a.snapshot_numpy_states()
    b_states = b.snapshot_numpy_states()
    for ai, bi in zip(a_states, b_states):
        for attr in (
            "buildings_alive", "buildings_owner", "buildings_type",
            "buildings_garrison", "buildings_capacity", "buildings_x", "buildings_y",
            "groups_alive", "groups_owner", "groups_src", "groups_tgt",
            "groups_count", "groups_progress", "groups_travel",
        ):
            if not np.array_equal(getattr(ai, attr), getattr(bi, attr)):
                return False
        if int(ai.tick) != int(bi.tick) or int(ai.phase) != int(bi.phase):
            return False
    return True


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_step_chunk_K1_matches_step(seed):
    """K=1 must equal a single .step() byte-for-byte over many ticks."""
    n_envs = 8
    base = 100 + seed * 1000
    env_chunk = JaxVecEnv(n_envs=n_envs, level_name="random_6_10", base_seed=base)
    env_step  = JaxVecEnv(n_envs=n_envs, level_name="random_6_10", base_seed=base)

    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)

    for t in range(40):
        a_chunk = _build_random_action_batch(env_chunk, rng_a)
        a_step  = _build_random_action_batch(env_step,  rng_b)
        assert np.array_equal(a_chunk, a_step), f"action mismatch at t={t}"

        r_chunk = env_chunk.step_chunk(a_chunk, K=1)
        r_step  = env_step.step(a_step)

        assert np.allclose(r_chunk["rewards"],    r_step.rewards,    atol=1e-6), (
            f"reward p1 diff at t={t}: chunk={r_chunk['rewards']} step={r_step.rewards}"
        )
        assert np.allclose(r_chunk["rewards_p2"], r_step.rewards_p2, atol=1e-6), (
            f"reward p2 diff at t={t}"
        )
        assert np.array_equal(r_chunk["dones"], r_step.terminated), (
            f"done mismatch at t={t}: chunk={r_chunk['dones']} step={r_step.terminated}"
        )
        assert _states_equal_batch(env_chunk, env_step), f"state diverged at t={t}"


@pytest.mark.parametrize("K", [2, 4, 8])
def test_step_chunk_matches_K_sequential_steps_with_noop_fill(K):
    """K>1 must equal K sequential .step() calls where ticks 1..K-1 are NOOP."""
    n_envs = 4
    base_seed = 500
    env_chunk = JaxVecEnv(n_envs=n_envs, level_name="random_6_10", base_seed=base_seed)
    env_seq   = JaxVecEnv(n_envs=n_envs, level_name="random_6_10", base_seed=base_seed)

    rng = np.random.default_rng(7)

    for chunk_idx in range(10):
        # One real action pair per env.
        actions = _build_random_action_batch(env_chunk, rng)

        # Chunk path.
        r_chunk = env_chunk.step_chunk(actions, K=K)

        # Sequential reference: tick 0 = real actions, ticks 1..K-1 = NOOP.
        noop_actions = np.zeros_like(actions)
        # NOOP encoding is [ACTION_KIND_NOOP, 0, 0, 0]; np.zeros gives that already.

        # Track per-env summed rewards + OR-folded done across the K sub-ticks.
        # We have to use the inner JaxVecEnv.step() which already handles
        # auto-reset on done. To keep parity with step_chunk's behaviour
        # (which resets only at chunk end), we manually replicate by stepping
        # through K ticks but passing NOOP after tick 0, and combining results.
        r1_sum = np.zeros(n_envs, dtype=np.float32)
        r2_sum = np.zeros(n_envs, dtype=np.float32)
        done_any = np.zeros(n_envs, dtype=bool)

        for k in range(K):
            a = actions if k == 0 else noop_actions
            # We can't use env_seq.step() because that auto-resets done envs
            # mid-chunk, while step_chunk does NOT — step_chunk's `_step_tick_impl`
            # early-outs on terminal phase and the auto-reset only happens after
            # the whole chunk. So replicate that semantics: skip auto-reset
            # by reaching into the internal `_step_batched` directly.
            from sim.envs.jax_vec_env import _step_batched
            import jax.numpy as jnp
            a1 = jnp.asarray(a[:, 0, :], dtype=jnp.int32)
            a2 = jnp.asarray(a[:, 1, :], dtype=jnp.int32)
            env_seq.state, r1, r2, d = _step_batched(env_seq.state, a1, a2)
            r1_sum   += np.asarray(r1)
            r2_sum   += np.asarray(r2)
            done_any |= np.asarray(d)

        # Now apply the same end-of-chunk auto-reset that step_chunk does.
        if done_any.any():
            env_seq._auto_reset(done_any)

        assert np.allclose(r_chunk["rewards"], r1_sum, atol=1e-6), (
            f"K={K} chunk={chunk_idx}: r1 chunk={r_chunk['rewards']} seq={r1_sum}"
        )
        assert np.allclose(r_chunk["rewards_p2"], r2_sum, atol=1e-6), (
            f"K={K} chunk={chunk_idx}: r2 mismatch"
        )
        assert np.array_equal(r_chunk["dones"], done_any), (
            f"K={K} chunk={chunk_idx}: done mismatch"
        )
        assert _states_equal_batch(env_chunk, env_seq), (
            f"K={K} chunk={chunk_idx}: state diverged"
        )


def test_step_chunk_caches_K():
    """Different K's compile separately; same K hits the cache (no recompile)."""
    from sim.envs.jax_vec_env import _step_chunk_cache, _step_chunk_batched

    _step_chunk_cache.clear()
    env = JaxVecEnv(n_envs=2, level_name="crossroads_6", base_seed=0)
    actions = np.zeros((2, 2, ACTION_DIM), dtype=np.int32)

    env.step_chunk(actions, K=1)
    env.step_chunk(actions, K=4)
    env.step_chunk(actions, K=1)  # should hit cache, no new compile

    assert set(_step_chunk_cache.keys()) == {1, 4}
