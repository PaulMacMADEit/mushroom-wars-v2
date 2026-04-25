"""Phase G1+G2: parity tests for sim/actions_jax.

G1: compute_mask_batched_jax — byte-identical bool masks vs numpy oracle.
G2: decode_to_slot_jax — byte-identical action decoder vs numpy oracle.
G2: random_legal_opponent_jax — distribution parity (KL) + mask compliance.
G2: pack_action_batch_jax — full action-pack parity vs numpy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sim import config as C
from sim.actions import (
    ACTION_SPACE_SIZE,
    NOOP_INDEX,
    compute_mask,
    compute_mask_batched,
    decode,
)
from sim.actions_jax import (
    compute_mask_batched_jax,
    decode_to_slot_jax,
    pack_action_batch_jax,
    random_legal_opponent_jax,
)
from sim.engine import step_tick
from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP
from sim.levels import reset
from sim.state_jax import StateJax, from_numpy_state
from training.fused_rollout import _pack_action_batch_with_p2_mask


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


# ---------------------------------------------------------------------------
# G2: decoder parity
# ---------------------------------------------------------------------------

def _np_decode_into_slot(flat: np.ndarray) -> np.ndarray:
    """Reference decoder — same logic as training/fused_rollout._decode_into_slot."""
    from sim.actions import SLOTS_SQ
    from sim.engine_jax import ACTION_KIND_SEND
    flat = np.asarray(flat, dtype=np.int64)
    is_noop = flat == NOOP_INDEX
    type_idx = (flat // SLOTS_SQ).astype(np.int32)
    rem      = (flat %  SLOTS_SQ).astype(np.int32)
    src_idx  = (rem // C.MAX_BUILDING_SLOTS).astype(np.int32)
    tgt_idx  = (rem %  C.MAX_BUILDING_SLOTS).astype(np.int32)
    out = np.zeros((flat.shape[0], 4), dtype=np.int32)
    out[:, 0] = np.where(is_noop, ACTION_KIND_NOOP, ACTION_KIND_SEND)
    out[:, 1] = np.where(is_noop, 0, type_idx)
    out[:, 2] = np.where(is_noop, 0, src_idx)
    out[:, 3] = np.where(is_noop, 0, tgt_idx)
    return out


def test_decode_to_slot_jax_parity():
    """JAX decoder == numpy decoder, byte-identical."""
    rng = np.random.default_rng(0)
    flat = rng.integers(0, ACTION_SPACE_SIZE, size=2048, dtype=np.int64)
    flat[::13] = NOOP_INDEX

    np_out = _np_decode_into_slot(flat)
    jx_out = np.asarray(decode_to_slot_jax(jnp.asarray(flat)))

    np.testing.assert_array_equal(np_out, jx_out)


# ---------------------------------------------------------------------------
# G2: random_legal_opponent_jax — mask compliance + distribution parity
# ---------------------------------------------------------------------------

def _build_random_masks(n_envs: int, seed: int) -> np.ndarray:
    """Build a mix of varied-density legality masks (NOOP always legal)."""
    rng = np.random.default_rng(seed)
    densities = rng.uniform(0.01, 0.5, size=n_envs)
    mask = np.zeros((n_envs, ACTION_SPACE_SIZE), dtype=bool)
    for i in range(n_envs):
        mask[i] = rng.random(ACTION_SPACE_SIZE) < densities[i]
    mask[:, NOOP_INDEX] = True
    return mask


def test_random_legal_jax_mask_compliance():
    """Every sampled action must be a legal action (mask[idx]==True)."""
    mask = _build_random_masks(n_envs=512, seed=7)
    key = jax.random.PRNGKey(123)
    sampled = np.asarray(random_legal_opponent_jax(jnp.asarray(mask), key))

    assert sampled.shape == (512,)
    assert np.all(mask[np.arange(512), sampled]), "JAX random_legal sampled an illegal action"


def _build_fixed_density_masks(legal_counts: list[int], seed: int) -> np.ndarray:
    """One mask per entry in legal_counts, each with that many random True
    positions (NOOP always one of them). Lets us test small-k distributions
    cleanly without sparse-bin noise."""
    rng = np.random.default_rng(seed)
    n_masks = len(legal_counts)
    mask = np.zeros((n_masks, ACTION_SPACE_SIZE), dtype=bool)
    for i, k in enumerate(legal_counts):
        assert 1 <= k <= ACTION_SPACE_SIZE
        idx = rng.choice(ACTION_SPACE_SIZE - 1, size=k - 1, replace=False)
        mask[i, idx] = True
        mask[i, NOOP_INDEX] = True
    return mask


def test_random_legal_jax_uniform_over_legal():
    """JAX sampler must be uniform over the legal entries of each mask.

    Chi-squared goodness-of-fit against the uniform-over-legal hypothesis:
    chi-sq < critical value at p=0.001. Tests across a range of legality
    densities so we hit both the small-k and larger-k regimes.
    """
    legal_counts = [4, 8, 16, 32, 64, 128]
    masks = _build_fixed_density_masks(legal_counts, seed=21)
    n_samples = 20000

    key = jax.random.PRNGKey(2025)
    counts = np.zeros((len(legal_counts), ACTION_SPACE_SIZE), dtype=np.int64)
    for _ in range(n_samples):
        key, sub = jax.random.split(key)
        idx = np.asarray(random_legal_opponent_jax(jnp.asarray(masks), sub))
        for i in range(len(legal_counts)):
            counts[i, idx[i]] += 1

    # Critical chi-sq values at p=0.001 for df = k-1, computed from
    # scipy.stats.chi2.ppf(0.999, k-1) and rounded up to give margin.
    crit = {3: 16.3, 7: 24.3, 15: 37.7, 31: 61.1, 63: 103.4, 127: 178.0}

    for i, k in enumerate(legal_counts):
        legal = np.where(masks[i])[0]
        assert legal.size == k
        observed = counts[i, legal]
        expected = n_samples / k
        chi_sq = float(((observed - expected) ** 2 / expected).sum())
        assert chi_sq < crit[k - 1], (
            f"random_legal not uniform over legal for k={k}: chi-sq={chi_sq:.2f} "
            f"vs critical {crit[k-1]} at p=0.001"
        )




# ---------------------------------------------------------------------------
# G2: pack_action_batch_jax — full pack parity vs numpy
# ---------------------------------------------------------------------------

def test_pack_action_batch_jax_noop_parity():
    """For opponent_name='noop', JAX pack must be byte-identical to numpy pack."""
    rng = np.random.default_rng(0)
    p1 = rng.integers(0, ACTION_SPACE_SIZE, size=64, dtype=np.int64)
    p1[::7] = NOOP_INDEX
    p2_mask = _build_random_masks(n_envs=64, seed=11)

    np_pack = _pack_action_batch_with_p2_mask(p1, p2_mask, "noop", rng, 64)

    key = jax.random.PRNGKey(0)
    jx_pack = np.asarray(pack_action_batch_jax(jnp.asarray(p1), jnp.asarray(p2_mask), key, "noop"))

    np.testing.assert_array_equal(np_pack, jx_pack)


def test_pack_action_batch_jax_random_legal_compliance():
    """For random_legal opponent, JAX pack must (a) match P1 row byte-identically,
    (b) emit only legal P2 actions per p2_mask."""
    rng = np.random.default_rng(0)
    p1 = rng.integers(0, ACTION_SPACE_SIZE, size=128, dtype=np.int64)
    p2_mask = _build_random_masks(n_envs=128, seed=42)

    np_pack = _pack_action_batch_with_p2_mask(p1, p2_mask, "random_legal", rng, 128)

    key = jax.random.PRNGKey(7)
    jx_pack = np.asarray(pack_action_batch_jax(jnp.asarray(p1), jnp.asarray(p2_mask), key, "random_legal"))

    np.testing.assert_array_equal(np_pack[:, 0, :], jx_pack[:, 0, :])

    for i in range(128):
        kind, type_, src, tgt = jx_pack[i, 1, :]
        if kind == ACTION_KIND_NOOP:
            assert p2_mask[i, NOOP_INDEX], f"env {i}: jax picked NOOP but NOOP not legal"
        else:
            slots = C.MAX_BUILDING_SLOTS
            from sim.actions import SLOTS_SQ
            flat = int(type_) * SLOTS_SQ + int(src) * slots + int(tgt)
            assert p2_mask[i, flat], (
                f"env {i}: jax picked illegal action flat={flat} "
                f"(type={type_}, src={src}, tgt={tgt})"
            )
