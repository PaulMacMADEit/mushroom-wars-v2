"""
JAX-batched mirror of `sim.actions.compute_mask_batched`.

Lives at the boundary between `JaxVecEnv.step_chunk` and the policy/opponent
in `training/fused_rollout.py`. Lets the legality mask stay on device across
the rollout instead of bouncing through host numpy each tick.

The numpy `compute_mask_batched` (sim/actions.py:152) stays as the reference
oracle; this module is parity-tested against it.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX, NUM_TYPES, SLOTS_SQ
from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND
from sim.state_jax import StateJax


@partial(jax.jit, static_argnames=("player",))
def compute_mask_batched_jax(state: StateJax, player: int) -> jnp.ndarray:
    """Vectorised legality mask, on device.

    Returns (N, ACTION_SPACE_SIZE) bool. Semantics byte-identical to
    `sim.actions.compute_mask_batched(...)` (numpy) for the same state and
    player.

    `player` is a static argument so the JIT specialises per-player; the call
    sites in fused_rollout.py call this twice per rollout step (P1 + P2) and
    both specialisations live in the cache.
    """
    N, MAX_B = state.buildings_alive.shape

    has_free_group = jnp.any(state.groups_alive == 0, axis=1)         # (N,) bool

    alive     = state.buildings_alive == 1                            # (N, MAX_B) bool
    owned     = alive & (state.buildings_owner == player)             # (N, MAX_B) bool
    valid_tgt = alive                                                  # (N, MAX_B) bool

    garrison  = state.buildings_garrison.astype(jnp.int32)            # (N, MAX_B)
    diag_mask = ~jnp.eye(MAX_B, dtype=bool)                            # (MAX_B, MAX_B)

    max_sendable = jnp.maximum(0, garrison - C.MIN_GARRISON_AFTER_SEND)

    type_blocks = []
    for pct in C.SEND_PERCENTAGES:
        real_units = (max_sendable * pct) // (100 * C.SCALE)
        enough     = (real_units * C.SCALE) >= C.MIN_SEND_INTERNAL    # (N, MAX_B) bool
        src_ok     = owned & enough                                    # (N, MAX_B) bool
        pair_ok    = (src_ok[:, :, None] & valid_tgt[:, None, :]) & diag_mask[None, :, :]
        type_blocks.append(pair_ok.reshape(N, SLOTS_SQ))

    send_region = jnp.concatenate(type_blocks, axis=1)                # (N, NUM_TYPES*SLOTS_SQ)
    assert send_region.shape == (N, NUM_TYPES * SLOTS_SQ)

    send_region = jnp.where(has_free_group[:, None], send_region, False)

    noop_col = jnp.ones((N, 1), dtype=bool)
    return jnp.concatenate([send_region, noop_col], axis=1)           # (N, ACTION_SPACE_SIZE)


# ---------------------------------------------------------------------------
# Phase G2: on-device action decoder + random_legal opponent + action pack.
# ---------------------------------------------------------------------------

@jax.jit
def decode_to_slot_jax(flat_actions: jnp.ndarray) -> jnp.ndarray:
    """Vectorised decoder mirror of `_decode_into_slot` (training/fused_rollout.py).

    flat_actions: (N,) int (any int dtype). Returns (N, ACTION_DIM) int32 with
    columns [kind, type, src, tgt]. NOOP entries decode to [ACTION_KIND_NOOP, 0, 0, 0].
    Byte-identical to the numpy mirror.
    """
    flat = flat_actions.astype(jnp.int64)
    is_noop  = flat == NOOP_INDEX
    type_idx = (flat // SLOTS_SQ).astype(jnp.int32)
    rem      = (flat %  SLOTS_SQ).astype(jnp.int32)
    src_idx  = (rem // C.MAX_BUILDING_SLOTS).astype(jnp.int32)
    tgt_idx  = (rem %  C.MAX_BUILDING_SLOTS).astype(jnp.int32)

    kind     = jnp.where(is_noop, ACTION_KIND_NOOP, ACTION_KIND_SEND).astype(jnp.int32)
    type_out = jnp.where(is_noop, jnp.int32(0), type_idx)
    src_out  = jnp.where(is_noop, jnp.int32(0), src_idx)
    tgt_out  = jnp.where(is_noop, jnp.int32(0), tgt_idx)

    return jnp.stack([kind, type_out, src_out, tgt_out], axis=1)


@jax.jit
def random_legal_opponent_jax(
    p2_mask: jnp.ndarray,        # (N, ACTION_SPACE_SIZE) bool
    key:     jax.Array,          # PRNGKey
) -> jnp.ndarray:
    """JAX mirror of `random_legal_opponent_batched`.

    Uses the same noise-and-argmax trick as the numpy oracle: per row, mask
    out illegal entries with a -1.0 sentinel, take argmax over uniform noise.
    NOOP is always legal so the all-illegal fallback never fires; the
    sentinel pattern is preserved for parity with the numpy version.

    RNG sequences differ between numpy and JAX, so distribution parity (not
    byte-identical) is the bar; tested via KL divergence in the test file.
    """
    noise = jax.random.uniform(key, p2_mask.shape, dtype=jnp.float32)
    masked = jnp.where(p2_mask, noise, -1.0)
    return masked.argmax(axis=1).astype(jnp.int64)


@partial(jax.jit, static_argnames=("opponent_name",))
def pack_action_batch_jax(
    p1_actions: jnp.ndarray,    # (N,) int — flat P1 action indices
    p2_mask:    jnp.ndarray,    # (N, ACTION_SPACE_SIZE) bool — on device
    key:        jax.Array,      # PRNGKey for random_legal sampling
    opponent_name: str,
) -> jnp.ndarray:
    """On-device mirror of `_pack_action_batch_with_p2_mask`.

    Returns (N, 2, ACTION_DIM) int32 — P1 action in row 0, P2 action in row 1.
    `opponent_name` is a static argument; supported values: "noop",
    "random_legal". Neural opponents take a separate code path on host.
    """
    p1_decoded = decode_to_slot_jax(p1_actions)
    N = p1_actions.shape[0]

    if opponent_name == "noop":
        kind  = jnp.full((N,), ACTION_KIND_NOOP, dtype=jnp.int32)
        zeros = jnp.zeros((N,), dtype=jnp.int32)
        p2_decoded = jnp.stack([kind, zeros, zeros, zeros], axis=1)
    elif opponent_name == "random_legal":
        p2_idx = random_legal_opponent_jax(p2_mask, key)
        p2_decoded = decode_to_slot_jax(p2_idx)
    else:
        raise ValueError(
            f"opponent_name {opponent_name!r} not supported in jax pack; "
            "use the host-side _pack_action_batch_neural for neural opponents."
        )

    return jnp.stack([p1_decoded, p2_decoded], axis=1)
