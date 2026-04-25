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
from sim.actions import ACTION_SPACE_SIZE, NUM_TYPES, SLOTS_SQ
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
