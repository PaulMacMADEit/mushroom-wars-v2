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


@jax.jit
def greedy_capacity_aware_opponent_jax(
    buildings_owner:    jnp.ndarray,   # (N, MAX_B) int8
    buildings_garrison: jnp.ndarray,   # (N, MAX_B) int16
    buildings_alive:    jnp.ndarray,   # (N, MAX_B) int8
    p2_mask:            jnp.ndarray,   # (N, ACTION_SPACE_SIZE) bool
) -> jnp.ndarray:
    """JAX mirror of `greedy_capacity_aware_opponent` (sim/envs/opponents.py).

    Logic per env:
      - Source = highest-garrison alive P2 building.
      - Phase A (any alive neutral): target = lowest-garrison alive neutral.
      - Phase B (no neutrals): target = lowest-garrison alive enemy (P1).
      - Capture iff: 0.75 * src_garrison > target_garrison * (1.0 if neutral else DEF_BONUS).
      - Else NOOP.

    Returns (N,) int64 flat action indices.
    """
    N, MAX_B = buildings_alive.shape
    alive   = buildings_alive == 1
    owners  = buildings_owner.astype(jnp.int32)
    garr    = buildings_garrison.astype(jnp.int32)

    # ----- Source: highest-garrison alive P2 building per env -----
    p2_owned = alive & (owners == C.OWNER_P2)
    has_p2_src = jnp.any(p2_owned, axis=1)                                    # (N,)
    SENT_LO  = jnp.iinfo(jnp.int32).min
    src_pool = jnp.where(p2_owned, garr, SENT_LO)
    src_idx  = jnp.argmax(src_pool, axis=1)                                   # (N,)
    src_g    = jnp.take_along_axis(garr, src_idx[:, None], axis=1).squeeze(1) # (N,)

    # ----- Phase A target: lowest-garrison alive neutral -----
    neutral_alive = alive & (owners == C.OWNER_NEUTRAL)
    has_neutral   = jnp.any(neutral_alive, axis=1)                            # (N,)
    SENT_HI       = jnp.iinfo(jnp.int32).max
    neutral_pool  = jnp.where(neutral_alive, garr, SENT_HI)
    neutral_tgt   = jnp.argmin(neutral_pool, axis=1)                          # (N,)

    # ----- Phase B target: lowest-garrison alive enemy (P1) -----
    p1_alive = alive & (owners == C.OWNER_P1)
    has_p1   = jnp.any(p1_alive, axis=1)                                      # (N,)
    p1_pool  = jnp.where(p1_alive, garr, SENT_HI)
    p1_tgt   = jnp.argmin(p1_pool, axis=1)                                    # (N,)

    target_is_neutral = has_neutral
    has_target = has_neutral | has_p1
    tgt_idx = jnp.where(has_neutral, neutral_tgt, p1_tgt)                     # (N,)
    tgt_g   = jnp.take_along_axis(garr, tgt_idx[:, None], axis=1).squeeze(1)  # (N,)

    # ----- Capture feasibility: 0.75 * src_g > tgt_g * (1.0 or DEF_BONUS) -----
    attacker = (src_g * 75) // 100
    def_neutral = tgt_g
    def_enemy   = (tgt_g * C.DEF_BONUS_NUM) // C.DEF_BONUS_DEN
    defender    = jnp.where(target_is_neutral, def_neutral, def_enemy)
    can_capture = attacker > defender

    # ----- Build action: encode(TYPE_75=2, src, tgt) -----
    TYPE_75 = jnp.int64(2)
    action = TYPE_75 * SLOTS_SQ + src_idx.astype(jnp.int64) * C.MAX_BUILDING_SLOTS + tgt_idx.astype(jnp.int64)

    # NOOP if any precondition fails OR src==tgt OR action illegal per mask.
    src_eq_tgt = src_idx == tgt_idx
    use_noop_pre = (~has_p2_src) | (~has_target) | (~can_capture) | src_eq_tgt
    # Mask check — clamp action index to safe range first to avoid OOB read.
    safe_action = jnp.where(use_noop_pre, jnp.int64(NOOP_INDEX), action)
    is_legal = jnp.take_along_axis(p2_mask, safe_action[:, None], axis=1).squeeze(1)
    use_noop = use_noop_pre | (~is_legal)

    return jnp.where(use_noop, jnp.int64(NOOP_INDEX), action)


@partial(jax.jit, static_argnames=("opponent_name",))
def pack_action_batch_jax(
    p1_actions: jnp.ndarray,    # (N,) int — flat P1 action indices
    p2_mask:    jnp.ndarray,    # (N, ACTION_SPACE_SIZE) bool — on device
    key:        jax.Array,      # PRNGKey for random_legal sampling
    opponent_name: str,
    # Optional state arrays for opponents that need them (greedy_capacity_aware).
    # Default None so noop/random_legal callers don't need to thread state.
    buildings_owner:    jnp.ndarray | None = None,
    buildings_garrison: jnp.ndarray | None = None,
    buildings_alive:    jnp.ndarray | None = None,
) -> jnp.ndarray:
    """On-device mirror of `_pack_action_batch_with_p2_mask`.

    Returns (N, 2, ACTION_DIM) int32 — P1 action in row 0, P2 action in row 1.
    `opponent_name` is a static argument; supported values: "noop",
    "random_legal", "greedy_capacity_aware". Neural opponents take a separate
    code path on host (`_pack_action_batch_neural` in fused_rollout.py).

    `buildings_owner`/`buildings_garrison`/`buildings_alive` only required
    when opponent_name == "greedy_capacity_aware".
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
    elif opponent_name == "greedy_capacity_aware":
        if buildings_owner is None:
            raise ValueError(
                "greedy_capacity_aware requires state arrays "
                "(buildings_owner / buildings_garrison / buildings_alive)."
            )
        p2_idx = greedy_capacity_aware_opponent_jax(
            buildings_owner, buildings_garrison, buildings_alive, p2_mask,
        )
        p2_decoded = decode_to_slot_jax(p2_idx)
    else:
        raise ValueError(
            f"opponent_name {opponent_name!r} not supported in jax pack; "
            "use the host-side _pack_action_batch_neural for neural opponents."
        )

    return jnp.stack([p1_decoded, p2_decoded], axis=1)
