"""
JAX-batched mirror of `training.encoder.encode_obs`.

`encode_obs_batched(state)` consumes a batched StateJax pytree (leading
dim = n_envs) and returns `(n_envs, OBS_DIM) jnp.float32` matching the
numpy `encode_obs` output element-for-element (within float32 epsilon).

Why a separate file: the numpy encoder stays the reference oracle for
non-fused training and the parity test. This module exists so the fused
rollout (`training/fused_rollout.py`) can keep encoded obs on device
across the chunked rollout instead of a per-tick host roundtrip.

Semantic note: the per-group "incoming flight" loop in the numpy
encoder is rewritten as a `segment_sum` over (tgt, owner). Result is
byte-identical because addition order doesn't matter for the small
integer counts involved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim import config as C
from sim.state_jax import StateJax
from training.encoder import (
    BUILDING_COUNT_NORM,
    BUILDING_FEATS,
    CAP_NORM,
    COUNT_SUM_NORM,
    GLOBAL_FEATS,
    GROUP_FEATS,
    N_BUILDINGS,
    N_GROUPS,
    OBS_DIM,
    POS_NORM,
    TIMEOUT_NORM,
)


# ---------------------------------------------------------------------------
# Batched encoder
# ---------------------------------------------------------------------------

def encode_obs_batched(state: StateJax) -> jnp.ndarray:
    """Encode a batched StateJax into (N, OBS_DIM) float32.

    Mirrors `training.encoder.encode_obs` exactly. Lives at the boundary
    between `JaxVecEnv.step_chunk` and the torch policy in the fused
    rollout — outputs stay on device.
    """
    # Leading dim handling: every state field is (N, …); we operate
    # element-wise plus a few `axis=-1` reductions, so vmap is unnecessary.
    b_alive    = state.buildings_alive.astype(jnp.float32)        # (N, MAX_B)
    b_owner    = state.buildings_owner                            # (N, MAX_B) int8
    b_type     = state.buildings_type.astype(jnp.float32)
    b_garrison = state.buildings_garrison.astype(jnp.float32)
    b_capacity = state.buildings_capacity.astype(jnp.float32)
    b_x        = state.buildings_x.astype(jnp.float32)
    b_y        = state.buildings_y.astype(jnp.float32)

    g_alive    = state.groups_alive.astype(jnp.float32)           # (N, MAX_G)
    g_owner    = state.groups_owner
    g_src      = state.groups_src.astype(jnp.int32)
    g_tgt      = state.groups_tgt.astype(jnp.int32)
    g_count    = state.groups_count.astype(jnp.float32)
    g_progress = state.groups_progress.astype(jnp.float32)
    g_travel   = state.groups_travel.astype(jnp.float32)

    is_p1 = (b_owner == C.OWNER_P1).astype(jnp.float32) * b_alive
    is_p2 = (b_owner == C.OWNER_P2).astype(jnp.float32) * b_alive
    is_n  = (b_owner == C.OWNER_NEUTRAL).astype(jnp.float32) * b_alive
    cap_safe = jnp.where(b_capacity > 0, b_capacity, 1.0)
    garr_ratio = b_garrison / cap_safe

    travel_safe = jnp.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive
    g_is_p1 = (g_owner == C.OWNER_P1).astype(jnp.float32) * g_alive
    g_is_p2 = (g_owner == C.OWNER_P2).astype(jnp.float32) * g_alive

    # ---- Per-building incoming flight aggregates -----------------------
    # Numpy version: Python loop over alive groups, scatter-add into
    # incoming_p1/incoming_p2 by tgt slot.
    # JAX version: scatter via jax.ops.segment_sum on the per-batch dim.
    # Build a flat (N * MAX_G,) contribution; segment by (env, tgt, owner).
    # Easier: do it per-env via vmap — keeps the segment count static.
    def _per_env_incoming(g_alive_e, g_owner_e, g_tgt_e, g_count_e):
        # Mask alive groups by owner.
        contrib_p1 = jnp.where(
            (g_alive_e > 0) & (g_owner_e == C.OWNER_P1),
            g_count_e, jnp.float32(0.0),
        )
        contrib_p2 = jnp.where(
            (g_alive_e > 0) & (g_owner_e == C.OWNER_P2),
            g_count_e, jnp.float32(0.0),
        )
        # tgt may be out of [0, N_BUILDINGS); clip so segment_sum is safe.
        tgt_clip = jnp.clip(g_tgt_e, 0, N_BUILDINGS - 1)
        in_p1 = jax.ops.segment_sum(contrib_p1, tgt_clip, num_segments=N_BUILDINGS)
        in_p2 = jax.ops.segment_sum(contrib_p2, tgt_clip, num_segments=N_BUILDINGS)
        return in_p1, in_p2

    incoming_p1, incoming_p2 = jax.vmap(_per_env_incoming)(
        g_alive, g_owner, g_tgt, g_count
    )  # both (N, MAX_B)

    incoming_friendly = (
        jnp.where(is_p1 > 0, incoming_p1, jnp.float32(0.0))
        + jnp.where(is_p2 > 0, incoming_p2, jnp.float32(0.0))
    )
    incoming_hostile = (
        jnp.where(is_p1 > 0, incoming_p2, jnp.float32(0.0))
        + jnp.where(is_p2 > 0, incoming_p1, jnp.float32(0.0))
    )
    # Neutrals: all incoming is hostile.
    incoming_hostile = jnp.where(
        is_n > 0, incoming_p1 + incoming_p2, incoming_hostile
    )

    # ---- Globals (10) --------------------------------------------------
    tick      = state.tick.astype(jnp.float32)             # (N,)
    p1_bldgs  = is_p1.sum(axis=-1)
    p2_bldgs  = is_p2.sum(axis=-1)
    n_bldgs   = is_n.sum(axis=-1)
    p1_garr   = jnp.where(is_p1 > 0, b_garrison, 0.0).sum(axis=-1)
    p2_garr   = jnp.where(is_p2 > 0, b_garrison, 0.0).sum(axis=-1)
    p1_flight = jnp.where(
        (g_owner == C.OWNER_P1) & (g_alive > 0), g_count, 0.0
    ).sum(axis=-1)
    p2_flight = jnp.where(
        (g_owner == C.OWNER_P2) & (g_alive > 0), g_count, 0.0
    ).sum(axis=-1)
    p1_total = p1_garr + p1_flight
    p2_total = p2_garr + p2_flight
    tot_total = p1_total + p2_total + jnp.float32(1e-6)

    globals_block = jnp.stack([
        tick / TIMEOUT_NORM,                                     # 0
        jnp.float32(1.0) - tick / TIMEOUT_NORM,                  # 1
        p1_bldgs / BUILDING_COUNT_NORM,                          # 2
        p2_bldgs / BUILDING_COUNT_NORM,                          # 3
        n_bldgs  / BUILDING_COUNT_NORM,                          # 4
        p1_total / COUNT_SUM_NORM,                               # 5
        p2_total / COUNT_SUM_NORM,                               # 6
        p1_total / tot_total,                                    # 7
        (p1_bldgs - p2_bldgs) / BUILDING_COUNT_NORM,             # 8
        (p1_total - p2_total) / COUNT_SUM_NORM,                  # 9
    ], axis=-1)  # (N, 10)

    # ---- Per-building block (32 × 22) ----------------------------------
    over_cap = ((b_garrison > b_capacity) & (b_alive > 0)).astype(jnp.float32)
    # type_oh: (N, MAX_B, 5). One-hot via comparison with arange.
    tidx = jnp.clip(b_type.astype(jnp.int32), 0, 4)              # (N, MAX_B)
    type_idx_grid = jnp.arange(5, dtype=jnp.int32)               # (5,)
    type_oh = (tidx[..., None] == type_idx_grid[None, None, :]).astype(jnp.float32) * b_alive[..., None]

    building_block = jnp.stack([
        b_alive,                                                  # 1
        is_p1,                                                    # 2
        is_p2,                                                    # 3
        is_n,                                                     # 4
        b_garrison / CAP_NORM,                                    # 5
        garr_ratio,                                               # 6
        b_capacity / CAP_NORM,                                    # 7
        over_cap,                                                 # 8
        b_x / POS_NORM,                                           # 9
        b_y / POS_NORM,                                           # 10
        type_oh[..., 0],                                          # 11
        type_oh[..., 1],                                          # 12
        type_oh[..., 2],                                          # 13
        type_oh[..., 3],                                          # 14
        type_oh[..., 4],                                          # 15
        incoming_p1 / CAP_NORM,                                   # 16
        incoming_p2 / CAP_NORM,                                   # 17
        incoming_friendly / CAP_NORM,                             # 18
        incoming_hostile  / CAP_NORM,                             # 19
        jnp.minimum(incoming_hostile, b_garrison) / CAP_NORM,     # 20
        (incoming_hostile > b_garrison).astype(jnp.float32) * b_alive,  # 21
        (garr_ratio > jnp.float32(0.95)).astype(jnp.float32) * b_alive,  # 22
    ], axis=-1)  # (N, MAX_B, 22)

    # ---- Per-group block (32 × 9) --------------------------------------
    # Gather b_x[g_src] / POS_NORM. b_x is (N, MAX_B), g_src is (N, MAX_G).
    src_x = jnp.take_along_axis(b_x, g_src, axis=-1) / POS_NORM * g_alive
    src_y = jnp.take_along_axis(b_y, g_src, axis=-1) / POS_NORM * g_alive
    tgt_x = jnp.take_along_axis(b_x, g_tgt, axis=-1) / POS_NORM * g_alive
    tgt_y = jnp.take_along_axis(b_y, g_tgt, axis=-1) / POS_NORM * g_alive

    group_block = jnp.stack([
        g_alive,                                                  # 1
        g_is_p1,                                                  # 2
        g_is_p2,                                                  # 3
        g_frac,                                                   # 4
        (g_count / CAP_NORM) * g_alive,                           # 5
        src_x,                                                    # 6
        src_y,                                                    # 7
        tgt_x,                                                    # 8
        tgt_y,                                                    # 9
    ], axis=-1)  # (N, MAX_G, 9)

    # ---- Concatenate to (N, OBS_DIM) -----------------------------------
    n_envs = b_alive.shape[0]
    out = jnp.concatenate([
        globals_block,
        building_block.reshape(n_envs, N_BUILDINGS * BUILDING_FEATS),
        group_block.reshape(n_envs, N_GROUPS * GROUP_FEATS),
    ], axis=-1)
    return out


# JIT-compiled wrapper. Static no-op arguments — shape is fixed by StateJax.
encode_obs_batched_jit = jax.jit(encode_obs_batched)


__all__ = ["encode_obs_batched", "encode_obs_batched_jit", "OBS_DIM"]
