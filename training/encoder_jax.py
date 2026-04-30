"""
JAX-batched mirror of `training.encoder.encode_obs` (v10).

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

v10 additions vs v9.0: prod/wasted/share_live/reward_delta globals,
action history (5 own + 5 opp × 6 dims = 60), and per-bldg
hostile_landed / friendly_landed / ownership_changed. The `type_oh`
block is dropped (dead under TYPE_BASIC). See encoder.py docstring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np

from sim import config as C
from sim.state_jax import StateJax
from training.encoder import (
    ACTION_HIST_FEATS,
    BUILDING_COUNT_NORM,
    BUILDING_FEATS,
    CAP_NORM,
    COUNT_SUM_NORM,
    GLOBAL_FEATS,
    GROUP_FEATS,
    HISTORY_K,
    N_BUILDINGS,
    N_GROUPS,
    OBS_DIM,
    PCT_NORM,
    POS_NORM,
    TIMEOUT_NORM,
)


# Send-percentage lookup as a JAX device constant — same source as the
# numpy encoder's `_PCT_LOOKUP` so the numbers are byte-identical.
_PCT_LOOKUP_JAX = jnp.asarray(_np.asarray(C.SEND_PERCENTAGES, dtype=_np.float32))


# ---------------------------------------------------------------------------
# Action-history encoding (one env)
# ---------------------------------------------------------------------------

def _encode_action_history_jax(history, b_x, b_y):
    """Mirror of `training.encoder._encode_action_history` for a single env.

    history: (HISTORY_K, 4) int8.
    b_x, b_y: (N_BUILDINGS,) float32.
    Returns: (HISTORY_K * 6,) float32.

    Per row: [src_x, src_y, tgt_x, tgt_y, pct, was_real]. Empty/noop rows
    zeroed out via was_real * everything.
    """
    kind     = history[:, 0].astype(jnp.int32)
    type_idx = history[:, 1].astype(jnp.int32)
    src      = history[:, 2].astype(jnp.int32)
    tgt      = history[:, 3].astype(jnp.int32)
    was_real = (kind == 1).astype(jnp.float32)

    src_c  = jnp.clip(src, 0, N_BUILDINGS - 1)
    tgt_c  = jnp.clip(tgt, 0, N_BUILDINGS - 1)
    type_c = jnp.clip(type_idx, 0, _PCT_LOOKUP_JAX.shape[0] - 1)

    src_x = (b_x[src_c] / POS_NORM) * was_real
    src_y = (b_y[src_c] / POS_NORM) * was_real
    tgt_x = (b_x[tgt_c] / POS_NORM) * was_real
    tgt_y = (b_y[tgt_c] / POS_NORM) * was_real
    pct   = (_PCT_LOOKUP_JAX[type_c] / PCT_NORM) * was_real

    rows = jnp.stack([src_x, src_y, tgt_x, tgt_y, pct, was_real], axis=-1)  # (K, 6)
    return rows.reshape(-1)


# ---------------------------------------------------------------------------
# Batched encoder
# ---------------------------------------------------------------------------

def encode_obs_batched(state: StateJax) -> jnp.ndarray:
    """Encode a batched StateJax into (N, OBS_DIM) float32.

    Mirrors `training.encoder.encode_obs` (v10) exactly. Lives at the
    boundary between `JaxVecEnv.step_chunk` and the torch policy in the
    fused rollout — outputs stay on device.
    """
    b_alive    = state.buildings_alive.astype(jnp.float32)        # (N, MAX_B)
    b_owner    = state.buildings_owner                            # (N, MAX_B) int8
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

    arrivals_p1 = state.arrivals_p1.astype(jnp.float32)
    arrivals_p2 = state.arrivals_p2.astype(jnp.float32)
    prev_owner  = state.prev_buildings_owner
    prev_p1_units = state.prev_p1_units_total.astype(jnp.float32)
    prev_p2_units = state.prev_p2_units_total.astype(jnp.float32)

    is_mine    = (b_owner == C.OWNER_P1).astype(jnp.float32) * b_alive
    is_enemy   = (b_owner == C.OWNER_P2).astype(jnp.float32) * b_alive
    is_neutral = (b_owner == C.OWNER_NEUTRAL).astype(jnp.float32) * b_alive
    cap_safe = jnp.where(b_capacity > 0, b_capacity, 1.0)
    garr_ratio = b_garrison / cap_safe

    travel_safe = jnp.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive
    g_is_mine  = (g_owner == C.OWNER_P1).astype(jnp.float32) * g_alive
    g_is_enemy = (g_owner == C.OWNER_P2).astype(jnp.float32) * g_alive

    # ---- Per-building incoming flight aggregates -----------------------
    # Numpy version: Python loop over alive groups, scatter-add into
    # incoming_p1/incoming_p2 by tgt slot.
    # JAX version: scatter via jax.ops.segment_sum on the per-batch dim.
    def _per_env_incoming(g_alive_e, g_owner_e, g_tgt_e, g_count_e):
        contrib_p1 = jnp.where(
            (g_alive_e > 0) & (g_owner_e == C.OWNER_P1),
            g_count_e, jnp.float32(0.0),
        )
        contrib_p2 = jnp.where(
            (g_alive_e > 0) & (g_owner_e == C.OWNER_P2),
            g_count_e, jnp.float32(0.0),
        )
        tgt_clip = jnp.clip(g_tgt_e, 0, N_BUILDINGS - 1)
        in_p1 = jax.ops.segment_sum(contrib_p1, tgt_clip, num_segments=N_BUILDINGS)
        in_p2 = jax.ops.segment_sum(contrib_p2, tgt_clip, num_segments=N_BUILDINGS)
        return in_p1, in_p2

    incoming_mine, incoming_enemy = jax.vmap(_per_env_incoming)(
        g_alive, g_owner, g_tgt, g_count
    )  # both (N, MAX_B)

    incoming_friendly = (
        jnp.where(is_mine  > 0, incoming_mine,  jnp.float32(0.0))
        + jnp.where(is_enemy > 0, incoming_enemy, jnp.float32(0.0))
    )
    incoming_hostile = (
        jnp.where(is_mine  > 0, incoming_enemy, jnp.float32(0.0))
        + jnp.where(is_enemy > 0, incoming_mine,  jnp.float32(0.0))
    )
    incoming_hostile = jnp.where(
        is_neutral > 0, incoming_mine + incoming_enemy, incoming_hostile
    )

    # ---- v10 event-explicit per-bldg features --------------------------
    friendly_landed = (
        jnp.where(is_mine  > 0, arrivals_p1, jnp.float32(0.0))
        + jnp.where(is_enemy > 0, arrivals_p2, jnp.float32(0.0))
    )
    hostile_landed = (
        jnp.where(is_mine  > 0, arrivals_p2, jnp.float32(0.0))
        + jnp.where(is_enemy > 0, arrivals_p1, jnp.float32(0.0))
    )
    hostile_landed = jnp.where(
        is_neutral > 0, arrivals_p1 + arrivals_p2, hostile_landed
    )
    ownership_changed = (
        (prev_owner != b_owner).astype(jnp.float32) * b_alive
    )

    # ---- Globals base (10) ---------------------------------------------
    tick      = state.tick.astype(jnp.float32)             # (N,)
    mine_bldgs    = is_mine.sum(axis=-1)
    enemy_bldgs   = is_enemy.sum(axis=-1)
    neutral_bldgs = is_neutral.sum(axis=-1)
    mine_garr     = jnp.where(is_mine  > 0, b_garrison, 0.0).sum(axis=-1)
    enemy_garr    = jnp.where(is_enemy > 0, b_garrison, 0.0).sum(axis=-1)
    mine_flight   = jnp.where(
        (g_owner == C.OWNER_P1) & (g_alive > 0), g_count, 0.0
    ).sum(axis=-1)
    enemy_flight  = jnp.where(
        (g_owner == C.OWNER_P2) & (g_alive > 0), g_count, 0.0
    ).sum(axis=-1)
    mine_total  = mine_garr  + mine_flight
    enemy_total = enemy_garr + enemy_flight
    tot_total   = mine_total + enemy_total + jnp.float32(1e-6)

    base_block = jnp.stack([
        tick / TIMEOUT_NORM,                                     # 0
        jnp.float32(1.0) - tick / TIMEOUT_NORM,                  # 1
        mine_bldgs    / BUILDING_COUNT_NORM,                     # 2
        enemy_bldgs   / BUILDING_COUNT_NORM,                     # 3
        neutral_bldgs / BUILDING_COUNT_NORM,                     # 4
        mine_total    / COUNT_SUM_NORM,                          # 5
        enemy_total   / COUNT_SUM_NORM,                          # 6
        mine_total    / tot_total,                               # 7
        (mine_bldgs - enemy_bldgs) / BUILDING_COUNT_NORM,        # 8
        (mine_total - enemy_total) / COUNT_SUM_NORM,             # 9
    ], axis=-1)  # (N, 10)

    # ---- v10 prod / wasted / topology / delta --------------------------
    below_cap   = b_garrison < b_capacity
    own_alive   = is_mine  > 0
    enemy_alive = is_enemy > 0
    prod_mine    = (own_alive   & below_cap).sum(axis=-1).astype(jnp.float32)
    prod_enemy   = (enemy_alive & below_cap).sum(axis=-1).astype(jnp.float32)
    wasted_mine  = (own_alive   & (~below_cap)).sum(axis=-1).astype(jnp.float32)
    wasted_enemy = (enemy_alive & (~below_cap)).sum(axis=-1).astype(jnp.float32)

    total_alive = mine_bldgs + enemy_bldgs + neutral_bldgs
    total_alive_safe = jnp.where(total_alive > 0, total_alive, jnp.float32(1.0))

    delta_mine  = mine_total  - prev_p1_units
    delta_enemy = enemy_total - prev_p2_units

    extra_block = jnp.stack([
        prod_mine    / BUILDING_COUNT_NORM,                      # 10
        prod_enemy   / BUILDING_COUNT_NORM,                      # 11
        wasted_mine  / BUILDING_COUNT_NORM,                      # 12
        wasted_enemy / BUILDING_COUNT_NORM,                      # 13
        total_alive  / BUILDING_COUNT_NORM,                      # 14
        mine_bldgs    / total_alive_safe,                        # 15
        enemy_bldgs   / total_alive_safe,                        # 16
        neutral_bldgs / total_alive_safe,                        # 17
        delta_mine  / COUNT_SUM_NORM,                            # 18
        delta_enemy / COUNT_SUM_NORM,                            # 19
    ], axis=-1)  # (N, 10)

    # ---- v10 action history (60 dims = 2 × HISTORY_K × 6) --------------
    # vmap the per-env encoder over the batch axis. Each call returns
    # (HISTORY_K * 6,); stack two for own + opp.
    own_hist = jax.vmap(_encode_action_history_jax)(
        state.last_actions_p1, b_x, b_y
    )  # (N, K*6)
    opp_hist = jax.vmap(_encode_action_history_jax)(
        state.last_actions_p2, b_x, b_y
    )

    globals_block = jnp.concatenate(
        [base_block, extra_block, own_hist, opp_hist], axis=-1
    )  # (N, GLOBAL_FEATS=80)

    # ---- Per-building block (32 × 20) ----------------------------------
    over_cap = ((b_garrison > b_capacity) & (b_alive > 0)).astype(jnp.float32)

    building_block = jnp.stack([
        b_alive,                                                  # 1
        is_mine,                                                  # 2
        is_enemy,                                                 # 3
        is_neutral,                                               # 4
        b_garrison / CAP_NORM,                                    # 5
        garr_ratio,                                               # 6
        b_capacity / CAP_NORM,                                    # 7
        over_cap,                                                 # 8
        b_x / POS_NORM,                                           # 9
        b_y / POS_NORM,                                           # 10
        incoming_mine  / CAP_NORM,                                # 11
        incoming_enemy / CAP_NORM,                                # 12
        incoming_friendly / CAP_NORM,                             # 13
        incoming_hostile  / CAP_NORM,                             # 14
        jnp.minimum(incoming_hostile, b_garrison) / CAP_NORM,     # 15
        (incoming_hostile > b_garrison).astype(jnp.float32) * b_alive,  # 16 will_fall
        (garr_ratio > jnp.float32(0.95)).astype(jnp.float32) * b_alive, # 17 near_cap
        hostile_landed   / CAP_NORM,                              # 18 v10
        friendly_landed  / CAP_NORM,                              # 19 v10
        ownership_changed,                                         # 20 v10
    ], axis=-1)  # (N, MAX_B, 20)

    # ---- Per-group block (32 × 9) --------------------------------------
    src_x = jnp.take_along_axis(b_x, g_src, axis=-1) / POS_NORM * g_alive
    src_y = jnp.take_along_axis(b_y, g_src, axis=-1) / POS_NORM * g_alive
    tgt_x = jnp.take_along_axis(b_x, g_tgt, axis=-1) / POS_NORM * g_alive
    tgt_y = jnp.take_along_axis(b_y, g_tgt, axis=-1) / POS_NORM * g_alive

    group_block = jnp.stack([
        g_alive,                                                  # 1
        g_is_mine,                                                # 2
        g_is_enemy,                                               # 3
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
