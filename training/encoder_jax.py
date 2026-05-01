"""JAX-batched mirror of `training.encoder.encode_obs` (v12).

Same shape/contract as the numpy reference encoder. Used by fused rollout
to keep encoded obs on device across the chunked rollout instead of a
per-tick host roundtrip.

The per-group "incoming flight" loop in numpy is rewritten as a
`segment_sum` over (tgt, owner). Result is byte-identical because
addition order doesn't matter for the small integer counts involved.
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
    OWNER_ID_ENEMY,
    OWNER_ID_MINE,
    OWNER_ID_NEUTRAL,
    PCT_NORM,
    POS_NORM,
    TIMEOUT_NORM,
)


_PCT_LOOKUP_JAX = jnp.asarray(_np.asarray(C.SEND_PERCENTAGES, dtype=_np.float32))


def _encode_action_history_jax(history, b_x, b_y):
    """history: (HISTORY_K, 4) int8. b_x, b_y: (N_BUILDINGS,) float32.
    Returns (HISTORY_K * 6,) float32."""
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

    rows = jnp.stack([src_x, src_y, tgt_x, tgt_y, pct, was_real], axis=-1)
    return rows.reshape(-1)


def encode_obs_batched(state: StateJax) -> jnp.ndarray:
    """Encode a batched StateJax into (N, OBS_DIM) float32."""
    b_alive    = state.buildings_alive.astype(jnp.float32)
    b_owner    = state.buildings_owner
    b_garrison = state.buildings_garrison.astype(jnp.float32)
    b_capacity = state.buildings_capacity.astype(jnp.float32)
    b_x        = state.buildings_x.astype(jnp.float32)
    b_y        = state.buildings_y.astype(jnp.float32)

    g_alive    = state.groups_alive.astype(jnp.float32)
    g_owner    = state.groups_owner
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

    travel_safe = jnp.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive

    # ---- Per-building incoming flight aggregates (vmap'd segment_sum) ----
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
    )

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

    # ---- Owner-ID scalar per building / group ----
    owner_id_b = jnp.full(b_alive.shape, jnp.float32(OWNER_ID_NEUTRAL))
    owner_id_b = jnp.where(is_mine  > 0, jnp.float32(OWNER_ID_MINE),  owner_id_b)
    owner_id_b = jnp.where(is_enemy > 0, jnp.float32(OWNER_ID_ENEMY), owner_id_b)

    g_owner_id = jnp.full(g_alive.shape, jnp.float32(OWNER_ID_NEUTRAL))
    g_owner_id = jnp.where(
        (g_owner == C.OWNER_P1) & (g_alive > 0), jnp.float32(OWNER_ID_MINE), g_owner_id
    )
    g_owner_id = jnp.where(
        (g_owner == C.OWNER_P2) & (g_alive > 0), jnp.float32(OWNER_ID_ENEMY), g_owner_id
    )

    # ---- Globals ---------------------------------------------------------
    tick = state.tick.astype(jnp.float32)
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
        tick / TIMEOUT_NORM,
        jnp.float32(1.0) - tick / TIMEOUT_NORM,
        mine_bldgs    / BUILDING_COUNT_NORM,
        enemy_bldgs   / BUILDING_COUNT_NORM,
        neutral_bldgs / BUILDING_COUNT_NORM,
        mine_total    / COUNT_SUM_NORM,
        enemy_total   / COUNT_SUM_NORM,
        mine_total    / tot_total,
        (mine_bldgs - enemy_bldgs) / BUILDING_COUNT_NORM,
        (mine_total - enemy_total) / COUNT_SUM_NORM,
    ], axis=-1)

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
        prod_mine    / BUILDING_COUNT_NORM,
        prod_enemy   / BUILDING_COUNT_NORM,
        wasted_mine  / BUILDING_COUNT_NORM,
        wasted_enemy / BUILDING_COUNT_NORM,
        total_alive  / BUILDING_COUNT_NORM,
        mine_bldgs    / total_alive_safe,
        enemy_bldgs   / total_alive_safe,
        neutral_bldgs / total_alive_safe,
        delta_mine  / COUNT_SUM_NORM,
        delta_enemy / COUNT_SUM_NORM,
    ], axis=-1)

    own_hist = jax.vmap(_encode_action_history_jax)(
        state.last_actions_p1, b_x, b_y
    )
    opp_hist = jax.vmap(_encode_action_history_jax)(
        state.last_actions_p2, b_x, b_y
    )

    globals_block = jnp.concatenate(
        [base_block, extra_block, own_hist, opp_hist], axis=-1
    )

    # ---- Per-building block (N × MAX_B × 11) ----
    building_block = jnp.stack([
        b_alive,
        owner_id_b,
        b_garrison / CAP_NORM,
        b_capacity / CAP_NORM,
        b_x / POS_NORM,
        b_y / POS_NORM,
        incoming_friendly / CAP_NORM,
        incoming_hostile  / CAP_NORM,
        friendly_landed   / CAP_NORM,
        hostile_landed    / CAP_NORM,
        ownership_changed,
    ], axis=-1)

    # ---- Per-group block (N × MAX_G × 6) ----
    tgt_x = jnp.take_along_axis(b_x, g_tgt, axis=-1) / POS_NORM * g_alive
    tgt_y = jnp.take_along_axis(b_y, g_tgt, axis=-1) / POS_NORM * g_alive

    group_block = jnp.stack([
        g_alive,
        g_owner_id,
        g_frac,
        (g_count / CAP_NORM) * g_alive,
        tgt_x,
        tgt_y,
    ], axis=-1)

    n_envs = b_alive.shape[0]
    out = jnp.concatenate([
        globals_block,
        building_block.reshape(n_envs, N_BUILDINGS * BUILDING_FEATS),
        group_block.reshape(n_envs, N_GROUPS * GROUP_FEATS),
    ], axis=-1)
    return out


encode_obs_batched_jit = jax.jit(encode_obs_batched)


__all__ = ["encode_obs_batched", "encode_obs_batched_jit", "OBS_DIM"]
