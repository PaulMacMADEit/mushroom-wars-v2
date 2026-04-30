"""v10 observation encoder.

Differences vs v9.0 (CURRICULUM_PLAN.md / encoder design discussion 2026-04-29):

  - Drop the `type_oh` block (5 cols × 32 buildings = 160 dead features). All
    current levels are TYPE_BASIC; col 0 duplicated `b_alive` and 1..4 were
    always zero.
  - Rename `is_p1` / `is_p2` / `is_n` (and group equivalents) to
    `is_mine` / `is_enemy` / `is_neutral`. Semantics unchanged — the env
    already mirrors P1↔P2 via `opponents._mirror_ownership` so the active
    player always reads as P1.
  - Globals: add `prod_rate_mine|enemy`, `wasted_prod_mine|enemy`,
    `total_alive_buildings`, `mine|enemy|neutral_share_live`,
    `reward_delta_mine|enemy` (units gained/lost since last decision),
    `last_5_own_actions` (5 × 6 = 30) and `last_5_opponent_actions` (5 × 6).
  - Per-building: add `hostile_units_landed_this_interval`,
    `friendly_units_landed_this_interval`, `ownership_changed_this_interval`.
    These are the v10 fix for the close-map signal-loss problem
    (DECISION_INTERVAL_TICKS=2, MIN_TRAVEL_TICKS=1 → groups can launch and
    arrive between two decisions, leaving no `incoming_*` trail).
  - Per-group: unchanged shape, just renamed fields to is_mine / is_enemy.

Shape:
  globals      (80)            10 base + 2 prod + 2 wasted + 1 total_live
                              + 3 share_live + 2 reward_delta
                              + 30 own history + 30 opp history
  per-building (32 × 20)       17 base (was 22 minus 5 type_oh)
                              + 3 event-explicit (hostile_landed, friendly_landed, flipped)
  per-group    (32 × 9)        unchanged

Total OBS_DIM = 80 + 640 + 288 = 1008.

The action-history rows come from `last_actions_p1/p2` (HISTORY_K, 4) int8
buffers maintained on State by the env at decision-interval boundaries.
Encoded per slot as [src_x, src_y, tgt_x, tgt_y, pct, was_real] (6 floats).
"""

from __future__ import annotations

import numpy as np

from sim import config as C


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

N_BUILDINGS = C.MAX_BUILDING_SLOTS     # 32
N_GROUPS    = C.MAX_UNIT_GROUP_SLOTS   # 32
HISTORY_K   = C.HISTORY_K              # 5

GLOBAL_BASE_FEATS    = 10
GLOBAL_PROD_FEATS    = 4   # prod_rate_mine, prod_rate_enemy, wasted_mine, wasted_enemy
GLOBAL_TOPO_FEATS    = 4   # total_alive + 3 × share_live
GLOBAL_DELTA_FEATS   = 2   # reward_delta_mine, reward_delta_enemy
ACTION_HIST_FEATS    = 6   # per stored action: src_x, src_y, tgt_x, tgt_y, pct, was_real
GLOBAL_HIST_FEATS    = 2 * HISTORY_K * ACTION_HIST_FEATS   # 60

GLOBAL_FEATS    = (
    GLOBAL_BASE_FEATS
    + GLOBAL_PROD_FEATS
    + GLOBAL_TOPO_FEATS
    + GLOBAL_DELTA_FEATS
    + GLOBAL_HIST_FEATS
)  # 10 + 4 + 4 + 2 + 60 = 80

BUILDING_FEATS  = 20
GROUP_FEATS     = 9

OBS_DIM = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS + N_GROUPS * GROUP_FEATS
# 80 + 640 + 288 = 1008

# Normalizers — chosen so all features land roughly in [0, ~3]. Welford in
# training/obs_norm.py re-centers after a few rollouts so exact scales don't
# matter much for learning — only for first-epoch magnitudes.
CAP_NORM         = float(C.DEFAULT_CAPACITY)       # 300 internal
POS_NORM         = 700.0                           # map spans 0..~700 map units
TIMEOUT_NORM     = float(C.GAME_TIMEOUT_TICKS)     # 200 ticks
TRAVEL_NORM      = float(C.MAX_TRAVEL_TICKS)       # 8
COUNT_SUM_NORM   = float(C.DEFAULT_CAPACITY * 4)   # ~total units on a side
BUILDING_COUNT_NORM = float(N_BUILDINGS)
PCT_NORM         = 100.0


# Send-percentage lookup as a numpy array — for vectorised pct lookup in
# the action-history encode path. type_idx in {0..3} → percentage in {25,50,75,100}.
_PCT_LOOKUP = np.asarray(C.SEND_PERCENTAGES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Action-history encoding
# ---------------------------------------------------------------------------

def _encode_action_history(
    history: np.ndarray,    # (HISTORY_K, 4) int8 — [kind, type_idx, src, tgt]
    b_x: np.ndarray,        # (N_BUILDINGS,) float32 — already cast
    b_y: np.ndarray,        # (N_BUILDINGS,) float32
) -> np.ndarray:
    """Return (HISTORY_K * 6,) float32. Per row: [src_x, src_y, tgt_x, tgt_y,
    pct, was_real]. Empty/noop rows are all zero (mask via was_real).
    """
    kind     = history[:, 0].astype(np.int32)
    type_idx = history[:, 1].astype(np.int32)
    src      = history[:, 2].astype(np.int32)
    tgt      = history[:, 3].astype(np.int32)
    was_real = (kind == 1).astype(np.float32)

    # Clamp src/tgt into valid range so the gathers are safe even on noop rows;
    # multiplying by was_real zeroes those out anyway.
    src_c = np.clip(src, 0, N_BUILDINGS - 1)
    tgt_c = np.clip(tgt, 0, N_BUILDINGS - 1)
    type_c = np.clip(type_idx, 0, _PCT_LOOKUP.shape[0] - 1)

    src_x = (b_x[src_c] / POS_NORM) * was_real
    src_y = (b_y[src_c] / POS_NORM) * was_real
    tgt_x = (b_x[tgt_c] / POS_NORM) * was_real
    tgt_y = (b_y[tgt_c] / POS_NORM) * was_real
    pct   = (_PCT_LOOKUP[type_c] / PCT_NORM) * was_real

    rows = np.stack([src_x, src_y, tgt_x, tgt_y, pct, was_real], axis=1)  # (K, 6)
    return rows.reshape(-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_obs(obs: dict) -> np.ndarray:
    """Convert a MushroomEnv obs dict into (OBS_DIM,) float32.

    `obs` is whatever MushroomEnv.step/reset produced for a single env. For
    vec-env obs (dict of stacked arrays), the caller loops and encodes each.

    Caller-perspective: when this is called by the opponent path,
    `_mirror_ownership` has already swapped P1↔P2 throughout — so reading
    `buildings_owner == OWNER_P1` consistently means "is mine".
    """
    out = np.empty(OBS_DIM, dtype=np.float32)

    # ---- Unpack + normalize arrays once (shared across blocks) ----------
    b_alive    = obs["buildings_alive"].astype(np.float32)        # (N,)
    b_owner    = obs["buildings_owner"]                           # (N,) int8
    b_garrison = obs["buildings_garrison"].astype(np.float32)
    b_capacity = obs["buildings_capacity"].astype(np.float32)
    b_x        = obs["buildings_x"].astype(np.float32)
    b_y        = obs["buildings_y"].astype(np.float32)

    g_alive    = obs["groups_alive"].astype(np.float32)           # (M,)
    g_owner    = obs["groups_owner"]
    g_src      = obs["groups_src"].astype(np.int64)
    g_tgt      = obs["groups_tgt"].astype(np.int64)
    g_count    = obs["groups_count"].astype(np.float32)
    g_progress = obs["groups_progress"].astype(np.float32)
    g_travel   = obs["groups_travel"].astype(np.float32)

    arrivals_p1          = obs["arrivals_p1"].astype(np.float32)
    arrivals_p2          = obs["arrivals_p2"].astype(np.float32)
    prev_owner           = obs["prev_buildings_owner"]
    prev_p1_units        = float(obs["prev_p1_units_total"])
    prev_p2_units        = float(obs["prev_p2_units_total"])
    last_actions_p1      = np.asarray(obs["last_actions_p1"], dtype=np.int8)
    last_actions_p2      = np.asarray(obs["last_actions_p2"], dtype=np.int8)

    is_mine     = (b_owner == C.OWNER_P1).astype(np.float32) * b_alive
    is_enemy    = (b_owner == C.OWNER_P2).astype(np.float32) * b_alive
    is_neutral  = (b_owner == C.OWNER_NEUTRAL).astype(np.float32) * b_alive
    cap_safe = np.where(b_capacity > 0, b_capacity, 1.0)
    garr_ratio = b_garrison / cap_safe

    travel_safe = np.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive
    g_is_mine  = (g_owner == C.OWNER_P1).astype(np.float32) * g_alive
    g_is_enemy = (g_owner == C.OWNER_P2).astype(np.float32) * g_alive

    # ---- Per-building incoming flight aggregates ------------------------
    # For each building b, sum counts of alive groups targeting b, split by
    # friend/foe relative to b's owner.
    incoming_mine  = np.zeros(N_BUILDINGS, dtype=np.float32)
    incoming_enemy = np.zeros(N_BUILDINGS, dtype=np.float32)
    inflight_alive_idx = np.where(g_alive > 0)[0]
    for gi in inflight_alive_idx:
        tgt = int(g_tgt[gi])
        if 0 <= tgt < N_BUILDINGS:
            if g_owner[gi] == C.OWNER_P1:
                incoming_mine[tgt] += g_count[gi]
            elif g_owner[gi] == C.OWNER_P2:
                incoming_enemy[tgt] += g_count[gi]

    incoming_friendly = (
        np.where(is_mine > 0,  incoming_mine,  0.0)
        + np.where(is_enemy > 0, incoming_enemy, 0.0)
    )
    incoming_hostile = (
        np.where(is_mine > 0,  incoming_enemy, 0.0)
        + np.where(is_enemy > 0, incoming_mine,  0.0)
    )
    # Neutrals: all incoming is hostile.
    incoming_hostile = np.where(is_neutral > 0, incoming_mine + incoming_enemy, incoming_hostile)

    # ---- v10 event-explicit per-bldg features ---------------------------
    friendly_landed = (
        np.where(is_mine  > 0, arrivals_p1, 0.0)
        + np.where(is_enemy > 0, arrivals_p2, 0.0)
    )
    hostile_landed = (
        np.where(is_mine  > 0, arrivals_p2, 0.0)
        + np.where(is_enemy > 0, arrivals_p1, 0.0)
    )
    hostile_landed = np.where(
        is_neutral > 0, arrivals_p1 + arrivals_p2, hostile_landed
    )
    ownership_changed = (
        (prev_owner != b_owner).astype(np.float32) * b_alive
    )

    # ---- Globals base (10) ----------------------------------------------
    tick      = float(obs["tick"])
    mine_bldgs    = float(is_mine.sum())
    enemy_bldgs   = float(is_enemy.sum())
    neutral_bldgs = float(is_neutral.sum())
    mine_garr     = float(b_garrison[is_mine  > 0].sum())
    enemy_garr    = float(b_garrison[is_enemy > 0].sum())
    mine_flight   = float(g_count[(g_owner == C.OWNER_P1) & (g_alive > 0)].sum())
    enemy_flight  = float(g_count[(g_owner == C.OWNER_P2) & (g_alive > 0)].sum())
    mine_total    = mine_garr  + mine_flight
    enemy_total   = enemy_garr + enemy_flight
    tot_total     = mine_total + enemy_total + 1e-6

    out[0] = tick / TIMEOUT_NORM
    out[1] = 1.0 - tick / TIMEOUT_NORM
    out[2] = mine_bldgs    / BUILDING_COUNT_NORM
    out[3] = enemy_bldgs   / BUILDING_COUNT_NORM
    out[4] = neutral_bldgs / BUILDING_COUNT_NORM
    out[5] = mine_total    / COUNT_SUM_NORM
    out[6] = enemy_total   / COUNT_SUM_NORM
    out[7] = mine_total    / tot_total
    out[8] = (mine_bldgs - enemy_bldgs) / BUILDING_COUNT_NORM
    out[9] = (mine_total - enemy_total) / COUNT_SUM_NORM

    # ---- v10 globals: prod / wasted / topology / delta / history --------
    # prod_rate per side = number of own buildings currently regenerating
    # (alive AND owned AND below capacity). All buildings produce 1 real
    # unit/tick under TYPE_BASIC, so the count IS the rate.
    below_cap = b_garrison < b_capacity
    own_alive_mask  = is_mine  > 0
    enemy_alive_mask = is_enemy > 0
    prod_mine    = float((own_alive_mask  & below_cap).sum())
    prod_enemy   = float((enemy_alive_mask & below_cap).sum())
    # Wasted production: own/enemy bldgs at-or-over capacity (no regen this tick).
    wasted_mine  = float((own_alive_mask  & (~below_cap)).sum())
    wasted_enemy = float((enemy_alive_mask & (~below_cap)).sum())

    out[10] = prod_mine    / BUILDING_COUNT_NORM
    out[11] = prod_enemy   / BUILDING_COUNT_NORM
    out[12] = wasted_mine  / BUILDING_COUNT_NORM
    out[13] = wasted_enemy / BUILDING_COUNT_NORM

    # Topology (4 dims).
    total_alive = mine_bldgs + enemy_bldgs + neutral_bldgs
    total_alive_safe = total_alive if total_alive > 0 else 1.0
    out[14] = total_alive / BUILDING_COUNT_NORM
    out[15] = mine_bldgs    / total_alive_safe
    out[16] = enemy_bldgs   / total_alive_safe
    out[17] = neutral_bldgs / total_alive_safe

    # Reward delta (2 dims): change in total internal units since last
    # decision-interval boundary, normalised by COUNT_SUM_NORM. Positive
    # for mine = "we gained units"; positive for enemy = "they gained".
    delta_mine  = mine_total  - prev_p1_units
    delta_enemy = enemy_total - prev_p2_units
    out[18] = delta_mine  / COUNT_SUM_NORM
    out[19] = delta_enemy / COUNT_SUM_NORM

    # Action history (60 dims).
    own_hist = _encode_action_history(last_actions_p1, b_x, b_y)
    opp_hist = _encode_action_history(last_actions_p2, b_x, b_y)
    out[20:20 + HISTORY_K * ACTION_HIST_FEATS]                              = own_hist
    out[20 + HISTORY_K * ACTION_HIST_FEATS:GLOBAL_FEATS]                    = opp_hist

    # ---- Per-building block (32 × 20 = 640) -----------------------------
    over_cap = ((b_garrison > b_capacity) & (b_alive > 0)).astype(np.float32)

    building_block = np.stack([
        b_alive,                                          # 1
        is_mine,                                          # 2
        is_enemy,                                         # 3
        is_neutral,                                       # 4
        b_garrison / CAP_NORM,                            # 5
        garr_ratio,                                       # 6
        b_capacity / CAP_NORM,                            # 7
        over_cap,                                         # 8
        b_x / POS_NORM,                                   # 9
        b_y / POS_NORM,                                   # 10
        incoming_mine  / CAP_NORM,                        # 11
        incoming_enemy / CAP_NORM,                        # 12
        incoming_friendly / CAP_NORM,                     # 13
        incoming_hostile  / CAP_NORM,                     # 14
        np.minimum(incoming_hostile, b_garrison) / CAP_NORM,           # 15
        (incoming_hostile > b_garrison).astype(np.float32) * b_alive,  # 16  will_fall
        (garr_ratio > 0.95).astype(np.float32) * b_alive,              # 17  near_cap
        hostile_landed   / CAP_NORM,                                   # 18 v10
        friendly_landed  / CAP_NORM,                                   # 19 v10
        ownership_changed,                                              # 20 v10
    ], axis=1)  # (N, 20)
    out[GLOBAL_FEATS : GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS] = building_block.reshape(-1)

    # ---- Per-group block (32 × 9 = 288) --------------------------------
    src_x = b_x[g_src] / POS_NORM * g_alive
    src_y = b_y[g_src] / POS_NORM * g_alive
    tgt_x = b_x[g_tgt] / POS_NORM * g_alive
    tgt_y = b_y[g_tgt] / POS_NORM * g_alive

    group_block = np.stack([
        g_alive,                                          # 1
        g_is_mine,                                        # 2
        g_is_enemy,                                       # 3
        g_frac,                                           # 4
        (g_count / CAP_NORM) * g_alive,                   # 5
        src_x,                                            # 6
        src_y,                                            # 7
        tgt_x,                                            # 8
        tgt_y,                                            # 9
    ], axis=1)  # (M, 9)
    out[GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS :] = group_block.reshape(-1)

    return out
