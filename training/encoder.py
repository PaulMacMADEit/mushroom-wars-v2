"""v12 observation encoder — slot-token layout for set-transformer body.

Differences vs v10:

  - 8 building slots (was 32), 4 group slots (was 32). Smaller maps; mask
    compute scales as O(N²), so the slot cuts dominate sim throughput.
  - Per-building features cut 20 → 11. The shallow MLP body needed
    pre-computed shortcut flags (will_fall, near_cap, over_cap, threat_cap,
    is_mine/is_enemy/is_neutral one-hot, garr_ratio, raw incoming_p1/p2).
    Attention encoder can compute these on the fly during its rounds of
    inter-token communication. We keep the irreducible signals: alive,
    owner_id (single scalar; net embeds), garrison, capacity, position,
    incoming friendly/hostile, landed friendly/hostile, ownership_changed.
  - Per-group features cut 9 → 6. Drop src_x/y (the source token already
    carries that info); collapse is_mine/is_enemy → owner_id scalar.
  - Globals unchanged at 80 (base + prod/wasted + topo + delta + history).

Output is a FLAT (OBS_DIM,) float32 vector, same packing convention as v10
so PPOTrainer / RunningNorm / replay buffer code keeps working. The net
reshapes back into [GLOBAL ⨁ N_BLDG × B_FEATS ⨁ N_GROUP × G_FEATS] tokens.

Shape:
  globals      (80)             10 base + 2 prod + 2 wasted + 1 total_live
                                + 3 share_live + 2 reward_delta
                                + 30 own history + 30 opp history
  per-building (8 × 11 = 88)
  per-group    (4 × 6  = 24)

OBS_DIM = 80 + 88 + 24 = 192.

Caller-perspective: when this is called by the opponent path,
`_mirror_ownership` has already swapped P1↔P2 throughout — so reading
`buildings_owner == OWNER_P1` consistently means "is mine".
"""

from __future__ import annotations

import numpy as np

from sim import config as C


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

N_BUILDINGS = C.MAX_BUILDING_SLOTS     # 8
N_GROUPS    = C.MAX_UNIT_GROUP_SLOTS   # 4
HISTORY_K   = C.HISTORY_K              # 5

GLOBAL_BASE_FEATS    = 10
GLOBAL_PROD_FEATS    = 4
GLOBAL_TOPO_FEATS    = 4
GLOBAL_DELTA_FEATS   = 2
ACTION_HIST_FEATS    = 6   # per stored action: src_x, src_y, tgt_x, tgt_y, pct, was_real
GLOBAL_HIST_FEATS    = 2 * HISTORY_K * ACTION_HIST_FEATS   # 60

GLOBAL_FEATS    = (
    GLOBAL_BASE_FEATS
    + GLOBAL_PROD_FEATS
    + GLOBAL_TOPO_FEATS
    + GLOBAL_DELTA_FEATS
    + GLOBAL_HIST_FEATS
)  # 80

BUILDING_FEATS  = 11
GROUP_FEATS     = 6

OBS_DIM = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS + N_GROUPS * GROUP_FEATS
# 80 + 88 + 24 = 192


# Building-token field offsets (consumed by net.py to slice owner_id out
# for embedding lookup). See `building_block` below for the column order.
BLDG_FEAT_ALIVE              = 0
BLDG_FEAT_OWNER_ID           = 1
BLDG_FEAT_GARRISON           = 2
BLDG_FEAT_CAPACITY           = 3
BLDG_FEAT_X                  = 4
BLDG_FEAT_Y                  = 5
BLDG_FEAT_INCOMING_FRIENDLY  = 6
BLDG_FEAT_INCOMING_HOSTILE   = 7
BLDG_FEAT_FRIENDLY_LANDED    = 8
BLDG_FEAT_HOSTILE_LANDED     = 9
BLDG_FEAT_OWNERSHIP_CHANGED  = 10

# Group-token field offsets.
GRP_FEAT_ALIVE     = 0
GRP_FEAT_OWNER_ID  = 1
GRP_FEAT_PROGRESS  = 2
GRP_FEAT_COUNT     = 3
GRP_FEAT_TGT_X     = 4
GRP_FEAT_TGT_Y     = 5


# Normalizers — all features land roughly in [0, ~3].
CAP_NORM         = float(C.DEFAULT_CAPACITY)       # 300 internal
POS_NORM         = 700.0
TIMEOUT_NORM     = float(C.GAME_TIMEOUT_TICKS)
TRAVEL_NORM      = float(C.MAX_TRAVEL_TICKS)
COUNT_SUM_NORM   = float(C.DEFAULT_CAPACITY * 4)
BUILDING_COUNT_NORM = float(N_BUILDINGS)
PCT_NORM         = 100.0


# Owner-ID encoding for the encoder output. The net side embeds these into
# a small learned vector. Mapping:
#   neutral → 0,  mine → 1,  enemy → 2.
# Different ordering from C.OWNER_* so 0 = "no signal" matches alive=0.
OWNER_ID_NEUTRAL = 0
OWNER_ID_MINE    = 1
OWNER_ID_ENEMY   = 2
NUM_OWNER_IDS    = 3


_PCT_LOOKUP = np.asarray(C.SEND_PERCENTAGES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Action-history encoding
# ---------------------------------------------------------------------------

def _encode_action_history(
    history: np.ndarray,
    b_x: np.ndarray,
    b_y: np.ndarray,
) -> np.ndarray:
    """Return (HISTORY_K * 6,) float32. Per row: [src_x, src_y, tgt_x, tgt_y,
    pct, was_real]. Empty/noop rows zero (mask via was_real)."""
    kind     = history[:, 0].astype(np.int32)
    type_idx = history[:, 1].astype(np.int32)
    src      = history[:, 2].astype(np.int32)
    tgt      = history[:, 3].astype(np.int32)
    was_real = (kind == 1).astype(np.float32)

    src_c = np.clip(src, 0, N_BUILDINGS - 1)
    tgt_c = np.clip(tgt, 0, N_BUILDINGS - 1)
    type_c = np.clip(type_idx, 0, _PCT_LOOKUP.shape[0] - 1)

    src_x = (b_x[src_c] / POS_NORM) * was_real
    src_y = (b_y[src_c] / POS_NORM) * was_real
    tgt_x = (b_x[tgt_c] / POS_NORM) * was_real
    tgt_y = (b_y[tgt_c] / POS_NORM) * was_real
    pct   = (_PCT_LOOKUP[type_c] / PCT_NORM) * was_real

    rows = np.stack([src_x, src_y, tgt_x, tgt_y, pct, was_real], axis=1)
    return rows.reshape(-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_obs_batched_numpy(batched: dict) -> np.ndarray:
    """Vectorised numpy mirror of `encode_obs` for N stacked states.

    `batched` is a dict-of-numpy-arrays where each array has a leading N axis.
    Returns (N, OBS_DIM) float32. Used by the neural-opponent batched path
    in sim.envs.opponents.batch_act — calling encode_obs in a Python loop
    over 1800 states was the dominant cost in self-play. This vectorises
    the inner ops over the batch axis.

    Required keys (all mirrored if the caller is encoding from the opponent's
    perspective): buildings_alive (N, MAX_B), buildings_owner, buildings_garrison,
    buildings_capacity, buildings_x, buildings_y; groups_alive (N, MAX_G),
    groups_owner, groups_tgt, groups_count, groups_progress, groups_travel;
    arrivals_p1 (N, MAX_B), arrivals_p2; prev_buildings_owner; prev_p1_units_total
    (N,), prev_p2_units_total; last_actions_p1 (N, K, 4), last_actions_p2; tick (N,).
    """
    b_alive    = batched["buildings_alive"].astype(np.float32)        # (N, MAX_B)
    b_owner    = batched["buildings_owner"]                           # (N, MAX_B) int8
    b_garrison = batched["buildings_garrison"].astype(np.float32)
    b_capacity = batched["buildings_capacity"].astype(np.float32)
    b_x        = batched["buildings_x"].astype(np.float32)
    b_y        = batched["buildings_y"].astype(np.float32)

    g_alive    = batched["groups_alive"].astype(np.float32)           # (N, MAX_G)
    g_owner    = batched["groups_owner"]
    g_tgt      = batched["groups_tgt"].astype(np.int64)
    g_count    = batched["groups_count"].astype(np.float32)
    g_progress = batched["groups_progress"].astype(np.float32)
    g_travel   = batched["groups_travel"].astype(np.float32)

    arrivals_p1 = batched["arrivals_p1"].astype(np.float32)
    arrivals_p2 = batched["arrivals_p2"].astype(np.float32)
    prev_owner  = batched["prev_buildings_owner"]
    prev_p1     = batched["prev_p1_units_total"].astype(np.float32)
    prev_p2     = batched["prev_p2_units_total"].astype(np.float32)
    last_actions_p1 = np.asarray(batched["last_actions_p1"], dtype=np.int8)
    last_actions_p2 = np.asarray(batched["last_actions_p2"], dtype=np.int8)
    tick = batched["tick"].astype(np.float32)                         # (N,)

    N = b_alive.shape[0]

    is_mine    = (b_owner == C.OWNER_P1).astype(np.float32) * b_alive
    is_enemy   = (b_owner == C.OWNER_P2).astype(np.float32) * b_alive
    is_neutral = (b_owner == C.OWNER_NEUTRAL).astype(np.float32) * b_alive

    travel_safe = np.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive

    # ---- Per-building incoming flight aggregates (vectorised scatter) ----
    incoming_mine  = np.zeros((N, N_BUILDINGS), dtype=np.float32)
    incoming_enemy = np.zeros((N, N_BUILDINGS), dtype=np.float32)
    g_alive_bool = g_alive > 0
    g_is_p1 = (g_owner == C.OWNER_P1) & g_alive_bool
    g_is_p2 = (g_owner == C.OWNER_P2) & g_alive_bool
    tgt_clip = np.clip(g_tgt, 0, N_BUILDINGS - 1)
    # np.add.at handles repeated indices correctly (unlike a[i] += ...).
    env_idx = np.repeat(np.arange(N), g_alive.shape[1]).reshape(N, -1)
    np.add.at(incoming_mine,  (env_idx[g_is_p1],  tgt_clip[g_is_p1]),  g_count[g_is_p1])
    np.add.at(incoming_enemy, (env_idx[g_is_p2],  tgt_clip[g_is_p2]),  g_count[g_is_p2])

    incoming_friendly = (
        np.where(is_mine  > 0, incoming_mine,  0.0)
        + np.where(is_enemy > 0, incoming_enemy, 0.0)
    )
    incoming_hostile = (
        np.where(is_mine  > 0, incoming_enemy, 0.0)
        + np.where(is_enemy > 0, incoming_mine,  0.0)
    )
    incoming_hostile = np.where(is_neutral > 0, incoming_mine + incoming_enemy, incoming_hostile)

    friendly_landed = (
        np.where(is_mine  > 0, arrivals_p1, 0.0)
        + np.where(is_enemy > 0, arrivals_p2, 0.0)
    )
    hostile_landed = (
        np.where(is_mine  > 0, arrivals_p2, 0.0)
        + np.where(is_enemy > 0, arrivals_p1, 0.0)
    )
    hostile_landed = np.where(is_neutral > 0, arrivals_p1 + arrivals_p2, hostile_landed)
    ownership_changed = (prev_owner != b_owner).astype(np.float32) * b_alive

    owner_id_b = np.full((N, N_BUILDINGS), float(OWNER_ID_NEUTRAL), dtype=np.float32)
    owner_id_b = np.where(is_mine  > 0, np.float32(OWNER_ID_MINE),  owner_id_b)
    owner_id_b = np.where(is_enemy > 0, np.float32(OWNER_ID_ENEMY), owner_id_b)

    g_owner_id = np.full((N, N_GROUPS), float(OWNER_ID_NEUTRAL), dtype=np.float32)
    g_owner_id = np.where((g_owner == C.OWNER_P1) & g_alive_bool, np.float32(OWNER_ID_MINE),  g_owner_id)
    g_owner_id = np.where((g_owner == C.OWNER_P2) & g_alive_bool, np.float32(OWNER_ID_ENEMY), g_owner_id)

    # ---- Globals ---------------------------------------------------------
    mine_bldgs    = is_mine.sum(axis=-1)
    enemy_bldgs   = is_enemy.sum(axis=-1)
    neutral_bldgs = is_neutral.sum(axis=-1)
    mine_garr     = np.where(is_mine  > 0, b_garrison, 0.0).sum(axis=-1)
    enemy_garr    = np.where(is_enemy > 0, b_garrison, 0.0).sum(axis=-1)
    mine_flight   = np.where((g_owner == C.OWNER_P1) & g_alive_bool, g_count, 0.0).sum(axis=-1)
    enemy_flight  = np.where((g_owner == C.OWNER_P2) & g_alive_bool, g_count, 0.0).sum(axis=-1)
    mine_total  = mine_garr  + mine_flight
    enemy_total = enemy_garr + enemy_flight
    tot_total   = mine_total + enemy_total + 1e-6

    base_block = np.stack([
        tick / TIMEOUT_NORM,
        1.0 - tick / TIMEOUT_NORM,
        mine_bldgs    / BUILDING_COUNT_NORM,
        enemy_bldgs   / BUILDING_COUNT_NORM,
        neutral_bldgs / BUILDING_COUNT_NORM,
        mine_total    / COUNT_SUM_NORM,
        enemy_total   / COUNT_SUM_NORM,
        mine_total    / tot_total,
        (mine_bldgs - enemy_bldgs) / BUILDING_COUNT_NORM,
        (mine_total - enemy_total) / COUNT_SUM_NORM,
    ], axis=-1).astype(np.float32)

    below_cap   = b_garrison < b_capacity
    own_alive   = is_mine  > 0
    enemy_alive = is_enemy > 0
    prod_mine    = (own_alive   & below_cap).sum(axis=-1).astype(np.float32)
    prod_enemy   = (enemy_alive & below_cap).sum(axis=-1).astype(np.float32)
    wasted_mine  = (own_alive   & (~below_cap)).sum(axis=-1).astype(np.float32)
    wasted_enemy = (enemy_alive & (~below_cap)).sum(axis=-1).astype(np.float32)
    total_alive  = mine_bldgs + enemy_bldgs + neutral_bldgs
    total_alive_safe = np.where(total_alive > 0, total_alive, 1.0)
    delta_mine  = mine_total  - prev_p1
    delta_enemy = enemy_total - prev_p2

    extra_block = np.stack([
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
    ], axis=-1).astype(np.float32)

    # ---- Action history per side -----------------------------------------
    def _hist_block(history):  # (N, K, 4) → (N, K * 6)
        kind     = history[..., 0].astype(np.int32)
        type_idx = history[..., 1].astype(np.int32)
        src      = history[..., 2].astype(np.int32)
        tgt      = history[..., 3].astype(np.int32)
        was_real = (kind == 1).astype(np.float32)
        src_c  = np.clip(src, 0, N_BUILDINGS - 1)
        tgt_c  = np.clip(tgt, 0, N_BUILDINGS - 1)
        type_c = np.clip(type_idx, 0, _PCT_LOOKUP.shape[0] - 1)
        # Gather per-env per-K x/y/pct
        env_arange = np.arange(history.shape[0])[:, None]
        sx = (b_x[env_arange, src_c] / POS_NORM) * was_real
        sy = (b_y[env_arange, src_c] / POS_NORM) * was_real
        tx = (b_x[env_arange, tgt_c] / POS_NORM) * was_real
        ty = (b_y[env_arange, tgt_c] / POS_NORM) * was_real
        pct = (_PCT_LOOKUP[type_c] / PCT_NORM) * was_real
        rows = np.stack([sx, sy, tx, ty, pct, was_real], axis=-1)  # (N, K, 6)
        return rows.reshape(rows.shape[0], -1).astype(np.float32)

    own_hist = _hist_block(last_actions_p1)
    opp_hist = _hist_block(last_actions_p2)

    globals_block = np.concatenate([base_block, extra_block, own_hist, opp_hist], axis=-1)

    # ---- Per-building block (N, MAX_B, 11) -------------------------------
    building_block = np.stack([
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
    ], axis=-1).astype(np.float32)

    # ---- Per-group block (N, MAX_G, 6) -----------------------------------
    env_arange = np.arange(N)[:, None]
    g_tgt_clip = np.clip(g_tgt, 0, N_BUILDINGS - 1)
    tgt_x = b_x[env_arange, g_tgt_clip] / POS_NORM * g_alive
    tgt_y = b_y[env_arange, g_tgt_clip] / POS_NORM * g_alive

    group_block = np.stack([
        g_alive,
        g_owner_id,
        g_frac,
        (g_count / CAP_NORM) * g_alive,
        tgt_x,
        tgt_y,
    ], axis=-1).astype(np.float32)

    out = np.concatenate([
        globals_block,
        building_block.reshape(N, N_BUILDINGS * BUILDING_FEATS),
        group_block.reshape(N, N_GROUPS * GROUP_FEATS),
    ], axis=-1)
    return out


def encode_obs(obs: dict) -> np.ndarray:
    """Convert a MushroomEnv obs dict into (OBS_DIM,) float32."""
    out = np.empty(OBS_DIM, dtype=np.float32)

    b_alive    = obs["buildings_alive"].astype(np.float32)
    b_owner    = obs["buildings_owner"]
    b_garrison = obs["buildings_garrison"].astype(np.float32)
    b_capacity = obs["buildings_capacity"].astype(np.float32)
    b_x        = obs["buildings_x"].astype(np.float32)
    b_y        = obs["buildings_y"].astype(np.float32)

    g_alive    = obs["groups_alive"].astype(np.float32)
    g_owner    = obs["groups_owner"]
    g_tgt      = obs["groups_tgt"].astype(np.int64)
    g_count    = obs["groups_count"].astype(np.float32)
    g_progress = obs["groups_progress"].astype(np.float32)
    g_travel   = obs["groups_travel"].astype(np.float32)

    arrivals_p1     = obs["arrivals_p1"].astype(np.float32)
    arrivals_p2     = obs["arrivals_p2"].astype(np.float32)
    prev_owner      = obs["prev_buildings_owner"]
    prev_p1_units   = float(obs["prev_p1_units_total"])
    prev_p2_units   = float(obs["prev_p2_units_total"])
    last_actions_p1 = np.asarray(obs["last_actions_p1"], dtype=np.int8)
    last_actions_p2 = np.asarray(obs["last_actions_p2"], dtype=np.int8)

    is_mine    = (b_owner == C.OWNER_P1).astype(np.float32) * b_alive
    is_enemy   = (b_owner == C.OWNER_P2).astype(np.float32) * b_alive
    is_neutral = (b_owner == C.OWNER_NEUTRAL).astype(np.float32) * b_alive

    travel_safe = np.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive

    # ---- Per-building incoming flight aggregates ------------------------
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
    incoming_hostile = np.where(is_neutral > 0, incoming_mine + incoming_enemy, incoming_hostile)

    friendly_landed = (
        np.where(is_mine  > 0, arrivals_p1, 0.0)
        + np.where(is_enemy > 0, arrivals_p2, 0.0)
    )
    hostile_landed = (
        np.where(is_mine  > 0, arrivals_p2, 0.0)
        + np.where(is_enemy > 0, arrivals_p1, 0.0)
    )
    hostile_landed = np.where(is_neutral > 0, arrivals_p1 + arrivals_p2, hostile_landed)
    ownership_changed = (prev_owner != b_owner).astype(np.float32) * b_alive

    # ---- Owner-ID scalar (one column per slot, looked up to embedding in net) ----
    # is_mine / is_enemy are mirrored at the env boundary, so this scalar is
    # already in self-centred frame.
    owner_id = np.full(N_BUILDINGS, OWNER_ID_NEUTRAL, dtype=np.float32)
    owner_id = np.where(is_mine  > 0, OWNER_ID_MINE,    owner_id)
    owner_id = np.where(is_enemy > 0, OWNER_ID_ENEMY,   owner_id)
    # Dead slots: kept at neutral; alive=0 mask handles them downstream.

    g_owner_id = np.full(N_GROUPS, OWNER_ID_NEUTRAL, dtype=np.float32)
    g_owner_id = np.where((g_owner == C.OWNER_P1) & (g_alive > 0), OWNER_ID_MINE,  g_owner_id)
    g_owner_id = np.where((g_owner == C.OWNER_P2) & (g_alive > 0), OWNER_ID_ENEMY, g_owner_id)

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

    # ---- prod / wasted / topology / delta -------------------------------
    below_cap = b_garrison < b_capacity
    own_alive_mask  = is_mine  > 0
    enemy_alive_mask = is_enemy > 0
    prod_mine    = float((own_alive_mask  & below_cap).sum())
    prod_enemy   = float((enemy_alive_mask & below_cap).sum())
    wasted_mine  = float((own_alive_mask  & (~below_cap)).sum())
    wasted_enemy = float((enemy_alive_mask & (~below_cap)).sum())

    out[10] = prod_mine    / BUILDING_COUNT_NORM
    out[11] = prod_enemy   / BUILDING_COUNT_NORM
    out[12] = wasted_mine  / BUILDING_COUNT_NORM
    out[13] = wasted_enemy / BUILDING_COUNT_NORM

    total_alive = mine_bldgs + enemy_bldgs + neutral_bldgs
    total_alive_safe = total_alive if total_alive > 0 else 1.0
    out[14] = total_alive / BUILDING_COUNT_NORM
    out[15] = mine_bldgs    / total_alive_safe
    out[16] = enemy_bldgs   / total_alive_safe
    out[17] = neutral_bldgs / total_alive_safe

    delta_mine  = mine_total  - prev_p1_units
    delta_enemy = enemy_total - prev_p2_units
    out[18] = delta_mine  / COUNT_SUM_NORM
    out[19] = delta_enemy / COUNT_SUM_NORM

    own_hist = _encode_action_history(last_actions_p1, b_x, b_y)
    opp_hist = _encode_action_history(last_actions_p2, b_x, b_y)
    out[20:20 + HISTORY_K * ACTION_HIST_FEATS]                 = own_hist
    out[20 + HISTORY_K * ACTION_HIST_FEATS:GLOBAL_FEATS]       = opp_hist

    # ---- Per-building block (N_BUILDINGS × 11) --------------------------
    building_block = np.stack([
        b_alive,                                          # 0  alive
        owner_id,                                         # 1  owner_id ∈ {0,1,2}
        b_garrison / CAP_NORM,                            # 2  garrison
        b_capacity / CAP_NORM,                            # 3  capacity
        b_x / POS_NORM,                                   # 4  x
        b_y / POS_NORM,                                   # 5  y
        incoming_friendly / CAP_NORM,                     # 6  incoming friendly
        incoming_hostile  / CAP_NORM,                     # 7  incoming hostile
        friendly_landed   / CAP_NORM,                     # 8  friendly landed this interval
        hostile_landed    / CAP_NORM,                     # 9  hostile landed this interval
        ownership_changed,                                # 10 changed this interval
    ], axis=1)  # (N, 11)
    out[GLOBAL_FEATS : GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS] = building_block.reshape(-1)

    # ---- Per-group block (N_GROUPS × 6) ---------------------------------
    tgt_x = b_x[g_tgt] / POS_NORM * g_alive
    tgt_y = b_y[g_tgt] / POS_NORM * g_alive

    group_block = np.stack([
        g_alive,                                          # 0  alive
        g_owner_id,                                       # 1  owner_id ∈ {0,1,2}
        g_frac,                                           # 2  progress
        (g_count / CAP_NORM) * g_alive,                   # 3  count
        tgt_x,                                            # 4  target x
        tgt_y,                                            # 5  target y
    ], axis=1)  # (M, 6)
    out[GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS :] = group_block.reshape(-1)

    return out
