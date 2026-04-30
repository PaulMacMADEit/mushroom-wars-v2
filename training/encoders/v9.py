"""Full v9.0 observation encoder.

Three-block layout (ARCHITECTURE §9.2 — dimensions trimmed from the spec's
~1150 to ~1010 here by dropping action-history / reward-delta / rolling-stats
fields. Those require env-side tracking we don't have yet; they can be added
later under a new model_id without touching this one):

  globals      (10)           tick, side counts, totals, ratios
  per-building (32 × 22)      ownership, garrison, capacity, type, position,
                              incoming flight aggregates, basic derived flags
  per-group    (32 × 9)       ownership, progress, count, endpoint coordinates

Total = 10 + 704 + 288 = 1002 float32s.

Downstream consumers:
  - training.net.ActorCritic — takes OBS_DIM-wide float32 vectors
  - training.obs_norm.RunningNorm — (OBS_DIM,) running mean/std
  - training.trainer.PPOTrainer — batched per-env encode

We deliberately keep the encoder *stateless*: each call depends only on the
obs dict passed in. Training-time state (action history, reward deltas) can
live on the trainer side and be concatenated into a wider obs later.
"""

from __future__ import annotations

import numpy as np

from sim import config as C


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

N_BUILDINGS = C.MAX_BUILDING_SLOTS     # 32
N_GROUPS    = C.MAX_UNIT_GROUP_SLOTS   # 32

GLOBAL_FEATS    = 10
BUILDING_FEATS  = 22
GROUP_FEATS     = 9

OBS_DIM = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS + N_GROUPS * GROUP_FEATS
# 10 + 704 + 288 = 1002

# Normalizers — chosen so all features land roughly in [0, ~3]. Welford in
# training/obs_norm.py re-centers after a few rollouts so exact scales don't
# matter much for learning — only for first-epoch magnitudes.
CAP_NORM         = float(C.DEFAULT_CAPACITY)       # 300 internal
POS_NORM         = 700.0                           # map spans 0..~700 map units
TIMEOUT_NORM     = float(C.GAME_TIMEOUT_TICKS)     # 200 ticks
TRAVEL_NORM      = float(C.MAX_TRAVEL_TICKS)       # 8
COUNT_SUM_NORM   = float(C.DEFAULT_CAPACITY * 4)   # ~total units on a side
BUILDING_COUNT_NORM = float(N_BUILDINGS)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_obs(obs: dict) -> np.ndarray:
    """Convert a MushroomEnv obs dict into (OBS_DIM,) float32.

    `obs` is whatever MushroomEnv.step/reset produced for a single env. For
    vec-env obs (dict of stacked arrays), the caller loops and encodes each.
    """
    out = np.empty(OBS_DIM, dtype=np.float32)

    # ---- Unpack + normalize arrays once (shared across blocks) ----------
    b_alive    = obs["buildings_alive"].astype(np.float32)        # (N,)
    b_owner    = obs["buildings_owner"]                           # (N,) int8
    b_type     = obs["buildings_type"].astype(np.float32)         # (N,)
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

    is_p1 = (b_owner == C.OWNER_P1).astype(np.float32) * b_alive
    is_p2 = (b_owner == C.OWNER_P2).astype(np.float32) * b_alive
    is_n  = (b_owner == C.OWNER_NEUTRAL).astype(np.float32) * b_alive
    cap_safe = np.where(b_capacity > 0, b_capacity, 1.0)
    garr_ratio = b_garrison / cap_safe

    travel_safe = np.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe) * g_alive
    g_is_p1 = (g_owner == C.OWNER_P1).astype(np.float32) * g_alive
    g_is_p2 = (g_owner == C.OWNER_P2).astype(np.float32) * g_alive

    # ---- Per-building incoming flight aggregates ------------------------
    # For each building b, sum counts of alive groups targeting b, split by
    # friend/foe relative to b's owner (and absolute p1/p2 for downstream).
    incoming_p1 = np.zeros(N_BUILDINGS, dtype=np.float32)
    incoming_p2 = np.zeros(N_BUILDINGS, dtype=np.float32)
    inflight_alive_idx = np.where(g_alive > 0)[0]
    for gi in inflight_alive_idx:
        tgt = int(g_tgt[gi])
        if 0 <= tgt < N_BUILDINGS:
            if g_owner[gi] == C.OWNER_P1:
                incoming_p1[tgt] += g_count[gi]
            elif g_owner[gi] == C.OWNER_P2:
                incoming_p2[tgt] += g_count[gi]

    incoming_friendly = np.where(is_p1 > 0, incoming_p1, 0.0) + np.where(is_p2 > 0, incoming_p2, 0.0)
    incoming_hostile  = np.where(is_p1 > 0, incoming_p2, 0.0) + np.where(is_p2 > 0, incoming_p1, 0.0)
    # Neutrals: all incoming is hostile (to them) — approx useful for the
    # agent; it reads as "this neutral is under threat".
    incoming_hostile = np.where(is_n > 0, incoming_p1 + incoming_p2, incoming_hostile)

    # ---- Globals (10) ---------------------------------------------------
    tick      = float(obs["tick"])
    p1_bldgs  = float(is_p1.sum())
    p2_bldgs  = float(is_p2.sum())
    n_bldgs   = float(is_n.sum())
    p1_garr   = float(b_garrison[is_p1 > 0].sum())
    p2_garr   = float(b_garrison[is_p2 > 0].sum())
    p1_flight = float(g_count[(g_owner == C.OWNER_P1) & (g_alive > 0)].sum())
    p2_flight = float(g_count[(g_owner == C.OWNER_P2) & (g_alive > 0)].sum())
    p1_total  = p1_garr + p1_flight
    p2_total  = p2_garr + p2_flight

    out[0] = tick / TIMEOUT_NORM
    out[1] = 1.0 - tick / TIMEOUT_NORM            # time remaining
    out[2] = p1_bldgs / BUILDING_COUNT_NORM
    out[3] = p2_bldgs / BUILDING_COUNT_NORM
    out[4] = n_bldgs  / BUILDING_COUNT_NORM
    out[5] = p1_total / COUNT_SUM_NORM
    out[6] = p2_total / COUNT_SUM_NORM
    tot_total = p1_total + p2_total + 1e-6
    out[7] = p1_total / tot_total                  # share of units (p1)
    out[8] = (p1_bldgs - p2_bldgs) / BUILDING_COUNT_NORM   # building margin
    out[9] = (p1_total - p2_total) / COUNT_SUM_NORM        # unit margin

    # ---- Per-building block (32 × 22 = 704) -----------------------------
    over_cap = ((b_garrison > b_capacity) & (b_alive > 0)).astype(np.float32)
    type_oh = np.zeros((N_BUILDINGS, 5), dtype=np.float32)
    tidx = np.clip(b_type.astype(np.int64), 0, 4)
    type_oh[np.arange(N_BUILDINGS), tidx] = b_alive

    building_block = np.stack([
        b_alive,                                 # 1
        is_p1,                                   # 2
        is_p2,                                   # 3
        is_n,                                    # 4
        b_garrison / CAP_NORM,                   # 5   raw garrison magnitude
        garr_ratio,                              # 6   garrison / own capacity
        b_capacity / CAP_NORM,                   # 7
        over_cap,                                # 8   reinforced beyond cap
        b_x / POS_NORM,                          # 9
        b_y / POS_NORM,                          # 10
        type_oh[:, 0],                           # 11
        type_oh[:, 1],                           # 12
        type_oh[:, 2],                           # 13
        type_oh[:, 3],                           # 14
        type_oh[:, 4],                           # 15
        incoming_p1 / CAP_NORM,                  # 16
        incoming_p2 / CAP_NORM,                  # 17
        incoming_friendly / CAP_NORM,            # 18
        incoming_hostile  / CAP_NORM,            # 19
        np.minimum(incoming_hostile, b_garrison) / CAP_NORM,  # 20  threat cap
        (incoming_hostile > b_garrison).astype(np.float32) * b_alive,  # 21 "will fall"
        (garr_ratio > 0.95).astype(np.float32) * b_alive,     # 22  near-cap flag
    ], axis=1)  # (N, 22)
    out[GLOBAL_FEATS : GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS] = building_block.reshape(-1)

    # ---- Per-group block (32 × 9 = 288) --------------------------------
    # src/tgt encoded as their (x, y) positions (normalized) — positional,
    # more useful than slot ids for a non-attention body. Dead groups
    # masked to zero via multiplication by g_alive.
    src_x = b_x[g_src] / POS_NORM * g_alive
    src_y = b_y[g_src] / POS_NORM * g_alive
    tgt_x = b_x[g_tgt] / POS_NORM * g_alive
    tgt_y = b_y[g_tgt] / POS_NORM * g_alive

    group_block = np.stack([
        g_alive,                                 # 1
        g_is_p1,                                 # 2
        g_is_p2,                                 # 3
        g_frac,                                  # 4   fraction of travel done
        (g_count / CAP_NORM) * g_alive,          # 5
        src_x,                                   # 6
        src_y,                                   # 7
        tgt_x,                                   # 8
        tgt_y,                                   # 9
    ], axis=1)  # (M, 9)
    out[GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS :] = group_block.reshape(-1)

    return out
