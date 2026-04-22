"""Minimal observation encoder for Phase 2 smoke training.

Flattens the env's raw dict observation into a fixed-size float tensor. Scope
is intentionally narrow — just enough state to distinguish positions and
drive basic learning. The full v9.0 encoder (§9.2, ~1150 floats) arrives
after the loop is proven end-to-end.

Layout (289 floats):
  - 32 buildings × 5 = 160      [is_p1, is_p2, is_neutral, garrison/cap, alive]
  - 32 groups    × 4 = 128      [is_p1, is_p2, progress/travel, count_norm]
  - 1 global                    [tick / GAME_TIMEOUT_TICKS]
"""

from __future__ import annotations

import numpy as np

from sim import config as C


N = C.MAX_BUILDING_SLOTS
M = C.MAX_UNIT_GROUP_SLOTS
BUILDING_FEATS = 5
GROUP_FEATS = 4
GLOBAL_FEATS = 1
OBS_DIM = N * BUILDING_FEATS + M * GROUP_FEATS + GLOBAL_FEATS   # = 289

# Normalizer — keep consistent with sim's internal scale. Game capacity is
# typically 30 real units (300 internal) but friendly reinforcement can exceed
# it, so we use a fixed denominator that covers the observed range.
COUNT_NORM = float(C.DEFAULT_CAPACITY)        # 300
TIMEOUT_NORM = float(C.GAME_TIMEOUT_TICKS)    # 200


def encode_obs(obs: dict) -> np.ndarray:
    """Dict obs → (OBS_DIM,) float32 vector.

    `obs` is the dict returned by MushroomEnv.step()/reset(). All fields are
    small numpy arrays; this function is called once per decision per env,
    so it's allowed to be a little chatty.
    """
    out = np.empty(OBS_DIM, dtype=np.float32)

    # --- Buildings ---------------------------------------------------------
    b_alive    = obs["buildings_alive"]
    b_owner    = obs["buildings_owner"]
    b_garrison = obs["buildings_garrison"].astype(np.float32)
    b_capacity = obs["buildings_capacity"].astype(np.float32)

    is_p1 = (b_owner == C.OWNER_P1).astype(np.float32)
    is_p2 = (b_owner == C.OWNER_P2).astype(np.float32)
    is_n  = (b_owner == C.OWNER_NEUTRAL).astype(np.float32) * b_alive.astype(np.float32)
    # Safe division: dead slots have capacity=0 → fall back to 1 denom.
    cap_safe = np.where(b_capacity > 0, b_capacity, 1.0)
    garr_ratio = (b_garrison / cap_safe).astype(np.float32)

    block = np.stack(
        [is_p1, is_p2, is_n, garr_ratio, b_alive.astype(np.float32)],
        axis=1,
    )  # (N, 5)
    out[: N * BUILDING_FEATS] = block.reshape(-1)

    # --- Groups ------------------------------------------------------------
    g_alive    = obs["groups_alive"].astype(np.float32)
    g_owner    = obs["groups_owner"]
    g_progress = obs["groups_progress"].astype(np.float32)
    g_travel   = obs["groups_travel"].astype(np.float32)
    g_count    = obs["groups_count"].astype(np.float32)

    g_is_p1 = (g_owner == C.OWNER_P1).astype(np.float32) * g_alive
    g_is_p2 = (g_owner == C.OWNER_P2).astype(np.float32) * g_alive
    travel_safe = np.where(g_travel > 0, g_travel, 1.0)
    g_frac = (g_progress / travel_safe).astype(np.float32) * g_alive
    g_count_norm = (g_count / COUNT_NORM).astype(np.float32) * g_alive

    gblock = np.stack([g_is_p1, g_is_p2, g_frac, g_count_norm], axis=1)  # (M, 4)
    out[N * BUILDING_FEATS : N * BUILDING_FEATS + M * GROUP_FEATS] = gblock.reshape(-1)

    # --- Globals -----------------------------------------------------------
    out[-1] = float(obs["tick"]) / TIMEOUT_NORM

    return out
