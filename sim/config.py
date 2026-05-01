"""
Sim constants. Everything tunable for training lives here.

Units:
- Internal storage uses fixed-point integers at SCALE (10) per real unit.
  garrison=35 internal  == 3.5 display units.
  This keeps combat math exact (no float rounding) while staying integer-fast.
- Time is integer ticks. 1 tick = 1 second (TICK_HZ = 1).
- Positions are integer "map units" (int16). Renderer scales to pixels.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Capacity (shape-defining — changing these means a Model version bump)
# ---------------------------------------------------------------------------
MAX_BUILDING_SLOTS    = 8     # v12: cut from 32. Levels above this fail validation.
MAX_UNIT_GROUP_SLOTS  = 4     # v12: cut from 32. Typical map has ≤4 in-flight groups.
HISTORY_K             = 5     # v10+ encoder: last K actions per side fed as obs

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
TICK_HZ               = 1             # 1 tick = 1 second
DECISION_INTERVAL_TICKS = 2           # agent is polled every N ticks (2 sec)
GAME_TIMEOUT_TICKS    = 200           # hard cap — then tiebreak by buildings→units

# ---------------------------------------------------------------------------
# Fixed-point scale (storage resolution)
# ---------------------------------------------------------------------------
SCALE                 = 10            # 1 real unit = 10 internal
# Helpers: internal = real * SCALE; real = internal / SCALE (display only)

# ---------------------------------------------------------------------------
# Production / capacity
# ---------------------------------------------------------------------------
PRODUCTION_PER_TICK   = 1 * SCALE     # +1 real unit per second per owned building
DEFAULT_CAPACITY      = 30 * SCALE    # 30 real units max garrison

# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
TRAVEL_SPEED          = 100           # map-units moved per sim tick (v9.1: halved
                                      # from 200 so transit time differentiates
                                      # close vs far sources at action_repeat=2,
                                      # i.e. close pairs land within 1 AI tick
                                      # while far pairs cost 2+ AI ticks of
                                      # defender regen and opponent reaction.)
MIN_TRAVEL_TICKS      = 1
MAX_TRAVEL_TICKS      = 12            # ceiling raised from 8 to absorb v9.1's 2×
                                      # slowdown on existing-size maps without
                                      # clipping any pair to the same value.

# ---------------------------------------------------------------------------
# Send action
# ---------------------------------------------------------------------------
SEND_PERCENTAGES      = (50, 100)           # v12: 2 types (was 4). Net change vs 25/75 is small in practice.
MIN_GARRISON_AFTER_SEND = 0           # keeps path open for "leave 1 behind" rule
MIN_SEND_INTERNAL     = 1 * SCALE     # must send at least 1 real unit; else invalid

# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------
# Defense bonus as rational: garrison × DEF_BONUS_NUM // DEF_BONUS_DEN
# 13/10 = 1.3  (1.3× defender strength when garrisoned in a building)
DEF_BONUS_NUM         = 13
DEF_BONUS_DEN         = 10

# ---------------------------------------------------------------------------
# Ownership codes
# ---------------------------------------------------------------------------
OWNER_NEUTRAL         = 0
OWNER_P1              = 1
OWNER_P2              = 2

# ---------------------------------------------------------------------------
# Building types (v0.1: only BASIC — but the table already supports more)
# ---------------------------------------------------------------------------
TYPE_BASIC            = 0
# Future: TYPE_CAPITAL=1, TYPE_MILITARY=2, TYPE_FORGE=3, TYPE_PRODUCTION=4

# Per-type stats. v0.1 has one row — adding a type later = append a row,
# plus engine.py reads stats[type_id] instead of the module-level constant.
BUILDING_STATS = {
    TYPE_BASIC: {
        "prod_per_tick": PRODUCTION_PER_TICK,
        "capacity":      DEFAULT_CAPACITY,
        "def_num":       DEF_BONUS_NUM,
        "def_den":       DEF_BONUS_DEN,
    },
}

# ---------------------------------------------------------------------------
# Game phase codes
# ---------------------------------------------------------------------------
PHASE_PLAYING         = 0
PHASE_P1_WINS         = 1
PHASE_P2_WINS         = 2
PHASE_DRAW            = 3

# ---------------------------------------------------------------------------
# Rewards (training-time only; sim emits these on step)
# ---------------------------------------------------------------------------
# Three reward schemes are supported; State carries an int8 `reward_version`
# (0 = v1.2 default, 1 = v1.3, 2 = v1.4) and the engine indexes into the
# lookup arrays below. v1.3 rebalances toward winning quickly: WIN/LOSE 5×,
# halved capture/loss, mildly-bad draw, and a 4× speed bonus. v1.4 keeps
# v1.3 terminal/event rewards and adds per-tick shaping based on the
# (own − opponent) delta in buildings owned and real units held — designed
# to break the mutual-noop equilibrium observed under v1.3 + random_legal.
# See CURRICULUM_PLAN.md §3.1.
REWARD_VERSION_V12 = 0
REWARD_VERSION_V13 = 1
REWARD_VERSION_V14 = 2
REWARD_VERSION_V15 = 3   # 2026-04-30: v1.4 + asymmetric capture/loss
                         # (enemy 4x neutral). Designed to break the
                         # 37% timeout_rate observed under v1.4 on big
                         # maps — agent dominates territory but never
                         # finishes games. Adds explicit signal that
                         # enemy buildings are 4x more valuable.
REWARD_VERSION_V16 = 4   # 2026-05-01: v1.5 + harsher loss + much harsher draw.
                         # LOSE -5.0 → -7.5 (50% more costly); DRAW -0.5 → -1.25
                         # (150% more costly). Designed to push the policy past
                         # "stalemate is fine" equilibria — losing should hurt
                         # more than half a win, and draws should sting.

# Per-version reward tables. Index with REWARD_VERSION_V*.
# Tuple positions:                    v1.2,  v1.3,  v1.4,  v1.5,  v1.6
REWARD_CAPTURE_BY_VERSION     = (0.1,   0.05,  0.05,  0.05,  0.05)
REWARD_LOSS_BY_VERSION        = (-0.1,  -0.05, -0.05, -0.05, -0.05)
REWARD_WIN_BY_VERSION         = (1.0,   5.0,   5.0,   5.0,   5.0)
REWARD_LOSE_BY_VERSION        = (-1.0,  -5.0,  -5.0,  -5.0,  -7.5)   # v1.6: 50% more
REWARD_DRAW_BY_VERSION        = (0.0,   -0.5,  -0.5,  -0.5,  -1.25)  # v1.6: 150% more
# Bonus added to the winner that scales linearly with how quickly they won.
# Final terminal reward (winner) = REWARD_WIN + REWARD_SPEED_BONUS * (1 - tick / GAME_TIMEOUT_TICKS)
# At tick=0 the bonus is REWARD_SPEED_BONUS; at timeout it is 0.
REWARD_SPEED_BONUS_BY_VERSION = (0.5,   2.0,   2.0,   2.0,   2.0)

# Per-tick shaping (v1.4+ only — zero for v1.2/v1.3). Symmetric: at end of
# each tick the engine adds COEF_B*(b_p1−b_p2) + COEF_U*(u_p1_real−u_p2_real)
# to r1 and the negation to r2. Coefficients are tuned so total per-game
# shaping is ~±1.0 = ~20% of REWARD_WIN(v14)=5.0, big enough to bias toward
# active play without dominating terminal outcomes.
#   buildings: ±4 typical × 80 ticks × 0.0010 ≈ ±0.32 per game
#   units:     ±50 real typical × 80 ticks × 0.0002 ≈ ±0.80 per game
REWARD_TICK_BUILDINGS_COEF_BY_VERSION = (0.0, 0.0, 0.0010, 0.0010, 0.0010)
REWARD_TICK_UNITS_COEF_BY_VERSION     = (0.0, 0.0, 0.0002, 0.0002, 0.0002)

# v1.5+ — asymmetric capture/loss bonus when ownership transitions
# directly between the two players (not via mutual wipeout to neutral):
#   - Capture FROM enemy player: r_capture += +0.15  → total +0.20 (4× neutral)
#   - Lost TO enemy player:      r_loss    += -0.15  → total -0.20 (4× neutral)
# Mutual-wipeout transitions (owner → NEUTRAL) keep the base loss only.
# These give explicit signal: enemy buildings matter more than neutrals.
REWARD_ENEMY_CAPTURE_BONUS_BY_VERSION = (0.0, 0.0, 0.0,  0.15,  0.15)
REWARD_ENEMY_LOSS_PENALTY_BY_VERSION  = (0.0, 0.0, 0.0, -0.15, -0.15)

# Module-level scalar constants (= v1.2). Kept for backward-compat reads from
# any code path that doesn't yet thread `reward_version` through; new code
# should index into the *_BY_VERSION tuples above.
REWARD_CAPTURE        = REWARD_CAPTURE_BY_VERSION[REWARD_VERSION_V12]
REWARD_LOSS           = REWARD_LOSS_BY_VERSION[REWARD_VERSION_V12]
REWARD_WIN            = REWARD_WIN_BY_VERSION[REWARD_VERSION_V12]
REWARD_LOSE           = REWARD_LOSE_BY_VERSION[REWARD_VERSION_V12]
REWARD_DRAW           = REWARD_DRAW_BY_VERSION[REWARD_VERSION_V12]
REWARD_SPEED_BONUS    = REWARD_SPEED_BONUS_BY_VERSION[REWARD_VERSION_V12]
