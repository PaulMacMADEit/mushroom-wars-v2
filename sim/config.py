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
MAX_BUILDING_SLOTS    = 32    # fixed per ARCHITECTURE §9.1; unused slots masked
MAX_UNIT_GROUP_SLOTS  = 32

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
TRAVEL_SPEED          = 200           # map-units moved per tick (1 sec)
MIN_TRAVEL_TICKS      = 1
MAX_TRAVEL_TICKS      = 8             # map is sized so corner-to-corner ≤ 8

# ---------------------------------------------------------------------------
# Send action
# ---------------------------------------------------------------------------
SEND_PERCENTAGES      = (25, 50, 75, 100)   # 4 action types — matches v9.0 arch
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
REWARD_CAPTURE        = 0.1           # capturing any building
REWARD_LOSS           = -0.1          # losing a building
REWARD_WIN            = 1.0
REWARD_LOSE           = -1.0
REWARD_DRAW           = 0.0
# Bonus added to the winner that scales linearly with how quickly they won.
# Final terminal reward (winner) = REWARD_WIN + REWARD_SPEED_BONUS * (1 - tick / GAME_TIMEOUT_TICKS)
# At tick=0 the bonus is REWARD_SPEED_BONUS; at timeout it is 0.
REWARD_SPEED_BONUS    = 0.5
