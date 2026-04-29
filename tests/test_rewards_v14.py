"""Tests for sim-v1.4 per-tick shaping rewards (KARPATHY_LOG fire 21).

v1.4 = v1.3 terminal/capture rewards + per-tick shaping based on
(buildings_owned_p1 − buildings_owned_p2) and (units_real_p1 − units_real_p2),
designed to break the mutual-noop equilibrium under random_legal training.

Tests:
  - Constants present and non-zero for v14 only
  - v13 path emits zero per-tick shaping (back-compat)
  - v14 path emits non-zero shaping when buildings/units are asymmetric
  - JAX engine matches numpy byte-for-byte under v14
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.engine import step_tick as step_numpy
from sim.engine_jax import (
    ACTION_KIND_NOOP,
    encode_action,
    step_tick_single,
)
from sim.state import empty_state, precompute_distances
from sim.state_jax import from_numpy_state


# ---------------------------------------------------------------------------
# Constant sanity
# ---------------------------------------------------------------------------

def test_v14_constants_match_design():
    """v14 reuses v13 terminal/capture rewards + adds per-tick shaping."""
    rv = C.REWARD_VERSION_V14
    assert rv == 2
    # Terminal/capture inherit from v13.
    assert C.REWARD_WIN_BY_VERSION[rv]         == 5.0
    assert C.REWARD_LOSE_BY_VERSION[rv]        == -5.0
    assert C.REWARD_DRAW_BY_VERSION[rv]        == -0.5
    assert C.REWARD_CAPTURE_BY_VERSION[rv]     == 0.05
    assert C.REWARD_LOSS_BY_VERSION[rv]        == -0.05
    assert C.REWARD_SPEED_BONUS_BY_VERSION[rv] == 2.0
    # Per-tick shaping coefficients are non-zero only on v14.
    assert C.REWARD_TICK_BUILDINGS_COEF_BY_VERSION[rv] == 0.0010
    assert C.REWARD_TICK_UNITS_COEF_BY_VERSION[rv]     == 0.0002


def test_per_tick_shaping_zero_on_v12_v13():
    """v12 and v13 emit zero per-tick shaping (back-compat)."""
    for rv in (C.REWARD_VERSION_V12, C.REWARD_VERSION_V13):
        assert C.REWARD_TICK_BUILDINGS_COEF_BY_VERSION[rv] == 0.0
        assert C.REWARD_TICK_UNITS_COEF_BY_VERSION[rv]     == 0.0


# ---------------------------------------------------------------------------
# Engine-driven scenarios
# ---------------------------------------------------------------------------

def _build_asymmetric_state(reward_version: int):
    """p1 has 2 buildings + 100 real units; p2 has 1 building + 50 real units.

    Buildings alive but distant enough that a noop tick won't trigger combat.
    """
    s = empty_state()
    # p1: building 0 with 50 garrison (real units = 50)
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 50 * C.SCALE
    s.buildings_capacity[0] = 100 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0

    # p1: building 1 with 50 garrison
    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_P1
    s.buildings_garrison[1] = 50 * C.SCALE
    s.buildings_capacity[1] = 100 * C.SCALE
    s.buildings_x[1] = 100
    s.buildings_y[1] = 0

    # p2: building 2 with 50 garrison
    s.buildings_alive[2] = 1
    s.buildings_owner[2] = C.OWNER_P2
    s.buildings_garrison[2] = 50 * C.SCALE
    s.buildings_capacity[2] = 100 * C.SCALE
    s.buildings_x[2] = 5000
    s.buildings_y[2] = 0

    precompute_distances(s)
    s.tick = 0
    s.phase = C.PHASE_PLAYING
    s.reward_version = int(reward_version)
    return s


def test_v14_per_tick_shaping_numpy():
    """Asymmetric state (p1 ahead in both buildings + units) → v14 emits
    positive shaping for p1 and equal-magnitude negative for p2."""
    s = _build_asymmetric_state(C.REWARD_VERSION_V14)
    # No actions, no combat (buildings far apart) → only shaping should fire.
    r1, r2, done = step_numpy(s, None, None)
    assert not done

    # Production tick may have grown garrisons by PRODUCTION_PER_TICK; that's
    # fine — we test the *sign* and approximate magnitude from the asymmetry.
    coef_b = C.REWARD_TICK_BUILDINGS_COEF_BY_VERSION[2]
    coef_u = C.REWARD_TICK_UNITS_COEF_BY_VERSION[2]
    # Buildings delta is exactly +1 (p1 has 2, p2 has 1)
    expected_b_part = coef_b * (2 - 1)
    # Units real: ~50 each on p1 (some growth from production), 50ish on p2
    # — assert positive but bound the range loosely
    assert r1 > 0.0
    assert r2 < 0.0
    assert r1 == pytest.approx(-r2, abs=1e-6)
    assert r1 >= expected_b_part * 0.5  # at least the building part


def test_v13_no_per_tick_shaping_numpy():
    """Same asymmetric state under v13 → zero reward (no combat, no terminal)."""
    s = _build_asymmetric_state(C.REWARD_VERSION_V13)
    r1, r2, done = step_numpy(s, None, None)
    assert not done
    assert r1 == 0.0
    assert r2 == 0.0


def test_v14_jax_matches_numpy():
    """JAX path produces the same per-tick shaping as numpy."""
    s_np = _build_asymmetric_state(C.REWARD_VERSION_V14)
    s_jx = from_numpy_state(s_np)
    a_noop = encode_action(ACTION_KIND_NOOP)

    r1_np, r2_np, _ = step_numpy(s_np, None, None)
    _, r1_jx, r2_jx, _ = step_tick_single(s_jx, a_noop, a_noop)

    assert float(r1_jx) == pytest.approx(r1_np, abs=1e-5)
    assert float(r2_jx) == pytest.approx(r2_np, abs=1e-5)


def test_v14_no_shaping_on_terminal_tick():
    """Per-tick shaping must NOT fire on the terminal tick — the terminal
    reward (WIN/LOSE) should be the only signal at game end."""
    s = empty_state()
    # Set up a state where p2 has no buildings → next tick ends the game.
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 50 * C.SCALE
    s.buildings_capacity[0] = 100 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0
    # No p2 buildings, no in-flight groups → p2 already eliminated.
    precompute_distances(s)
    s.tick = 0
    s.phase = C.PHASE_PLAYING
    s.reward_version = int(C.REWARD_VERSION_V14)

    r1, r2, done = step_numpy(s, None, None)
    assert done
    # On terminal tick, p1 gets WIN(5.0) + speed_bonus only — no shaping.
    speed_bonus = C.REWARD_SPEED_BONUS_BY_VERSION[2] * (1.0 - 1.0 / C.GAME_TIMEOUT_TICKS)
    expected_r1 = C.REWARD_WIN_BY_VERSION[2] + speed_bonus
    assert r1 == pytest.approx(expected_r1, abs=1e-5)
    assert r2 == pytest.approx(C.REWARD_LOSE_BY_VERSION[2], abs=1e-5)
