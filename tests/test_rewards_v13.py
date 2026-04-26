"""Tests for sim-v1.3 reward rebalance (CURRICULUM_PLAN.md §3.1).

Each test runs a small scenario under both reward_version=0 (v1.2) and
reward_version=1 (v1.3) and asserts the v1.3 constants take effect (5×
WIN/LOSE, 0.5× capture/loss, -0.5 draw, 4× speed bonus, etc.).

Strategy: drive the same scripted game under both versions and compare the
*ratios* of reward emitted, since the engine math is identical except for
the constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import Action
from sim.engine import step_tick as step_numpy
from sim.engine_jax import (
    ACTION_KIND_NOOP,
    ACTION_KIND_SEND,
    encode_action,
    step_tick_single,
)
from sim.state import empty_state, precompute_distances
from sim.state_jax import from_numpy_state, to_numpy_state


# ---------------------------------------------------------------------------
# Constant sanity (no engine driving)
# ---------------------------------------------------------------------------

def test_v13_constants_match_plan():
    """Sanity: the v1.3 numbers in config match CURRICULUM_PLAN.md §3.1."""
    rv = C.REWARD_VERSION_V13
    assert C.REWARD_WIN_BY_VERSION[rv]         == 5.0
    assert C.REWARD_LOSE_BY_VERSION[rv]        == -5.0
    assert C.REWARD_DRAW_BY_VERSION[rv]        == -0.5
    assert C.REWARD_CAPTURE_BY_VERSION[rv]     == 0.05
    assert C.REWARD_LOSS_BY_VERSION[rv]        == -0.05
    assert C.REWARD_SPEED_BONUS_BY_VERSION[rv] == 2.0


def test_v12_module_constants_unchanged():
    """Default REWARD_* (v1.2) constants are unchanged for back-compat."""
    assert C.REWARD_WIN         == 1.0
    assert C.REWARD_LOSE        == -1.0
    assert C.REWARD_DRAW        == 0.0
    assert C.REWARD_CAPTURE     == 0.1
    assert C.REWARD_LOSS        == -0.1
    assert C.REWARD_SPEED_BONUS == 0.5


def test_state_default_reward_version_is_v12():
    """A freshly built state defaults to v1.2 so callers that don't opt in
    keep the old reward semantics."""
    s = empty_state()
    assert s.reward_version == C.REWARD_VERSION_V12 == 0


def test_jax_state_default_reward_version_is_v12():
    s = empty_state()
    sj = from_numpy_state(s)
    assert int(sj.reward_version) == 0
    back = to_numpy_state(sj)
    assert back.reward_version == 0


# ---------------------------------------------------------------------------
# Engine-driven scenarios
# ---------------------------------------------------------------------------

def _build_capture_state(reward_version: int):
    """One p1 base, one neutral 1-garrison target, one p2 fortress.

    p1 sends 100% (10 units), captures the neutral on tick 0 (distance=200,
    travel_speed=200 → travel_ticks=1 means group spawns at tick 0 and
    arrives during the same tick's _advance_movement; engine emits capture
    reward immediately).
    """
    s = empty_state()
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 10 * C.SCALE
    s.buildings_capacity[0] = 100 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0

    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_NEUTRAL
    s.buildings_garrison[1] = 1 * C.SCALE
    s.buildings_capacity[1] = 100 * C.SCALE
    s.buildings_x[1] = 200
    s.buildings_y[1] = 0

    # Add p2 building so the game stays "playing" through these ticks.
    s.buildings_alive[2] = 1
    s.buildings_owner[2] = C.OWNER_P2
    s.buildings_garrison[2] = 50 * C.SCALE
    s.buildings_capacity[2] = 100 * C.SCALE
    s.buildings_x[2] = 1000
    s.buildings_y[2] = 0

    precompute_distances(s)
    s.tick = 0
    s.phase = C.PHASE_PLAYING
    s.reward_version = int(reward_version)
    return s


@pytest.mark.parametrize("rv", [0, 1])
def test_capture_reward_by_version_numpy(rv):
    s = _build_capture_state(rv)
    a1 = Action(kind="send", type_idx=3, src=0, tgt=1)
    r1, r2, _ = step_numpy(s, a1, None)
    expected = C.REWARD_CAPTURE_BY_VERSION[rv]
    assert r1 == pytest.approx(expected, abs=1e-6), (
        f"v{rv}: numpy capture reward = {r1!r}, expected {expected!r}"
    )
    assert r2 == 0.0


@pytest.mark.parametrize("rv", [0, 1])
def test_capture_reward_by_version_jax(rv):
    s = _build_capture_state(rv)
    sj = from_numpy_state(s)
    a1 = encode_action(ACTION_KIND_SEND, 3, 0, 1)
    a_noop = encode_action(ACTION_KIND_NOOP)

    sj, r1, r2, _ = step_tick_single(sj, a1, a_noop)
    expected = C.REWARD_CAPTURE_BY_VERSION[rv]
    assert float(r1) == pytest.approx(expected, abs=1e-5), (
        f"v{rv}: jax capture reward = {float(r1)}, expected {expected}"
    )


def _build_p2_capture_p1_state(reward_version: int):
    """p2 captures a p1-owned building → p1 should get REWARD_LOSS, p2 REWARD_CAPTURE.

    Both p2's send and the capture happen on the same tick (distance 200,
    travel=1 tick).
    """
    s = empty_state()
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P2
    s.buildings_garrison[0] = 50 * C.SCALE
    s.buildings_capacity[0] = 100 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0
    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_P1
    s.buildings_garrison[1] = 1 * C.SCALE
    s.buildings_capacity[1] = 100 * C.SCALE
    s.buildings_x[1] = 200
    s.buildings_y[1] = 0
    # extra p1 building so the game doesn't end on capture
    s.buildings_alive[2] = 1
    s.buildings_owner[2] = C.OWNER_P1
    s.buildings_garrison[2] = 5 * C.SCALE
    s.buildings_capacity[2] = 100 * C.SCALE
    s.buildings_x[2] = 1000
    s.buildings_y[2] = 0
    precompute_distances(s)
    s.reward_version = int(reward_version)
    return s


@pytest.mark.parametrize("rv", [0, 1])
def test_loss_reward_by_version_numpy(rv):
    s = _build_p2_capture_p1_state(rv)
    a2 = Action(kind="send", type_idx=2, src=0, tgt=1)  # send 75%
    r1, r2, _ = step_numpy(s, None, a2)
    exp_loss    = C.REWARD_LOSS_BY_VERSION[rv]
    exp_capture = C.REWARD_CAPTURE_BY_VERSION[rv]
    assert r1 == pytest.approx(exp_loss,    abs=1e-6), f"v{rv}: p1 loss = {r1!r}"
    assert r2 == pytest.approx(exp_capture, abs=1e-6), f"v{rv}: p2 capture = {r2!r}"


def _build_eliminate_p2_state(reward_version: int):
    """p1 sends 100% to p2's only building; capture eliminates p2 same tick."""
    s = empty_state()
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 50 * C.SCALE
    s.buildings_capacity[0] = 100 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0

    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_P2
    s.buildings_garrison[1] = 1 * C.SCALE
    s.buildings_capacity[1] = 100 * C.SCALE
    s.buildings_x[1] = 200
    s.buildings_y[1] = 0

    precompute_distances(s)
    s.reward_version = int(reward_version)
    return s


@pytest.mark.parametrize("rv", [0, 1])
def test_win_reward_with_speed_bonus_numpy(rv):
    s = _build_eliminate_p2_state(rv)
    a1 = Action(kind="send", type_idx=3, src=0, tgt=1)
    r1, r2, done = step_numpy(s, a1, None)
    assert done, "p2 should be eliminated on tick 0"
    # state.tick is 1 after the increment in step_tick.
    rv_capture = C.REWARD_CAPTURE_BY_VERSION[rv]
    rv_win     = C.REWARD_WIN_BY_VERSION[rv]
    rv_lose    = C.REWARD_LOSE_BY_VERSION[rv]
    rv_speed   = C.REWARD_SPEED_BONUS_BY_VERSION[rv]
    expected_speed = rv_speed * max(0.0, 1.0 - 1 / C.GAME_TIMEOUT_TICKS)
    rv_loss_per_bldg = C.REWARD_LOSS_BY_VERSION[rv]
    # When p2's building is captured, p2 also gets REWARD_LOSS for losing it,
    # in addition to the eliminate-end REWARD_LOSE.
    expected_r1 = rv_capture + rv_win + expected_speed
    expected_r2 = rv_loss_per_bldg + rv_lose
    assert r1 == pytest.approx(expected_r1, abs=1e-5), (
        f"v{rv}: r1 win = {r1!r}, expected {expected_r1!r}"
    )
    assert r2 == pytest.approx(expected_r2, abs=1e-5), (
        f"v{rv}: r2 lose = {r2!r}, expected {expected_r2!r}"
    )


@pytest.mark.parametrize("rv", [0, 1])
def test_win_reward_with_speed_bonus_jax(rv):
    """Same scenario, JAX backend."""
    s = _build_eliminate_p2_state(rv)
    sj = from_numpy_state(s)
    a1 = encode_action(ACTION_KIND_SEND, 3, 0, 1)
    a_noop = encode_action(ACTION_KIND_NOOP)
    sj, r1, r2, done = step_tick_single(sj, a1, a_noop)
    assert bool(done)
    rv_capture = C.REWARD_CAPTURE_BY_VERSION[rv]
    rv_win     = C.REWARD_WIN_BY_VERSION[rv]
    rv_lose    = C.REWARD_LOSE_BY_VERSION[rv]
    rv_speed   = C.REWARD_SPEED_BONUS_BY_VERSION[rv]
    expected_speed = rv_speed * max(0.0, 1.0 - 1 / C.GAME_TIMEOUT_TICKS)
    rv_loss_per_bldg = C.REWARD_LOSS_BY_VERSION[rv]
    # When p2's building is captured, p2 also gets REWARD_LOSS for losing it,
    # in addition to the eliminate-end REWARD_LOSE.
    expected_r1 = rv_capture + rv_win + expected_speed
    expected_r2 = rv_loss_per_bldg + rv_lose
    assert float(r1) == pytest.approx(expected_r1, abs=1e-5)
    assert float(r2) == pytest.approx(expected_r2, abs=1e-5)


def _build_timeout_p1_more_units(reward_version: int):
    """At timeout, p1 has more units → p1 wins, no speed bonus."""
    s = empty_state()
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 30 * C.SCALE
    s.buildings_capacity[0] = 30 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0
    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_P2
    s.buildings_garrison[1] = 5 * C.SCALE
    s.buildings_capacity[1] = 30 * C.SCALE
    s.buildings_x[1] = 600
    s.buildings_y[1] = 0
    precompute_distances(s)
    # Set tick to TIMEOUT-1 so the next tick triggers timeout.
    s.tick = C.GAME_TIMEOUT_TICKS - 1
    s.reward_version = int(reward_version)
    return s


@pytest.mark.parametrize("rv", [0, 1])
def test_timeout_win_reward_numpy(rv):
    s = _build_timeout_p1_more_units(rv)
    r1, r2, done = step_numpy(s, None, None)
    assert done
    rv_win  = C.REWARD_WIN_BY_VERSION[rv]
    rv_lose = C.REWARD_LOSE_BY_VERSION[rv]
    assert r1 == pytest.approx(rv_win,  abs=1e-6), f"v{rv}: timeout r1 = {r1!r}"
    assert r2 == pytest.approx(rv_lose, abs=1e-6), f"v{rv}: timeout r2 = {r2!r}"


def _build_timeout_draw(reward_version: int):
    """Both players have equal buildings AND equal units at timeout → draw."""
    s = empty_state()
    s.buildings_alive[0] = 1
    s.buildings_owner[0] = C.OWNER_P1
    s.buildings_garrison[0] = 5 * C.SCALE
    s.buildings_capacity[0] = 30 * C.SCALE
    s.buildings_x[0] = 0
    s.buildings_y[0] = 0
    s.buildings_alive[1] = 1
    s.buildings_owner[1] = C.OWNER_P2
    s.buildings_garrison[1] = 5 * C.SCALE
    s.buildings_capacity[1] = 30 * C.SCALE
    s.buildings_x[1] = 600
    s.buildings_y[1] = 0
    precompute_distances(s)
    # Both bases at cap so production doesn't break the symmetry.
    s.buildings_capacity[0] = 5 * C.SCALE
    s.buildings_capacity[1] = 5 * C.SCALE
    s.tick = C.GAME_TIMEOUT_TICKS - 1
    s.reward_version = int(reward_version)
    return s


@pytest.mark.parametrize("rv", [0, 1])
def test_timeout_draw_reward_numpy(rv):
    s = _build_timeout_draw(rv)
    r1, r2, done = step_numpy(s, None, None)
    assert done
    assert s.phase == C.PHASE_DRAW, f"phase={s.phase}"
    rv_draw = C.REWARD_DRAW_BY_VERSION[rv]
    assert r1 == pytest.approx(rv_draw, abs=1e-6), f"v{rv}: r1 draw = {r1!r}"
    assert r2 == pytest.approx(rv_draw, abs=1e-6), f"v{rv}: r2 draw = {r2!r}"


@pytest.mark.parametrize("rv", [0, 1])
def test_timeout_draw_reward_jax(rv):
    s = _build_timeout_draw(rv)
    sj = from_numpy_state(s)
    a_noop = encode_action(ACTION_KIND_NOOP)
    sj, r1, r2, done = step_tick_single(sj, a_noop, a_noop)
    assert bool(done)
    rv_draw = C.REWARD_DRAW_BY_VERSION[rv]
    assert float(r1) == pytest.approx(rv_draw, abs=1e-5)
    assert float(r2) == pytest.approx(rv_draw, abs=1e-5)


def test_v13_winning_dominates_capture_grind():
    """v1.3's WIN reward should dwarf the per-capture gradient.

    Quick win (eliminate p2 in 1 tick after capturing 1 building) under v1.3
    must exceed the maximum theoretical "stall and capture all 32 slots"
    sequence under v1.2. This is the high-level invariant the rebalance
    targets — ensure the test guards it numerically.
    """
    # Quick win under v1.3:
    s = _build_eliminate_p2_state(C.REWARD_VERSION_V13)
    a1 = Action(kind="send", type_idx=3, src=0, tgt=1)
    r1, _, done = step_numpy(s, a1, None)
    assert done
    quick_v13 = r1

    # Theoretical max v1.2 capture spam (no win):
    # 32 captures × REWARD_CAPTURE = 32 × 0.1 = 3.2
    capture_grind_v12_max = 32 * C.REWARD_CAPTURE_BY_VERSION[C.REWARD_VERSION_V12]

    assert quick_v13 > capture_grind_v12_max, (
        f"v1.3 quick win = {quick_v13:.3f} should exceed v1.2 capture grind max "
        f"= {capture_grind_v12_max:.3f}"
    )
