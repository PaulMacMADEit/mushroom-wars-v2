"""Tests for the v0.1 basic sim.

Run: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import Action, compute_mask, decode, encode, send_amount, NOOP_INDEX
from sim.engine import step_tick, _combat
from sim.levels import reset
from sim.state import count_owned_buildings, count_owned_units


# ---------------------------------------------------------------------------
# Action encode/decode
# ---------------------------------------------------------------------------

def test_action_roundtrip():
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for src in [0, 5, 31]:
            for tgt in [0, 10, 31]:
                idx = encode(type_idx, src, tgt)
                a = decode(idx)
                assert a.kind == "send"
                assert a.type_idx == type_idx
                assert a.src == src
                assert a.tgt == tgt


def test_noop_decode():
    a = decode(NOOP_INDEX)
    assert a.kind == "noop"


# ---------------------------------------------------------------------------
# Send amount (fixed-point, whole real units)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("garrison, pct, expected", [
    (200, 25, 50),    # 20 real, 25% → 5 real = 50 internal
    (200, 50, 100),   # 20 real, 50% → 10 real
    (200, 100, 200),  # 20 real, 100% → 20 real
    (35, 25, 0),      # 3.5 real, 25% → 0.875 real → floor 0 (will be masked)
    (35, 50, 10),     # 3.5 real, 50% → 1.75 → floor 1 real = 10 internal
    (35, 100, 30),    # 3.5 real, 100% → 3.5 → floor 3 real (half unit left behind)
    (0,   100, 0),
])
def test_send_amount(garrison, pct, expected):
    assert send_amount(garrison, pct) == expected


# ---------------------------------------------------------------------------
# Combat formula
# ---------------------------------------------------------------------------

def test_combat_defender_holds():
    # 10 real (100 internal) garrison, 5 real (50) attackers → defender holds.
    # eff_def = 100 * 13 // 10 = 130; 50 < 130 → holds, new = 130 - 50 = 80.
    ng, owner = _combat(100, 50, C.OWNER_P1, C.OWNER_P2)
    assert ng == 80
    assert owner == C.OWNER_P2


def test_combat_attacker_takes():
    # 5 real (50) garrison, 10 real (100) attackers → attacker takes.
    # eff_def = 65; 100 > 65 → new = 100 - 65 = 35.
    ng, owner = _combat(50, 100, C.OWNER_P1, C.OWNER_P2)
    assert ng == 35
    assert owner == C.OWNER_P1


def test_combat_tie_goes_neutral():
    # attackers == eff_def → both wiped, neutral.
    eff = (100 * C.DEF_BONUS_NUM) // C.DEF_BONUS_DEN  # = 130
    ng, owner = _combat(100, eff, C.OWNER_P1, C.OWNER_P2)
    assert ng == 0
    assert owner == C.OWNER_NEUTRAL


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

def test_production_grows_then_caps():
    state = reset()
    # P1 base starts at 10 real = 100 internal, capacity 30 real = 300.
    b = state.buildings
    p1_base = 0
    start = int(b["garrison"][p1_base])
    assert start == 100

    for _ in range(5):
        step_tick(state)
    # +1 real per tick * 5 ticks = +50 internal.
    assert int(state.buildings["garrison"][p1_base]) == 150

    # Run past the cap.
    for _ in range(50):
        step_tick(state)
    assert int(state.buildings["garrison"][p1_base]) == 300


def test_neutrals_do_not_produce():
    state = reset()
    # Neutral at slot 2 starts at 1 real = 10 internal.
    start = int(state.buildings["garrison"][2])
    assert start == 10
    for _ in range(10):
        step_tick(state)
    assert int(state.buildings["garrison"][2]) == 10


# ---------------------------------------------------------------------------
# Send + arrival + capture
# ---------------------------------------------------------------------------

def test_send_capture_neutral():
    state = reset()
    # P1 base (slot 0, 10 real garrison) → N1 (slot 2, 1 real garrison).
    # 100% send from P1 base → 10 real attackers vs 1 real neutral.
    # eff_def = 10 * 13/10 = 13; attackers=100 > 13 → new_garrison = 87.
    action = Action(kind="send", type_idx=3, src=0, tgt=2)  # 100%
    step_tick(state, action_p1=action)

    # Unit group should be in flight.
    g = state.unit_groups
    assert np.any(g["alive"] == 1)

    # Travel from (100,100) to (350,100) = 250 → 3 ticks.
    # Already stepped 1 tick above; step 2 more to reach arrival.
    step_tick(state)
    r1, r2, done = step_tick(state)

    # N1 should now be P1-owned.
    assert int(state.buildings["owner"][2]) == C.OWNER_P1
    assert r1 == pytest.approx(C.REWARD_CAPTURE)
    # Garrison: 100 attackers - 13 eff_def = 87.
    assert int(state.buildings["garrison"][2]) == 87


def test_send_defender_holds_neutral():
    state = reset()
    # Reduce P1 base so we send fewer than we need.
    # 25% of 10 real = 2.5 → floor = 2 real = 20 internal attackers vs N3 (5 real).
    # eff_def_N3 = 50 * 1.3 = 65; 20 < 65 → holds, new = 65 - 20 = 45.
    action = Action(kind="send", type_idx=0, src=0, tgt=4)  # 25% to N3
    step_tick(state, action_p1=action)

    # Travel 0→4 is (100,100)→(100,350) = 250 → 3 ticks.
    for _ in range(3):
        step_tick(state)

    assert int(state.buildings["owner"][4]) == C.OWNER_NEUTRAL
    assert int(state.buildings["garrison"][4]) == 45


# ---------------------------------------------------------------------------
# Field cancellation (same-tick opposing arrivals)
# ---------------------------------------------------------------------------

def test_field_cancel_reinforce_only():
    state = reset()
    # Set up: P1 at slot 0 sends 50% (5 real) to N1. P2 also sends 50% to N1.
    # But timings differ (different distances). To force same-tick arrival,
    # we manually set up unit groups.
    g = state.unit_groups
    # Clear any existing groups.
    g[:] = 0

    # Two in-flight groups arriving next tick at slot 2 (neutral, 10 internal garrison).
    g[0]["alive"] = 1
    g[0]["owner"] = C.OWNER_P1
    g[0]["src_slot"] = 0
    g[0]["tgt_slot"] = 2
    g[0]["count"] = 100   # 10 real
    g[0]["progress"] = 0
    g[0]["travel_ticks"] = 1

    g[1]["alive"] = 1
    g[1]["owner"] = C.OWNER_P2
    g[1]["src_slot"] = 1
    g[1]["tgt_slot"] = 2
    g[1]["count"] = 60    # 6 real
    g[1]["progress"] = 0
    g[1]["travel_ticks"] = 1

    # Reset tick + perf so this is a clean step.
    state.tick = 0

    r1, r2, done = step_tick(state)

    # Field cancel: 100 vs 60 → P1 survivors = 40, P2 = 0.
    # Combat: 40 vs 10 * 1.3 = 13 → attacker wins, new_garrison = 40 - 13 = 27.
    assert int(state.buildings["owner"][2]) == C.OWNER_P1
    assert int(state.buildings["garrison"][2]) == 27
    # P1 captured a neutral → +0.1.
    assert r1 == pytest.approx(C.REWARD_CAPTURE)


def test_field_cancel_perfect_wipe():
    state = reset()
    g = state.unit_groups
    g[:] = 0

    g[0]["alive"] = 1; g[0]["owner"] = C.OWNER_P1; g[0]["tgt_slot"] = 2
    g[0]["count"] = 50; g[0]["travel_ticks"] = 1
    g[1]["alive"] = 1; g[1]["owner"] = C.OWNER_P2; g[1]["tgt_slot"] = 2
    g[1]["count"] = 50; g[1]["travel_ticks"] = 1

    step_tick(state)

    # Perfect cancel → neutral garrison unchanged (still 10 internal).
    assert int(state.buildings["owner"][2]) == C.OWNER_NEUTRAL
    assert int(state.buildings["garrison"][2]) == 10


# ---------------------------------------------------------------------------
# Victory
# ---------------------------------------------------------------------------

def test_timeout_tiebreak_by_buildings():
    state = reset()
    # Force P1 to own 4 buildings, P2 to own 2, run to timeout.
    state.buildings["owner"][2] = C.OWNER_P1
    state.buildings["owner"][3] = C.OWNER_P1
    state.buildings["owner"][4] = C.OWNER_P2  # leave neutral->P2
    state.buildings["owner"][5] = C.OWNER_NEUTRAL

    state.tick = C.GAME_TIMEOUT_TICKS - 1
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P1_WINS
    assert r1 == pytest.approx(C.REWARD_WIN)
    assert r2 == pytest.approx(C.REWARD_LOSE)


def test_early_win_elimination():
    state = reset()
    # Wipe P2 — no buildings, no in-flight.
    state.buildings["owner"][1] = C.OWNER_NEUTRAL
    state.buildings["garrison"][1] = 0

    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P1_WINS


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def test_mask_basic():
    state = reset()
    mask = compute_mask(state, C.OWNER_P1)
    assert mask[NOOP_INDEX]                                 # noop always legal
    # P1 owns slot 0. Sending from 0 to any other alive building with 25% should be valid.
    # P1 garrison = 100; 25% * 100 / 1000 = 2 real = 20 internal ≥ MIN_SEND_INTERNAL (10). OK.
    idx = encode(0, src=0, tgt=2)
    assert mask[idx]
    # Sending from a slot P1 doesn't own should be invalid.
    idx = encode(0, src=1, tgt=2)
    assert not mask[idx]
    # Self-pair invalid.
    idx = encode(0, src=0, tgt=0)
    assert not mask[idx]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    # Same initial state + same action sequence → byte-identical final state.
    actions_p1 = [
        (5, Action(kind="send", type_idx=3, src=0, tgt=2)),
        (12, Action(kind="send", type_idx=1, src=2, tgt=3)),
    ]

    def run():
        state = reset(seed=42)
        tick_to_action = dict(actions_p1)
        for _ in range(50):
            a = tick_to_action.get(state.tick)
            step_tick(state, action_p1=a)
        return state

    s1 = run()
    s2 = run()
    assert np.array_equal(s1.buildings, s2.buildings)
    assert np.array_equal(s1.unit_groups, s2.unit_groups)
    assert s1.tick == s2.tick
    assert s1.phase == s2.phase


# ---------------------------------------------------------------------------
# Sanity: random-play game terminates
# ---------------------------------------------------------------------------

def test_random_game_terminates():
    rng = np.random.default_rng(123)
    state = reset(seed=123)
    for _ in range(C.GAME_TIMEOUT_TICKS + 10):
        # Random legal actions for both players.
        a1 = _random_action(state, C.OWNER_P1, rng)
        a2 = _random_action(state, C.OWNER_P2, rng)
        r1, r2, done = step_tick(state, a1, a2)
        if done:
            break
    assert done
    assert state.phase in (C.PHASE_P1_WINS, C.PHASE_P2_WINS, C.PHASE_DRAW)


def _random_action(state, player, rng):
    mask = compute_mask(state, player)
    legal = np.where(mask)[0]
    idx = int(rng.choice(legal))
    return decode(idx)
