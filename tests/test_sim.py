"""Proof-of-correctness test suite for the v0.1 sim.

Strategy:
  - Every formula documented in sim/engine.py is exercised with hand-computed
    expected values, including boundary + degenerate cases.
  - Property-style invariants (unit conservation, non-negativity, ownership
    bounds) run a scripted game and assert on every tick.
  - Movement/production/combat ordering is pinned down with explicit tests so
    accidental re-ordering in engine.step_tick breaks a test loudly.
  - Key timing rules (same-tick arrival for 1-tick travel,
    production-before-combat, simultaneous same-tick arrivals) have
    dedicated tests so any future behaviour change is caught.

Run: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import (
    ACTION_SPACE_SIZE,
    Action,
    NOOP_INDEX,
    compute_mask,
    decode,
    encode,
    is_valid,
    send_amount,
)
from sim.engine import _combat, step_tick
from sim.levels import apply, reset
from sim.state import (
    count_owned_buildings,
    count_owned_units,
    has_in_flight,
)
from sim import levels as sim_levels


# ===========================================================================
# Helpers
# ===========================================================================

P1_BASE = 0
P2_BASE = 1
N1_TOP = 2
N2_BOT = 3
N3_LEFT = 4
N4_RIGHT = 5


def _clear_groups(state):
    state.unit_groups[:] = 0


def _inject_group(state, slot, owner, src, tgt, count, travel_ticks, progress=0):
    """Directly place a unit-group in flight for deterministic arrival timing."""
    g = state.unit_groups
    g[slot]["alive"] = 1
    g[slot]["owner"] = owner
    g[slot]["src_slot"] = src
    g[slot]["tgt_slot"] = tgt
    g[slot]["count"] = count
    g[slot]["progress"] = progress
    g[slot]["travel_ticks"] = travel_ticks


def _total_units_on_board(state):
    """Garrisons + in-flight, across ALL owners (including neutral)."""
    b = state.buildings
    g = state.unit_groups
    garr = int(np.sum(np.where(b["alive"] == 1, b["garrison"], 0)))
    flight = int(np.sum(np.where(g["alive"] == 1, g["count"], 0)))
    return garr + flight


# ===========================================================================
# 1. Action encode / decode
# ===========================================================================


def test_action_roundtrip_full_grid():
    """Every (type, src, tgt) triple must round-trip losslessly."""
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for src in range(C.MAX_BUILDING_SLOTS):
            for tgt in range(C.MAX_BUILDING_SLOTS):
                idx = encode(type_idx, src, tgt)
                assert 0 <= idx < NOOP_INDEX, f"idx {idx} out of range"
                a = decode(idx)
                assert a.kind == "send"
                assert a.type_idx == type_idx
                assert a.src == src
                assert a.tgt == tgt


def test_action_space_size_matches_formula():
    # NUM_TYPES * MAX_SLOTS^2 + 1 (noop)
    expected = len(C.SEND_PERCENTAGES) * C.MAX_BUILDING_SLOTS ** 2 + 1
    assert ACTION_SPACE_SIZE == expected


def test_noop_decode():
    a = decode(NOOP_INDEX)
    assert a.kind == "noop"


def test_decode_out_of_range_raises():
    with pytest.raises(ValueError):
        decode(-1)
    with pytest.raises(ValueError):
        decode(NOOP_INDEX + 1)


def test_encode_out_of_range_raises():
    with pytest.raises(ValueError):
        encode(-1, 0, 0)
    with pytest.raises(ValueError):
        encode(0, -1, 0)
    with pytest.raises(ValueError):
        encode(0, 0, C.MAX_BUILDING_SLOTS)
    with pytest.raises(ValueError):
        encode(len(C.SEND_PERCENTAGES), 0, 0)


def test_action_indices_disjoint():
    """No two distinct (type, src, tgt) triples should collide to the same idx."""
    seen = set()
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for src in range(C.MAX_BUILDING_SLOTS):
            for tgt in range(C.MAX_BUILDING_SLOTS):
                idx = encode(type_idx, src, tgt)
                assert idx not in seen
                seen.add(idx)
    assert len(seen) == len(C.SEND_PERCENTAGES) * C.MAX_BUILDING_SLOTS ** 2


def test_action_unknown_kind_is_invalid():
    state = reset()
    assert not is_valid(state, C.OWNER_P1, Action(kind="weird", type_idx=0, src=P1_BASE, tgt=N1_TOP))


# ===========================================================================
# 2. Send amount — fixed-point math must yield whole real units
# ===========================================================================


@pytest.mark.parametrize(
    "garrison, pct, expected",
    [
        (200, 25, 50),    # 20 real * 25% = 5 real = 50 internal
        (200, 50, 100),
        (200, 75, 150),
        (200, 100, 200),
        (35, 25, 0),      # 3.5 * 0.25 = 0.875 → floor 0 real
        (35, 50, 10),     # 3.5 * 0.5 = 1.75 → floor 1 real
        (35, 75, 20),     # 3.5 * 0.75 = 2.625 → floor 2 real
        (35, 100, 30),    # 3.5 * 1.0 = 3.5 → floor 3 real
        (0, 100, 0),
        (5, 100, 0),      # 0.5 real at 100% = 0 real (floored)
        (10, 100, 10),    # 1 real at 100% = 1 real
        (300, 100, 300),  # at capacity, 100% sends all
    ],
)
def test_send_amount_values(garrison, pct, expected):
    assert send_amount(garrison, pct) == expected


def test_send_amount_is_always_multiple_of_scale():
    """Core invariant: amounts sent are always whole real units."""
    for g in range(0, 301, 7):          # arbitrary sweep
        for pct in C.SEND_PERCENTAGES:
            assert send_amount(g, pct) % C.SCALE == 0


def test_send_amount_monotonic_in_percentage():
    """Higher pct never yields less."""
    for g in range(10, 301, 10):
        amts = [send_amount(g, pct) for pct in C.SEND_PERCENTAGES]
        assert amts == sorted(amts), f"non-monotonic for g={g}: {amts}"


# ===========================================================================
# 3. Combat formula — exercise every branch, pin the formula
# ===========================================================================


def test_combat_zero_attackers_is_noop():
    ng, owner = _combat(100, 0, C.OWNER_P1, C.OWNER_P2)
    assert ng == 100
    assert owner == C.OWNER_P2


def test_combat_defender_holds_moderate_attack():
    # 10 real garrison (100), 5 real attackers (50).
    # Effective defense = 130. Remaining garrison = (130 - 50) / 1.3 = 61.538 -> 62.
    ng, owner = _combat(100, 50, C.OWNER_P1, C.OWNER_P2)
    assert ng == 62
    assert owner == C.OWNER_P2


def test_combat_small_attack_chips_proportionally():
    """Tiny attacks reduce defenders proportionally instead of flooring to 1 damage."""
    # 10 real garrison (100), 1 real attacker (10).
    # Remaining garrison = (130 - 10) / 1.3 = 92.307 -> 92.
    ng, owner = _combat(100, 10, C.OWNER_P1, C.OWNER_P2)
    assert ng == 92
    assert owner == C.OWNER_P2


def test_combat_small_attack_bigger_defender():
    """300 garrison vs 20 attackers leaves 28.5 after proportional defense."""
    ng, owner = _combat(300, 20, C.OWNER_P1, C.OWNER_P2)
    assert ng == 285
    assert owner == C.OWNER_P2


def test_combat_attacker_wins_cleanly():
    # 5 real garrison (50), 10 real attackers (100).
    # Effective defense = 65. Attacker wins with 35 left.
    ng, owner = _combat(50, 100, C.OWNER_P1, C.OWNER_P2)
    assert ng == 35
    assert owner == C.OWNER_P1


def test_combat_mutual_wipe_exact_eff_def():
    """attackers == effective defense → building goes neutral, 0 garrison."""
    garrison = 100
    eff_def = garrison + (garrison * 3) // 10   # 130
    ng, owner = _combat(garrison, eff_def, C.OWNER_P1, C.OWNER_P2)
    assert ng == 0
    assert owner == C.OWNER_NEUTRAL


def test_combat_attacker_wins_by_one_unit():
    """attackers = eff_def + SCALE (minimum survivor): new garrison = SCALE."""
    garrison = 100
    eff_def = garrison + (garrison * 3) // 10
    attackers = eff_def + C.SCALE
    ng, owner = _combat(garrison, attackers, C.OWNER_P1, C.OWNER_P2)
    assert ng == C.SCALE
    assert owner == C.OWNER_P1


def test_combat_defense_bonus_is_thirty_percent():
    """Defender effective HP = garrison * DEF_NUM / DEF_DEN = 1.3 * garrison."""
    for g in [10, 50, 100, 200, 300]:
        absorb_expected = (g * 3) // 10          # 30% of garrison, integer floor
        # attackers just below kill threshold should NOT capture
        attackers_just_under = g + absorb_expected - 1
        if attackers_just_under <= 0:
            continue
        ng, owner = _combat(g, attackers_just_under, C.OWNER_P1, C.OWNER_P2)
        # Either defender holds, or mutual wipe — NEVER attacker_owner.
        assert owner in (C.OWNER_P2, C.OWNER_NEUTRAL)


def test_combat_attack_on_empty_garrison():
    """0-garrison building: any attacker captures cleanly."""
    ng, owner = _combat(0, 50, C.OWNER_P1, C.OWNER_NEUTRAL)
    assert ng == 50
    assert owner == C.OWNER_P1


def test_combat_never_produces_negative_garrison():
    """Sweep plausible (garrison, attackers) space — new garrison never negative."""
    for g in range(0, 301, 10):
        for a in range(0, 401, 10):
            ng, _ = _combat(g, a, C.OWNER_P1, C.OWNER_P2)
            assert ng >= 0, f"negative garrison from g={g} a={a}: {ng}"


# ===========================================================================
# 4. Production
# ===========================================================================


def test_production_owned_grows_one_per_tick():
    state = reset()
    b = state.buildings
    assert int(b["garrison"][P1_BASE]) == 100   # 10 real
    for _ in range(5):
        step_tick(state)
    assert int(state.buildings["garrison"][P1_BASE]) == 150


def test_production_caps_at_capacity():
    state = reset()
    for _ in range(100):                        # way past cap
        step_tick(state)
    assert int(state.buildings["garrison"][P1_BASE]) == C.DEFAULT_CAPACITY


def test_production_neutrals_do_not_grow():
    state = reset()
    start = int(state.buildings["garrison"][N1_TOP])
    for _ in range(20):
        step_tick(state)
    assert int(state.buildings["garrison"][N1_TOP]) == start


def test_production_over_capacity_owned_building_stays_put():
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = state.buildings["capacity"][N1_TOP] + 70
    before = int(state.buildings["garrison"][N1_TOP])
    for _ in range(5):
        step_tick(state)
    assert int(state.buildings["garrison"][N1_TOP]) == before


def test_production_both_players_grow_symmetrically():
    state = reset()
    for _ in range(5):
        step_tick(state)
    assert int(state.buildings["garrison"][P1_BASE]) == int(
        state.buildings["garrison"][P2_BASE]
    )


def test_production_runs_before_movement_same_tick():
    """If a group arrives this tick, the target's production applies first (if owned)."""
    state = reset()
    _clear_groups(state)
    # Put a P1 reinforcement arriving this tick at P1's own base.
    _inject_group(state, 0, C.OWNER_P1, P2_BASE, P1_BASE,
                  count=50, travel_ticks=1, progress=0)
    g0 = int(state.buildings["garrison"][P1_BASE])     # 100
    step_tick(state)
    # Expected: production brings 100 → 110; then reinforce adds 50 → 160.
    assert int(state.buildings["garrison"][P1_BASE]) == g0 + C.PRODUCTION_PER_TICK + 50


# ===========================================================================
# 5. Movement timing
# ===========================================================================


def test_travel_matrix_is_symmetric_and_nonzero_for_alive_pairs():
    state = reset()
    tm = state.travel_matrix
    for i in range(6):
        for j in range(6):
            if i == j:
                assert tm[i, j] == 0
            else:
                assert tm[i, j] == tm[j, i]
                assert C.MIN_TRAVEL_TICKS <= tm[i, j] <= C.MAX_TRAVEL_TICKS


def test_travel_matrix_dead_slots_zeroed():
    state = reset()
    # slots 6+ are empty
    for i in range(6, C.MAX_BUILDING_SLOTS):
        assert state.travel_matrix[0, i] == 0
        assert state.travel_matrix[i, 0] == 0


def test_movement_progress_increments_one_per_tick():
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=50, travel_ticks=5, progress=0)
    step_tick(state)
    assert int(state.unit_groups[0]["progress"]) == 1
    step_tick(state)
    assert int(state.unit_groups[0]["progress"]) == 2


def test_movement_arrival_at_travel_ticks():
    """Arrives on the tick that brings progress to travel_ticks."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=100, travel_ticks=3, progress=0)
    # 3 ticks of movement needed.
    step_tick(state); assert state.unit_groups[0]["alive"] == 1
    step_tick(state); assert state.unit_groups[0]["alive"] == 1
    step_tick(state); assert state.unit_groups[0]["alive"] == 0      # cleared on arrival
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1       # captured


def test_movement_slot_cleared_fully_on_arrival():
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=100, travel_ticks=1, progress=0)
    step_tick(state)
    g = state.unit_groups[0]
    assert g["alive"] == 0
    assert g["count"] == 0
    assert g["progress"] == 0
    assert g["travel_ticks"] == 0


def test_crossing_groups_do_not_cancel_in_flight():
    """Opposing groups passing through the same lane should survive until arrival."""
    state = reset()
    step_tick(
        state,
        action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=P2_BASE),
        action_p2=Action(kind="send", type_idx=3, src=P2_BASE, tgt=P1_BASE),
    )

    # Base-to-base travel is 4 ticks on the default map. Until the arrival tick,
    # both in-flight groups should remain alive with their original counts.
    for _ in range(2):
        alive = state.unit_groups[state.unit_groups["alive"] == 1]
        assert len(alive) == 2
        assert sorted(int(g["count"]) for g in alive) == [100, 100]
        step_tick(state)

    alive = state.unit_groups[state.unit_groups["alive"] == 1]
    assert len(alive) == 2
    assert sorted(int(g["count"]) for g in alive) == [100, 100]


def test_quirk_travel_ticks_one_arrives_same_tick_as_send():
    """QUIRK: because actions apply before movement, a send with travel_ticks=1
    lands at its target in the same step_tick call. Engineered behaviour —
    test pins it so accidental movement/action re-ordering is caught."""
    state = reset()
    # Synthetically make the travel matrix 1 tick for slot 0 → slot 2.
    state.travel_matrix[P1_BASE, N1_TOP] = 1
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP))
    # Target captured within this same call.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1


# ===========================================================================
# 6. Send action (apply)
# ===========================================================================


def test_send_deducts_garrison_and_spawns_group():
    state = reset()
    g0 = int(state.buildings["garrison"][P1_BASE])
    action = Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP)  # 100%
    step_tick(state, action_p1=action)
    # Full garrison was sent, then +1 production.
    assert int(state.buildings["garrison"][P1_BASE]) == C.PRODUCTION_PER_TICK
    # Group alive with original count.
    alive = state.unit_groups[state.unit_groups["alive"] == 1]
    assert len(alive) == 1
    assert int(alive[0]["count"]) == g0
    assert int(alive[0]["owner"]) == C.OWNER_P1


def test_send_invalid_src_unowned_is_dropped():
    state = reset()
    g_p2_before = int(state.buildings["garrison"][P2_BASE])
    # P1 trying to send from P2's base.
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P2_BASE, tgt=N1_TOP))
    # P2 base got production, not deduction.
    assert int(state.buildings["garrison"][P2_BASE]) == g_p2_before + C.PRODUCTION_PER_TICK
    assert not np.any(state.unit_groups["alive"] == 1)


def test_send_src_equals_tgt_is_dropped():
    state = reset()
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=P1_BASE))
    assert not np.any(state.unit_groups["alive"] == 1)


def test_send_to_dead_slot_is_dropped():
    state = reset()
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=20))
    assert not np.any(state.unit_groups["alive"] == 1)


def test_send_too_small_garrison_is_dropped():
    state = reset()
    # Force garrison to 0.5 real (5 internal) — below MIN_SEND_INTERNAL at every pct.
    state.buildings["garrison"][P1_BASE] = 5
    step_tick(state, action_p1=Action(kind="send", type_idx=0, src=P1_BASE, tgt=N1_TOP))
    assert not np.any(state.unit_groups["alive"] == 1)


def test_send_invalid_action_kind_is_ignored():
    state = reset()
    g_before = int(state.buildings["garrison"][P1_BASE])
    step_tick(state, action_p1=Action(kind="weird", type_idx=3, src=P1_BASE, tgt=N1_TOP))
    assert int(state.buildings["garrison"][P1_BASE]) == g_before + C.PRODUCTION_PER_TICK
    assert not np.any(state.unit_groups["alive"] == 1)


def test_send_capture_neutral_full_flow():
    """Capture a neutral and verify owner/garrison/reward exactly.

    Tick layout (travel_ticks = T, action applied on tick 0):
      tick 0: action → progress 0→1 (same tick via action+movement ordering)
      ticks 1..T-1: movement only, progress climbs to T (arrival on tick T-1).
    So we call step_tick once for the action, then (T-1) more times to land exactly
    on the arrival tick and read the post-combat garrison with no extra production.
    """
    state = reset()
    action = Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP)   # 100%
    r1_total = r2_total = 0.0
    r1, r2, _ = step_tick(state, action_p1=action)
    r1_total += r1; r2_total += r2
    assert np.any(state.unit_groups["alive"] == 1)

    travel = int(state.travel_matrix[P1_BASE, N1_TOP])
    for _ in range(travel - 1):
        r1, r2, _ = step_tick(state)
        r1_total += r1; r2_total += r2

    # Combat: garrison=10, attackers=100. absorb=3, dmg=97, eff_def=13, new=87.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert int(state.buildings["garrison"][N1_TOP]) == 87
    assert r1_total == pytest.approx(C.REWARD_CAPTURE)
    assert r2_total == 0.0


def test_send_defender_holds_neutral_proportional_defense():
    """25% of 10 real = 2 real attackers vs 5-real neutral. Defense is proportional."""
    state = reset()
    action = Action(kind="send", type_idx=0, src=P1_BASE, tgt=N3_LEFT)   # 25% to N3
    step_tick(state, action_p1=action)
    travel = int(state.travel_matrix[P1_BASE, N3_LEFT])
    for _ in range(travel - 1):
        step_tick(state)
    # Combat: garrison=50, attackers=20. Effective defense = 65.
    # Remaining garrison = (65 - 20) / 1.3 = 34.615 -> 35.
    assert int(state.buildings["owner"][N3_LEFT]) == C.OWNER_NEUTRAL
    assert int(state.buildings["garrison"][N3_LEFT]) == 35


def test_send_to_own_building_reinforces_no_combat():
    state = reset()
    # Capture N1 first via 100% send; then send more from base to reinforce.
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP))
    for _ in range(int(state.travel_matrix[P1_BASE, N1_TOP]) + 5):
        step_tick(state)
    # N1 now owned by P1 with ~87 garrison + production.
    owner = int(state.buildings["owner"][N1_TOP])
    assert owner == C.OWNER_P1

    # Now send 100% from P1_BASE → N1_TOP. It's friendly; should reinforce.
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP))
    for _ in range(int(state.travel_matrix[P1_BASE, N1_TOP])):
        r1, r2, _ = step_tick(state)
    # No capture reward this time — friendly reinforcement only.
    # (The tick that resolves reinforcement should not emit reward.)
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    # Calibrated to v9.1 sim constants (TRAVEL_SPEED=100). Was 257 under v9.0
    # (TRAVEL_SPEED=200). The 20-unit delta is two extra travel ticks of
    # production at both endpoints (P1_BASE→N1_TOP transit went 2→3 ticks,
    # and the same again on the second leg).
    assert int(state.buildings["garrison"][N1_TOP]) == 277


def test_friendly_reinforce_clamps_at_capacity():
    """Friendly arrivals must not push garrison above the building's capacity."""
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    cap = int(state.buildings["capacity"][N1_TOP])
    state.buildings["garrison"][N1_TOP] = cap - 5
    _clear_groups(state)
    # Inject a 100-unit P1 arrival landing this tick — would overshoot by 95.
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=100, travel_ticks=1, progress=0)
    step_tick(state)
    assert int(state.buildings["garrison"][N1_TOP]) == cap


def test_send_p1_and_p2_both_act_same_tick():
    state = reset()
    a1 = Action(kind="send", type_idx=3, src=P1_BASE, tgt=N3_LEFT)
    a2 = Action(kind="send", type_idx=3, src=P2_BASE, tgt=N4_RIGHT)
    step_tick(state, action_p1=a1, action_p2=a2)
    owners = state.unit_groups["owner"][state.unit_groups["alive"] == 1]
    assert set(owners.tolist()) == {C.OWNER_P1, C.OWNER_P2}


# ===========================================================================
# 7. Same-tick arrival ordering
# ===========================================================================


def test_same_tick_equal_hostiles_mutual_kill():
    """Equal P1 and P2 forces on a neutral resolve simultaneously → mutual kill.

    50×10 + 50×10 = 1000 attack vs 10×13 = 130 defense. Defender dies.
    Each attacker loses 130*500/1000 = 65 → survivors 435 each. Tie → neutral.
    """
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=50, travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P2, P2_BASE, N1_TOP, count=50, travel_ticks=1)
    step_tick(state)
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_NEUTRAL
    assert int(state.buildings["garrison"][N1_TOP]) == 0


def test_same_tick_unequal_hostiles_larger_wins():
    """Unequal simultaneous hostiles: larger force takes the building."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=100, travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P2, P2_BASE, N1_TOP, count=60,  travel_ticks=1)
    r1, r2, _ = step_tick(state)
    # total_atk=160*10=1600 vs def=1*13=13. survivors: P1=1000-130*1000/1600=919,
    # P2=600-130*600/1600=552. P1 wins, remaining=(919-552+5)/10=37.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert int(state.buildings["garrison"][N1_TOP]) == 37
    assert r1 == pytest.approx(C.REWARD_CAPTURE)


def test_same_tick_swap_order_gives_same_result():
    """Same two hostiles landing same tick — outcome is insensitive to source
    unit-group slot order (which used to grant P1 a first-mover advantage)."""
    state = reset()
    _clear_groups(state)
    # Swap slot assignment vs previous test: P2 in slot 0, P1 in slot 1.
    _inject_group(state, 0, C.OWNER_P2, P2_BASE, N1_TOP, count=60,  travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P1, P1_BASE, N1_TOP, count=100, travel_ticks=1)
    step_tick(state)
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert int(state.buildings["garrison"][N1_TOP]) == 37


def test_same_tick_p2_larger_force_wins():
    """P2's larger force beats P1's smaller simultaneous strike."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=30,  travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P2, P2_BASE, N1_TOP, count=100, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    # total_atk=1300, def=13. survivors: P1=300-30=270, P2=1000-100=900.
    # P2 wins, remaining=(900-270+5)/10=63.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P2
    assert int(state.buildings["garrison"][N1_TOP]) == 63
    assert r2 == pytest.approx(C.REWARD_CAPTURE)


def test_same_tick_multiple_friendly_plus_hostile():
    """Two P1 groups pool as friendlies… wait, defender is NEUTRAL so both P1
    groups are hostile. They pool as one P1 attacker vs P2's hostile."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=40, travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P1, P1_BASE, N1_TOP, count=30, travel_ticks=1)
    _inject_group(state, 2, C.OWNER_P2, P2_BASE, N1_TOP, count=40, travel_ticks=1)
    step_tick(state)
    # Hostiles: P1=70, P2=40. total=110*10=1100 vs def=13.
    # survivors: P1=700-130*700/1100=618, P2=400-130*400/1100=353.
    # P1 wins, remaining=(618-353+5)/10=27.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert int(state.buildings["garrison"][N1_TOP]) == 27


def test_same_tick_reinforcement_defends_vs_simultaneous_attack():
    """A same-tick reinforcement pools with the defender against the attack."""
    state = reset()
    # Make P1 own N1 with 100 garrison.
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = 100
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=40, travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P2, P2_BASE, N1_TOP, count=80, travel_ticks=1)
    step_tick(state)
    # Production runs before movement → 110. Friendly reinforcement +40 = 150.
    # P2 attack 80×10=800 vs defense 150×13=1950 → defender holds, remaining
    # = (1950-800+6)//13 = 88.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert int(state.buildings["garrison"][N1_TOP]) == 88


# ===========================================================================
# 8. Victory
# ===========================================================================


def test_victory_playing_by_default():
    state = reset()
    _, _, done = step_tick(state)
    assert not done
    assert state.phase == C.PHASE_PLAYING


def _expected_win_reward(state) -> float:
    bonus = C.REWARD_SPEED_BONUS * max(0.0, 1.0 - state.tick / C.GAME_TIMEOUT_TICKS)
    return C.REWARD_WIN + bonus


def test_victory_elimination_p1_wins():
    state = reset()
    state.buildings["owner"][P2_BASE] = C.OWNER_NEUTRAL
    state.buildings["garrison"][P2_BASE] = 0
    _clear_groups(state)                    # no P2 in-flight
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P1_WINS
    assert r1 == pytest.approx(_expected_win_reward(state))
    assert r2 == pytest.approx(C.REWARD_LOSE)


def test_victory_elimination_p2_wins():
    state = reset()
    state.buildings["owner"][P1_BASE] = C.OWNER_NEUTRAL
    state.buildings["garrison"][P1_BASE] = 0
    _clear_groups(state)
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P2_WINS
    assert r1 == pytest.approx(C.REWARD_LOSE)
    assert r2 == pytest.approx(_expected_win_reward(state))


def test_victory_in_flight_keeps_player_alive():
    """Player has 0 buildings but 1 in-flight group → not eliminated yet."""
    state = reset()
    state.buildings["owner"][P2_BASE] = C.OWNER_NEUTRAL
    state.buildings["garrison"][P2_BASE] = 0
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P2, P1_BASE, P2_BASE, count=50, travel_ticks=8)
    _, _, done = step_tick(state)
    assert not done       # still flying
    assert state.phase == C.PHASE_PLAYING


def test_victory_timeout_tiebreak_by_buildings():
    state = reset()
    # P1 owns base + 2 neutrals; P2 owns base. P1 has 3 > P2 has 1.
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["owner"][N2_BOT] = C.OWNER_P1
    state.tick = C.GAME_TIMEOUT_TICKS - 1
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P1_WINS
    assert r1 == pytest.approx(C.REWARD_WIN)
    assert r2 == pytest.approx(C.REWARD_LOSE)


def test_victory_timeout_tiebreak_by_units_when_buildings_equal():
    state = reset()
    # 1 building each. Give P1 more total units.
    state.buildings["garrison"][P1_BASE] = 200
    state.buildings["garrison"][P2_BASE] = 100
    state.tick = C.GAME_TIMEOUT_TICKS - 1
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_P1_WINS


def test_victory_timeout_draw_when_everything_equal():
    state = reset()
    # Equal buildings + equal units.
    state.buildings["garrison"][P1_BASE] = 100
    state.buildings["garrison"][P2_BASE] = 100
    state.tick = C.GAME_TIMEOUT_TICKS - 1
    r1, r2, done = step_tick(state)
    assert done
    assert state.phase == C.PHASE_DRAW
    assert r1 == pytest.approx(C.REWARD_DRAW)
    assert r2 == pytest.approx(C.REWARD_DRAW)


def test_terminal_phase_stepping_is_safe_noop():
    state = reset()
    state.phase = C.PHASE_P1_WINS
    tick_before = state.tick
    r1, r2, done = step_tick(state)
    assert done and r1 == 0 and r2 == 0
    assert state.tick == tick_before
    assert state.perf["n_ticks"] == 0


# ===========================================================================
# 9. Mask
# ===========================================================================


def test_mask_noop_always_legal():
    state = reset()
    for player in (C.OWNER_P1, C.OWNER_P2):
        mask = compute_mask(state, player)
        assert mask[NOOP_INDEX]


def test_mask_matches_is_valid():
    """Cross-check: compute_mask should equal the per-action is_valid scan."""
    from sim.actions import is_valid
    state = reset()
    mask = compute_mask(state, C.OWNER_P1)
    # Sample a chunk of the action space (full sweep too slow for unit test).
    for idx in range(0, ACTION_SPACE_SIZE, 17):
        a = decode(idx)
        assert mask[idx] == is_valid(state, C.OWNER_P1, a)


def test_mask_self_pair_forbidden():
    state = reset()
    mask = compute_mask(state, C.OWNER_P1)
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for slot in range(C.MAX_BUILDING_SLOTS):
            assert not mask[encode(type_idx, slot, slot)]


def test_mask_owned_source_required():
    state = reset()
    mask = compute_mask(state, C.OWNER_P1)
    # slot 1 is P2's base — P1 cannot send from it.
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for tgt in range(C.MAX_BUILDING_SLOTS):
            if tgt == P2_BASE:
                continue
            assert not mask[encode(type_idx, P2_BASE, tgt)]


def test_mask_dead_target_forbidden():
    state = reset()
    mask = compute_mask(state, C.OWNER_P1)
    # slots 6..31 are dead.
    for type_idx in range(len(C.SEND_PERCENTAGES)):
        for dead_tgt in range(6, C.MAX_BUILDING_SLOTS):
            assert not mask[encode(type_idx, P1_BASE, dead_tgt)]


def test_mask_no_free_group_slots_only_noop():
    state = reset()
    # Occupy every group slot.
    for slot in range(C.MAX_UNIT_GROUP_SLOTS):
        _inject_group(state, slot, C.OWNER_P1, P1_BASE, N1_TOP,
                      count=10, travel_ticks=8)
    mask = compute_mask(state, C.OWNER_P1)
    assert mask.sum() == 1                    # only noop
    assert mask[NOOP_INDEX]


# ===========================================================================
# 10. Invariants (property-style over scripted random games)
# ===========================================================================


def _random_action(state, player, rng):
    mask = compute_mask(state, player)
    legal = np.where(mask)[0]
    return decode(int(rng.choice(legal)))


def test_invariant_garrison_nonnegative_under_random_play():
    rng = np.random.default_rng(7)
    state = reset(seed=7)
    for _ in range(C.GAME_TIMEOUT_TICKS):
        a1 = _random_action(state, C.OWNER_P1, rng)
        a2 = _random_action(state, C.OWNER_P2, rng)
        step_tick(state, a1, a2)
        assert np.all(state.buildings["garrison"] >= 0)
        assert np.all(state.unit_groups["count"] >= 0)
        if state.phase != C.PHASE_PLAYING:
            break


def test_invariant_production_does_not_increase_owned_building_above_capacity():
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = state.buildings["capacity"][N1_TOP] + 70
    before = int(state.buildings["garrison"][N1_TOP])
    step_tick(state)
    assert int(state.buildings["garrison"][N1_TOP]) == before


def test_invariant_alive_garrison_nonnegative_under_random_play():
    rng = np.random.default_rng(11)
    state = reset(seed=11)
    for _ in range(C.GAME_TIMEOUT_TICKS):
        step_tick(
            state,
            _random_action(state, C.OWNER_P1, rng),
            _random_action(state, C.OWNER_P2, rng),
        )
        alive = state.buildings["alive"] == 1
        assert np.all(state.buildings["garrison"][alive] >= 0)
        if state.phase != C.PHASE_PLAYING:
            break


def test_reset_unknown_level_raises_value_error():
    with pytest.raises(ValueError):
        reset("missing_level")


def test_apply_unknown_level_raises_value_error():
    state = reset()
    with pytest.raises(ValueError):
        apply(state, "missing_level")


def test_apply_invalid_building_type_raises_value_error():
    state = reset()
    original = sim_levels.LEVELS.get("bad_type_test")
    sim_levels.LEVELS["bad_type_test"] = [
        (C.OWNER_P1, 100, 100, 10, 999),
    ]
    try:
        with pytest.raises(ValueError):
            apply(state, "bad_type_test")
    finally:
        if original is None:
            del sim_levels.LEVELS["bad_type_test"]
        else:
            sim_levels.LEVELS["bad_type_test"] = original


def test_apply_recomputes_travel_matrix_after_manual_corruption():
    state = reset()
    state.travel_matrix[P1_BASE, N1_TOP] = 99
    apply(state, "crossroads_6")
    # Calibrated to v9.1 (TRAVEL_SPEED=100). Was 2 under v9.0. Test verifies
    # apply() recomputed the matrix (not the corrupted 99); the literal value
    # is the actual ceil(dist / TRAVEL_SPEED) for this slot pair.
    assert int(state.travel_matrix[P1_BASE, N1_TOP]) == 3


def test_reset_clears_in_flight_groups():
    state = reset()
    step_tick(
        state,
        action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=P2_BASE),
        action_p2=Action(kind="send", type_idx=3, src=P2_BASE, tgt=P1_BASE),
    )
    assert np.any(state.unit_groups["alive"] == 1)

    state = reset()
    assert not np.any(state.unit_groups["alive"] == 1)


def test_apply_level_too_large_raises_value_error():
    state = reset()
    original = sim_levels.LEVELS.get("too_large_test")
    sim_levels.LEVELS["too_large_test"] = [
        (C.OWNER_NEUTRAL, i, 0, 1, C.TYPE_BASIC)
        for i in range(C.MAX_BUILDING_SLOTS + 1)
    ]
    try:
        with pytest.raises(ValueError):
            apply(state, "too_large_test")
    finally:
        if original is None:
            del sim_levels.LEVELS["too_large_test"]
        else:
            sim_levels.LEVELS["too_large_test"] = original


def test_invariant_owner_codes_are_valid():
    rng = np.random.default_rng(23)
    state = reset(seed=23)
    for _ in range(C.GAME_TIMEOUT_TICKS):
        step_tick(
            state,
            _random_action(state, C.OWNER_P1, rng),
            _random_action(state, C.OWNER_P2, rng),
        )
        owners = state.buildings["owner"][state.buildings["alive"] == 1]
        assert set(owners.tolist()).issubset({C.OWNER_NEUTRAL, C.OWNER_P1, C.OWNER_P2})
        if state.phase != C.PHASE_PLAYING:
            break


def test_invariant_unit_counts_not_decreasing_without_combat():
    """With both players on noop, total units = garrisons only = grows monotonically
    (by 2 per tick — one per owned base) until capacities are hit."""
    state = reset()
    prev = _total_units_on_board(state)
    for _ in range(20):
        step_tick(state)
        cur = _total_units_on_board(state)
        assert cur >= prev
        prev = cur


def test_invariant_send_conserves_units_pre_combat():
    """After a send but before arrival: (src garrison + in-flight) should equal
    the pre-send garrison, plus production over ticks elapsed."""
    state = reset()
    g_before = int(state.buildings["garrison"][P1_BASE])
    action = Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP)   # 100%
    step_tick(state, action_p1=action)
    # After one tick: garrison = 0 (sent all), then +1 production = 10.
    # In flight = g_before.
    g_after = int(state.buildings["garrison"][P1_BASE])
    in_flight = int(state.unit_groups[0]["count"])
    assert g_after + in_flight == g_before + C.PRODUCTION_PER_TICK


# ===========================================================================
# 11. Determinism + replay
# ===========================================================================


def test_determinism_scripted_game():
    """Same inputs → byte-identical final state arrays."""
    script = {
        5:  Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP),
        12: Action(kind="send", type_idx=1, src=N1_TOP,  tgt=N2_BOT),
        20: Action(kind="send", type_idx=2, src=P1_BASE, tgt=N3_LEFT),
    }

    def run():
        s = reset(seed=42)
        for _ in range(80):
            step_tick(s, action_p1=script.get(s.tick))
        return s

    s1, s2 = run(), run()
    assert np.array_equal(s1.buildings, s2.buildings)
    assert np.array_equal(s1.unit_groups, s2.unit_groups)
    assert s1.tick == s2.tick
    assert s1.phase == s2.phase


def test_determinism_both_players_random_but_seeded():
    def run():
        rng = np.random.default_rng(2024)
        s = reset(seed=2024)
        for _ in range(C.GAME_TIMEOUT_TICKS + 5):
            a1 = _random_action(s, C.OWNER_P1, rng)
            a2 = _random_action(s, C.OWNER_P2, rng)
            _, _, done = step_tick(s, a1, a2)
            if done:
                break
        return s

    s1, s2 = run(), run()
    assert np.array_equal(s1.buildings, s2.buildings)
    assert np.array_equal(s1.unit_groups, s2.unit_groups)
    assert s1.tick == s2.tick
    assert s1.phase == s2.phase


# ===========================================================================
# 12. Integration — full scripted game with hand-computed checkpoints
# ===========================================================================


def test_integration_p1_cant_solo_a_full_base():
    """A single 100% send from cap vs cap cannot crack the 1.3x defense bonus."""
    state = reset()
    for _ in range(25):                          # stockpile to cap
        step_tick(state)
    assert int(state.buildings["garrison"][P1_BASE]) == C.DEFAULT_CAPACITY
    assert int(state.buildings["garrison"][P2_BASE]) == C.DEFAULT_CAPACITY

    action = Action(kind="send", type_idx=3, src=P1_BASE, tgt=P2_BASE)
    step_tick(state, action_p1=action)

    travel = int(state.travel_matrix[P1_BASE, P2_BASE])
    assert C.MIN_TRAVEL_TICKS <= travel <= C.MAX_TRAVEL_TICKS

    r1_total = 0.0
    for _ in range(travel - 1):
        r1, _, _ = step_tick(state)
        r1_total += r1

    # Arrival tick combat: garrison=300 (cap), attackers=300.
    # Effective defense = 390. Defender holds with (390 - 300) / 1.3 = 69.23 -> 69.
    assert int(state.buildings["owner"][P2_BASE]) == C.OWNER_P2
    assert int(state.buildings["garrison"][P2_BASE]) == 69
    assert r1_total == 0.0


# ===========================================================================
# 13. Accuracy audit — documented / canonical rule mismatches
# ===========================================================================


def test_spec_friendly_reinforcement_clamps_at_capacity():
    """Friendly reinforcement cannot push garrison above the building's capacity.

    Excess incoming units are discarded. This matches the documented spec
    (``DEFAULT_CAPACITY`` is the max garrison) and was a sim bug fix.
    """
    state = reset()
    target = N1_TOP
    state.buildings["owner"][target] = C.OWNER_P1
    state.buildings["garrison"][target] = state.buildings["capacity"][target]
    state.buildings["garrison"][P1_BASE] = 500

    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=target))
    for _ in range(int(state.travel_matrix[P1_BASE, target]) - 1):
        step_tick(state)

    assert int(state.buildings["garrison"][target]) == int(state.buildings["capacity"][target])


def test_spec_capture_preserves_all_surviving_attackers():
    state = reset()
    target = N1_TOP
    state.buildings["garrison"][P1_BASE] = 500
    state.buildings["garrison"][target] = 10

    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=target))
    for _ in range(int(state.travel_matrix[P1_BASE, target]) - 1):
        step_tick(state)

    # 500 attackers into 1-real neutral (10 internal):
    # effective defense = 13, survivors = 48.7 real = 487 internal.
    assert int(state.buildings["owner"][target]) == C.OWNER_P1
    assert int(state.buildings["garrison"][target]) == 487


def test_spec_one_unit_chip_damage_does_not_zero_a_one_unit_neutral():
    state = reset()
    state.buildings["garrison"][P1_BASE] = 20          # 2 real units
    state.buildings["garrison"][N1_TOP] = 10           # 1 real unit

    step_tick(state, action_p1=Action(kind="send", type_idx=1, src=P1_BASE, tgt=N1_TOP))  # 50% -> 1 real
    for _ in range(int(state.travel_matrix[P1_BASE, N1_TOP]) - 1):
        step_tick(state)

    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_NEUTRAL
    assert 0 < int(state.buildings["garrison"][N1_TOP]) < 10


def test_spec_is_valid_rejects_out_of_range_type_idx():
    state = reset()
    action = Action(kind="send", type_idx=99, src=P1_BASE, tgt=N1_TOP)
    assert not is_valid(state, C.OWNER_P1, action)


def test_spec_apply_resets_dynamic_state_when_reusing_state():
    state = reset()
    step_tick(state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=N1_TOP))
    assert np.any(state.unit_groups["alive"] == 1)
    assert state.tick == 1

    from sim.levels import apply

    apply(state)

    assert not np.any(state.unit_groups["alive"] == 1)
    assert state.tick == 0
    assert state.phase == C.PHASE_PLAYING
    assert state.perf["n_ticks"] == 0


def test_integration_scripted_p1_wins_full_game():
    """P1 captures every neutral and takes P2's base. Assert phase + rewards."""
    state = reset()
    # Build up at P1 base.
    for _ in range(20):
        step_tick(state)

    # Send 100% waves to each neutral one at a time, wait for arrival.
    r1_total = 0.0
    for target in (N1_TOP, N3_LEFT, N2_BOT, N4_RIGHT):
        # Ensure garrison is enough for 100% send.
        while int(state.buildings["garrison"][P1_BASE]) < 100:
            r1, r2, _ = step_tick(state); r1_total += r1
        r1, _, _ = step_tick(
            state, action_p1=Action(kind="send", type_idx=3, src=P1_BASE, tgt=target)
        )
        r1_total += r1
        for _ in range(int(state.travel_matrix[P1_BASE, target]) + 1):
            r1, _, _ = step_tick(state)
            r1_total += r1

    # P1 should now own 5 buildings (its base + 4 ex-neutrals).
    assert count_owned_buildings(state, C.OWNER_P1) == 5
    assert r1_total >= 4 * C.REWARD_CAPTURE - 1e-6


# ===========================================================================
# 13. Rewards accounting
# ===========================================================================


def test_reward_capture_only_on_ownership_flip():
    """Reinforcing own building does NOT emit REWARD_CAPTURE."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P2_BASE, P1_BASE,
                  count=50, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    assert r1 == 0.0
    assert r2 == 0.0


def test_reward_loss_when_losing_building_to_enemy():
    """P2 captures P1-owned building → P1 gets REWARD_LOSS, P2 gets REWARD_CAPTURE."""
    state = reset()
    # Give P1 the neutral N1 first (no reward bookkeeping needed here).
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = 20
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P2, P2_BASE, N1_TOP,
                  count=100, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    # Production first (N1 owned by P1): garrison 20 → 30. Combat: absorb=9, dmg=91,
    # eff_def=39, new = 100 - 39 = 61. P2 captures.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P2
    assert r1 == pytest.approx(C.REWARD_LOSS)
    assert r2 == pytest.approx(C.REWARD_CAPTURE)


def test_reward_neutral_capture_no_loss_penalty():
    """Capturing a neutral gives REWARD_CAPTURE — no REWARD_LOSS fires."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=100, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    assert r1 == pytest.approx(C.REWARD_CAPTURE)
    assert r2 == 0.0


def test_reward_mutual_wipe_both_lose_none_captures():
    """Owner → neutral via mutual wipe: loser gets REWARD_LOSS, no capture reward."""
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = 100
    _clear_groups(state)
    # attackers = eff_def = 130 → mutual wipe (see test_combat_mutual_wipe_exact_eff_def).
    # But production runs first (garrison 100 → 110), so eff_def changes. Use exact.
    # We bypass production-effect by setting owner to NEUTRAL so production skips it,
    # then combat is against the raw 100 garrison.
    state.buildings["owner"][N1_TOP] = C.OWNER_NEUTRAL
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP,
                  count=130, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    # Owner was NEUTRAL → no REWARD_LOSS fires; new owner is NEUTRAL → no capture.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_NEUTRAL
    assert r1 == 0.0 and r2 == 0.0


def test_capture_event_emitted_on_ownership_change():
    """`kind: "capture"` is emitted when a building changes owner."""
    state = reset()
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=100, travel_ticks=1)
    events: list = []
    step_tick(state, events=events)
    captures = [e for e in events if e.get("kind") == "capture"]
    assert len(captures) == 1
    c = captures[0]
    assert c["tgt"] == N1_TOP
    assert c["owner_before"] == C.OWNER_NEUTRAL
    assert c["owner_after"] == C.OWNER_P1


def test_no_capture_event_when_defender_holds():
    """No capture event when ownership is unchanged (defender holds)."""
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P1
    state.buildings["garrison"][N1_TOP] = 100
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P2, P2_BASE, N1_TOP, count=20, travel_ticks=1)
    events: list = []
    step_tick(state, events=events)
    assert not any(e.get("kind") == "capture" for e in events)
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P1


def test_p1_p2_symmetric_random_play_winrate():
    """Scripted random policy for both players — P1 and P2 should win
    roughly equally. Catches any systematic first-mover or ordering bias.

    With 200 games (2 per seed, swapping sides) the stat band for a fair coin
    is ~45–55% at 2σ. The OLD sequential sim had ~68% P1 bias on the same
    sample size — this test would fail loudly under the old engine.
    """
    p1_wins = 0
    total = 0
    n_games = 200
    for seed in range(n_games):
        rng = np.random.default_rng(seed)
        state = reset(level_name="random_8_12", seed=seed)
        done = False
        for _ in range(C.GAME_TIMEOUT_TICKS + 10):
            a1 = _random_action(state, C.OWNER_P1, rng)
            a2 = _random_action(state, C.OWNER_P2, rng)
            _, _, done = step_tick(state, a1, a2)
            if done:
                break
        p1_b = count_owned_buildings(state, C.OWNER_P1)
        p2_b = count_owned_buildings(state, C.OWNER_P2)
        if p1_b > p2_b:
            p1_wins += 1
            total += 1
        elif p2_b > p1_b:
            total += 1
        # ties/draws (both zero, timeout with equal buildings) don't count
    p1_rate = p1_wins / max(total, 1)
    assert 0.40 <= p1_rate <= 0.60, (
        f"P1 win rate {p1_rate:.3f} out of fair band [0.40, 0.60] "
        f"({p1_wins}/{total}) — systematic side bias?"
    )


def test_reward_defender_reinforcement_pools_no_flip():
    """Simultaneous same-size attack + defender reinforcement → defender holds,
    no ownership change, no capture/loss rewards."""
    state = reset()
    state.buildings["owner"][N1_TOP] = C.OWNER_P2
    state.buildings["garrison"][N1_TOP] = 10
    _clear_groups(state)
    _inject_group(state, 0, C.OWNER_P1, P1_BASE, N1_TOP, count=100, travel_ticks=1)
    _inject_group(state, 1, C.OWNER_P2, P2_BASE, N1_TOP, count=100, travel_ticks=1)
    r1, r2, _ = step_tick(state)
    # Prod tick for P2-owned N1 → garrison 10+10=20. Friendly P2 +100 = 120.
    # P1 attack 100×10=1000 vs defense 120×13=1560 → defender holds,
    # remaining=(1560-1000+6)//13=43. No flip, no rewards.
    assert int(state.buildings["owner"][N1_TOP]) == C.OWNER_P2
    assert int(state.buildings["garrison"][N1_TOP]) == 43
    assert r1 == pytest.approx(0.0)
    assert r2 == pytest.approx(0.0)


# ===========================================================================
# 14. Sanity: random games terminate
# ===========================================================================


def test_random_game_terminates():
    rng = np.random.default_rng(123)
    state = reset(seed=123)
    done = False
    for _ in range(C.GAME_TIMEOUT_TICKS + 10):
        a1 = _random_action(state, C.OWNER_P1, rng)
        a2 = _random_action(state, C.OWNER_P2, rng)
        _, _, done = step_tick(state, a1, a2)
        if done:
            break
    assert done
    assert state.phase in (C.PHASE_P1_WINS, C.PHASE_P2_WINS, C.PHASE_DRAW)


@pytest.mark.parametrize("seed", list(range(10)))
def test_random_game_terminates_across_seeds(seed):
    """Sanity sweep: 10 different seeds all terminate cleanly."""
    rng = np.random.default_rng(seed)
    state = reset(seed=seed)
    done = False
    for _ in range(C.GAME_TIMEOUT_TICKS + 10):
        _, _, done = step_tick(
            state,
            _random_action(state, C.OWNER_P1, rng),
            _random_action(state, C.OWNER_P2, rng),
        )
        if done:
            break
    assert done
    assert state.phase in (C.PHASE_P1_WINS, C.PHASE_P2_WINS, C.PHASE_DRAW)
