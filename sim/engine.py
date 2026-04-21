"""
Core sim step loop. Pure numpy — numba can wrap the hot functions later.

Per-tick phases:
  1. apply actions       (if caller provided any)
  2. advance production  (owned buildings regenerate garrison up to capacity)
  3. advance movement    (unit groups progress; collect arrivals)
  4. resolve arrivals    (field cancel → reinforce or combat)
  5. check victory

All five phases are wrapped in perf_counter timers (state.perf). The total cost
is ~1 µs/tick — essentially free — and lets the benchmarker show exact phase
breakdown without a separate profiler run.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from sim import config as C
from sim.actions import Action, is_valid, send_amount
from sim.state import State, count_owned_buildings, has_in_flight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def step_tick(
    state: State,
    action_p1: Optional[Action] = None,
    action_p2: Optional[Action] = None,
) -> tuple[float, float, bool]:
    """Advance the sim by exactly one tick.

    Actions (if provided) are applied BEFORE production/movement this tick.
    Pass None for a player if they have no action this tick (e.g. between
    decision intervals, or they chose noop).

    Returns (reward_p1, reward_p2, done).
    """
    t0 = time.perf_counter_ns()

    if state.phase != C.PHASE_PLAYING:
        # Terminal — nothing to do.
        return 0.0, 0.0, True

    r1 = r2 = 0.0

    # 1. Actions
    ta = time.perf_counter_ns()
    if action_p1 is not None and action_p1.kind == "send":
        _apply_send(state, C.OWNER_P1, action_p1)
    if action_p2 is not None and action_p2.kind == "send":
        _apply_send(state, C.OWNER_P2, action_p2)
    tb = time.perf_counter_ns()
    state.perf["actions_ns"] += tb - ta

    # 2. Production
    ta = time.perf_counter_ns()
    _advance_production(state)
    tb = time.perf_counter_ns()
    state.perf["production_ns"] += tb - ta

    # 3. Movement → collect arrivals
    ta = time.perf_counter_ns()
    arrivals = _advance_movement(state)
    tb = time.perf_counter_ns()
    state.perf["movement_ns"] += tb - ta

    # 4. Resolve arrivals (field cancel + combat). Produces capture/loss rewards.
    ta = time.perf_counter_ns()
    dr1, dr2 = _resolve_arrivals(state, arrivals)
    r1 += dr1
    r2 += dr2
    tb = time.perf_counter_ns()
    state.perf["combat_ns"] += tb - ta

    state.tick += 1

    # 5. Victory check
    ta = time.perf_counter_ns()
    dr1, dr2, done = _check_victory(state)
    r1 += dr1
    r2 += dr2
    tb = time.perf_counter_ns()
    state.perf["victory_ns"] += tb - ta

    state.perf["n_ticks"] += 1
    state.perf["total_ns"] += time.perf_counter_ns() - t0

    return r1, r2, done


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def _apply_send(state: State, player: int, action: Action) -> None:
    """Validate + spawn a unit group for a send action. Silently drops invalid."""
    if not is_valid(state, player, action):
        return

    b = state.buildings
    g = state.unit_groups
    src, tgt = action.src, action.tgt
    pct = C.SEND_PERCENTAGES[action.type_idx]

    amount = send_amount(int(b["garrison"][src]), pct)
    if amount <= 0:
        return

    # Find a free unit-group slot.
    free_idx = np.where(g["alive"] == 0)[0]
    if free_idx.size == 0:
        return
    slot = int(free_idx[0])

    # Deduct from source garrison; spawn the group.
    b["garrison"][src] -= amount
    g[slot]["alive"]        = 1
    g[slot]["owner"]        = player
    g[slot]["src_slot"]     = src
    g[slot]["tgt_slot"]     = tgt
    g[slot]["count"]        = amount
    g[slot]["progress"]     = 0
    g[slot]["travel_ticks"] = int(state.travel_matrix[src, tgt])


def _advance_production(state: State) -> None:
    """Each owned + alive building regenerates garrison, capped at capacity."""
    b = state.buildings
    alive = b["alive"] == 1
    owned = (b["owner"] == C.OWNER_P1) | (b["owner"] == C.OWNER_P2)
    eligible = alive & owned

    # v0.1: all buildings are TYPE_BASIC with one rate. When more types land,
    # this becomes a per-type rate lookup (cheap — 32-entry gather).
    garrison = b["garrison"].astype(np.int32)
    capacity = b["capacity"].astype(np.int32)
    new_garrison = np.minimum(garrison + C.PRODUCTION_PER_TICK, capacity)
    b["garrison"] = np.where(eligible, new_garrison, garrison).astype(np.int16)


def _advance_movement(state: State) -> list:
    """Advance progress on every alive unit group. Returns list of arrivals.

    An arrival is (tgt_slot, owner, count). Freed slots are cleared.
    """
    g = state.unit_groups
    alive_idx = np.where(g["alive"] == 1)[0]
    arrivals = []

    for idx in alive_idx:
        g[idx]["progress"] += 1
        if g[idx]["progress"] >= g[idx]["travel_ticks"]:
            arrivals.append((
                int(g[idx]["tgt_slot"]),
                int(g[idx]["owner"]),
                int(g[idx]["count"]),
            ))
            # Clear the slot.
            g[idx]["alive"] = 0
            g[idx]["owner"] = 0
            g[idx]["count"] = 0
            g[idx]["progress"] = 0
            g[idx]["travel_ticks"] = 0

    return arrivals


def _resolve_arrivals(state: State, arrivals: list) -> tuple[float, float]:
    """Apply field cancellation + combat for all arrivals this tick.

    Group arrivals by target. For each target:
      1. Sum arrivals by owner (A1 from P1, A2 from P2).
      2. Field cancel: A1' = max(0, A1-A2), A2' = max(0, A2-A1).
      3. If survivors are friendly to the building → reinforce (cap by capacity).
      4. If survivors are enemy → combat with defense bonus.

    Returns (reward_p1, reward_p2) from capture/loss events.
    """
    if not arrivals:
        return 0.0, 0.0

    b = state.buildings
    r1 = r2 = 0.0

    # Aggregate per target.
    by_target: dict[int, list[int]] = {}   # tgt → [A1_total, A2_total]
    for tgt, owner, count in arrivals:
        entry = by_target.setdefault(tgt, [0, 0])
        if owner == C.OWNER_P1:
            entry[0] += count
        elif owner == C.OWNER_P2:
            entry[1] += count

    for tgt, (a1, a2) in by_target.items():
        if not b["alive"][tgt]:
            continue

        # Field cancel.
        a1p = max(0, a1 - a2)
        a2p = max(0, a2 - a1)
        if a1p == 0 and a2p == 0:
            continue

        owner_before = int(b["owner"][tgt])
        garrison = int(b["garrison"][tgt])
        capacity = int(b["capacity"][tgt])

        attacker = C.OWNER_P1 if a1p > 0 else C.OWNER_P2
        survivors = a1p if a1p > 0 else a2p

        if owner_before == attacker:
            # Reinforce — not combat.
            b["garrison"][tgt] = min(capacity, garrison + survivors)
            continue

        new_garrison, new_owner = _combat(garrison, survivors, attacker, owner_before)
        b["owner"][tgt] = new_owner
        b["garrison"][tgt] = min(capacity, new_garrison)

        # Reward bookkeeping.
        if new_owner == attacker:
            if attacker == C.OWNER_P1:
                r1 += C.REWARD_CAPTURE
            else:
                r2 += C.REWARD_CAPTURE
        if new_owner != owner_before and owner_before != C.OWNER_NEUTRAL:
            if owner_before == C.OWNER_P1:
                r1 += C.REWARD_LOSS
            else:
                r2 += C.REWARD_LOSS

    return r1, r2


def _combat(garrison: int, attackers: int, attacker_owner: int, owner_before: int) -> tuple[int, int]:
    """Ratio-fight combat with integer fixed-point. Returns (new_garrison, new_owner).

    effective_defender = garrison * DEF_NUM // DEF_DEN
    if attackers < eff_def: defender holds, new_garrison = eff_def − attackers
    if attackers > eff_def: attacker takes it, new_garrison = attackers − eff_def
    if equal: building becomes neutral with 0 garrison
    """
    eff_def = (garrison * C.DEF_BONUS_NUM) // C.DEF_BONUS_DEN
    if attackers < eff_def:
        return (eff_def - attackers, owner_before)
    if attackers > eff_def:
        return (attackers - eff_def, attacker_owner)
    return (0, C.OWNER_NEUTRAL)


def _check_victory(state: State) -> tuple[float, float, bool]:
    """Apply early-win, elimination, and timeout rules.

    Returns (reward_p1, reward_p2, done). Sets state.phase on terminal.
    """
    p1_bldgs = count_owned_buildings(state, C.OWNER_P1)
    p2_bldgs = count_owned_buildings(state, C.OWNER_P2)

    # Elimination: player owns 0 buildings AND has 0 in-flight groups.
    p1_alive = p1_bldgs > 0 or has_in_flight(state, C.OWNER_P1)
    p2_alive = p2_bldgs > 0 or has_in_flight(state, C.OWNER_P2)

    if not p1_alive and not p2_alive:
        state.phase = C.PHASE_DRAW
        return C.REWARD_DRAW, C.REWARD_DRAW, True
    if not p1_alive:
        state.phase = C.PHASE_P2_WINS
        return C.REWARD_LOSE, C.REWARD_WIN, True
    if not p2_alive:
        state.phase = C.PHASE_P1_WINS
        return C.REWARD_WIN, C.REWARD_LOSE, True

    # Timeout: tiebreak buildings → units → draw.
    if state.tick >= C.GAME_TIMEOUT_TICKS:
        if p1_bldgs > p2_bldgs:
            state.phase = C.PHASE_P1_WINS
            return C.REWARD_WIN, C.REWARD_LOSE, True
        if p2_bldgs > p1_bldgs:
            state.phase = C.PHASE_P2_WINS
            return C.REWARD_LOSE, C.REWARD_WIN, True
        # Buildings tied — compare units.
        from sim.state import count_owned_units
        u1 = count_owned_units(state, C.OWNER_P1)
        u2 = count_owned_units(state, C.OWNER_P2)
        if u1 > u2:
            state.phase = C.PHASE_P1_WINS
            return C.REWARD_WIN, C.REWARD_LOSE, True
        if u2 > u1:
            state.phase = C.PHASE_P2_WINS
            return C.REWARD_LOSE, C.REWARD_WIN, True
        state.phase = C.PHASE_DRAW
        return C.REWARD_DRAW, C.REWARD_DRAW, True

    return 0.0, 0.0, False
