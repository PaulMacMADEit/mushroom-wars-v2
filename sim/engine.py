"""
Core sim step loop. Pure numpy — numba can wrap the hot functions later.

Per-tick phases:
  1. apply actions       (if caller provided any)
  2. advance production  (owned buildings regenerate garrison up to capacity)
  3. advance movement    (unit groups progress; collect arrivals)
  4. resolve arrivals    (sequential reinforce/combat)
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
    events: Optional[list] = None,
) -> tuple[float, float, bool]:
    """Advance the sim by exactly one tick.

    Actions (if provided) are applied BEFORE production/movement this tick.
    Pass None for a player if they have no action this tick (e.g. between
    decision intervals, or they chose noop).

    If `events` is a list, low-level engine events (spawn/arrive/end) are
    appended to it for the replay recorder to consume. No-op when None.

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
        _apply_send(state, C.OWNER_P1, action_p1, events)
    if action_p2 is not None and action_p2.kind == "send":
        _apply_send(state, C.OWNER_P2, action_p2, events)
    tb = time.perf_counter_ns()
    state.perf["actions_ns"] += tb - ta

    # 2. Production
    ta = time.perf_counter_ns()
    _advance_production(state)
    tb = time.perf_counter_ns()
    state.perf["production_ns"] += tb - ta

    # 3. Movement → collect arrivals
    ta = time.perf_counter_ns()
    arrivals = _advance_movement(state, events)
    tb = time.perf_counter_ns()
    state.perf["movement_ns"] += tb - ta

    # 4. Resolve arrivals (simultaneous per target). Produces capture/loss rewards.
    ta = time.perf_counter_ns()
    dr1, dr2 = _resolve_arrivals(state, arrivals, events)
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

    if done and events is not None:
        events.append({"kind": "end", "phase": int(state.phase)})

    state.perf["n_ticks"] += 1
    state.perf["total_ns"] += time.perf_counter_ns() - t0

    return r1, r2, done


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def _apply_send(state: State, player: int, action: Action, events: Optional[list] = None) -> None:
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

    travel_ticks = int(state.travel_matrix[src, tgt])

    # Deduct from source garrison; spawn the group.
    b["garrison"][src] -= amount
    g[slot]["alive"]        = 1
    g[slot]["owner"]        = player
    g[slot]["src_slot"]     = src
    g[slot]["tgt_slot"]     = tgt
    g[slot]["count"]        = amount
    g[slot]["progress"]     = 0
    g[slot]["travel_ticks"] = travel_ticks

    if events is not None:
        events.append({
            "kind": "spawn",
            "slot": slot,
            "owner": int(player),
            "src": int(src),
            "tgt": int(tgt),
            "count": int(amount),
            "travel_ticks": travel_ticks,
        })


def _advance_production(state: State) -> None:
    """Each owned + alive building regenerates garrison up to capacity.

    Buildings may already sit above capacity due to friendly reinforcement or a
    large capture. In that case production should stop, not clamp them back
    down.
    """
    b = state.buildings
    alive = b["alive"] == 1
    owned = (b["owner"] == C.OWNER_P1) | (b["owner"] == C.OWNER_P2)
    below_cap = b["garrison"] < b["capacity"]
    eligible = alive & owned & below_cap

    # v0.1: all buildings are TYPE_BASIC with one rate. When more types land,
    # this becomes a per-type rate lookup (cheap — 32-entry gather).
    garrison = b["garrison"].astype(np.int32)
    capacity = b["capacity"].astype(np.int32)
    new_garrison = np.minimum(garrison + C.PRODUCTION_PER_TICK, capacity)
    b["garrison"] = np.where(eligible, new_garrison, garrison).astype(np.int16)


def _advance_movement(state: State, events: Optional[list] = None) -> list:
    """Advance progress on every alive unit group. Returns list of arrivals.

    An arrival is (tgt_slot, owner, count). Freed slots are cleared.
    """
    g = state.unit_groups
    alive_idx = np.where(g["alive"] == 1)[0]
    arrivals = []

    for idx in alive_idx:
        g[idx]["progress"] += 1
        if g[idx]["progress"] >= g[idx]["travel_ticks"]:
            tgt = int(g[idx]["tgt_slot"])
            owner = int(g[idx]["owner"])
            count = int(g[idx]["count"])
            arrivals.append((tgt, owner, count))
            if events is not None:
                events.append({
                    "kind": "arrive",
                    "slot": int(idx),
                    "owner": owner,
                    "tgt": tgt,
                    "count": count,
                })
            # Clear the slot.
            g[idx]["alive"] = 0
            g[idx]["owner"] = 0
            g[idx]["count"] = 0
            g[idx]["progress"] = 0
            g[idx]["travel_ticks"] = 0

    return arrivals


def _resolve_arrivals(state: State, arrivals: list, events: Optional[list] = None) -> tuple[float, float]:
    """Apply arrivals simultaneously per target.

    All groups landing on the same target this tick resolve as one event:
    friendly groups reinforce together (clamped at capacity); hostile groups
    attack the (possibly-reinforced) defender simultaneously. When two
    different hostile owners land the same tick on the same target, the
    resolution is symmetric — no systematic first-mover advantage.

    Targets are independent — order across different targets doesn't matter.
    Within a single target, friendlies reinforce before hostile combat.

    Emits `kind:"capture"` events whenever ownership changes.

    Returns (reward_p1, reward_p2) from capture/loss events.
    """
    if not arrivals:
        return 0.0, 0.0

    b = state.buildings
    r1 = r2 = 0.0

    # Group arrivals by target so the per-target resolution is
    # order-independent across the full arrival list.
    from collections import defaultdict
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for tgt, owner, count in arrivals:
        by_target[int(tgt)].append((int(owner), int(count)))

    for tgt, groups in by_target.items():
        if not b["alive"][tgt]:
            continue

        owner_before = int(b["owner"][tgt])
        garrison = int(b["garrison"][tgt])
        capacity = int(b["capacity"][tgt])

        # Friendlies match the current owner. Everyone else is hostile.
        friendlies = sum(c for o, c in groups if o == owner_before)
        hostile_by_owner: dict[int, int] = defaultdict(int)
        for o, c in groups:
            if o != owner_before:
                hostile_by_owner[o] += c

        # Friendly reinforcement first — clamped at capacity, excess discarded.
        if friendlies > 0:
            garrison = min(garrison + friendlies, capacity)

        if not hostile_by_owner:
            b["garrison"][tgt] = garrison
            continue

        new_garrison, new_owner = _simultaneous_combat(
            garrison, owner_before, hostile_by_owner
        )
        b["owner"][tgt] = new_owner
        b["garrison"][tgt] = new_garrison

        if new_owner != owner_before:
            if events is not None:
                events.append({
                    "kind": "capture",
                    "tgt": int(tgt),
                    "owner_before": owner_before,
                    "owner_after": int(new_owner),
                    "garrison_after": int(new_garrison),
                })
            if new_owner == C.OWNER_P1:
                r1 += C.REWARD_CAPTURE
            elif new_owner == C.OWNER_P2:
                r2 += C.REWARD_CAPTURE
            if owner_before == C.OWNER_P1:
                r1 += C.REWARD_LOSS
            elif owner_before == C.OWNER_P2:
                r2 += C.REWARD_LOSS

    return r1, r2


def _simultaneous_combat(
    garrison: int,
    owner_before: int,
    hostile_by_owner: dict[int, int],
) -> tuple[int, int]:
    """Resolve one defender against N hostile owners landing simultaneously.

    Model (order-independent, symmetric across hostile owners):

      D  = garrison * DEF_BONUS_NUM                        (defender strength)
      Ai = count_i  * DEF_BONUS_DEN                        (attacker i strength)
      A  = sum(Ai)

      A <  D : defender holds, remaining = (D - A) / DEF_BONUS_NUM.
      A == D : mutual wipeout, defender goes neutral with 0 garrison.
      A >  D : defender dies. Each attacker loses D * Ai / A (proportional
               share of defender damage). The attacker with the largest
               surviving force takes the building; runner-up forces collide
               1:1 with the winner's survivors. Ties go neutral.

    Reduces to the pre-change single-hostile combat when len(hostile_by_owner)==1.
    """
    if not hostile_by_owner:
        return (garrison, owner_before)

    total_attack = sum(hostile_by_owner.values()) * C.DEF_BONUS_DEN
    defense = garrison * C.DEF_BONUS_NUM

    if total_attack < defense:
        remaining_scaled = defense - total_attack
        remaining = (remaining_scaled + (C.DEF_BONUS_NUM // 2)) // C.DEF_BONUS_NUM
        return (int(remaining), owner_before)
    if total_attack == defense:
        return (0, C.OWNER_NEUTRAL)

    # Attackers overwhelm the defender; split defender damage proportionally.
    survivors_scaled: dict[int, int] = {}
    for owner, count in hostile_by_owner.items():
        attack_i = count * C.DEF_BONUS_DEN
        # Integer proportional share — floor; residual rounding error is
        # bounded by len(hostile_by_owner) and is part of the fixed-point model.
        damage_share = (defense * attack_i) // total_attack
        survivors_scaled[owner] = attack_i - damage_share

    ordered = sorted(survivors_scaled.items(), key=lambda kv: -kv[1])
    winner, winner_force = ordered[0]
    runner_up_force = sum(f for _, f in ordered[1:])

    if winner_force > runner_up_force:
        remaining_scaled = winner_force - runner_up_force
        remaining = (remaining_scaled + (C.DEF_BONUS_DEN // 2)) // C.DEF_BONUS_DEN
        return (int(remaining), winner)
    # Tied top survivors → mutual kill, neutral.
    return (0, C.OWNER_NEUTRAL)


def _combat(garrison: int, attackers: int, attacker_owner: int, owner_before: int) -> tuple[int, int]:
    """Proportional combat with integer fixed-point. Returns (new_garrison, new_owner).

    Buildings defend at a constant multiplier:

      effective_defense = garrison * DEF_NUM / DEF_DEN

    Outcomes:
      attack <  effective_defense → defender holds, remaining garrison is reduced
                                  proportionally: (defense - attack) / defense_multiplier
      attack == effective_defense → both wiped, neutral
      attack >  effective_defense → attacker captures, survivors = attack - defense

    We round to the nearest internal unit so the sim can preserve chip damage
    using fixed-point integers instead of floats.
    """
    if attackers <= 0:
        return (garrison, owner_before)

    attack_scaled = attackers * C.DEF_BONUS_DEN
    defense_scaled = garrison * C.DEF_BONUS_NUM

    if attack_scaled < defense_scaled:
        remaining_scaled = defense_scaled - attack_scaled
        remaining = (remaining_scaled + (C.DEF_BONUS_NUM // 2)) // C.DEF_BONUS_NUM
        return (int(remaining), owner_before)
    if attack_scaled == defense_scaled:
        return (0, C.OWNER_NEUTRAL)

    remaining_scaled = attack_scaled - defense_scaled
    remaining = (remaining_scaled + (C.DEF_BONUS_DEN // 2)) // C.DEF_BONUS_DEN
    return (int(remaining), attacker_owner)


def _check_victory(state: State) -> tuple[float, float, bool]:
    """Apply early-win, elimination, and timeout rules.

    Returns (reward_p1, reward_p2, done). Sets state.phase on terminal.
    """
    p1_bldgs = count_owned_buildings(state, C.OWNER_P1)
    p2_bldgs = count_owned_buildings(state, C.OWNER_P2)

    # Elimination: player owns 0 buildings AND has 0 in-flight groups.
    p1_alive = p1_bldgs > 0 or has_in_flight(state, C.OWNER_P1)
    p2_alive = p2_bldgs > 0 or has_in_flight(state, C.OWNER_P2)

    # Linear speed bonus: faster wins earn more. Decays from REWARD_SPEED_BONUS
    # at tick=0 to 0 at timeout. Loser/draw rewards are unchanged.
    speed_bonus = C.REWARD_SPEED_BONUS * max(0.0, 1.0 - state.tick / C.GAME_TIMEOUT_TICKS)
    win_reward = C.REWARD_WIN + speed_bonus

    if not p1_alive and not p2_alive:
        state.phase = C.PHASE_DRAW
        return C.REWARD_DRAW, C.REWARD_DRAW, True
    if not p1_alive:
        state.phase = C.PHASE_P2_WINS
        return C.REWARD_LOSE, win_reward, True
    if not p2_alive:
        state.phase = C.PHASE_P1_WINS
        return win_reward, C.REWARD_LOSE, True

    # Timeout: tiebreak buildings → units → draw. No speed bonus on timeout.
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
