"""
Core sim step loop. Pure numpy — the JAX backend (sim/engine_jax.py) mirrors
this exactly; changes here must keep gameplay semantics byte-identical.

Per-tick phases:
  1. apply actions       (if caller provided any)
  2. advance production  (owned buildings regenerate garrison up to capacity)
  3. advance movement    (unit groups progress; collect arrivals)
  4. resolve arrivals    (per-target simultaneous reinforce/combat)
  5. check victory

All five phases are wrapped in perf_counter timers (state.perf). The total cost
is ~1 µs/tick and lets the bench show per-phase breakdown without a separate
profiler run.

Storage is parallel ndarrays (see sim/state.py). Combat resolves via a
fixed-shape 3-slot hostile-counts array (indexed by owner 0/1/2) rather than a
Python dict — this is what the JAX port needs and stays JIT-friendly.

Event emission (spawn/arrive/capture/end) is optional and gated on
`events is not None`. The hot training path passes `events=None`; the replay
path passes a list. Emitting events is not in the JAX hot path — replay runs
on the numpy backend.
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
    """Advance the sim by exactly one tick. Returns (reward_p1, reward_p2, done).

    Actions (if provided) are applied BEFORE production/movement this tick.
    Pass None for a player if they have no action this tick.

    If `events` is a list, low-level engine events (spawn/arrive/capture/end)
    are appended to it for the replay recorder to consume. No-op when None.
    """
    t0 = time.perf_counter_ns()

    if state.phase != C.PHASE_PLAYING:
        return 0.0, 0.0, True

    r1 = r2 = 0.0

    ta = time.perf_counter_ns()
    if action_p1 is not None and action_p1.kind == "send":
        _apply_send(state, C.OWNER_P1, action_p1, events)
    if action_p2 is not None and action_p2.kind == "send":
        _apply_send(state, C.OWNER_P2, action_p2, events)
    tb = time.perf_counter_ns()
    state.perf["actions_ns"] += tb - ta

    ta = time.perf_counter_ns()
    _advance_production(state)
    tb = time.perf_counter_ns()
    state.perf["production_ns"] += tb - ta

    ta = time.perf_counter_ns()
    arrivals = _advance_movement(state, events)
    tb = time.perf_counter_ns()
    state.perf["movement_ns"] += tb - ta

    ta = time.perf_counter_ns()
    dr1, dr2 = _resolve_arrivals(state, arrivals, events)
    r1 += dr1
    r2 += dr2
    tb = time.perf_counter_ns()
    state.perf["combat_ns"] += tb - ta

    state.tick += 1

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

    src, tgt = action.src, action.tgt
    pct = C.SEND_PERCENTAGES[action.type_idx]

    amount = send_amount(int(state.buildings_garrison[src]), pct)
    if amount <= 0:
        return

    free_idx = np.where(state.groups_alive == 0)[0]
    if free_idx.size == 0:
        return
    slot = int(free_idx[0])

    travel_ticks = int(state.travel_matrix[src, tgt])

    state.buildings_garrison[src] -= amount
    state.groups_alive[slot]    = 1
    state.groups_owner[slot]    = player
    state.groups_src[slot]      = src
    state.groups_tgt[slot]      = tgt
    state.groups_count[slot]    = amount
    state.groups_progress[slot] = 0
    state.groups_travel[slot]   = travel_ticks

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

    Buildings may already sit above capacity (friendly reinforcement / large
    capture). In that case production stops, not clamps them back down.
    """
    alive = state.buildings_alive == 1
    owned = (state.buildings_owner == C.OWNER_P1) | (state.buildings_owner == C.OWNER_P2)
    below_cap = state.buildings_garrison < state.buildings_capacity
    eligible = alive & owned & below_cap

    garrison = state.buildings_garrison.astype(np.int32)
    capacity = state.buildings_capacity.astype(np.int32)
    new_garrison = np.minimum(garrison + C.PRODUCTION_PER_TICK, capacity)
    state.buildings_garrison[:] = np.where(eligible, new_garrison, garrison).astype(np.int16)


def _advance_movement(state: State, events: Optional[list] = None) -> list:
    """Advance progress on every alive unit group. Returns list of arrivals.

    An arrival is (tgt_slot, owner, count). Freed slots are cleared.
    """
    alive_idx = np.where(state.groups_alive == 1)[0]
    arrivals: list = []

    for idx in alive_idx:
        state.groups_progress[idx] += 1
        if state.groups_progress[idx] >= state.groups_travel[idx]:
            tgt = int(state.groups_tgt[idx])
            owner = int(state.groups_owner[idx])
            count = int(state.groups_count[idx])
            arrivals.append((tgt, owner, count))
            if events is not None:
                events.append({
                    "kind": "arrive",
                    "slot": int(idx),
                    "owner": owner,
                    "tgt": tgt,
                    "count": count,
                })
            state.groups_alive[idx]    = 0
            state.groups_owner[idx]    = 0
            state.groups_count[idx]    = 0
            state.groups_progress[idx] = 0
            state.groups_travel[idx]   = 0

    return arrivals


def _resolve_arrivals(
    state: State,
    arrivals: list,
    events: Optional[list] = None,
) -> tuple[float, float]:
    """Apply arrivals simultaneously per target.

    All groups landing on the same target this tick resolve as one event:
    friendly groups reinforce together (clamped at capacity); hostile groups
    attack the (possibly-reinforced) defender simultaneously. When two
    different hostile owners land the same tick on the same target, the
    resolution is symmetric — no systematic first-mover advantage.

    Targets are independent — order across different targets doesn't matter.
    Within a single target, friendlies reinforce before hostile combat.

    Emits `kind:"capture"` events whenever ownership changes.

    Returns (reward_p1, reward_p2).
    """
    if not arrivals:
        return 0.0, 0.0

    r1 = r2 = 0.0

    # Group arrivals by target. Use per-target fixed-shape (3,) int arrays
    # indexed by owner (0=neutral, 1=P1, 2=P2) so combat resolution is
    # arithmetic-only — no Python dict in the hot path.
    by_target: dict[int, np.ndarray] = {}
    for tgt, owner, count in arrivals:
        tgt = int(tgt)
        by_owner = by_target.get(tgt)
        if by_owner is None:
            by_owner = np.zeros(3, dtype=np.int64)
            by_target[tgt] = by_owner
        by_owner[int(owner)] += int(count)

    for tgt, by_owner in by_target.items():
        if not state.buildings_alive[tgt]:
            continue

        owner_before = int(state.buildings_owner[tgt])
        garrison = int(state.buildings_garrison[tgt])
        capacity = int(state.buildings_capacity[tgt])

        friendlies = int(by_owner[owner_before]) if 0 <= owner_before <= 2 else 0
        # Hostile totals — a fixed-shape (3,) array with the friendly slot zeroed.
        hostile = by_owner.copy()
        if 0 <= owner_before <= 2:
            hostile[owner_before] = 0
        hostile_total = int(hostile.sum())

        if friendlies > 0:
            garrison = min(garrison + friendlies, capacity)

        if hostile_total == 0:
            state.buildings_garrison[tgt] = garrison
            continue

        new_garrison, new_owner = _simultaneous_combat(
            garrison, owner_before, hostile
        )
        state.buildings_owner[tgt] = new_owner
        state.buildings_garrison[tgt] = new_garrison

        if new_owner != owner_before:
            if events is not None:
                events.append({
                    "kind": "capture",
                    "tgt": int(tgt),
                    "owner_before": owner_before,
                    "owner_after": int(new_owner),
                    "garrison_after": int(new_garrison),
                })
            rv = int(state.reward_version)
            r_capture = C.REWARD_CAPTURE_BY_VERSION[rv]
            r_loss    = C.REWARD_LOSS_BY_VERSION[rv]
            if new_owner == C.OWNER_P1:
                r1 += r_capture
            elif new_owner == C.OWNER_P2:
                r2 += r_capture
            if owner_before == C.OWNER_P1:
                r1 += r_loss
            elif owner_before == C.OWNER_P2:
                r2 += r_loss

    return r1, r2


def _simultaneous_combat(
    garrison: int,
    owner_before: int,
    hostile: np.ndarray,   # shape (3,) int64 — counts by owner, friendly slot = 0
) -> tuple[int, int]:
    """Resolve one defender against simultaneous hostile arrivals.

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
    """
    total_hostile = int(hostile.sum())
    if total_hostile == 0:
        return (garrison, owner_before)

    total_attack = total_hostile * C.DEF_BONUS_DEN
    defense = garrison * C.DEF_BONUS_NUM

    if total_attack < defense:
        remaining_scaled = defense - total_attack
        remaining = (remaining_scaled + (C.DEF_BONUS_NUM // 2)) // C.DEF_BONUS_NUM
        return (int(remaining), owner_before)
    if total_attack == defense:
        return (0, C.OWNER_NEUTRAL)

    # Attackers overwhelm — fixed-shape proportional damage across 3 owner slots.
    attack_i = hostile.astype(np.int64) * C.DEF_BONUS_DEN
    damage_share = (defense * attack_i) // total_attack
    survivors = attack_i - damage_share  # (3,) int64; friendly slot stays 0.

    # Pick winner = argmax of survivors; tie detection via sorted order.
    order = np.argsort(-survivors, kind="stable")
    winner = int(order[0])
    winner_force = int(survivors[winner])
    # Runner-up and below — everything else that's hostile.
    runner_up_force = int(survivors.sum() - winner_force)

    if winner_force > runner_up_force:
        remaining_scaled = winner_force - runner_up_force
        remaining = (remaining_scaled + (C.DEF_BONUS_DEN // 2)) // C.DEF_BONUS_DEN
        return (int(remaining), winner)
    return (0, C.OWNER_NEUTRAL)


def _combat(garrison: int, attackers: int, attacker_owner: int, owner_before: int) -> tuple[int, int]:
    """Single-hostile combat shim. Preserved for tests that exercise the
    proportional-damage math in isolation. Delegates to _simultaneous_combat
    with a one-owner hostile vector so the model stays single-sourced.
    """
    if attackers <= 0:
        return (garrison, owner_before)
    hostile = np.zeros(3, dtype=np.int64)
    hostile[int(attacker_owner)] = int(attackers)
    return _simultaneous_combat(garrison, owner_before, hostile)


def _check_victory(state: State) -> tuple[float, float, bool]:
    """Apply early-win, elimination, and timeout rules.

    Returns (reward_p1, reward_p2, done). Sets state.phase on terminal.
    """
    p1_bldgs = count_owned_buildings(state, C.OWNER_P1)
    p2_bldgs = count_owned_buildings(state, C.OWNER_P2)

    p1_alive = p1_bldgs > 0 or has_in_flight(state, C.OWNER_P1)
    p2_alive = p2_bldgs > 0 or has_in_flight(state, C.OWNER_P2)

    rv = int(state.reward_version)
    r_win   = C.REWARD_WIN_BY_VERSION[rv]
    r_lose  = C.REWARD_LOSE_BY_VERSION[rv]
    r_draw  = C.REWARD_DRAW_BY_VERSION[rv]
    r_speed = C.REWARD_SPEED_BONUS_BY_VERSION[rv]

    speed_bonus = r_speed * max(0.0, 1.0 - state.tick / C.GAME_TIMEOUT_TICKS)
    win_reward = r_win + speed_bonus

    if not p1_alive and not p2_alive:
        state.phase = C.PHASE_DRAW
        return r_draw, r_draw, True
    if not p1_alive:
        state.phase = C.PHASE_P2_WINS
        return r_lose, win_reward, True
    if not p2_alive:
        state.phase = C.PHASE_P1_WINS
        return win_reward, r_lose, True

    if state.tick >= C.GAME_TIMEOUT_TICKS:
        if p1_bldgs > p2_bldgs:
            state.phase = C.PHASE_P1_WINS
            return r_win, r_lose, True
        if p2_bldgs > p1_bldgs:
            state.phase = C.PHASE_P2_WINS
            return r_lose, r_win, True
        from sim.state import count_owned_units
        u1 = count_owned_units(state, C.OWNER_P1)
        u2 = count_owned_units(state, C.OWNER_P2)
        if u1 > u2:
            state.phase = C.PHASE_P1_WINS
            return r_win, r_lose, True
        if u2 > u1:
            state.phase = C.PHASE_P2_WINS
            return r_lose, r_win, True
        state.phase = C.PHASE_DRAW
        return r_draw, r_draw, True

    return 0.0, 0.0, False
