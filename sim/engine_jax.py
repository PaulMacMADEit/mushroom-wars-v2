"""
JAX reimplementation of the numpy engine.step_tick.

Semantics are byte-identical to sim/engine.py (see the parity harness in
tests/test_backend_parity.py). Differences are purely mechanical:

- Pure functions over a StateJax pytree (no in-place mutation).
- Branchless: every decision is a `jnp.where` or array-indexed update.
- Fixed-shape arrays everywhere: (MAX_BUILDING_SLOTS, …) and (MAX_UNIT_GROUP_SLOTS, …).
- Events (spawn/arrive/capture/end) are NOT emitted — the JAX hot path is
  event-free by design. Replay runs on the numpy backend.
- Actions are encoded as a (4,) int32 array [kind, type_idx, src, tgt]
  rather than the Python `Action` dataclass, so the whole tick is JIT-able.

Action encoding (shared with the JaxVecEnv caller in Phase 3):

    kind=0, ...    = noop
    kind=1, t,s,t  = send(type_idx=t, src=s, tgt=t)

Rewards returned per tick: float32 scalars (P1, P2). Matches the numpy path.
"""

from __future__ import annotations

from jax import config as _jax_config
# Combat math needs int64 for intermediates: defense * attack_i can reach ~4e10
# which overflows int32. Enable x64 BEFORE importing jax.numpy so downstream
# int64 requests stay honoured.
_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as _np

from sim import config as C
from sim.state_jax import StateJax


# Saturating bound for arrivals_* accumulators (int16 storage).
_I16_MAX_HOST = int(_np.iinfo(_np.int16).max)


# ---------------------------------------------------------------------------
# Reward lookups (version-indexed)
# ---------------------------------------------------------------------------
# Sim-v1.3 introduced per-State `reward_version`. The engine indexes into
# these constant device-resident lookup arrays via state.reward_version, so
# both v1.2 and v1.3 reward schemes can run in the same JIT'd graph without
# retracing.
_REWARD_CAPTURE_VEC     = jnp.asarray(C.REWARD_CAPTURE_BY_VERSION,     dtype=jnp.float32)
_REWARD_LOSS_VEC        = jnp.asarray(C.REWARD_LOSS_BY_VERSION,        dtype=jnp.float32)
_REWARD_WIN_VEC         = jnp.asarray(C.REWARD_WIN_BY_VERSION,         dtype=jnp.float32)
_REWARD_LOSE_VEC        = jnp.asarray(C.REWARD_LOSE_BY_VERSION,        dtype=jnp.float32)
_REWARD_DRAW_VEC        = jnp.asarray(C.REWARD_DRAW_BY_VERSION,        dtype=jnp.float32)
_REWARD_SPEED_BONUS_VEC = jnp.asarray(C.REWARD_SPEED_BONUS_BY_VERSION, dtype=jnp.float32)
_REWARD_TICK_BUILDINGS_COEF_VEC = jnp.asarray(C.REWARD_TICK_BUILDINGS_COEF_BY_VERSION, dtype=jnp.float32)
_REWARD_TICK_UNITS_COEF_VEC     = jnp.asarray(C.REWARD_TICK_UNITS_COEF_BY_VERSION,     dtype=jnp.float32)
# v1.5+ asymmetric bonuses (zero for v1.2/v1.3/v1.4).
_REWARD_ENEMY_CAPTURE_BONUS_VEC = jnp.asarray(C.REWARD_ENEMY_CAPTURE_BONUS_BY_VERSION, dtype=jnp.float32)
_REWARD_ENEMY_LOSS_PENALTY_VEC  = jnp.asarray(C.REWARD_ENEMY_LOSS_PENALTY_BY_VERSION,  dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

ACTION_KIND_NOOP = 0
ACTION_KIND_SEND = 1

# Shape (4,) int32: [kind, type_idx, src, tgt].
ACTION_DIM = 4


def encode_action(kind: int, type_idx: int = 0, src: int = 0, tgt: int = 0) -> jnp.ndarray:
    return jnp.asarray([kind, type_idx, src, tgt], dtype=jnp.int32)


def noop_action() -> jnp.ndarray:
    return encode_action(ACTION_KIND_NOOP)


# ---------------------------------------------------------------------------
# Send-amount helper (matches sim.actions.send_amount semantics)
# ---------------------------------------------------------------------------

def _send_amount(garrison_internal: jnp.ndarray, percentage: jnp.ndarray) -> jnp.ndarray:
    """Internal units to send for (garrison, pct). Always a multiple of SCALE.

    Matches `sim.actions.send_amount` exactly. Integer math, no floats.
    """
    max_sendable = jnp.maximum(0, garrison_internal - C.MIN_GARRISON_AFTER_SEND)
    real_units = (max_sendable * percentage) // (100 * C.SCALE)
    return real_units * C.SCALE


# ---------------------------------------------------------------------------
# Phase 1 — apply send (one player at a time)
# ---------------------------------------------------------------------------

def _apply_send(state: StateJax, player: int, action: jnp.ndarray) -> StateJax:
    """Branchless send. Silently no-ops on invalid actions.

    Validity mirrors sim.actions.is_valid:
      - kind == SEND
      - type_idx in [0, NUM_TYPES)
      - src != tgt, both in bounds
      - both src and tgt alive
      - state.buildings_owner[src] == player
      - send_amount >= MIN_SEND_INTERNAL
      - at least one free group slot exists
    """
    NUM_TYPES = len(C.SEND_PERCENTAGES)
    N = C.MAX_BUILDING_SLOTS

    kind     = action[0]
    type_idx = action[1]
    src      = action[2]
    tgt      = action[3]

    # Clamp src/tgt into valid index range before gathers (avoids out-of-bounds
    # even in the invalid path — the validity mask zeros out effects regardless).
    src_c = jnp.clip(src, 0, N - 1)
    tgt_c = jnp.clip(tgt, 0, N - 1)
    type_c = jnp.clip(type_idx, 0, NUM_TYPES - 1)

    percentages = jnp.asarray(C.SEND_PERCENTAGES, dtype=jnp.int32)
    pct = percentages[type_c]

    src_alive = state.buildings_alive[src_c] == 1
    tgt_alive = state.buildings_alive[tgt_c] == 1
    src_owned = state.buildings_owner[src_c] == player
    type_ok   = (type_idx >= 0) & (type_idx < NUM_TYPES)
    idx_ok    = (src == src_c) & (tgt == tgt_c) & (src != tgt)
    is_send   = kind == ACTION_KIND_SEND

    garrison_src = state.buildings_garrison[src_c].astype(jnp.int32)
    amount = _send_amount(garrison_src, pct).astype(jnp.int32)
    amount_ok = amount >= C.MIN_SEND_INTERNAL

    # First free group slot, or 0 if none. `any_free` tracks whether the 0 is real.
    free_mask = state.groups_alive == 0
    any_free = jnp.any(free_mask)
    # argmax on a bool returns the index of the first True (or 0 if all False).
    slot = jnp.argmax(free_mask).astype(jnp.int32)

    valid = is_send & type_ok & idx_ok & src_alive & tgt_alive & src_owned & amount_ok & any_free

    # Apply the effects *conditioned on valid* via jnp.where.
    new_garrison_src = jnp.where(
        valid,
        (garrison_src - amount).astype(jnp.int16),
        state.buildings_garrison[src_c],
    )
    buildings_garrison = state.buildings_garrison.at[src_c].set(new_garrison_src)

    travel_ticks = state.travel_matrix[src_c, tgt_c].astype(jnp.int16)

    groups_alive    = state.groups_alive.at[slot].set(
        jnp.where(valid, jnp.int8(1), state.groups_alive[slot])
    )
    groups_owner    = state.groups_owner.at[slot].set(
        jnp.where(valid, jnp.int8(player), state.groups_owner[slot])
    )
    groups_src      = state.groups_src.at[slot].set(
        jnp.where(valid, src_c.astype(jnp.int8), state.groups_src[slot])
    )
    groups_tgt      = state.groups_tgt.at[slot].set(
        jnp.where(valid, tgt_c.astype(jnp.int8), state.groups_tgt[slot])
    )
    groups_count    = state.groups_count.at[slot].set(
        jnp.where(valid, amount.astype(jnp.int16), state.groups_count[slot])
    )
    groups_progress = state.groups_progress.at[slot].set(
        jnp.where(valid, jnp.int16(0), state.groups_progress[slot])
    )
    groups_travel   = state.groups_travel.at[slot].set(
        jnp.where(valid, travel_ticks, state.groups_travel[slot])
    )

    return state.replace(
        buildings_garrison=buildings_garrison,
        groups_alive=groups_alive,
        groups_owner=groups_owner,
        groups_src=groups_src,
        groups_tgt=groups_tgt,
        groups_count=groups_count,
        groups_progress=groups_progress,
        groups_travel=groups_travel,
    )


# ---------------------------------------------------------------------------
# Phase 2 — advance production
# ---------------------------------------------------------------------------

def _advance_production(state: StateJax) -> StateJax:
    """Vectorised production: each owned alive below-capacity building +1/tick."""
    alive     = state.buildings_alive == 1
    owned     = (state.buildings_owner == C.OWNER_P1) | (state.buildings_owner == C.OWNER_P2)
    below_cap = state.buildings_garrison < state.buildings_capacity
    eligible  = alive & owned & below_cap

    garrison = state.buildings_garrison.astype(jnp.int32)
    capacity = state.buildings_capacity.astype(jnp.int32)
    new_garrison = jnp.minimum(garrison + C.PRODUCTION_PER_TICK, capacity)
    new_garrison = jnp.where(eligible, new_garrison, garrison).astype(jnp.int16)
    return state.replace(buildings_garrison=new_garrison)


# ---------------------------------------------------------------------------
# Phase 3 — advance movement, produce (target, owner, count, valid) per slot
# ---------------------------------------------------------------------------

def _advance_movement(state: StateJax) -> tuple[StateJax, jnp.ndarray]:
    """Advance progress; return (new_state, incoming) where

        incoming[tgt, owner]  = total units arriving at `tgt` this tick from `owner`

    shape (MAX_BUILDING_SLOTS, 3) int64. Cleared group slots reset to zero.
    """
    N = C.MAX_BUILDING_SLOTS

    # Advance progress on alive groups (progress += 1).
    alive_mask = state.groups_alive == 1
    progress = jnp.where(
        alive_mask,
        state.groups_progress + jnp.int16(1),
        state.groups_progress,
    )

    # An arrival this tick: alive AND progress >= travel_ticks.
    arrived = alive_mask & (progress >= state.groups_travel)

    # Build (MAX_BUILDING_SLOTS, 3) aggregation of arriving counts per (tgt, owner).
    # Use jnp.zeros + scatter-add over arrived groups. One-hot by tgt AND owner.
    group_tgt   = state.groups_tgt.astype(jnp.int32)        # (M,)
    group_owner = state.groups_owner.astype(jnp.int32)      # (M,)
    group_count = state.groups_count.astype(jnp.int64)      # (M,)

    contrib = jnp.where(arrived, group_count, jnp.int64(0)) # (M,)

    # Scatter-add into (N, 3). `segment_sum` is the vmap-friendly primitive; we
    # combine tgt+owner into a single linear index then unflatten.
    linear_idx = group_tgt * 3 + group_owner                 # (M,)
    flat_incoming = jax.ops.segment_sum(
        contrib, linear_idx, num_segments=N * 3,
    )                                                        # (N*3,) int64
    incoming = flat_incoming.reshape(N, 3)                   # (N, 3) int64

    # Clear arrived groups. The numpy engine clears alive/owner/count/progress/
    # travel but leaves src/tgt with their old values (dead data since alive=0).
    # Match that exactly so parity holds.
    keep = ~arrived
    groups_alive    = jnp.where(keep, state.groups_alive,    jnp.int8(0))
    groups_owner    = jnp.where(keep, state.groups_owner,    jnp.int8(0))
    groups_count    = jnp.where(keep, state.groups_count,    jnp.int16(0))
    groups_progress = jnp.where(keep, progress,              jnp.int16(0))
    groups_travel   = jnp.where(keep, state.groups_travel,   jnp.int16(0))
    # groups_src / groups_tgt: left untouched on arrival (matches numpy).

    new_state = state.replace(
        groups_alive=groups_alive,
        groups_owner=groups_owner,
        groups_count=groups_count,
        groups_progress=groups_progress,
        groups_travel=groups_travel,
    )
    return new_state, incoming


# ---------------------------------------------------------------------------
# Phase 4 — resolve arrivals per target (branchless, vmapped internally)
# ---------------------------------------------------------------------------

def _resolve_one_target(
    alive: jnp.ndarray,        # int8 scalar
    owner_before: jnp.ndarray, # int8 scalar
    garrison: jnp.ndarray,     # int16 scalar
    capacity: jnp.ndarray,     # int16 scalar
    incoming: jnp.ndarray,     # (3,) int64 — counts by owner
    reward_version: jnp.ndarray,  # int8 scalar (0=v1.2, 1=v1.3)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve one target slot. Returns (new_garrison, new_owner, r1_delta, r2_delta).

    Branchless implementation of sim.engine._resolve_arrivals +
    _simultaneous_combat. Semantics must match byte-for-byte.
    """
    owner_i32 = owner_before.astype(jnp.int32)  # 0/1/2
    garrison_i32 = garrison.astype(jnp.int32)
    capacity_i32 = capacity.astype(jnp.int32)

    # Friendlies: incoming at owner_before's slot. Groups are never neutral-owned,
    # so incoming[0] is always 0; when owner_before==0 (neutral), friendlies=0
    # correctly.
    friendlies = incoming[owner_i32].astype(jnp.int32)
    hostile = incoming.at[owner_i32].set(jnp.int64(0))  # zero-out friendly slot
    hostile_total = hostile.sum().astype(jnp.int64)

    # Friendly reinforcement: only clamp when friendlies > 0. Numpy leaves an
    # over-capacity garrison alone when no friendlies arrive — a building that
    # survived a capture with garrison > capacity stays that way until either
    # friendly reinforcement lands (then it's clamped at capacity) or hostiles
    # knock it down.
    reinforced = jnp.minimum(garrison_i32 + friendlies, capacity_i32)
    g_after_reinforce = jnp.where(friendlies > 0, reinforced, garrison_i32)

    # --- Simultaneous combat ---
    total_attack = hostile_total.astype(jnp.int64) * C.DEF_BONUS_DEN
    defense      = g_after_reinforce.astype(jnp.int64) * C.DEF_BONUS_NUM

    # Case A: defender holds (total_attack < defense).
    remaining_scaled_A = defense - total_attack
    remaining_A = (remaining_scaled_A + (C.DEF_BONUS_NUM // 2)) // C.DEF_BONUS_NUM
    # new_owner_A = owner_before (unchanged).

    # Case B: mutual wipe (total_attack == defense).
    # new_garrison = 0; new_owner = NEUTRAL.

    # Case C: attackers overwhelm (total_attack > defense).
    # Avoid divide-by-zero when hostile_total == 0 — in that path we don't go
    # into combat (fall-through below), but the intermediate math is still
    # evaluated. Use jnp.maximum(total_attack, 1) in the denominator.
    denom_safe = jnp.maximum(total_attack, jnp.int64(1))
    attack_i = hostile.astype(jnp.int64) * C.DEF_BONUS_DEN            # (3,)
    damage_share = (defense * attack_i) // denom_safe                  # (3,)
    survivors = attack_i - damage_share                                # (3,)

    # Winner = argmax survivors. In sim/engine.py the numpy path sorts stable by
    # -survivors (tie-break by owner index ascending) and picks the first.
    # jnp.argmax on equal values returns the first index → same result.
    winner = jnp.argmax(survivors).astype(jnp.int32)
    winner_force = survivors[winner]
    runner_up_force = survivors.sum() - winner_force

    # Sub-case C1: winner_force > runner_up_force → winner captures.
    remaining_scaled_C = winner_force - runner_up_force
    remaining_C = (remaining_scaled_C + (C.DEF_BONUS_DEN // 2)) // C.DEF_BONUS_DEN

    # Pick the right case by masks.
    no_hostiles = hostile_total == 0
    holds       = (~no_hostiles) & (total_attack <  defense)
    mutual      = (~no_hostiles) & (total_attack == defense)
    overwhelm   = (~no_hostiles) & (total_attack >  defense)
    overwhelm_capture = overwhelm & (winner_force > runner_up_force)
    overwhelm_neutral = overwhelm & (winner_force <= runner_up_force)

    new_garrison_i32 = jnp.where(
        no_hostiles,       g_after_reinforce,
        jnp.where(holds,             remaining_A.astype(jnp.int32),
        jnp.where(mutual,            jnp.int32(0),
        jnp.where(overwhelm_capture, remaining_C.astype(jnp.int32),
        jnp.where(overwhelm_neutral, jnp.int32(0),
                                     g_after_reinforce)))))
    new_owner_i32 = jnp.where(
        no_hostiles,       owner_i32,
        jnp.where(holds,             owner_i32,
        jnp.where(mutual,            jnp.int32(C.OWNER_NEUTRAL),
        jnp.where(overwhelm_capture, winner,
        jnp.where(overwhelm_neutral, jnp.int32(C.OWNER_NEUTRAL),
                                     owner_i32)))))

    # Dead / empty target — no effect at all.
    alive_mask = alive == 1
    new_garrison_i32 = jnp.where(alive_mask, new_garrison_i32, garrison_i32)
    new_owner_i32    = jnp.where(alive_mask, new_owner_i32,    owner_i32)

    # Rewards: emitted only if ownership changed AND the target was alive.
    changed = alive_mask & (new_owner_i32 != owner_i32)
    # Version-indexed reward lookups (per-state reward_version).
    rv = reward_version.astype(jnp.int32)
    r_capture_v       = _REWARD_CAPTURE_VEC[rv]
    r_loss_v          = _REWARD_LOSS_VEC[rv]
    r_enemy_cap_v     = _REWARD_ENEMY_CAPTURE_BONUS_VEC[rv]
    r_enemy_loss_v    = _REWARD_ENEMY_LOSS_PENALTY_VEC[rv]
    # Base capture/loss (any ownership change of an alive slot).
    r_capture_p1 = jnp.where(changed & (new_owner_i32 == C.OWNER_P1), r_capture_v, jnp.float32(0.0))
    r_capture_p2 = jnp.where(changed & (new_owner_i32 == C.OWNER_P2), r_capture_v, jnp.float32(0.0))
    r_loss_p1    = jnp.where(changed & (owner_i32     == C.OWNER_P1), r_loss_v,    jnp.float32(0.0))
    r_loss_p2    = jnp.where(changed & (owner_i32     == C.OWNER_P2), r_loss_v,    jnp.float32(0.0))
    # v1.5 asymmetric bonus: only when ownership transitions DIRECTLY between
    # the two players (excludes mutual wipeout to neutral).
    p1_took_from_p2 = changed & (new_owner_i32 == C.OWNER_P1) & (owner_i32 == C.OWNER_P2)
    p2_took_from_p1 = changed & (new_owner_i32 == C.OWNER_P2) & (owner_i32 == C.OWNER_P1)
    r_enemy_cap_p1  = jnp.where(p1_took_from_p2, r_enemy_cap_v,  jnp.float32(0.0))
    r_enemy_cap_p2  = jnp.where(p2_took_from_p1, r_enemy_cap_v,  jnp.float32(0.0))
    r_enemy_loss_p1 = jnp.where(p2_took_from_p1, r_enemy_loss_v, jnp.float32(0.0))   # P1 lost to P2
    r_enemy_loss_p2 = jnp.where(p1_took_from_p2, r_enemy_loss_v, jnp.float32(0.0))   # P2 lost to P1
    r1_delta = r_capture_p1 + r_loss_p1 + r_enemy_cap_p1 + r_enemy_loss_p1
    r2_delta = r_capture_p2 + r_loss_p2 + r_enemy_cap_p2 + r_enemy_loss_p2

    return new_garrison_i32.astype(jnp.int16), new_owner_i32.astype(jnp.int8), r1_delta, r2_delta


def _resolve_arrivals(state: StateJax, incoming: jnp.ndarray) -> tuple[StateJax, jnp.ndarray, jnp.ndarray]:
    """Vectorise _resolve_one_target across all target slots. Returns
    (new_state, reward_p1, reward_p2).
    """
    new_garrison, new_owner, r1_per_tgt, r2_per_tgt = jax.vmap(
        _resolve_one_target, in_axes=(0, 0, 0, 0, 0, None)
    )(
        state.buildings_alive,
        state.buildings_owner,
        state.buildings_garrison,
        state.buildings_capacity,
        incoming,
        state.reward_version,
    )
    new_state = state.replace(
        buildings_garrison=new_garrison,
        buildings_owner=new_owner,
    )
    return new_state, r1_per_tgt.sum(), r2_per_tgt.sum()


# ---------------------------------------------------------------------------
# Phase 5 — check victory
# ---------------------------------------------------------------------------

def _check_victory(state: StateJax) -> tuple[StateJax, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Branchless victory check. Returns (new_state, r1_delta, r2_delta, done).

    Semantics match sim.engine._check_victory byte-for-byte.
    """
    alive = state.buildings_alive == 1
    owned_p1 = alive & (state.buildings_owner == C.OWNER_P1)
    owned_p2 = alive & (state.buildings_owner == C.OWNER_P2)
    p1_bldgs = owned_p1.sum()
    p2_bldgs = owned_p2.sum()

    g_alive = state.groups_alive == 1
    p1_inflight = (g_alive & (state.groups_owner == C.OWNER_P1)).any()
    p2_inflight = (g_alive & (state.groups_owner == C.OWNER_P2)).any()

    p1_alive_player = (p1_bldgs > 0) | p1_inflight
    p2_alive_player = (p2_bldgs > 0) | p2_inflight

    # Version-indexed reward lookups.
    rv = state.reward_version.astype(jnp.int32)
    r_win   = _REWARD_WIN_VEC[rv]
    r_lose  = _REWARD_LOSE_VEC[rv]
    r_draw  = _REWARD_DRAW_VEC[rv]
    r_speed = _REWARD_SPEED_BONUS_VEC[rv]

    # Speed bonus (matches numpy).
    speed_bonus = r_speed * jnp.maximum(
        jnp.float32(0.0),
        jnp.float32(1.0) - state.tick.astype(jnp.float32) / jnp.float32(C.GAME_TIMEOUT_TICKS),
    )
    win_reward = r_win + speed_bonus

    both_dead = (~p1_alive_player) & (~p2_alive_player)
    p1_dead   = (~p1_alive_player) & p2_alive_player
    p2_dead   = p1_alive_player & (~p2_alive_player)

    # Unit counts for timeout tiebreak.
    p1_garrison = jnp.where(owned_p1, state.buildings_garrison, jnp.int16(0)).sum()
    p2_garrison = jnp.where(owned_p2, state.buildings_garrison, jnp.int16(0)).sum()
    p1_flight_count = jnp.where(
        g_alive & (state.groups_owner == C.OWNER_P1), state.groups_count, jnp.int16(0)
    ).sum()
    p2_flight_count = jnp.where(
        g_alive & (state.groups_owner == C.OWNER_P2), state.groups_count, jnp.int16(0)
    ).sum()
    u1 = p1_garrison.astype(jnp.int32) + p1_flight_count.astype(jnp.int32)
    u2 = p2_garrison.astype(jnp.int32) + p2_flight_count.astype(jnp.int32)

    timed_out = state.tick >= C.GAME_TIMEOUT_TICKS
    timeout_p1_wins = timed_out & ((p1_bldgs > p2_bldgs) | ((p1_bldgs == p2_bldgs) & (u1 > u2)))
    timeout_p2_wins = timed_out & ((p2_bldgs > p1_bldgs) | ((p1_bldgs == p2_bldgs) & (u2 > u1)))
    timeout_draw    = timed_out & (~timeout_p1_wins) & (~timeout_p2_wins)

    terminal_p1_wins = p2_dead   | timeout_p1_wins
    terminal_p2_wins = p1_dead   | timeout_p2_wins
    terminal_draw    = both_dead | timeout_draw

    done = terminal_p1_wins | terminal_p2_wins | terminal_draw

    new_phase = (
        jnp.where(terminal_p1_wins, jnp.int8(C.PHASE_P1_WINS),
        jnp.where(terminal_p2_wins, jnp.int8(C.PHASE_P2_WINS),
        jnp.where(terminal_draw,    jnp.int8(C.PHASE_DRAW),
                                    state.phase)))
    )

    # Reward rules (by case):
    #   both_dead (elimination draw)           : REWARD_DRAW both
    #   p1_dead (P2 eliminates P1)             : LOSE, WIN+speed_bonus
    #   p2_dead (P1 eliminates P2)             : WIN+speed_bonus, LOSE
    #   timeout_p1_wins                        : WIN, LOSE   (no speed bonus per numpy)
    #   timeout_p2_wins                        : LOSE, WIN
    #   timeout_draw                           : DRAW both
    #   not done                               : 0, 0
    r1_delta = jnp.where(
        both_dead,        r_draw,
        jnp.where(p1_dead, r_lose,
        jnp.where(p2_dead, win_reward,
        jnp.where(timeout_p1_wins, r_win,
        jnp.where(timeout_p2_wins, r_lose,
        jnp.where(timeout_draw,    r_draw,
                                   jnp.float32(0.0)))))),
    )
    r2_delta = jnp.where(
        both_dead,        r_draw,
        jnp.where(p1_dead, win_reward,
        jnp.where(p2_dead, r_lose,
        jnp.where(timeout_p1_wins, r_lose,
        jnp.where(timeout_p2_wins, r_win,
        jnp.where(timeout_draw,    r_draw,
                                   jnp.float32(0.0)))))),
    )

    new_state = state.replace(phase=new_phase)
    return new_state, r1_delta, r2_delta, done


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _step_tick_impl(
    state: StateJax,
    action_p1: jnp.ndarray,
    action_p2: jnp.ndarray,
) -> tuple[StateJax, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One tick, pure function. Semantics match sim.engine.step_tick.

    Returns (new_state, r1, r2, done). All outputs are jnp arrays.
    If the state is already terminal, returns (state_unchanged, 0.0, 0.0, True).
    """
    was_terminal = state.phase != C.PHASE_PLAYING

    # Phase 1: actions.
    s = _apply_send(state, C.OWNER_P1, action_p1)
    s = _apply_send(s,     C.OWNER_P2, action_p2)

    # Phase 2: production.
    s = _advance_production(s)

    # Phase 3: movement → incoming table.
    s, incoming = _advance_movement(s)

    # v10: accumulate per-bldg landing counts into the state's arrival
    # counters (reset by the env each decision interval). Saturate at int16
    # max defensively — a single decision interval can't realistically
    # produce >32k landings on one building.
    incoming_p1 = jnp.minimum(
        s.arrivals_p1.astype(jnp.int32) + incoming[:, C.OWNER_P1].astype(jnp.int32),
        jnp.int32(_I16_MAX_HOST),
    ).astype(jnp.int16)
    incoming_p2 = jnp.minimum(
        s.arrivals_p2.astype(jnp.int32) + incoming[:, C.OWNER_P2].astype(jnp.int32),
        jnp.int32(_I16_MAX_HOST),
    ).astype(jnp.int16)
    s = s.replace(arrivals_p1=incoming_p1, arrivals_p2=incoming_p2)

    # Phase 4: resolve arrivals → combat rewards.
    s, r1_combat, r2_combat = _resolve_arrivals(s, incoming)

    # Tick advances between combat and victory check (matches numpy).
    s = s.replace(tick=s.tick + jnp.int32(1))

    # Phase 5: victory check → terminal rewards + new phase.
    s, r1_v, r2_v, done = _check_victory(s)

    # v1.4 per-tick shaping (zero-coefficient on v1.2/v1.3): rewards holding
    # more buildings + units than the opponent. Skipped on terminal tick to
    # keep the terminal signal clean.
    rv = s.reward_version.astype(jnp.int32)
    coef_b = _REWARD_TICK_BUILDINGS_COEF_VEC[rv]
    coef_u = _REWARD_TICK_UNITS_COEF_VEC[rv]
    alive = s.buildings_alive == 1
    owned_p1 = alive & (s.buildings_owner == C.OWNER_P1)
    owned_p2 = alive & (s.buildings_owner == C.OWNER_P2)
    b1 = owned_p1.sum().astype(jnp.float32)
    b2 = owned_p2.sum().astype(jnp.float32)
    g_alive = s.groups_alive == 1
    p1_garrison = jnp.where(owned_p1, s.buildings_garrison, jnp.int16(0)).sum()
    p2_garrison = jnp.where(owned_p2, s.buildings_garrison, jnp.int16(0)).sum()
    p1_flight = jnp.where(g_alive & (s.groups_owner == C.OWNER_P1), s.groups_count, jnp.int16(0)).sum()
    p2_flight = jnp.where(g_alive & (s.groups_owner == C.OWNER_P2), s.groups_count, jnp.int16(0)).sum()
    u1_real = (p1_garrison.astype(jnp.float32) + p1_flight.astype(jnp.float32)) / jnp.float32(C.SCALE)
    u2_real = (p2_garrison.astype(jnp.float32) + p2_flight.astype(jnp.float32)) / jnp.float32(C.SCALE)
    shaping_delta = coef_b * (b1 - b2) + coef_u * (u1_real - u2_real)
    shaping_delta = jnp.where(done, jnp.float32(0.0), shaping_delta)
    r1_shape = shaping_delta
    r2_shape = -shaping_delta

    r1 = r1_combat + r1_v + r1_shape
    r2 = r2_combat + r2_v + r2_shape

    # If we entered step_tick already terminal, the numpy path returns
    # (0.0, 0.0, True) without advancing. Mirror that via select_pytree.
    r1_out   = jnp.where(was_terminal, jnp.float32(0.0), r1)
    r2_out   = jnp.where(was_terminal, jnp.float32(0.0), r2)
    done_out = jnp.where(was_terminal, jnp.bool_(True),  done)
    state_out = jax.tree_util.tree_map(
        lambda new, old: jnp.where(was_terminal, old, new),
        s, state,
    )
    return state_out, r1_out, r2_out, done_out


step_tick_single = jax.jit(_step_tick_impl)


# ---------------------------------------------------------------------------
# Multi-tick fused step — pack T ticks into one XLA dispatch
# ---------------------------------------------------------------------------
#
# `step_tick_single` is small. On CUDA the ~microsecond kernel plus its
# per-launch overhead caps throughput at a few thousand ticks/sec
# irrespective of batch size (GPU sits at ~5% util). Fusing T ticks inside
# a jit'd `jax.lax.scan` collapses T launches into one, which is what we
# need for the ≥10× / ≥40% SM gate.
#
# Caller shape: actions_p1, actions_p2 both (T, 4) int32; or batched
# (T, n_envs, 4) when vmapped per-env. Returns (final_state, rewards_p1,
# rewards_p2, dones) of shape (T,).

def _scan_body(state, actions):
    a1, a2 = actions
    state, r1, r2, done = _step_tick_impl(state, a1, a2)
    return state, (r1, r2, done)


def _step_many_impl(state, actions_p1, actions_p2):
    """Run T ticks over a single game. `actions_p1`/`p2`: (T, 4) int32."""
    final, (r1s, r2s, dones) = jax.lax.scan(
        _scan_body, state, (actions_p1, actions_p2),
    )
    return final, r1s, r2s, dones


step_many_single = jax.jit(_step_many_impl)
