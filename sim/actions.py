"""
Action encoding + validation + masking.

Action space:
  action = type_idx * SLOTS_SQ + src * MAX_BUILDING_SLOTS + tgt   (send)
  action = NOOP_INDEX                                              (no-op)

Total size = NUM_TYPES * MAX_BUILDING_SLOTS² + 1.
v12 shape (current): 2 × 8 × 8 + 1 = 129. SEND_PERCENTAGES was cut from
4 → 2 (50/100) and MAX_BUILDING_SLOTS from 32 → 8.

One action per decision. A "decision" happens every DECISION_INTERVAL_TICKS ticks.
An action can only be issued for a player who owns the source building AND
whose source has enough garrison to send at least MIN_SEND_INTERNAL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import config as C
from sim.state import State


NUM_TYPES = len(C.SEND_PERCENTAGES)
SLOTS_SQ  = C.MAX_BUILDING_SLOTS * C.MAX_BUILDING_SLOTS
NOOP_INDEX = NUM_TYPES * SLOTS_SQ              # 4 * 1024 = 4096
ACTION_SPACE_SIZE = NOOP_INDEX + 1              # 4097


@dataclass(frozen=True)
class Action:
    """Decoded action. `kind` is "send" or "noop"."""
    kind: str
    type_idx: int = 0         # 0..NUM_TYPES-1, indexes SEND_PERCENTAGES
    src: int = 0
    tgt: int = 0

    @property
    def percentage(self) -> int:
        return C.SEND_PERCENTAGES[self.type_idx]


def encode(type_idx: int, src: int, tgt: int) -> int:
    """Pack (type, src, tgt) into a single action index."""
    if type_idx < 0 or type_idx >= NUM_TYPES:
        raise ValueError(f"type_idx out of range: {type_idx}")
    if src < 0 or src >= C.MAX_BUILDING_SLOTS:
        raise ValueError(f"src out of range: {src}")
    if tgt < 0 or tgt >= C.MAX_BUILDING_SLOTS:
        raise ValueError(f"tgt out of range: {tgt}")
    return type_idx * SLOTS_SQ + src * C.MAX_BUILDING_SLOTS + tgt


def decode(action_idx: int) -> Action:
    """Unpack action index into an Action struct."""
    if action_idx < 0 or action_idx > NOOP_INDEX:
        raise ValueError(f"action_idx out of range: {action_idx}")
    if action_idx == NOOP_INDEX:
        return Action(kind="noop")
    type_idx, rem = divmod(action_idx, SLOTS_SQ)
    src, tgt = divmod(rem, C.MAX_BUILDING_SLOTS)
    return Action(kind="send", type_idx=type_idx, src=src, tgt=tgt)


# ---------------------------------------------------------------------------
# Send amount — fixed-point, always integer real units (multiples of SCALE)
# ---------------------------------------------------------------------------

def send_amount(garrison_internal: int, percentage: int) -> int:
    """How many internal units to send for a (garrison, pct) pair.

    Always a multiple of SCALE (i.e. a whole number of real units).
    Respects MIN_GARRISON_AFTER_SEND (0 in v0.1; reserved for future rule).
    """
    max_sendable = max(0, garrison_internal - C.MIN_GARRISON_AFTER_SEND)
    real_units = (max_sendable * percentage) // (100 * C.SCALE)
    return real_units * C.SCALE


# ---------------------------------------------------------------------------
# Validation + masking
# ---------------------------------------------------------------------------

def is_valid(state: State, player: int, action: Action) -> bool:
    """Is this action legal for `player` in the current state?"""
    if action.kind == "noop":
        return True
    if action.kind != "send":
        return False
    if action.type_idx < 0 or action.type_idx >= NUM_TYPES:
        return False
    src, tgt = action.src, action.tgt
    if src == tgt:
        return False
    if src < 0 or src >= C.MAX_BUILDING_SLOTS:
        return False
    if tgt < 0 or tgt >= C.MAX_BUILDING_SLOTS:
        return False

    if not state.buildings_alive[src] or not state.buildings_alive[tgt]:
        return False
    if state.buildings_owner[src] != player:
        return False

    pct = C.SEND_PERCENTAGES[action.type_idx]
    if send_amount(int(state.buildings_garrison[src]), pct) < C.MIN_SEND_INTERNAL:
        return False

    if not np.any(state.groups_alive == 0):
        return False

    return True


def compute_mask(state: State, player: int) -> np.ndarray:
    """(ACTION_SPACE_SIZE,) bool mask — True where the action is legal.

    Used to zero out invalid-action logits before the agent samples.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[NOOP_INDEX] = True                     # noop always valid

    has_free_group = bool(np.any(state.groups_alive == 0))
    if not has_free_group:
        return mask

    alive = state.buildings_alive.astype(bool)
    owned = alive & (state.buildings_owner == player)
    valid_tgt = alive

    garrison = state.buildings_garrison
    for type_idx, pct in enumerate(C.SEND_PERCENTAGES):
        max_sendable = np.maximum(0, garrison - C.MIN_GARRISON_AFTER_SEND)
        real_units = (max_sendable.astype(np.int32) * pct) // (100 * C.SCALE)
        enough = (real_units * C.SCALE) >= C.MIN_SEND_INTERNAL
        src_ok = owned & enough

        if not np.any(src_ok):
            continue

        pair_ok = src_ok[:, None] & valid_tgt[None, :]
        np.fill_diagonal(pair_ok, False)

        base = type_idx * SLOTS_SQ
        flat = pair_ok.reshape(-1)
        mask[base:base + SLOTS_SQ] = flat

    return mask


def compute_mask_batched(
    buildings_alive:    np.ndarray,   # (N, MAX_B) int8
    buildings_owner:    np.ndarray,   # (N, MAX_B) int8
    buildings_garrison: np.ndarray,   # (N, MAX_B) int16
    groups_alive:       np.ndarray,   # (N, MAX_G) int8
    player: int,
) -> np.ndarray:
    """Vectorised `compute_mask` over N envs at once.

    Returns (N, ACTION_SPACE_SIZE) bool. Semantics byte-identical to
    calling `compute_mask(state, player)` N times.

    This exists because the per-env Python loop dominated the JaxVecAdapter
    hot path on CUDA — 64 iterations of ~150 µs of numpy + validity work
    = 10 ms CPU-bound, serialised against the GPU. Batched version stays
    in numpy-vectorised land and runs in a few hundred µs for n_envs=64.
    """
    N, MAX_B = buildings_alive.shape
    mask = np.zeros((N, ACTION_SPACE_SIZE), dtype=bool)
    mask[:, NOOP_INDEX] = True

    has_free_group = np.any(groups_alive == 0, axis=1)           # (N,)
    if not has_free_group.any():
        return mask

    alive = buildings_alive == 1                                  # (N, MAX_B)
    owned = alive & (buildings_owner == player)                   # (N, MAX_B)
    valid_tgt = alive                                             # (N, MAX_B)

    garrison = buildings_garrison.astype(np.int32)
    # Diagonal-zero template so we don't enable src==tgt actions. Shared
    # across envs; broadcasts via & in the batched fill below.
    diag_mask = ~np.eye(MAX_B, dtype=bool)                        # (MAX_B, MAX_B)

    for type_idx, pct in enumerate(C.SEND_PERCENTAGES):
        max_sendable = np.maximum(0, garrison - C.MIN_GARRISON_AFTER_SEND)
        real_units = (max_sendable * pct) // (100 * C.SCALE)
        enough = (real_units * C.SCALE) >= C.MIN_SEND_INTERNAL    # (N, MAX_B)
        src_ok = owned & enough                                    # (N, MAX_B)
        # (N, MAX_B, MAX_B)
        pair_ok = (src_ok[:, :, None] & valid_tgt[:, None, :]) & diag_mask[None, :, :]
        base = type_idx * SLOTS_SQ
        mask[:, base:base + SLOTS_SQ] = pair_ok.reshape(N, SLOTS_SQ)

    # Envs with no free group slot: only NOOP valid. Clear send-space.
    no_free = ~has_free_group                                      # (N,)
    if no_free.any():
        mask[no_free, :NOOP_INDEX] = False

    return mask
