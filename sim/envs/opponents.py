"""Opponent policies used inside MushroomEnv.

Three families:
  - noop_opponent          — always idle; for debugging.
  - random_legal_opponent  — uniform over the mask; weakest real opponent.
  - make_neural_opponent   — loads a frozen ActorCritic snapshot from disk
                             and uses it to pick P2's action. Self-play.

The neural opponent MIRRORS ownership before encoding so a net trained as P1
can act as P2 (is_p1 ↔ is_p2 swap). Slot *positions* aren't mirrored — on
180°-symmetric levels like crossroads_6 that's close enough; on asymmetric
levels the opponent has a small handicap that's worth fixing later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from sim import config as C
from sim.actions import NOOP_INDEX, compute_mask
from sim.state import State


Opponent = Callable[[State, np.random.Generator], int]


# ---------------------------------------------------------------------------
# Simple opponents
# ---------------------------------------------------------------------------

def noop_opponent(state: State, rng: np.random.Generator) -> int:
    del state, rng
    return NOOP_INDEX


def random_legal_opponent(state: State, rng: np.random.Generator) -> int:
    mask = compute_mask(state, C.OWNER_P2)
    legal = np.where(mask)[0]
    if legal.size == 0:
        return NOOP_INDEX
    return int(rng.choice(legal))


# ---------------------------------------------------------------------------
# Neural opponent (self-play)
# ---------------------------------------------------------------------------

def _state_to_obs(state: State, mask_player: int) -> dict:
    """Build the gym-style obs dict from raw State. Mirrors MushroomEnv._make_obs
    but callable without an env instance (opponents run inside env subprocs)."""
    b = state.buildings
    g = state.unit_groups
    return {
        "buildings_alive":    b["alive"].copy(),
        "buildings_owner":    b["owner"].copy(),
        "buildings_type":     b["type_id"].copy(),
        "buildings_garrison": b["garrison"].copy(),
        "buildings_capacity": b["capacity"].copy(),
        "buildings_x":        b["x"].copy(),
        "buildings_y":        b["y"].copy(),
        "groups_alive":       g["alive"].copy(),
        "groups_owner":       g["owner"].copy(),
        "groups_src":         g["src_slot"].copy(),
        "groups_tgt":         g["tgt_slot"].copy(),
        "groups_count":       g["count"].copy(),
        "groups_progress":    g["progress"].copy(),
        "groups_travel":      g["travel_ticks"].copy(),
        "travel_matrix":      state.travel_matrix.copy(),
        "tick":               np.int32(state.tick),
        "action_mask":        compute_mask(state, mask_player),
    }


def _mirror_ownership(obs: dict) -> dict:
    """Swap P1↔P2 in the owner fields. Returns a shallow-copied dict with
    fresh owner arrays; all other fields reference the originals unchanged."""
    mirrored = dict(obs)
    for key in ("buildings_owner", "groups_owner"):
        orig = obs[key]
        swapped = orig.copy()
        swapped = np.where(orig == C.OWNER_P1, C.OWNER_P2, swapped)
        swapped = np.where(orig == C.OWNER_P2, C.OWNER_P1, swapped)
        mirrored[key] = swapped.astype(orig.dtype)
    return mirrored


def make_neural_opponent(
    weights_path: str,
    obs_norm_path: Optional[str] = None,
    device: str = "cpu",
    recorder=None,
) -> Opponent:
    """Load a frozen ActorCritic snapshot and return an Opponent callable.

    Each vec-env subprocess constructs its own via make_env's factory; the
    `weights_path` is read once at construction (not on every step).

    When `recorder` is passed (replay capture path), the opponent uses
    `act_one_with_diag` and records P2's decision with the same schema as P1.

    IMPORTANT: when the parent process has CUDA initialized and spawns
    AsyncVectorEnv subprocs, each subproc re-imports torch and would also
    try to init CUDA. With 64 subprocs doing that at once, torch's CUDA
    init deadlocks (observed: main proc stuck on `unix_stream_read_generic`
    waiting for subprocs stuck on `futex_do_wait`). We hide the GPU from
    the subproc's torch *before* importing it so opponents stay CPU-only.
    """
    import os

    if device == "cpu":
        # Unconditional override — setdefault leaves parent's value in place.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Lazy imports: we don't want random-legal / noop paths to pull torch or
    # training code into every subprocess that doesn't need it.
    import torch

    from training.agent import PPOAgent
    from training.encoder import OBS_DIM, encode_obs
    from training.net import ActorCritic
    from training.obs_norm import RunningNorm

    net = ActorCritic()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    agent = PPOAgent(net, device=device)

    obs_norm: Optional[RunningNorm] = None
    if obs_norm_path and Path(str(obs_norm_path)).exists():
        obs_norm = RunningNorm(OBS_DIM)
        obs_norm.load(str(obs_norm_path))

    def opponent(state: State, rng: np.random.Generator) -> int:
        obs_dict = _state_to_obs(state, mask_player=C.OWNER_P2)
        mirrored = _mirror_ownership(obs_dict)
        # Mask is P2's (computed against real state, not the mirrored dict).
        mirrored["action_mask"] = obs_dict["action_mask"]

        x = encode_obs(mirrored)
        if obs_norm is not None:
            x = obs_norm.normalize(x)

        # Recording path: capture diag alongside the action.
        if recorder is not None:
            action, diag = agent.act_one_with_diag(x, mirrored["action_mask"])
            # Post-tick time, matching engine.py's event stamps.
            recorder.record_decision(
                tick=int(state.tick + 1),
                player=C.OWNER_P2,
                diag=diag,
            )
            return int(action)

        # Hot-path (training / eval without recording): stay on act_batch.
        action, *_ = agent.act_batch(x[None, :], mirrored["action_mask"][None, :])
        return int(action[0])

    return opponent
