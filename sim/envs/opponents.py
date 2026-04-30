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


def random_legal_opponent_batched(
    p2_mask: np.ndarray,               # (N, ACTION_SPACE_SIZE) bool
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick a uniform random legal action per env — all envs in one shot.

    Trick: `argmax(uniform_noise * mask)` picks a random index among the
    True entries in each row. Falls back to NOOP if a row has no legal
    action (shouldn't happen since NOOP is always legal, but defensive).
    """
    N, A = p2_mask.shape
    # Uniform noise; zero out illegal entries so argmax picks only legal ones.
    noise = rng.random((N, A), dtype=np.float32)
    noise = np.where(p2_mask, noise, -1.0)
    return noise.argmax(axis=1).astype(np.int64)


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
        # v10 decision-interval features.
        "arrivals_p1":          state.arrivals_p1.copy(),
        "arrivals_p2":          state.arrivals_p2.copy(),
        "prev_buildings_owner": state.prev_buildings_owner.copy(),
        "prev_p1_units_total":  np.int32(state.prev_p1_units_total),
        "prev_p2_units_total":  np.int32(state.prev_p2_units_total),
        "last_actions_p1":      state.last_actions_p1.copy(),
        "last_actions_p2":      state.last_actions_p2.copy(),
    }


def _mirror_ownership(obs: dict) -> dict:
    """Swap P1↔P2 in every owner-keyed field. Returns a shallow-copied dict
    with fresh owner arrays; all other fields reference the originals.

    v10: also swaps `arrivals_p1` ↔ `arrivals_p2`, prev_p*_units_total,
    last_actions_p1 ↔ p2, and re-codes prev_buildings_owner P1↔P2. So the
    encoder always sees the active player as "P1" regardless of true side.
    """
    mirrored = dict(obs)
    for key in ("buildings_owner", "groups_owner", "prev_buildings_owner"):
        if key not in obs:
            continue
        orig = obs[key]
        swapped = orig.copy()
        swapped = np.where(orig == C.OWNER_P1, C.OWNER_P2, swapped)
        swapped = np.where(orig == C.OWNER_P2, C.OWNER_P1, swapped)
        mirrored[key] = swapped.astype(orig.dtype)
    # Per-player arrays just swap labels.
    for key_p1, key_p2 in (
        ("arrivals_p1",         "arrivals_p2"),
        ("last_actions_p1",     "last_actions_p2"),
        ("prev_p1_units_total", "prev_p2_units_total"),
    ):
        if key_p1 in obs and key_p2 in obs:
            mirrored[key_p1] = obs[key_p2]
            mirrored[key_p2] = obs[key_p1]
    return mirrored


def preload_state_dict(
    weights_path: str,
    device: str = "cpu",
) -> tuple[dict, str]:
    """Read a weights .pt file once into RAM.

    Returns `(state_dict, encoder_version)`. Legacy raw saves (no wrapper)
    resolve to `DEFAULT_ENCODER_VERSION` ("v9.0") via `checkpoint`.

    Pair with `make_neural_opponent_cached` for fast per-update swaps
    under opponent_pool_mode=rotate_per_update.

    2026-04-29 fire 65: avoids re-reading the file on every swap; the
    trainer pre-loads every archive member at init, then cycles in-memory.
    """
    import os
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from training.checkpoint import load_state_dict_with_version
    # weights_only=False because new wrapped saves include the
    # `encoder_version` string field which torch's `weights_only` mode
    # rejects (it whitelists tensors only). The wrapper is project-internal
    # so this is safe.
    return load_state_dict_with_version(
        weights_path, map_location=device, weights_only=False,
    )


def preload_obs_norm(obs_norm_path: Optional[str], obs_dim: Optional[int] = None):
    """Read an obs_norm .pt file once into a RunningNorm instance, or None
    if path is missing/null.

    `obs_dim` is the expected dimension. RunningNorm.load_state_dict
    overwrites the shape from the file regardless, so passing the wrong
    `obs_dim` here is harmless — the file wins. Default uses the current
    encoder's OBS_DIM, fine for fresh trainer init.
    """
    if not obs_norm_path:
        return None
    if not Path(str(obs_norm_path)).exists():
        return None
    from training.encoder import OBS_DIM
    from training.obs_norm import RunningNorm
    norm = RunningNorm(obs_dim if obs_dim is not None else OBS_DIM)
    norm.load(str(obs_norm_path))
    return norm


def make_neural_opponent_cached(
    state_dict: dict,
    obs_norm,                            # already-loaded RunningNorm or None
    device: str = "cpu",
    recorder=None,
    encoder_version: Optional[str] = None,
) -> Opponent:
    """Build an Opponent callable from a pre-loaded state_dict + obs_norm.

    `encoder_version` selects which encoder to run on each call (must
    match the obs distribution the state_dict was trained on). When None,
    defaults to `DEFAULT_ENCODER_VERSION` — the v9.0 fallback for legacy
    unstamped checkpoints. Pass the value returned alongside the
    state_dict by `preload_state_dict`.

    Faster than `make_neural_opponent` for hot-swap paths because no disk
    I/O happens here — caller is expected to have called `preload_state_dict`
    + `preload_obs_norm` once at init time.

    Per-call cost: ~10-50ms (net construct + state_dict copy + .to(device)).
    Compare to ~30-100ms for the full disk-read variant.
    """
    from training.agent import PPOAgent
    from training.encoders import DEFAULT_ENCODER_VERSION, get_encoder
    from training.net import ActorCritic, infer_body_dim, infer_obs_dim

    if encoder_version is None:
        encoder_version = DEFAULT_ENCODER_VERSION
    encoder_entry = get_encoder(encoder_version)

    # Size the net to the saved trunk's actual obs_dim, not the current
    # encoder's. A v9.0 checkpoint has trunk.0 ∈ (body, 1002); a v10
    # checkpoint has (body, 1008). Default ActorCritic(obs_dim=OBS_DIM)
    # would always pick v10's 1008 and crash on a v9.0 state_dict load.
    body_dim = infer_body_dim(state_dict)
    obs_dim  = infer_obs_dim(state_dict)
    if obs_dim != encoder_entry.obs_dim:
        raise ValueError(
            f"checkpoint trunk obs_dim={obs_dim} but encoder_version="
            f"{encoder_version!r} expects obs_dim={encoder_entry.obs_dim}"
        )
    net = ActorCritic(obs_dim=obs_dim, body_dim=body_dim)
    net.load_state_dict(state_dict)
    agent = PPOAgent(net, device=device)
    return _build_opponent_callable(agent, obs_norm, recorder, encoder_entry.encode)


def make_neural_opponent(
    weights_path: str,
    obs_norm_path: Optional[str] = None,
    device: str = "cpu",
    recorder=None,
) -> Opponent:
    """Load a frozen ActorCritic snapshot and return an Opponent callable.

    Each vec-env subprocess constructs its own via make_env's factory; the
    `weights_path` is read once at construction (not on every step).

    Routes through the versioned encoder registry so a checkpoint trained
    against any past encoder version still loads against current code.

    When `recorder` is passed (replay capture path), the opponent uses
    `act_one_with_diag` and records P2's decision with the same schema as P1.

    IMPORTANT: when the parent process has CUDA initialized and spawns
    AsyncVectorEnv subprocs, each subproc re-imports torch and would also
    try to init CUDA. With 64 subprocs doing that at once, torch's CUDA
    init deadlocks (observed: main proc stuck on `unix_stream_read_generic`
    waiting for subprocs stuck on `futex_do_wait`). We hide the GPU from
    the subproc's torch *before* importing it so opponents stay CPU-only.
    """
    state_dict, encoder_version = preload_state_dict(weights_path, device=device)
    # Size obs_norm to whatever shape the saved file actually has — see
    # preload_obs_norm; the file's own shape wins regardless of the
    # constructor arg.
    from training.encoders import get_encoder
    obs_norm = preload_obs_norm(
        obs_norm_path, obs_dim=get_encoder(encoder_version).obs_dim,
    )
    return make_neural_opponent_cached(
        state_dict, obs_norm, device=device, recorder=recorder,
        encoder_version=encoder_version,
    )


def _build_opponent_callable(agent, obs_norm, recorder, encode_fn=None):
    """Closure factory shared between cached + uncached opponent makers.

    `encode_fn` selects which encoder to run on the mirrored obs dict.
    Defaults to the current (v10) encoder for back-compat with callers
    that haven't been updated to thread `encoder_version` through.
    """
    if encode_fn is None:
        from training.encoder import encode_obs as encode_fn

    def opponent(state: State, rng: np.random.Generator) -> int:
        obs_dict = _state_to_obs(state, mask_player=C.OWNER_P2)
        mirrored = _mirror_ownership(obs_dict)
        # Mask is P2's (computed against real state, not the mirrored dict).
        mirrored["action_mask"] = obs_dict["action_mask"]

        x = encode_fn(mirrored)
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
