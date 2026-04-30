"""Versioned encoder registry — adapter regression coverage.

Why this exists: the encoder registry is the *only* path through which
checkpoints saved under one encoder version remain loadable after a
shape-changing bump. Without these tests, a future bump can silently
delete a historical encoder and break every old archive checkpoint.

What's covered:
- Both registered versions produce vectors of the right OBS_DIM.
- Legacy (unstamped) checkpoints default to v9.0 and load cleanly into
  an ActorCritic shaped by `infer_obs_dim`.
- Wrapped (stamped) checkpoints round-trip through save → load with
  the version stamp preserved.
- Wrong-encoder-for-checkpoint raises with a useful message.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sim.envs.mushroom_env import MushroomEnv
from sim.envs.opponents import (
    make_neural_opponent_cached,
    noop_opponent,
    preload_state_dict,
)
from training.checkpoint import (
    load_state_dict_with_version,
    save_state_dict,
)
from training.encoders import (
    CURRENT_ENCODER_VERSION,
    DEFAULT_ENCODER_VERSION,
    get_encoder,
    known_versions,
)
from training.encoders.v9 import OBS_DIM as OBS_DIM_V9
from training.net import ActorCritic, infer_obs_dim


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lists_v9_and_v10():
    versions = known_versions()
    assert "v9.0" in versions
    assert "v10"  in versions


def test_default_version_is_v9_for_legacy_checkpoints():
    # Pre-stamping convention saves are by definition v9.0; the default must
    # match or every legacy checkpoint silently mis-encodes.
    assert DEFAULT_ENCODER_VERSION == "v9.0"


def test_current_version_is_v10():
    # Catches a bump where someone adds an encoder but forgets to bump the
    # constant new-saves stamp themselves with.
    assert CURRENT_ENCODER_VERSION == "v10"


def test_obs_dims_match_per_version():
    assert get_encoder("v9.0").obs_dim == 1002
    assert get_encoder("v10").obs_dim  == 1008


# ---------------------------------------------------------------------------
# Save → load round-trip
# ---------------------------------------------------------------------------

def test_legacy_save_loads_as_v9():
    """Raw torch.save (no wrapper) → preload_state_dict returns v9.0."""
    net = ActorCritic(obs_dim=OBS_DIM_V9, body_dim=128)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        torch.save(net.state_dict(), path)  # legacy: raw, no wrapper

        sd, enc_version = preload_state_dict(str(path), device="cpu")
        assert enc_version == "v9.0"
        assert infer_obs_dim(sd) == 1002


def test_wrapped_save_round_trips_version():
    """save_state_dict + load_state_dict_with_version preserve the stamp."""
    net = ActorCritic()  # current = v10
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        save_state_dict(net.state_dict(), path)

        sd, enc_version = load_state_dict_with_version(
            path, weights_only=False,
        )
        assert enc_version == "v10"
        assert infer_obs_dim(sd) == 1008


def test_explicit_version_override():
    """Caller can stamp a non-current version (e.g. converted backfill)."""
    net = ActorCritic()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        save_state_dict(net.state_dict(), path, encoder_version="v9.0")

        _, enc_version = load_state_dict_with_version(
            path, weights_only=False,
        )
        assert enc_version == "v9.0"


# ---------------------------------------------------------------------------
# End-to-end: load + run as an opponent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version,body_dim", [("v9.0", 128), ("v10", 128)])
def test_opponent_dispatch_runs_per_version(version, body_dim):
    """Build a fresh net at each version, save (legacy format for v9 to mimic
    the existing archive; new format for v10), load through the adapter,
    and run one decision against a real env state. No crash = the encoder
    dispatch is wired through.
    """
    obs_dim = get_encoder(version).obs_dim
    net = ActorCritic(obs_dim=obs_dim, body_dim=body_dim)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        if version == "v9.0":
            torch.save(net.state_dict(), path)  # legacy raw
        else:
            save_state_dict(net.state_dict(), path)  # stamped wrapper

        sd, enc_version = preload_state_dict(str(path), device="cpu")
        assert enc_version == version

        opp = make_neural_opponent_cached(
            sd, obs_norm=None, device="cpu", encoder_version=enc_version,
        )

        env = MushroomEnv(level_name="crossroads_6", opponent=noop_opponent, seed=42)
        env.reset(seed=42)
        rng = np.random.default_rng(7)
        action_idx = opp(env.state, rng)
        # NOOP_INDEX = 4096; sends are 0..4095. Just confirm legality range.
        assert 0 <= action_idx <= 4096


def test_loading_v9_state_dict_with_v10_dispatch_raises():
    """If a caller passes encoder_version='v10' with a v9.0 state_dict, the
    obs_dim mismatch is caught early with a useful error, not deep in
    `load_state_dict` with cryptic shape spam.
    """
    net_v9 = ActorCritic(obs_dim=OBS_DIM_V9, body_dim=128)
    with pytest.raises(ValueError, match="obs_dim=1002.*expects obs_dim=1008"):
        make_neural_opponent_cached(
            net_v9.state_dict(),
            obs_norm=None,
            device="cpu",
            encoder_version="v10",
        )
