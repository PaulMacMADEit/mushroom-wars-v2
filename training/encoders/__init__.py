"""Versioned encoder registry.

Reason this exists: the encoder is a shape-defining contract between the
simulator's State and the policy network's input layer. When we change it
(add fields, drop fields, reorder), every checkpoint trained against the
old shape stops loading against the new code. Without a registry, that
means we either keep dead branches in `encoder.py` forever, OR we lose
the ability to play / evaluate / fine-tune any pre-bump checkpoint.

Design (per encoder design discussion 2026-04-29):
- Each historical encoder lives at `training.encoders.v{N}` (numpy) and
  `training.encoders.v{N}_jax` (JAX mirror), self-contained — no imports
  from `training.encoder` (which always points at the *current* version).
- `ENCODERS` maps a string key to `EncoderEntry` carrying both the numpy
  and JAX entry points plus the OBS_DIM that built that net.
- Loaders read `state_dict["__encoder_version__"]` (added at save time
  by `training.checkpoint.save_checkpoint`); legacy checkpoints lacking
  the stamp default to "v9.0", which is the latest version that pre-dates
  the stamping convention.

Adding a new version (e.g. v11):
  1. Copy `training/encoder.py` (the current v10) to
     `training/encoders/v11.py` BEFORE editing it. Keep the file
     self-contained — no imports from `training.encoder`.
  2. Same for `encoder_jax.py` → `training/encoders/v11_jax.py`.
  3. Register the v11 entry in `ENCODERS` below.
  4. Edit `training/encoder.py` (and `encoder_jax.py`) freely — they
     remain the live "current" encoders that the trainer uses.
  5. Bump the default in `save_checkpoint` (or wherever CURRENT_ENCODER
     is read from) so new saves stamp v11.

This way the codebase only carries dead-code weight for as many
versions as are still in the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EncoderEntry:
    """One registry row.

    Attributes:
      version:    the string key (also stamped on saved state_dicts)
      obs_dim:    flat float32 vector length the encoder produces
      encode:     numpy entry point — `(obs_dict) -> (OBS_DIM,) float32`
      encode_jax: JAX entry point — `(StateJax) -> (n_envs, OBS_DIM) jnp.float32`
                  (already wrapped in `jax.jit` so callers can dispatch
                  through this entry without re-jitting per call).
    """
    version: str
    obs_dim: int
    encode: Callable[[dict], np.ndarray]
    encode_jax: Callable
    description: str = ""


# ---------------------------------------------------------------------------
# Build the registry. Imports are nested so that loading this module doesn't
# pay the cost of every encoder + its JAX jit at import time — only the ones
# the caller actually uses get built (via `get_encoder`).
# ---------------------------------------------------------------------------

def _build_v12() -> EncoderEntry:
    # v12 is the live encoder. Importing the live module also keeps
    # backward compat — `training.encoder` keeps working as the
    # "current" entry for trainer code that doesn't know about versions.
    from training import encoder as enc_v12
    from training import encoder_jax as enc_v12_jax
    return EncoderEntry(
        version="v12",
        obs_dim=enc_v12.OBS_DIM,
        encode=enc_v12.encode_obs,
        encode_jax=enc_v12_jax.encode_obs_batched_jit,
        description="v12 (192 dims): slot-token layout for set-transformer body. "
                    "8 buildings (× 11 features), 4 groups (× 6), 80 globals. "
                    "Action space cut to 2 send % + noop. Clean break from v10 — "
                    "old (32-slot) checkpoints incompatible.",
    )


_BUILDERS: dict[str, Callable[[], EncoderEntry]] = {
    "v12": _build_v12,
}

_CACHE: dict[str, EncoderEntry] = {}


# v12 is a clean break — old (v9.0/v10) checkpoints are incompatible with
# the 8-slot sim and won't load. Default version for unstamped checkpoints
# is now "v12" so any fresh save without an explicit stamp is interpreted
# correctly. Loading a pre-v12 archive entry will raise (intentional).
DEFAULT_ENCODER_VERSION = "v12"

# What new saves should stamp. Bump alongside `_BUILDERS` when shipping
# a new version so the trainer's saves carry the right version label.
CURRENT_ENCODER_VERSION = "v12"


def get_encoder(version: str | None) -> EncoderEntry:
    """Return the EncoderEntry for `version`. None → DEFAULT_ENCODER_VERSION."""
    if version is None:
        version = DEFAULT_ENCODER_VERSION
    if version not in _BUILDERS:
        raise ValueError(
            f"unknown encoder version {version!r}; known: {sorted(_BUILDERS)}"
        )
    if version not in _CACHE:
        _CACHE[version] = _BUILDERS[version]()
    return _CACHE[version]


def known_versions() -> list[str]:
    return sorted(_BUILDERS)


__all__ = [
    "EncoderEntry",
    "get_encoder",
    "known_versions",
    "CURRENT_ENCODER_VERSION",
    "DEFAULT_ENCODER_VERSION",
]
