"""Versioned net registry — mirrors training.encoders.

Reason this exists: when net topology changes (head structure, sampling
chain, body depth), the saved state_dict shape changes too. Without a
registry every checkpoint trained against the old shape stops loading.

Design (mirrors training/encoders/__init__.py):
- Each historical net lives at `training.nets.v{N}`. Self-contained — no
  imports from `training.net` (which always points at the *current* version).
- `NET_BUILDERS` maps a string key to a factory returning the ActorCritic
  class for that version.
- Loaders read `net_version` from the wrapped checkpoint payload. Legacy
  checkpoints lacking the stamp default to "v12" (the version that
  pre-dates net-version stamping).

Adding a new version (e.g. v14):
  1. Copy `training/net.py` (the current v13) to `training/nets/v13.py`
     BEFORE editing it. Self-contained — the archive stays frozen.
  2. Register the v13 entry in NET_BUILDERS below.
  3. Edit `training/net.py` freely — it stays the live "current" net.
  4. Bump CURRENT_NET_VERSION below.

Why a separate registry from encoders: the encoder is the obs contract
(shape of inputs); the net is the topology contract (shape of weights and
sampling chain). They evolve independently — a net change with the same
encoder is common (e.g. v12 → v13 keeps OBS_DIM=192 but reshapes heads).
"""

from __future__ import annotations

from typing import Callable, Type


# ---------------------------------------------------------------------------
# Version registry. Imports are nested so loading this module doesn't pay
# the cost of importing every archived net at startup.
# ---------------------------------------------------------------------------

def _build_v12() -> Type:
    """Return the v12 ActorCritic class (frozen archive)."""
    from training.nets.v12 import ActorCritic
    return ActorCritic


def _build_v13() -> Type:
    """Return the v13 ActorCritic class (live, in training/net.py)."""
    from training.net import ActorCritic
    return ActorCritic


NET_BUILDERS: dict[str, Callable[[], Type]] = {
    "v12": _build_v12,
    "v13": _build_v13,
}

_CACHE: dict[str, Type] = {}


# Default for unstamped checkpoints (predates the stamping convention).
DEFAULT_NET_VERSION = "v12"

# What new saves stamp. Bump alongside NET_BUILDERS when shipping a new
# version so the trainer's saves carry the right label.
CURRENT_NET_VERSION = "v13"


def get_net_class(version: str | None) -> Type:
    """Return the ActorCritic class for `version`. None → DEFAULT_NET_VERSION."""
    if version is None:
        version = DEFAULT_NET_VERSION
    if version not in NET_BUILDERS:
        raise ValueError(
            f"unknown net version {version!r}; known: {sorted(NET_BUILDERS)}"
        )
    if version not in _CACHE:
        _CACHE[version] = NET_BUILDERS[version]()
    return _CACHE[version]


def known_versions() -> list[str]:
    return sorted(NET_BUILDERS)


__all__ = [
    "get_net_class",
    "known_versions",
    "CURRENT_NET_VERSION",
    "DEFAULT_NET_VERSION",
    "NET_BUILDERS",
]
