"""Versioned checkpoint save/load.

Wraps `nn.Module.state_dict()` with two version stamps:
  - `encoder_version`: which obs-encoder produced the obs the net was
    trained on (controls input shape).
  - `net_version`: which ActorCritic topology the state_dict belongs to
    (controls head structure + sampling chain). Added in v13 alongside
    the chain-reorder + head-capacity bump; v12 and earlier checkpoints
    don't carry this stamp and default to "v12".

On-disk format (new saves):
    {
      "state_dict":      <flat torch.Tensor dict — what nn.Module.state_dict() returns>,
      "encoder_version": "v12",
      "net_version":     "v13",
    }

Legacy saves (no wrapper, or wrapper without net_version) are accepted by
`load_state_dict_with_version` for backward compat:
  - raw state_dict (oldest format) → (state_dict, DEFAULT_ENCODER_VERSION, DEFAULT_NET_VERSION)
  - wrapper without net_version (v10–v12 era) → uses DEFAULT_NET_VERSION ("v12")

Why a wrapper instead of stuffing `__net_version__` into the state_dict
itself: nn.Module.load_state_dict is strict by default, so any non-tensor
key would raise. The wrapper keeps the inner dict pure so it can flow
into `net.load_state_dict()` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.encoders import CURRENT_ENCODER_VERSION, DEFAULT_ENCODER_VERSION
from training.nets import CURRENT_NET_VERSION, DEFAULT_NET_VERSION


_REQUIRED_WRAPPER_KEYS = {"state_dict", "encoder_version"}


def save_state_dict(
    net_state_dict: dict[str, torch.Tensor],
    path: str | Path,
    encoder_version: str | None = None,
    net_version: str | None = None,
) -> None:
    """Save a net.state_dict() to `path` with the version stamps.

    `encoder_version` and `net_version` default to the project's current
    versions. Pass explicitly only when stamping a backfilled / converted
    checkpoint to a non-current version.
    """
    if encoder_version is None:
        encoder_version = CURRENT_ENCODER_VERSION
    if net_version is None:
        net_version = CURRENT_NET_VERSION
    payload: dict[str, Any] = {
        "state_dict":      net_state_dict,
        "encoder_version": encoder_version,
        "net_version":     net_version,
    }
    torch.save(payload, str(path))


def load_state_dict_with_version(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
    weights_only: bool = True,
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Load a checkpoint at `path`, returning (state_dict, encoder_version, net_version).

    Backward-compatible:
    - raw state_dict (no wrapper) → (state_dict, DEFAULT_ENCODER_VERSION, DEFAULT_NET_VERSION)
    - wrapper missing net_version (v10–v12 era) → uses DEFAULT_NET_VERSION ("v12")

    `weights_only=True` matches the security default torch nudges users
    toward; pass False only when loading wrapper dicts whose payload
    includes non-tensor metadata that pickle would otherwise reject.
    """
    obj = torch.load(str(path), map_location=map_location, weights_only=weights_only)
    if isinstance(obj, dict) and _REQUIRED_WRAPPER_KEYS.issubset(obj.keys()):
        state_dict      = obj["state_dict"]
        encoder_version = obj["encoder_version"]
        net_version     = obj.get("net_version", DEFAULT_NET_VERSION)
        return state_dict, encoder_version, net_version
    # Legacy raw state_dict — predates the stamping convention entirely.
    return obj, DEFAULT_ENCODER_VERSION, DEFAULT_NET_VERSION


__all__ = [
    "save_state_dict",
    "load_state_dict_with_version",
]
