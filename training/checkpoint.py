"""Versioned checkpoint save/load.

Wraps `nn.Module.state_dict()` with the encoder version that produced the
obs the net was trained on, so `make_neural_opponent_cached` (and any
future loader that wants to play / fine-tune an old checkpoint) can pick
the right encoder from `training.encoders.get_encoder()`.

On-disk format (new saves):
    {
      "state_dict":      <flat torch.Tensor dict — what nn.Module.state_dict() returns>,
      "encoder_version": "v10",
    }

Legacy saves (no wrapper) load fine via `load_state_dict_with_version` —
they're treated as `DEFAULT_ENCODER_VERSION` ("v9.0"), which is the
latest version trained before the stamping convention existed.

Why a wrapper instead of stuffing `__encoder_version__` into the state_dict
itself: nn.Module.load_state_dict is strict by default, so any
non-tensor key would raise. The wrapper keeps the inner dict pure so it
can flow into `net.load_state_dict()` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.encoders import CURRENT_ENCODER_VERSION, DEFAULT_ENCODER_VERSION


_WRAPPER_KEYS = {"state_dict", "encoder_version"}


def save_state_dict(
    net_state_dict: dict[str, torch.Tensor],
    path: str | Path,
    encoder_version: str | None = None,
) -> None:
    """Save a net.state_dict() to `path` with the encoder_version stamp.

    `encoder_version` defaults to the project's current version. Pass
    explicitly only when stamping a backfilled / converted checkpoint.
    """
    if encoder_version is None:
        encoder_version = CURRENT_ENCODER_VERSION
    payload: dict[str, Any] = {
        "state_dict":      net_state_dict,
        "encoder_version": encoder_version,
    }
    torch.save(payload, str(path))


def load_state_dict_with_version(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
    weights_only: bool = True,
) -> tuple[dict[str, torch.Tensor], str]:
    """Load a checkpoint at `path`, returning (state_dict, encoder_version).

    Backward-compatible: legacy saves (raw state_dict, no wrapper) return
    (state_dict, DEFAULT_ENCODER_VERSION).

    `weights_only=True` matches the security default torch nudges users
    toward; pass False only when loading wrapper dicts whose payload
    includes non-tensor metadata that pickle would otherwise reject.
    """
    obj = torch.load(str(path), map_location=map_location, weights_only=weights_only)
    if isinstance(obj, dict) and _WRAPPER_KEYS.issubset(obj.keys()):
        return obj["state_dict"], obj["encoder_version"]
    # Legacy raw state_dict — predates the stamping convention.
    return obj, DEFAULT_ENCODER_VERSION


__all__ = [
    "save_state_dict",
    "load_state_dict_with_version",
]
