"""Loader for configs/karpathy_loop.yaml.

Single source of truth for the hourly loop. Use:

    from cli.loop_config import load
    cfg = load()
    cfg.baseline_hyperparams        # dict
    cfg.next_axis(last_used="entropy_coef")  # picks next sweep axis
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "karpathy_loop.yaml"


@dataclass
class SweepAxis:
    axis: str
    cells: list[dict]   # [{label, value}, ...]


@dataclass
class LoopConfig:
    schedule:           dict
    queue_policy:       dict
    model:              dict
    training_opponent:  dict   # {name: str, kwargs: dict}
    baseline_hyperparams: dict
    sweep_axes:         list[SweepAxis]
    raw:                dict

    def axes(self) -> list[str]:
        return [a.axis for a in self.sweep_axes]

    def get_axis(self, name: str) -> SweepAxis:
        for a in self.sweep_axes:
            if a.axis == name:
                return a
        raise KeyError(f"axis {name!r} not in {self.axes()}")

    def next_axis(self, last_used: str | None) -> SweepAxis:
        """Round-robin pick. If last_used is None or unknown, returns first."""
        names = self.axes()
        if last_used is None or last_used not in names:
            return self.sweep_axes[0]
        i = names.index(last_used)
        return self.sweep_axes[(i + 1) % len(names)]


def load(path: Path | str | None = None) -> LoopConfig:
    p = Path(path) if path is not None else _CONFIG_PATH
    with p.open() as f:
        raw = yaml.safe_load(f)
    axes = [SweepAxis(axis=a["axis"], cells=list(a["cells"])) for a in raw.get("sweep_axes", [])]
    return LoopConfig(
        schedule=raw.get("schedule", {}),
        queue_policy=raw.get("queue_policy", {}),
        model=raw.get("model", {}),
        training_opponent=dict(raw.get("training_opponent", {"name": "random_legal", "kwargs": {}})),
        baseline_hyperparams=dict(raw.get("baseline_hyperparams", {})),
        sweep_axes=axes,
        raw=raw,
    )


if __name__ == "__main__":
    import json
    cfg = load()
    print(f"loaded {_CONFIG_PATH}")
    print(f"axes: {cfg.axes()}")
    print(f"baseline keys: {sorted(cfg.baseline_hyperparams.keys())}")
    print(f"schedule: {json.dumps(cfg.schedule, indent=2)}")
