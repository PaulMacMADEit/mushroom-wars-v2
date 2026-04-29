"""Loader for configs/worker.yaml.

Single source of truth for worker tunables. Use:

    from cli.worker_config import load
    cfg = load()
    cfg.metrics["upload_every"]      # 1
    cfg.auto_rate["games_per_match"] # 64
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "worker.yaml"


@dataclass
class WorkerConfig:
    metrics:    dict
    auto_rate:  dict
    admission:  dict
    baseline:   dict
    raw:        dict


def load(path: Path | str | None = None) -> WorkerConfig:
    p = Path(path) if path is not None else _CONFIG_PATH
    with p.open() as f:
        raw = yaml.safe_load(f)
    return WorkerConfig(
        metrics=dict(raw.get("metrics", {})),
        auto_rate=dict(raw.get("auto_rate", {})),
        admission=dict(raw.get("admission", {})),
        baseline=dict(raw.get("baseline", {})),
        raw=raw,
    )


if __name__ == "__main__":
    import json
    cfg = load()
    print(f"loaded {_CONFIG_PATH}")
    print(json.dumps(cfg.raw, indent=2))
