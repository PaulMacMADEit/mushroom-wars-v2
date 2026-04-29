"""Loader for configs/bench_eval.yaml.

Single source of truth for bench_eval tunables. Use:

    from cli.bench_config import load
    cfg = load()
    cfg.match["level_name"]         # "random_close_4_5"
    cfg.sweep["games_per_champion"] # 8
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "bench_eval.yaml"


@dataclass
class BenchConfig:
    match:          dict
    sweep:          dict
    promotion:      dict
    bootstrap_gate: dict
    archive:        dict
    raw:            dict


def load(path: Path | str | None = None) -> BenchConfig:
    p = Path(path) if path is not None else _CONFIG_PATH
    with p.open() as f:
        raw = yaml.safe_load(f)
    return BenchConfig(
        match=dict(raw.get("match", {})),
        sweep=dict(raw.get("sweep", {})),
        promotion=dict(raw.get("promotion", {})),
        bootstrap_gate=dict(raw.get("bootstrap_gate", {})),
        archive=dict(raw.get("archive", {})),
        raw=raw,
    )


if __name__ == "__main__":
    import json
    cfg = load()
    print(f"loaded {_CONFIG_PATH}")
    print(json.dumps(cfg.raw, indent=2))
