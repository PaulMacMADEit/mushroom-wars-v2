"""Running observation normalizer (Welford's algorithm).

Standard PPO stabilization trick (ARCHITECTURE §10.1). Without it, raw sim
observations span very different scales — `tick/GAME_TIMEOUT_TICKS` is in
[0,1], garrison ratios are in [0,~3.5] (buildings can exceed capacity via
reinforcement), one-hot ownership flags are {0,1}. Running per-feature
mean/std keeps the network's input distribution stable across training and
transferable across runs.

Stored alongside weights.pt as `obs_norm.pt` so a continuation or evaluation
run can load the exact same scaling.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningNorm:
    """Welford per-feature running statistics.

    Thread/process-unsafe; the trainer owns a single instance in the main
    process and updates it after each rollout.
    """

    def __init__(self, shape: tuple[int, ...] | int, epsilon: float = 1e-4):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        # M2 = sum of squared deltas (Welford). Variance = M2 / count.
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.M2    = np.zeros(shape, dtype=np.float64)
        self.count = float(epsilon)  # epsilon seed avoids div-by-zero before first update

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Accumulate stats from a batch `x` of shape (B, *self.shape).

        Uses Chan's parallel-algorithm variant of Welford so we can absorb a
        whole batch at once instead of looping per-sample.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x[None, ...]
        batch_n    = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_M2   = ((x - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self.mean
        tot_n = self.count + batch_n
        self.mean = self.mean + delta * (batch_n / tot_n)
        self.M2   = self.M2 + batch_M2 + (delta ** 2) * (self.count * batch_n / tot_n)
        self.count = tot_n

    # ------------------------------------------------------------------
    # Apply + serialize
    # ------------------------------------------------------------------

    @property
    def var(self) -> np.ndarray:
        return self.M2 / max(self.count, 1.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Center + scale. Clip protects against rare extreme inputs."""
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        if clip is not None:
            np.clip(out, -clip, clip, out=out)
        return out

    def state_dict(self) -> dict:
        return {"mean": self.mean, "M2": self.M2, "count": self.count, "shape": self.shape}

    def load_state_dict(self, state: dict) -> None:
        self.mean  = np.asarray(state["mean"],  dtype=np.float64)
        self.M2    = np.asarray(state["M2"],    dtype=np.float64)
        self.count = float(state["count"])
        self.shape = tuple(state["shape"])

    def save(self, path: str | "Path") -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | "Path") -> None:
        self.load_state_dict(torch.load(path, weights_only=False))
