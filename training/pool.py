"""Disk-backed opponent pool for self-play.

Each snapshot = one (weights.pt, obs_norm.pt) pair under a per-tag subdir.
The pool caps at `max_size`; oldest entries are evicted. Sampling is
weighted toward the latest snapshot (ARCHITECTURE §10.3: "80% latest, 20%
random pool member"); diversity against older frozen checkpoints prevents
the learner from chasing its own recent tactics.

We pass *paths* to the vec env factory rather than tensors: each sub-env
loads its opponent independently in its own process, which also means pool
membership can survive a vec-env restart.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import numpy as np
import torch


class OpponentPool:
    def __init__(self, root: Optional[str] = None, max_size: int = 20):
        self.root = Path(root) if root else Path(mkdtemp(prefix="mw2-pool-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        # Each entry: (weights.pt path, obs_norm.pt path or None)
        self.snapshots: list[tuple[Path, Optional[Path]]] = []

    # ------------------------------------------------------------------
    # Register / sample / evict
    # ------------------------------------------------------------------

    def register(
        self,
        net: torch.nn.Module,
        obs_norm,
        tag: str,
        net_version: Optional[str] = None,
    ) -> tuple[Path, Optional[Path]]:
        """Snapshot `net` (and optionally `obs_norm`) under `tag`. Returns the
        paths so the caller can pass them into the env factory right away.

        `net_version` stamps the topology version on the saved checkpoint so
        the loader can pick the right ActorCritic class. Defaults to None
        which falls through to `CURRENT_NET_VERSION` in `save_state_dict`."""
        snap_dir = self.root / f"snap-{tag}"
        snap_dir.mkdir(exist_ok=True)

        w_path = snap_dir / "weights.pt"
        # v10: wrap with encoder_version stamp so loaders dispatch to the
        # right encoder. v13: also stamp net_version. Legacy raw saves still
        # load via the back-compat path in checkpoint.load_state_dict_with_version.
        from training.checkpoint import save_state_dict
        save_state_dict(
            {k: v.detach().cpu() for k, v in net.state_dict().items()},
            w_path,
            net_version=net_version,
        )

        n_path: Optional[Path] = None
        if obs_norm is not None:
            n_path = snap_dir / "obs_norm.pt"
            obs_norm.save(n_path)

        self.snapshots.append((w_path, n_path))

        # Evict oldest beyond cap.
        while len(self.snapshots) > self.max_size:
            old_w, old_n = self.snapshots.pop(0)
            for p in (old_w, old_n):
                if p is not None:
                    p.unlink(missing_ok=True)
            try:
                old_w.parent.rmdir()
            except OSError:
                pass

        return w_path, n_path

    def sample(
        self,
        rng: np.random.Generator,
        latest_bias: float = 0.8,
    ) -> Optional[tuple[Path, Optional[Path]]]:
        if not self.snapshots:
            return None
        if len(self.snapshots) == 1 or rng.random() < latest_bias:
            return self.snapshots[-1]
        idx = int(rng.integers(0, len(self.snapshots) - 1))
        return self.snapshots[idx]

    def __len__(self) -> int:
        return len(self.snapshots)
