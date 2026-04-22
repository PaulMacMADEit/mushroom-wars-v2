"""Minimal PPO network for Phase 2 smoke training.

Flat 4097-way policy head with a mask-before-softmax. The v9.0 chained heads
(source → type → target, §9.4) are the eventual target, but for the smoke
run a single head is simpler to debug and equally valid as a loss surface.
"""

from __future__ import annotations

import torch
from torch import nn

from sim.actions import ACTION_SPACE_SIZE
from training.encoder import OBS_DIM


# Large negative used to zero masked logits after softmax. Not -inf because
# backward on (-inf + 0) can yield NaN in edge cases; this is ~0 after softmax
# without causing numerical issues.
MASK_FILL = -1e9


class ActorCritic(nn.Module):
    """Shared-body actor-critic. Body: OBS_DIM → 128 → 128. Heads: 4097 + 1."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init on the policy head with small gain keeps early
        # rollouts exploratory (standard PPO trick).
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2.0 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits, value). Logits are raw — caller applies mask + softmax."""
        z = self.body(obs)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value

    def masked_logits(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply action mask. Invalid positions get MASK_FILL."""
        return torch.where(mask, logits, torch.full_like(logits, MASK_FILL))
