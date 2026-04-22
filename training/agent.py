"""PPO agent wrapper around `ActorCritic`.

Responsibilities:
  - act(obs, mask): sample action, return (action, logprob, value, entropy)
  - evaluate(obs, action, mask): recompute logprob/entropy/value for a stored
    action (used in the PPO update loop)

Observation arrives as a numpy float array (OBS_DIM,). The agent converts
to/from tensors internally.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from training.net import ActorCritic


class PPOAgent:
    def __init__(self, net: ActorCritic, device: torch.device | str = "cpu"):
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float, float]:
        """Sample one action. Returns (action_idx, logprob, value)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        masked = self.net.masked_logits(logits, mask_t)
        dist = Categorical(logits=masked)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    @torch.no_grad()
    def act_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample N actions from N obs in one forward pass.

        Args:
            obs:  (N, OBS_DIM) float32
            mask: (N, ACTION_SPACE_SIZE) bool

        Returns (actions (N,) int64, logprobs (N,) float32, values (N,) float32).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        logits, value = self.net(obs_t)
        masked = self.net.masked_logits(logits, mask_t)
        dist = Categorical(logits=masked)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return (
            action.cpu().numpy(),
            logprob.cpu().numpy(),
            value.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched re-evaluation for PPO updates.

        Returns (logprob, entropy, value) — all shape (B,).
        """
        logits, value = self.net(obs)
        masked = self.net.masked_logits(logits, masks)
        dist = Categorical(logits=masked)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value
