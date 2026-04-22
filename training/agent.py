"""PPO agent for the chained actor-critic.

Sampling goes in three steps: source → type | source → target | source.
The `type` head has an extra noop slot (index = NUM_TYPES); if type=noop the
flat action emitted to the env is NOOP_INDEX regardless of the sampled src
and tgt (those values are "wasted" for env purposes but still affect log-
prob — that's fine, it's just factored exploration noise).

Masking is done inside the agent so the caller only has to hand over the
full (B, ACTION_SPACE_SIZE) env-legality mask. `_decompose_masks` derives
per-head masks from it. If no source has any legal send anywhere, we fall
back to letting the source head choose freely — the type head will then
force noop via its mask.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX
from training.net import ActorCritic, NUM_SRC, NUM_TGT, NUM_TYPES, NUM_TYPE_CHOICES


MASK_FILL = -1e9


def _decompose_masks(
    full_mask: torch.Tensor,
    src: torch.Tensor | None = None,
    type_: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Break the env (B, 4097) mask into chained-head masks.

    Returns whichever masks the caller has context for:
      - call with just `full_mask`              → (src_mask,)
      - call with `full_mask + src`             → (src_mask, type_mask)
      - call with `full_mask + src + type_`     → (src_mask, type_mask, tgt_mask)
    """
    B = full_mask.shape[0]
    send_3d = full_mask[:, :ACTION_SPACE_SIZE - 1].reshape(B, NUM_TYPES, NUM_SRC, NUM_TGT)
    noop_ok = full_mask[:, NOOP_INDEX]  # (B,)

    # src_mask: slot has ≥1 legal (type, tgt) pair.
    src_legal = send_3d.any(dim=1).any(dim=-1)          # (B, 32)
    any_src_legal = src_legal.any(dim=-1, keepdim=True)  # (B, 1)
    # If nobody's a legal source this step, allow every src so the agent can
    # still sample; the type head's mask will force noop.
    src_mask = src_legal | ~any_src_legal

    if src is None:
        return (src_mask,)

    # Gather send_3d[:, :, src, :] → (B, 4, 32) without a host-side loop.
    src_idx = src[:, None, None, None].expand(-1, NUM_TYPES, 1, NUM_TGT)
    send_for_src = send_3d.gather(2, src_idx).squeeze(2)   # (B, 4, 32)

    type_any_tgt = send_for_src.any(dim=-1)                  # (B, 4)
    type_mask = torch.cat([type_any_tgt, noop_ok[:, None]], dim=-1)  # (B, 5)

    if type_ is None:
        return src_mask, type_mask

    # For noop (type=4), tgt is a don't-care: allow everything so the
    # Categorical doesn't see an all-masked row.
    type_clamp = type_.clamp(max=NUM_TYPES - 1)
    tgt_legal = send_for_src.gather(
        1, type_clamp[:, None, None].expand(-1, 1, NUM_TGT)
    ).squeeze(1)                                             # (B, 32)
    is_noop = (type_ == NUM_TYPES)
    tgt_mask = tgt_legal | is_noop[:, None]
    return src_mask, type_mask, tgt_mask


def _compose_action(src: torch.Tensor, type_: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Pack (src, type, tgt) into the flat env action index."""
    is_noop = (type_ == NUM_TYPES)
    type_send = type_.clamp(max=NUM_TYPES - 1)
    send_action = type_send * (NUM_SRC * NUM_TGT) + src * NUM_TGT + tgt
    return torch.where(is_noop, torch.full_like(send_action, NOOP_INDEX), send_action)


class PPOAgent:
    def __init__(self, net: ActorCritic, device: torch.device | str = "cpu"):
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Factored sampling. Returns six numpy arrays:
          action (N,)   — flat env action index
          src    (N,)   — sampled source slot (0..31)
          type_  (N,)   — sampled type (0..4; 4=noop)
          tgt    (N,)   — sampled target slot (0..31)
          logp   (N,)   — summed log-prob of (src, type, tgt)
          value  (N,)   — V(obs)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        body = self.net.forward_body(obs_t)
        value = self.net.value(body)

        (src_mask,) = _decompose_masks(mask_t)
        src_logits = self.net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
        src_dist = Categorical(logits=src_logits)
        src = src_dist.sample()
        logp_src = src_dist.log_prob(src)

        type_logits, tgt_logits = self.net.cond_logits(body, src)

        _, type_mask = _decompose_masks(mask_t, src=src)
        type_logits = type_logits.masked_fill(~type_mask, MASK_FILL)
        type_dist = Categorical(logits=type_logits)
        type_ = type_dist.sample()
        logp_type = type_dist.log_prob(type_)

        _, _, tgt_mask = _decompose_masks(mask_t, src=src, type_=type_)
        tgt_logits = tgt_logits.masked_fill(~tgt_mask, MASK_FILL)
        tgt_dist = Categorical(logits=tgt_logits)
        tgt = tgt_dist.sample()
        logp_tgt = tgt_dist.log_prob(tgt)

        logp = logp_src + logp_type + logp_tgt
        action = _compose_action(src, type_, tgt)
        return (
            action.cpu().numpy(),
            src.cpu().numpy(),
            type_.cpu().numpy(),
            tgt.cpu().numpy(),
            logp.cpu().numpy(),
            value.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Single-decision diagnostics — for replay/introspection.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act_one_with_diag(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, dict]:
        """Batch-of-1 decision with the full policy breakdown exposed.

        Mirrors `act_batch`'s sampling chain exactly, but returns the picked
        action plus a diagnostic dict with per-head softmax, masks, picks,
        value estimate and entropy. Used by the replay recorder — training
        uses `act_batch` so the hot path stays untouched.
        """
        obs_t  = torch.as_tensor(obs[None, :],  dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool,    device=self.device)

        body = self.net.forward_body(obs_t)
        value = self.net.value(body).squeeze().item()

        (src_mask,) = _decompose_masks(mask_t)
        src_logits = self.net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
        src_dist = Categorical(logits=src_logits)
        src = src_dist.sample()
        src_entropy = src_dist.entropy().item()

        type_logits_raw, tgt_logits_raw = self.net.cond_logits(body, src)
        _, type_mask = _decompose_masks(mask_t, src=src)
        type_logits = type_logits_raw.masked_fill(~type_mask, MASK_FILL)
        type_dist = Categorical(logits=type_logits)
        type_ = type_dist.sample()
        type_entropy = type_dist.entropy().item()

        _, _, tgt_mask = _decompose_masks(mask_t, src=src, type_=type_)
        tgt_logits = tgt_logits_raw.masked_fill(~tgt_mask, MASK_FILL)
        tgt_dist = Categorical(logits=tgt_logits)
        tgt = tgt_dist.sample()
        tgt_entropy = tgt_dist.entropy().item()

        action = _compose_action(src, type_, tgt).item()

        diag = {
            "value":        float(value),
            "entropy":      float(src_entropy + type_entropy + tgt_entropy),
            "src_picked":   int(src.item()),
            "type_picked":  int(type_.item()),
            "tgt_picked":   int(tgt.item()),
            "src_probs":    src_dist.probs.squeeze(0).cpu().numpy(),
            "type_probs":   type_dist.probs.squeeze(0).cpu().numpy(),
            "tgt_probs":    tgt_dist.probs.squeeze(0).cpu().numpy(),
            "src_mask":     src_mask.squeeze(0).cpu().numpy(),
            "type_mask":    type_mask.squeeze(0).cpu().numpy(),
            "tgt_mask":     tgt_mask.squeeze(0).cpu().numpy(),
        }
        return int(action), diag

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs: torch.Tensor,
        src: torch.Tensor,
        type_: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-prob / entropy / value for stored (src, type, tgt)."""
        body = self.net.forward_body(obs)
        value = self.net.value(body)

        src_mask, type_mask, tgt_mask = _decompose_masks(mask, src=src, type_=type_)

        src_logits  = self.net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
        type_logits_raw, tgt_logits_raw = self.net.cond_logits(body, src)
        type_logits = type_logits_raw.masked_fill(~type_mask, MASK_FILL)
        tgt_logits  = tgt_logits_raw.masked_fill(~tgt_mask, MASK_FILL)

        src_dist  = Categorical(logits=src_logits)
        type_dist = Categorical(logits=type_logits)
        tgt_dist  = Categorical(logits=tgt_logits)

        logp = src_dist.log_prob(src) + type_dist.log_prob(type_) + tgt_dist.log_prob(tgt)
        entropy = src_dist.entropy() + type_dist.entropy() + tgt_dist.entropy()
        return logp, entropy, value
