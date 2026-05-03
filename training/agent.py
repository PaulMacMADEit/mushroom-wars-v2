"""PPO agent for the chained actor-critic — version-dispatched.

Two sampling chains live here:
  v12: source → type → target           (type carries the noop slot)
  v13: source → target → pct            (pct carries the noop slot)

Both emit into the same env-flat action space (encode(type_idx, src, tgt) or
NOOP_INDEX), so cross-version play "just works" — see V13_PLAN.md §
"Backward compatibility".

PPOAgent is a thin dispatcher. The version-specific sampling functions live
below and share the `_to_torch` helper. We picked two ~60-line impls over one
parameterised function because the chain order, conditioning, and action
packing differ enough that a unified branchy version got unreadable.

Public API (unchanged from v12 caller's perspective):
  act_batch(obs, mask, deterministic=False) → (action, c1, c2, c3, logp, value)
      v12: (c1, c2, c3) = (src, type, tgt)
      v13: (c1, c2, c3) = (src, tgt, pct)
  act_one_with_diag(obs, mask) → (action, diag)
      diag carries per-head probs/masks/picks; diag keys differ per version.
  evaluate(obs, c1, c2, c3, mask) → (logp, entropy, value)

Trainer code reads the (c1, c2, c3) tuple opaquely and passes it back to
evaluate() — it doesn't need to know what each component means.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX


MASK_FILL = -1e9


def _to_torch(x, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coerce numpy / jax / torch input to a torch tensor on `device`.

    Zero-copy via DLPack when the input is a JAX array (cuda) or already a
    torch tensor on the right device. Numpy inputs go through the standard
    host->device path.
    """
    if isinstance(x, torch.Tensor):
        if x.device != device:
            x = x.to(device)
        if x.dtype != dtype:
            x = x.to(dtype)
        return x

    try:
        import jax  # noqa: F401
        from jax import Array as JaxArray  # type: ignore
        is_jax = isinstance(x, JaxArray)
    except Exception:
        is_jax = False

    if is_jax:
        try:
            t = torch.from_dlpack(x)
            if t.device != device:
                t = t.to(device)
            if t.dtype != dtype:
                t = t.to(dtype)
            return t
        except Exception:
            x = np.asarray(x)

    return torch.as_tensor(x, dtype=dtype, device=device)


# =============================================================================
# v12 sampling chain: source → type → target  (type 0..NUM_TYPES-1 = sends, NUM_TYPES = noop)
# =============================================================================

# Constants pulled lazily — v12 archive owns its own copies. We import inside
# functions to avoid pulling the v12 module into PYTHONPATH at agent import
# (the v12 ActorCritic is heavy; the v13 trainer never imports it).

def _v12_constants():
    from training.nets.v12 import NUM_SRC, NUM_TGT, NUM_TYPES, NUM_TYPE_CHOICES
    return NUM_SRC, NUM_TGT, NUM_TYPES, NUM_TYPE_CHOICES


def _decompose_masks_v12(
    full_mask: torch.Tensor,
    src: torch.Tensor | None = None,
    type_: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """v12 mask decomposition: src → type(incl. noop) → tgt."""
    NUM_SRC, NUM_TGT, NUM_TYPES, _ = _v12_constants()
    B = full_mask.shape[0]
    send_3d = full_mask[:, :ACTION_SPACE_SIZE - 1].reshape(B, NUM_TYPES, NUM_SRC, NUM_TGT)
    noop_ok = full_mask[:, NOOP_INDEX]

    src_legal = send_3d.any(dim=1).any(dim=-1)
    any_src_legal = src_legal.any(dim=-1, keepdim=True)
    src_mask = src_legal | ~any_src_legal

    if src is None:
        return (src_mask,)

    src_idx = src[:, None, None, None].expand(-1, NUM_TYPES, 1, NUM_TGT)
    send_for_src = send_3d.gather(2, src_idx).squeeze(2)

    type_any_tgt = send_for_src.any(dim=-1)
    type_mask = torch.cat([type_any_tgt, noop_ok[:, None]], dim=-1)

    if type_ is None:
        return src_mask, type_mask

    type_clamp = type_.clamp(max=NUM_TYPES - 1)
    tgt_legal = send_for_src.gather(
        1, type_clamp[:, None, None].expand(-1, 1, NUM_TGT)
    ).squeeze(1)
    is_noop = (type_ == NUM_TYPES)
    tgt_mask = tgt_legal | is_noop[:, None]
    return src_mask, type_mask, tgt_mask


def _compose_action_v12(src: torch.Tensor, type_: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Pack v12 (src, type, tgt) into the flat env action index."""
    NUM_SRC, NUM_TGT, NUM_TYPES, _ = _v12_constants()
    is_noop = (type_ == NUM_TYPES)
    type_send = type_.clamp(max=NUM_TYPES - 1)
    send_action = type_send * (NUM_SRC * NUM_TGT) + src * NUM_TGT + tgt
    return torch.where(is_noop, torch.full_like(send_action, NOOP_INDEX), send_action)


@torch.no_grad()
def _act_batch_v12(net: nn.Module, device: torch.device, obs, mask, deterministic: bool):
    obs_t  = _to_torch(obs,  torch.float32, device)
    mask_t = _to_torch(mask, torch.bool,    device)

    body = net.forward_body(obs_t)
    value = net.value(body)

    (src_mask,) = _decompose_masks_v12(mask_t)
    src_logits = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    src_dist = Categorical(logits=src_logits)
    src = src_logits.argmax(dim=-1) if deterministic else src_dist.sample()
    logp_src = src_dist.log_prob(src)

    type_logits, tgt_logits = net.cond_logits(body, src)

    _, type_mask = _decompose_masks_v12(mask_t, src=src)
    type_logits = type_logits.masked_fill(~type_mask, MASK_FILL)
    type_dist = Categorical(logits=type_logits)
    type_ = type_logits.argmax(dim=-1) if deterministic else type_dist.sample()
    logp_type = type_dist.log_prob(type_)

    _, _, tgt_mask = _decompose_masks_v12(mask_t, src=src, type_=type_)
    tgt_logits = tgt_logits.masked_fill(~tgt_mask, MASK_FILL)
    tgt_dist = Categorical(logits=tgt_logits)
    tgt = tgt_logits.argmax(dim=-1) if deterministic else tgt_dist.sample()
    logp_tgt = tgt_dist.log_prob(tgt)

    logp = logp_src + logp_type + logp_tgt
    action = _compose_action_v12(src, type_, tgt)
    return (
        action.cpu().numpy(),
        src.cpu().numpy(),
        type_.cpu().numpy(),
        tgt.cpu().numpy(),
        logp.cpu().numpy(),
        value.cpu().numpy(),
    )


@torch.no_grad()
def _act_one_with_diag_v12(net: nn.Module, device: torch.device, obs, mask):
    obs_t  = torch.as_tensor(obs[None, :],  dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool,    device=device)

    body = net.forward_body(obs_t)
    value = net.value(body).squeeze().item()

    (src_mask,) = _decompose_masks_v12(mask_t)
    src_logits = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    src_dist = Categorical(logits=src_logits)
    src = src_dist.sample()
    src_entropy = src_dist.entropy().item()

    type_logits_raw, tgt_logits_raw = net.cond_logits(body, src)
    _, type_mask = _decompose_masks_v12(mask_t, src=src)
    type_logits = type_logits_raw.masked_fill(~type_mask, MASK_FILL)
    type_dist = Categorical(logits=type_logits)
    type_ = type_dist.sample()
    type_entropy = type_dist.entropy().item()

    _, _, tgt_mask = _decompose_masks_v12(mask_t, src=src, type_=type_)
    tgt_logits = tgt_logits_raw.masked_fill(~tgt_mask, MASK_FILL)
    tgt_dist = Categorical(logits=tgt_logits)
    tgt = tgt_dist.sample()
    tgt_entropy = tgt_dist.entropy().item()

    action = _compose_action_v12(src, type_, tgt).item()

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


def _evaluate_v12(net: nn.Module, obs, src, type_, tgt, mask):
    body = net.forward_body(obs)
    value = net.value(body)

    src_mask, type_mask, tgt_mask = _decompose_masks_v12(mask, src=src, type_=type_)

    src_logits  = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    type_logits_raw, tgt_logits_raw = net.cond_logits(body, src)
    type_logits = type_logits_raw.masked_fill(~type_mask, MASK_FILL)
    tgt_logits  = tgt_logits_raw.masked_fill(~tgt_mask, MASK_FILL)

    src_dist  = Categorical(logits=src_logits)
    type_dist = Categorical(logits=type_logits)
    tgt_dist  = Categorical(logits=tgt_logits)

    logp = src_dist.log_prob(src) + type_dist.log_prob(type_) + tgt_dist.log_prob(tgt)
    entropy = src_dist.entropy() + type_dist.entropy() + tgt_dist.entropy()
    return logp, entropy, value


# =============================================================================
# v13 sampling chain: source → target → pct (pct slot 0 = noop, 1.. = sends)
# =============================================================================

def _v13_constants():
    from training.net import NUM_SRC, NUM_TGT, NUM_SEND_PCTS, NUM_PCT_CHOICES, PCT_NOOP, PCT_FIRST_SEND
    return NUM_SRC, NUM_TGT, NUM_SEND_PCTS, NUM_PCT_CHOICES, PCT_NOOP, PCT_FIRST_SEND


def _decompose_masks_v13(
    full_mask: torch.Tensor,
    src: torch.Tensor | None = None,
    tgt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """v13 mask decomposition: src → tgt → pct.

    The env's flat mask is the same shape as v12 (NUM_TYPES * SRC * TGT + 1).
    What changes is the ORDER we factor it. v13 must give:
      src_mask: any building with at least one legal (type, tgt) pair
      tgt_mask|src: any tgt reachable from src by ANY send type
      pct_mask|src,tgt: noop always legal; pct=k+1 legal iff (type=k, src, tgt) is legal
    """
    NUM_SRC, NUM_TGT, NUM_SEND_PCTS, NUM_PCT_CHOICES, PCT_NOOP, PCT_FIRST_SEND = _v13_constants()
    B = full_mask.shape[0]
    # send_3d[b, type_idx, src, tgt] mirrors v12's encoding — env action space is unchanged.
    send_3d = full_mask[:, :ACTION_SPACE_SIZE - 1].reshape(B, NUM_SEND_PCTS, NUM_SRC, NUM_TGT)
    noop_ok = full_mask[:, NOOP_INDEX]

    # src_mask: any legal send anywhere from this src.
    src_legal = send_3d.any(dim=1).any(dim=-1)            # (B, NUM_SRC)
    any_src_legal = src_legal.any(dim=-1, keepdim=True)
    src_mask = src_legal | ~any_src_legal

    if src is None:
        return (src_mask,)

    # send_for_src[b, type, tgt] = mask[b, type, src[b], tgt]
    src_idx = src[:, None, None, None].expand(-1, NUM_SEND_PCTS, 1, NUM_TGT)
    send_for_src = send_3d.gather(2, src_idx).squeeze(2)  # (B, NUM_SEND_PCTS, NUM_TGT)

    # tgt_mask | src: any send type reaches this tgt from src.
    tgt_legal = send_for_src.any(dim=1)                   # (B, NUM_TGT)
    any_tgt_legal = tgt_legal.any(dim=-1, keepdim=True)
    # If no tgt is legal (src is fully noop-ish), allow all so Categorical
    # has a valid distribution; pct mask will force noop anyway.
    tgt_mask = tgt_legal | ~any_tgt_legal

    if tgt is None:
        return src_mask, tgt_mask

    # pct_mask | src, tgt: noop always legal; pct=k+1 legal iff send_for_src[type=k, tgt] is True.
    tgt_idx = tgt[:, None, None].expand(-1, NUM_SEND_PCTS, 1)
    legal_per_type = send_for_src.gather(2, tgt_idx).squeeze(2)   # (B, NUM_SEND_PCTS) bool
    # Build pct_mask: index 0 = noop (always noop_ok), 1.. = sends.
    pct_mask = torch.zeros((B, NUM_PCT_CHOICES), dtype=torch.bool, device=full_mask.device)
    pct_mask[:, PCT_NOOP] = noop_ok.bool()
    pct_mask[:, PCT_FIRST_SEND : PCT_FIRST_SEND + NUM_SEND_PCTS] = legal_per_type
    return src_mask, tgt_mask, pct_mask


def _compose_action_v13(src: torch.Tensor, tgt: torch.Tensor, pct: torch.Tensor) -> torch.Tensor:
    """Pack v13 (src, tgt, pct) into the flat env action index.

    pct ∈ {0=noop, 1=50%, 2=100%}. For sends, type_idx = pct - PCT_FIRST_SEND.
    The env action space is identical to v12 — this is what makes v13-vs-v12
    cross-play work without any env changes.
    """
    NUM_SRC, NUM_TGT, NUM_SEND_PCTS, _, PCT_NOOP, PCT_FIRST_SEND = _v13_constants()
    is_noop = (pct == PCT_NOOP)
    type_send = (pct - PCT_FIRST_SEND).clamp(min=0, max=NUM_SEND_PCTS - 1)
    send_action = type_send * (NUM_SRC * NUM_TGT) + src * NUM_TGT + tgt
    return torch.where(is_noop, torch.full_like(send_action, NOOP_INDEX), send_action)


@torch.no_grad()
def _act_batch_v13(net: nn.Module, device: torch.device, obs, mask, deterministic: bool):
    obs_t  = _to_torch(obs,  torch.float32, device)
    mask_t = _to_torch(mask, torch.bool,    device)

    body = net.forward_body(obs_t)
    value = net.value(body)

    (src_mask,) = _decompose_masks_v13(mask_t)
    src_logits = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    src_dist = Categorical(logits=src_logits)
    src = src_logits.argmax(dim=-1) if deterministic else src_dist.sample()
    logp_src = src_dist.log_prob(src)

    _, tgt_mask = _decompose_masks_v13(mask_t, src=src)
    tgt_logits = net.target_logits(body, src).masked_fill(~tgt_mask, MASK_FILL)
    tgt_dist = Categorical(logits=tgt_logits)
    tgt = tgt_logits.argmax(dim=-1) if deterministic else tgt_dist.sample()
    logp_tgt = tgt_dist.log_prob(tgt)

    _, _, pct_mask = _decompose_masks_v13(mask_t, src=src, tgt=tgt)
    pct_logits = net.pct_logits(body, src, tgt).masked_fill(~pct_mask, MASK_FILL)
    pct_dist = Categorical(logits=pct_logits)
    pct = pct_logits.argmax(dim=-1) if deterministic else pct_dist.sample()
    logp_pct = pct_dist.log_prob(pct)

    logp = logp_src + logp_tgt + logp_pct
    action = _compose_action_v13(src, tgt, pct)
    return (
        action.cpu().numpy(),
        src.cpu().numpy(),
        tgt.cpu().numpy(),    # buffer slot c2 = tgt under v13
        pct.cpu().numpy(),    # buffer slot c3 = pct under v13
        logp.cpu().numpy(),
        value.cpu().numpy(),
    )


@torch.no_grad()
def _act_one_with_diag_v13(net: nn.Module, device: torch.device, obs, mask):
    obs_t  = torch.as_tensor(obs[None, :],  dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool,    device=device)

    body = net.forward_body(obs_t)
    value = net.value(body).squeeze().item()

    (src_mask,) = _decompose_masks_v13(mask_t)
    src_logits = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    src_dist = Categorical(logits=src_logits)
    src = src_dist.sample()
    src_entropy = src_dist.entropy().item()

    _, tgt_mask = _decompose_masks_v13(mask_t, src=src)
    tgt_logits = net.target_logits(body, src).masked_fill(~tgt_mask, MASK_FILL)
    tgt_dist = Categorical(logits=tgt_logits)
    tgt = tgt_dist.sample()
    tgt_entropy = tgt_dist.entropy().item()

    _, _, pct_mask = _decompose_masks_v13(mask_t, src=src, tgt=tgt)
    pct_logits = net.pct_logits(body, src, tgt).masked_fill(~pct_mask, MASK_FILL)
    pct_dist = Categorical(logits=pct_logits)
    pct = pct_dist.sample()
    pct_entropy = pct_dist.entropy().item()

    action = _compose_action_v13(src, tgt, pct).item()

    diag = {
        "value":        float(value),
        "entropy":      float(src_entropy + tgt_entropy + pct_entropy),
        "src_picked":   int(src.item()),
        "tgt_picked":   int(tgt.item()),
        "pct_picked":   int(pct.item()),
        "src_probs":    src_dist.probs.squeeze(0).cpu().numpy(),
        "tgt_probs":    tgt_dist.probs.squeeze(0).cpu().numpy(),
        "pct_probs":    pct_dist.probs.squeeze(0).cpu().numpy(),
        "src_mask":     src_mask.squeeze(0).cpu().numpy(),
        "tgt_mask":     tgt_mask.squeeze(0).cpu().numpy(),
        "pct_mask":     pct_mask.squeeze(0).cpu().numpy(),
    }
    return int(action), diag


def _evaluate_v13(net: nn.Module, obs, src, tgt, pct, mask):
    body = net.forward_body(obs)
    value = net.value(body)

    src_mask, tgt_mask, pct_mask = _decompose_masks_v13(mask, src=src, tgt=tgt)

    src_logits = net.source_logits(body).masked_fill(~src_mask, MASK_FILL)
    tgt_logits = net.target_logits(body, src).masked_fill(~tgt_mask, MASK_FILL)
    pct_logits = net.pct_logits(body, src, tgt).masked_fill(~pct_mask, MASK_FILL)

    src_dist = Categorical(logits=src_logits)
    tgt_dist = Categorical(logits=tgt_logits)
    pct_dist = Categorical(logits=pct_logits)

    logp = src_dist.log_prob(src) + tgt_dist.log_prob(tgt) + pct_dist.log_prob(pct)
    entropy = src_dist.entropy() + tgt_dist.entropy() + pct_dist.entropy()
    return logp, entropy, value


# =============================================================================
# Dispatcher class — what callers see.
# =============================================================================

class PPOAgent:
    """Thin wrapper over an ActorCritic. Dispatches sampling to the right
    chain based on `net_version`.

    `net_version` defaults to "v13" because that's what `from training.net
    import ActorCritic` returns under v13 code (the live class). Loaders that
    instantiate via the registry (opponents.make_neural_opponent_cached,
    match_runner, tournament) pass the saved version explicitly so v12
    checkpoints still play correctly. See V13_PLAN.md backward-compat section.
    """

    def __init__(
        self,
        net: nn.Module,
        device: torch.device | str = "cpu",
        net_version: str = "v13",
    ):
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net_version = net_version
        if net_version not in ("v12", "v13"):
            raise ValueError(f"unknown net_version {net_version!r}")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def act_batch(self, obs, mask, deterministic: bool = False):
        """Factored sampling. Returns (action, c1, c2, c3, logp, value).

        v12: (c1, c2, c3) = (src, type, tgt)
        v13: (c1, c2, c3) = (src, tgt, pct)

        The trainer's rollout buffer stores (c1, c2, c3) opaquely and passes
        them back to evaluate() in the same positional order.
        """
        if self.net_version == "v13":
            return _act_batch_v13(self.net, self.device, obs, mask, deterministic)
        return _act_batch_v12(self.net, self.device, obs, mask, deterministic)

    # ------------------------------------------------------------------
    # Single-decision diagnostics — for replay/introspection.
    # ------------------------------------------------------------------

    def act_one_with_diag(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, dict]:
        """Batch-of-1 decision with the full policy breakdown exposed.

        diag dict keys differ per version:
          v12: src_*, type_*, tgt_*
          v13: src_*, tgt_*, pct_*
        Replay viewers should branch on the keys present, or read net_version
        from the run's config.
        """
        if self.net_version == "v13":
            return _act_one_with_diag_v13(self.net, self.device, obs, mask)
        return _act_one_with_diag_v12(self.net, self.device, obs, mask)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def evaluate(self, obs, c1, c2, c3, mask):
        """Recompute log-prob / entropy / value for stored (c1, c2, c3).

        v12: (c1, c2, c3) = (src, type, tgt)
        v13: (c1, c2, c3) = (src, tgt, pct)
        """
        if self.net_version == "v13":
            return _evaluate_v13(self.net, obs, c1, c2, c3, mask)
        return _evaluate_v12(self.net, obs, c1, c2, c3, mask)
