"""
Fused PPO rollout collector — Phase C of FUSED_ROLLOUT_PLAN.

Drop-in replacement for `PPOTrainer.collect_rollout()` when:
  - SIM_BACKEND=jax (the only backend that has step_chunk)
  - cfg.fused_rollout=True
  - cfg.self_play=False (per-env neural opponents not yet supported here)

Architecture: each "rollout step" represents `cfg.action_repeat` env ticks.
The agent decides once per chunk; the env runs the chosen action on tick 0
and NOOP on ticks 1..K-1 inside one fused XLA dispatch (`step_chunk`).
Rewards over the K ticks sum into one stored reward per chunk per env.

Returns the same dict shape as the per-tick collector so the PPO update
phase is unchanged. With cfg.action_repeat=1 the dict is byte-identical
to the per-tick path under the same seed (parity test in
tests/test_fused_rollout.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch

from sim import config as C
from sim.actions import (
    ACTION_SPACE_SIZE,
    NOOP_INDEX,
    SLOTS_SQ,
    compute_mask_batched,
)
from sim.engine_jax import (
    ACTION_DIM,
    ACTION_KIND_NOOP,
    ACTION_KIND_SEND,
)
from sim.envs.opponents import random_legal_opponent_batched
from training.encoder import OBS_DIM
from training.encoder_jax import encode_obs_batched_jit

if TYPE_CHECKING:
    from sim.envs.jax_vec_env import JaxVecEnv
    from training.agent import PPOAgent
    from training.obs_norm import RunningNorm
    from training.trainer import PPOConfig


def collect_rollout_fused(
    agent: "PPOAgent",
    vec_env: "JaxVecEnv",
    cfg: "PPOConfig",
    obs_norm: "RunningNorm | None",
    obs_clip: float,
    bookkeeping: dict,
    rng: np.random.Generator,
    opponent_name: str,
) -> dict:
    """Collect one PPO rollout under the fused path.

    Inputs:
      agent        — PPOAgent with .net and .device set up.
      vec_env      — JaxVecEnv (the inner sim, not the adapter). Holds
                     `cfg.n_envs` games as one batched StateJax.
      cfg          — PPOConfig. `rollout_steps` is the number of agent
                     decisions; effective env ticks = rollout_steps × cfg.action_repeat.
      obs_norm     — RunningNorm or None.
      obs_clip     — clip threshold for normalized obs.
      bookkeeping  — dict carrying per-env episode counters across rollouts
                     (mutated in place):
                        ep_return: (n_envs,) float32
                        ep_length: (n_envs,) int64
                        completed_episodes: list[(return, length, won_proxy)]
                        last_obs_dev: jnp.ndarray (n_envs, OBS_DIM) — encoded
                                       obs from end of previous rollout, or None.
                        last_mask_dev: jnp.ndarray (n_envs, ACTION_SPACE_SIZE).
      rng          — numpy RNG for the opponent. Threaded across rollouts so
                     opponent randomness is deterministic under cfg.seed.
      opponent_name — "random_legal" or "noop". Neural opponents NotImplemented.

    Returns the same dict shape as `PPOTrainer.collect_rollout`:
      {obs, mask, src, type, tgt, logprob, value, reward, done, advantage, return}
      each (T*N, …) flat where T = cfg.rollout_steps, N = cfg.n_envs.
    """
    if opponent_name not in ("random_legal", "noop"):
        raise NotImplementedError(
            f"fused rollout supports random_legal/noop opponents only; "
            f"got {opponent_name!r}"
        )

    T = cfg.rollout_steps
    N = cfg.n_envs
    K = max(1, int(getattr(cfg, "action_repeat", 1)))

    # Pre-allocated rollout buffers.
    obs_buf  = np.zeros((T, N, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((T, N, ACTION_SPACE_SIZE), dtype=bool)
    src_buf  = np.zeros((T, N), dtype=np.int64)
    type_buf = np.zeros((T, N), dtype=np.int64)
    tgt_buf  = np.zeros((T, N), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    val_buf  = np.zeros((T, N), dtype=np.float32)
    rew_buf  = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)

    # Pull current obs+masks from device (set by the previous rollout or
    # rebuilt below). We keep p1+p2 masks together because the rollout
    # needs P1 mask for the agent and P2 mask for the opponent on the
    # SAME state.
    obs_dev    = bookkeeping["last_obs_dev"]
    p1_mask_h  = bookkeeping["last_p1_mask"]   # numpy (N, A) bool or None
    p2_mask_h  = bookkeeping["last_p2_mask"]   # numpy (N, A) bool or None
    if obs_dev is None:
        obs_dev, p1_mask_h, p2_mask_h = _encode_and_masks(vec_env)

    # If RunningNorm is active, lift its (mean, std) to device once per
    # rollout. This trades a tiny lag (stats update at rollout boundaries
    # rather than per-tick) for keeping obs on device through the whole
    # rollout — necessary to actually saturate the GPU.
    norm_mean_dev, norm_std_dev = _norm_stats_to_device(obs_norm) if obs_norm is not None else (None, None)

    # On-device normalize. obs_dev stays on JAX device; agent.act_batch
    # consumes it via DLPack with no host roundtrip.
    obs_normed_dev = _normalize_on_device(obs_dev, norm_mean_dev, norm_std_dev, obs_clip)
    mask_dev = jnp.asarray(p1_mask_h)

    ep_return = bookkeeping["ep_return"]
    ep_length = bookkeeping["ep_length"]
    completed = bookkeeping["completed_episodes"]

    # We still need ONE host obs copy per rollout to feed RunningNorm.update.
    # Take a sample at the start (cheap; one per rollout) so stats keep moving.
    # The numpy buffer for the rollout-return obs is filled in batched at the
    # end so the inner loop has zero numpy-side obs allocs.
    obs_first_host = np.array(obs_normed_dev, dtype=np.float32)
    obs_buf[0] = obs_first_host
    if obs_norm is not None:
        obs_norm.update(np.array(obs_dev, dtype=np.float32))

    # Cache device-side obs/mask per rollout step so we can dump them to the
    # numpy rollout buffers in one go at the end (avoids per-tick host roundtrip
    # for the buffer write).
    obs_dev_per_t  = [None] * T
    mask_dev_per_t = [None] * T
    obs_dev_per_t[0]  = obs_normed_dev
    mask_dev_per_t[0] = mask_dev

    for t in range(T):
        # Agent picks one action per env. Inputs stay on device via DLPack.
        actions, srcs, types, tgts, logps, values = agent.act_batch(
            obs_normed_dev, mask_dev,
        )

        # P2 mask is still on host (compute_mask_batched is numpy).
        # Pack action batch on host.
        a_batch = _pack_action_batch_with_p2_mask(
            actions, p2_mask_h, opponent_name, rng, N,
        )

        # Run K env ticks fused.
        result = vec_env.step_chunk(a_batch, K=K)
        rewards    = result["rewards"]
        terminated = result["dones"]

        # Encode + masks for the post-chunk state.
        next_obs_dev, next_p1_mask, next_p2_mask = _encode_and_masks(vec_env)

        # Per-step buffers (host-side mostly; obs/mask deferred).
        src_buf[t]  = srcs
        type_buf[t] = types
        tgt_buf[t]  = tgts
        logp_buf[t] = logps
        val_buf[t]  = values
        rew_buf[t]  = rewards
        done_buf[t] = terminated.astype(np.float32)

        # Per-env episode bookkeeping.
        ep_return += rewards
        ep_length += K
        for i in range(N):
            if terminated[i]:
                won = bool(ep_return[i] > 0.5)
                completed.append((float(ep_return[i]), int(ep_length[i]), won))
                ep_return[i] = 0.0
                ep_length[i] = 0

        # Roll forward.
        obs_dev   = next_obs_dev
        p1_mask_h = next_p1_mask
        p2_mask_h = next_p2_mask
        mask_dev  = jnp.asarray(p1_mask_h)
        obs_normed_dev = _normalize_on_device(obs_dev, norm_mean_dev, norm_std_dev, obs_clip)

        if t + 1 < T:
            obs_dev_per_t[t + 1]  = obs_normed_dev
            mask_dev_per_t[t + 1] = mask_dev

    # Dump per-step device-side obs/masks to the host buffers in batched form
    # (one device->host transfer for each instead of T per-step transfers).
    # Skip t=0 since obs_buf[0] was filled above before the loop.
    for t in range(1, T):
        if obs_dev_per_t[t] is not None:
            obs_buf[t] = np.array(obs_dev_per_t[t], dtype=np.float32)
        if mask_dev_per_t[t] is not None:
            mask_buf[t] = np.array(mask_dev_per_t[t], dtype=bool)
    # mask_buf[0] still needs filling.
    mask_buf[0] = np.array(mask_dev_per_t[0], dtype=bool)

    # Bootstrap value for GAE on the truncated rollout — keep obs on device.
    obs_t  = _torch_from_jax(obs_normed_dev, agent.device, torch.float32)
    with torch.no_grad():
        body_t = agent.net.forward_body(obs_t)
        bootstrap = agent.net.value(body_t).cpu().numpy()

    adv, ret = _compute_gae(
        rew_buf, val_buf, done_buf, bootstrap, cfg.gamma, cfg.gae_lambda,
    )

    # Stash post-rollout obs + masks so the next call picks up where we
    # left off.
    bookkeeping["last_obs_dev"]  = obs_dev
    bookkeeping["last_p1_mask"]  = p1_mask_h
    bookkeeping["last_p2_mask"]  = p2_mask_h

    flat = T * N
    return {
        "obs":       obs_buf.reshape(flat, OBS_DIM),
        "mask":      mask_buf.reshape(flat, ACTION_SPACE_SIZE),
        "src":       src_buf.reshape(flat),
        "type":      type_buf.reshape(flat),
        "tgt":       tgt_buf.reshape(flat),
        "logprob":   logp_buf.reshape(flat),
        "value":     val_buf.reshape(flat),
        "reward":    rew_buf.reshape(flat),
        "done":      done_buf.reshape(flat),
        "advantage": adv.reshape(flat),
        "return":    ret.reshape(flat),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_stats_to_device(obs_norm) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lift RunningNorm's mean/std to device (Phase E zero-copy path).

    Done once per rollout — stats lag by T*K env ticks but at typical
    n_envs * T this is a tiny fraction of the running average's window.
    """
    mean = jnp.asarray(obs_norm.mean.astype(np.float32))
    std  = jnp.asarray(obs_norm.std.astype(np.float32))
    return mean, std


def _normalize_on_device(
    obs_dev: jnp.ndarray,
    mean: jnp.ndarray | None,
    std:  jnp.ndarray | None,
    clip: float,
) -> jnp.ndarray:
    """(obs - mean) / std clipped, all on device. No-op when mean is None."""
    if mean is None:
        return obs_dev
    out = (obs_dev - mean) / std
    return jnp.clip(out, -clip, clip)


def _torch_from_jax(arr: jnp.ndarray, device, dtype):
    """JAX -> torch via DLPack with a numpy fallback."""
    try:
        t = torch.from_dlpack(arr)
        if t.device != device:
            t = t.to(device)
        if t.dtype != dtype:
            t = t.to(dtype)
        return t
    except Exception:
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


def _encode_and_masks(vec_env) -> tuple[jnp.ndarray, np.ndarray, np.ndarray]:
    """Encode current state + compute P1 and P2 masks in one bulk pull.

    Returns (encoded_obs_jnp, p1_mask_np, p2_mask_np). Encoded obs stays on
    device for the agent's next forward; masks are numpy because the agent
    converts to torch host-side anyway and the opponent reads numpy.
    """
    state = vec_env.state
    obs_dev = encode_obs_batched_jit(state)

    # Mask uses 4 fields; pull them once.
    b_alive    = np.asarray(state.buildings_alive)
    b_owner    = np.asarray(state.buildings_owner)
    b_garrison = np.asarray(state.buildings_garrison)
    g_alive    = np.asarray(state.groups_alive)
    p1_mask = compute_mask_batched(b_alive, b_owner, b_garrison, g_alive, C.OWNER_P1)
    p2_mask = compute_mask_batched(b_alive, b_owner, b_garrison, g_alive, C.OWNER_P2)
    return obs_dev, p1_mask, p2_mask


def _pack_action_batch_with_p2_mask(
    p1_actions_flat: np.ndarray,   # (N,) int64
    p2_mask: np.ndarray,           # (N, A) bool
    opponent_name: str,
    rng: np.random.Generator,
    n_envs: int,
) -> np.ndarray:
    """Same as `_pack_action_batch` but with the P2 mask provided so we can
    do random_legal_opponent_batched in one numpy call."""
    a_batch = np.zeros((n_envs, 2, ACTION_DIM), dtype=np.int32)

    _decode_into_slot(p1_actions_flat, a_batch[:, 0, :])

    if opponent_name == "noop":
        a_batch[:, 1, 0] = ACTION_KIND_NOOP
        return a_batch

    p2_actions = random_legal_opponent_batched(p2_mask, rng)  # (N,) int64
    _decode_into_slot(p2_actions, a_batch[:, 1, :])
    return a_batch


def _decode_into_slot(flat: np.ndarray, out: np.ndarray) -> None:
    """Vectorised decode of (N,) flat action indices into (N, 4) [kind,type,src,tgt].

    Mirrors `sim.actions.decode` for the whole batch.
    """
    flat = np.asarray(flat, dtype=np.int64)
    is_noop = flat == NOOP_INDEX
    type_idx = (flat // SLOTS_SQ).astype(np.int32)
    rem      = (flat %  SLOTS_SQ).astype(np.int32)
    src_idx  = (rem // C.MAX_BUILDING_SLOTS).astype(np.int32)
    tgt_idx  = (rem %  C.MAX_BUILDING_SLOTS).astype(np.int32)

    out[:, 0] = np.where(is_noop, ACTION_KIND_NOOP, ACTION_KIND_SEND)
    out[:, 1] = np.where(is_noop, 0, type_idx)
    out[:, 2] = np.where(is_noop, 0, src_idx)
    out[:, 3] = np.where(is_noop, 0, tgt_idx)


def _compute_gae(
    rewards: np.ndarray,      # (T, N)
    values:  np.ndarray,      # (T, N)
    dones:   np.ndarray,      # (T, N)
    bootstrap: np.ndarray,    # (N,)
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE backward scan. Numpy version is fast enough at these sizes
    (T*N ~ 64*1024 = 64k); a JAX scan would buy <1ms."""
    T, N = rewards.shape
    adv = np.zeros_like(rewards)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = bootstrap if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    ret = adv + values
    return adv, ret
