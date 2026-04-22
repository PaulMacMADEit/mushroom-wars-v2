"""PPO trainer with vectorized rollouts.

Owns its own gymnasium vector env and runs batched NN forwards so GPU
kernel-launch overhead is amortized across envs. With single-env rollouts,
MPS/CUDA are *slower* than CPU at batch=1 (kernel dispatch dominates); with
~32-128 envs the GPU paths cross over and pull ahead (see
`scripts/bench_vec_games.py` and phase 2 polish notes).

Usage:
    agent = PPOAgent(ActorCritic(), device='cuda')
    trainer = PPOTrainer(agent, PPOConfig(n_envs=64), seed=0)
    for _ in range(n_updates):
        metrics = trainer.update()

Minibatch size note: with (T * N) samples per rollout, `minibatch_size`
should be ≥ N to keep minibatches well-populated across envs. Default 512
works well for n_envs up to 128.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from sim.actions import ACTION_SPACE_SIZE
from sim.envs import make_env
from training.agent import PPOAgent
from training.encoder import OBS_DIM, encode_obs
from training.obs_norm import RunningNorm


@dataclass
class PPOConfig:
    # Rollout
    n_envs:         int = 32            # parallel envs; GPU wins at ≥32-64
    vec_mode:       str = "async"       # "async" (procs) or "sync" (in-process)
    rollout_steps:  int = 128           # per-env steps → total samples = n_envs * rollout_steps
    # Update
    update_epochs:  int = 4
    minibatch_size: int = 512
    lr:             float = 3e-4
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_range:     float = 0.2
    value_coef:     float = 0.5
    entropy_coef:   float = 0.01
    max_grad_norm:  float = 0.5
    # Obs normalization
    normalize_obs:  bool = True         # Welford running mean/std; see training/obs_norm.py
    obs_clip:       float = 10.0        # clip normalized obs; None to disable


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        config: PPOConfig | None = None,
        seed: int = 0,
        opponent_name: str = "random_legal",
    ):
        self.agent = agent
        self.cfg = config or PPOConfig()
        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.cfg.lr)
        self.device = self.agent.device

        # Build the vec env. Each sub-env gets its own seed so opponents
        # don't move in lockstep.
        factories = [
            make_env(seed=seed + i, opponent_name=opponent_name)
            for i in range(self.cfg.n_envs)
        ]
        if self.cfg.vec_mode == "async":
            self.vec = gym.vector.AsyncVectorEnv(factories, shared_memory=False)
        elif self.cfg.vec_mode == "sync":
            self.vec = gym.vector.SyncVectorEnv(factories)
        else:
            raise ValueError(f"unknown vec_mode: {self.cfg.vec_mode!r}")

        N = self.cfg.n_envs
        obs_batch, _ = self.vec.reset(seed=seed)
        self.obs_norm = RunningNorm(OBS_DIM) if self.cfg.normalize_obs else None
        self._obs_raw, self._masks = self._encode_batch(obs_batch)
        if self.obs_norm is not None:
            self.obs_norm.update(self._obs_raw)
        self._obs = self._apply_norm(self._obs_raw)

        # Per-env running counters. Running returns/lengths persist across
        # rollouts — an episode that doesn't finish in this rollout finishes
        # in the next.
        self._ep_return = np.zeros(N, dtype=np.float32)
        self._ep_length = np.zeros(N, dtype=np.int64)
        # (return, length, won_proxy)
        self._completed_episodes: list[tuple[float, int, bool]] = []

    # ------------------------------------------------------------------
    # Obs encoding — vec env returns a dict of stacked arrays; turn it into
    # per-sample flat float vectors + stacked masks.
    # ------------------------------------------------------------------

    def _encode_batch(self, obs_batch: dict) -> tuple[np.ndarray, np.ndarray]:
        N = self.cfg.n_envs
        obs_out = np.empty((N, OBS_DIM), dtype=np.float32)
        mask_shape = obs_batch["action_mask"].shape[1]
        masks_out = np.empty((N, mask_shape), dtype=bool)
        for i in range(N):
            single = {k: v[i] for k, v in obs_batch.items()}
            obs_out[i] = encode_obs(single)
            masks_out[i] = single["action_mask"]
        return obs_out, masks_out

    def _apply_norm(self, obs_raw: np.ndarray) -> np.ndarray:
        if self.obs_norm is None:
            return obs_raw
        return self.obs_norm.normalize(obs_raw, clip=self.cfg.obs_clip)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict:
        T, N = self.cfg.rollout_steps, self.cfg.n_envs
        A = ACTION_SPACE_SIZE

        obs_buf  = np.zeros((T, N, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((T, N, A),       dtype=bool)
        act_buf  = np.zeros((T, N),          dtype=np.int64)
        logp_buf = np.zeros((T, N),          dtype=np.float32)
        val_buf  = np.zeros((T, N),          dtype=np.float32)
        rew_buf  = np.zeros((T, N),          dtype=np.float32)
        done_buf = np.zeros((T, N),          dtype=np.float32)

        for t in range(T):
            actions, logps, values = self.agent.act_batch(self._obs, self._masks)
            obs_buf[t]  = self._obs             # stored obs is normalized
            mask_buf[t] = self._masks
            act_buf[t]  = actions
            logp_buf[t] = logps
            val_buf[t]  = values

            next_obs_batch, rewards, terminated, truncated, infos = self.vec.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            done = np.logical_or(terminated, truncated)
            rew_buf[t]  = rewards
            done_buf[t] = done.astype(np.float32)

            # Per-env episode bookkeeping. `won` proxy: REWARD_WIN=+1 and
            # REWARD_LOSE=-1 dominate the per-episode return; any total >0.5
            # is effectively a win (captures alone can't cross that threshold
            # without the win bonus). Good enough for the training metric.
            self._ep_return += rewards
            self._ep_length += 1
            for i in range(N):
                if done[i]:
                    won = bool(self._ep_return[i] > 0.5)
                    self._completed_episodes.append((
                        float(self._ep_return[i]),
                        int(self._ep_length[i]),
                        won,
                    ))
                    self._ep_return[i] = 0.0
                    self._ep_length[i] = 0

            self._obs_raw, self._masks = self._encode_batch(next_obs_batch)
            if self.obs_norm is not None:
                # Update running stats with raw obs, then normalize for the
                # next forward pass + the rollout buffer.
                self.obs_norm.update(self._obs_raw)
            self._obs = self._apply_norm(self._obs_raw)

        # Bootstrap value for each env (for GAE on truncated rollout).
        # Use normalized obs — that's what the net was trained on.
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            _, val_t = self.agent.net(obs_t)
            bootstrap = val_t.cpu().numpy()

        adv, ret = self._compute_gae(rew_buf, val_buf, done_buf, bootstrap)

        # Flatten (T, N, …) → (T*N, …). PPO doesn't care about per-env
        # ordering once advantages are computed.
        flat = T * N
        return {
            "obs":       obs_buf.reshape(flat, OBS_DIM),
            "mask":      mask_buf.reshape(flat, A),
            "action":    act_buf.reshape(flat),
            "logprob":   logp_buf.reshape(flat),
            "value":     val_buf.reshape(flat),
            "reward":    rew_buf.reshape(flat),
            "done":      done_buf.reshape(flat),
            "advantage": adv.reshape(flat),
            "return":    ret.reshape(flat),
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,      # (T, N)
        values:  np.ndarray,      # (T, N)
        dones:   np.ndarray,      # (T, N)
        bootstrap: np.ndarray,    # (N,)
    ) -> tuple[np.ndarray, np.ndarray]:
        T, N = rewards.shape
        adv = np.zeros_like(rewards)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_v = bootstrap if t == T - 1 else values[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_v * nonterm - values[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + values
        return adv, ret

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        batch = self.collect_rollout()
        B = len(batch["obs"])

        obs_t     = torch.as_tensor(batch["obs"],       device=self.device)
        mask_t    = torch.as_tensor(batch["mask"],      device=self.device)
        action_t  = torch.as_tensor(batch["action"],    device=self.device)
        oldlogp_t = torch.as_tensor(batch["logprob"],   device=self.device)
        adv_t     = torch.as_tensor(batch["advantage"], device=self.device)
        ret_t     = torch.as_tensor(batch["return"],    device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        pol_losses, val_losses, ent_losses, approx_kls, clip_fracs = [], [], [], [], []
        idx = np.arange(B)
        mb_size = min(self.cfg.minibatch_size, B)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, mb_size):
                mb = idx[start : start + mb_size]
                mb_t = torch.as_tensor(mb, device=self.device)
                new_logp, entropy, new_value = self.agent.evaluate(
                    obs_t[mb_t], action_t[mb_t], mask_t[mb_t]
                )

                ratio = torch.exp(new_logp - oldlogp_t[mb_t])
                mb_adv = adv_t[mb_t]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(new_value, ret_t[mb_t])
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.net.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (oldlogp_t[mb_t] - new_logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item()

                pol_losses.append(policy_loss.item())
                val_losses.append(value_loss.item())
                ent_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        metrics = {
            "policy_loss":   float(np.mean(pol_losses)),
            "value_loss":    float(np.mean(val_losses)),
            "entropy_loss":  float(np.mean(ent_losses)),
            "approx_kl":     float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "mean_reward":   float(batch["reward"].mean()),
        }
        ep = self._completed_episodes
        if ep:
            metrics["episodes_completed"] = len(ep)
            metrics["mean_episode_return"] = float(np.mean([e[0] for e in ep]))
            metrics["mean_episode_length"] = float(np.mean([e[1] for e in ep]))
            metrics["win_rate"]            = float(np.mean([e[2] for e in ep]))
            self._completed_episodes = []
        return metrics

    def close(self):
        """Explicit teardown. AsyncVectorEnv leaks subprocesses if not closed."""
        try:
            self.vec.close()
        except Exception:
            pass
