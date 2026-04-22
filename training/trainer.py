"""Minimal PPO trainer for the Phase 2 smoke run.

Scope: single env, flat rollout buffer, GAE, clipped PPO update. No vec env,
no checkpointing, no schedulers, no self-play pool — those land after the
loop is proven end-to-end.

Usage:
    env = MushroomEnv(seed=0)
    net = ActorCritic()
    agent = PPOAgent(net)
    trainer = PPOTrainer(env, agent)
    for _ in range(n_updates):
        metrics = trainer.update()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from sim.actions import ACTION_SPACE_SIZE
from sim.envs import MushroomEnv
from training.agent import PPOAgent
from training.encoder import OBS_DIM, encode_obs


@dataclass
class PPOConfig:
    rollout_steps: int = 512
    update_epochs: int = 4
    minibatch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(
        self,
        env: MushroomEnv,
        agent: PPOAgent,
        config: PPOConfig | None = None,
    ):
        self.env = env
        self.agent = agent
        self.cfg = config or PPOConfig()
        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.cfg.lr)
        self.device = self.agent.device

        # Current env state (persisted across updates for seamless rollouts).
        obs_dict, _ = self.env.reset()
        self._obs = encode_obs(obs_dict)
        self._mask = obs_dict["action_mask"].copy()
        self._episode_return = 0.0
        self._episode_length = 0
        self._completed_episodes: list[tuple[float, int, bool]] = []
        # (return, length, won)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict:
        T = self.cfg.rollout_steps
        A = ACTION_SPACE_SIZE

        obs_buf    = np.zeros((T, OBS_DIM), dtype=np.float32)
        mask_buf   = np.zeros((T, A),       dtype=bool)
        act_buf    = np.zeros((T,),         dtype=np.int64)
        logp_buf   = np.zeros((T,),         dtype=np.float32)
        val_buf    = np.zeros((T,),         dtype=np.float32)
        rew_buf    = np.zeros((T,),         dtype=np.float32)
        done_buf   = np.zeros((T,),         dtype=np.float32)

        for t in range(T):
            action, logprob, value = self.agent.act(self._obs, self._mask)

            obs_buf[t]  = self._obs
            mask_buf[t] = self._mask
            act_buf[t]  = action
            logp_buf[t] = logprob
            val_buf[t]  = value

            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            rew_buf[t]  = reward
            done_buf[t] = float(done)

            self._episode_return += reward
            self._episode_length += 1

            if done:
                won = info.get("phase", 0) == 1   # C.PHASE_P1_WINS
                self._completed_episodes.append(
                    (self._episode_return, self._episode_length, won)
                )
                obs_dict, _ = self.env.reset()
                self._episode_return = 0.0
                self._episode_length = 0

            self._obs = encode_obs(obs_dict)
            self._mask = obs_dict["action_mask"].copy()

        # Bootstrap value for truncated rollout.
        with torch.no_grad():
            bootstrap = self.agent.net(
                torch.as_tensor(self._obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            )[1].item()

        adv, ret = self._compute_gae(rew_buf, val_buf, done_buf, bootstrap)
        return {
            "obs":       obs_buf,
            "mask":      mask_buf,
            "action":    act_buf,
            "logprob":   logp_buf,
            "value":     val_buf,
            "reward":    rew_buf,
            "done":      done_buf,
            "advantage": adv,
            "return":    ret,
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        bootstrap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            next_value = bootstrap if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last
            adv[t] = last
        ret = adv + values
        return adv, ret

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        batch = self.collect_rollout()
        T = len(batch["obs"])

        obs_t     = torch.as_tensor(batch["obs"],       device=self.device)
        mask_t    = torch.as_tensor(batch["mask"],      device=self.device)
        action_t  = torch.as_tensor(batch["action"],    device=self.device)
        oldlogp_t = torch.as_tensor(batch["logprob"],   device=self.device)
        adv_t     = torch.as_tensor(batch["advantage"], device=self.device)
        ret_t     = torch.as_tensor(batch["return"],    device=self.device)

        # Advantage normalization (standard PPO trick; stabilizes updates).
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        pol_losses, val_losses, ent_losses, approx_kls, clip_fracs = [], [], [], [], []
        idx = np.arange(T)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.cfg.minibatch_size):
                mb = idx[start : start + self.cfg.minibatch_size]
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
            "policy_loss":    float(np.mean(pol_losses)),
            "value_loss":     float(np.mean(val_losses)),
            "entropy_loss":   float(np.mean(ent_losses)),
            "approx_kl":      float(np.mean(approx_kls)),
            "clip_fraction":  float(np.mean(clip_fracs)),
            "mean_reward":    float(batch["reward"].mean()),
        }
        ep = self._completed_episodes
        if ep:
            metrics["episodes_completed"] = len(ep)
            metrics["mean_episode_return"] = float(np.mean([e[0] for e in ep]))
            metrics["mean_episode_length"] = float(np.mean([e[1] for e in ep]))
            metrics["win_rate"] = float(np.mean([e[2] for e in ep]))
            self._completed_episodes = []
        return metrics
