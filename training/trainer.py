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
from training.pool import OpponentPool


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
    # Self-play
    self_play:            bool  = False  # if False, opponent is random_legal forever
    snapshot_every:       int   = 10     # register a new pool entry every N updates
    pool_max_size:        int   = 20     # evict oldest beyond this
    latest_bias:          float = 0.8    # P(sample latest snapshot) per env
    initial_opponent:     str   = "random_legal"  # before first snapshot registers
    # Cross-lineage: include external leaderboard top-K as opponents.
    # P(env uses leaderboard opp) per snapshot = leaderboard_bias. Remainder
    # falls through to the self-play pool (latest_bias rules within).
    # Default 0.3 gives roughly: 30% leaderboard, 56% latest own, 14% older own.
    leaderboard_bias:     float = 0.3
    leaderboard_top_k:    int   = 10
    # Level
    # Static name (e.g. "crossroads_6") or dynamic "random_<min>_<max>".
    # Dynamic levels regenerate per reset so training sees varied geometry.
    level_name:           str   = "crossroads_6"


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        config: PPOConfig | None = None,
        seed: int = 0,
        opponent_name: str = "random_legal",
        pool_root: str | None = None,
        leaderboard_paths: list[tuple] | None = None,
    ):
        self.agent = agent
        self.cfg = config or PPOConfig()
        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.cfg.lr)
        self.device = self.agent.device
        self.seed = seed
        self._update_count = 0
        self._rng = np.random.default_rng(seed)

        # Self-play pool. Only populated when cfg.self_play is True.
        self.pool: OpponentPool | None = None
        if self.cfg.self_play:
            self.pool = OpponentPool(root=pool_root, max_size=self.cfg.pool_max_size)

        # Cross-lineage leaderboard opponents. Each entry: (weights_path, obs_norm_path|None).
        # Worker downloads top-K Elo snapshots before constructing us; when
        # leaderboard_bias > 0 we sample from this list for a fraction of envs.
        self._leaderboard: list[tuple] = list(leaderboard_paths or [])

        # Initial opponent: simple name for the very first rollout. After
        # the first snapshot registers (every cfg.snapshot_every updates),
        # the vec env gets rebuilt with neural opponents drawn from the pool.
        self._initial_opponent_name = (
            opponent_name if not self.cfg.self_play else self.cfg.initial_opponent
        )

        self.obs_norm = RunningNorm(OBS_DIM) if self.cfg.normalize_obs else None
        self._build_vec(opponent_specs=None)

        # Per-env running counters. Persist across rollouts.
        self._ep_return = np.zeros(self.cfg.n_envs, dtype=np.float32)
        self._ep_length = np.zeros(self.cfg.n_envs, dtype=np.int64)
        # (return, length, won_proxy)
        self._completed_episodes: list[tuple[float, int, bool]] = []

    # ------------------------------------------------------------------
    # Vec-env (re)build
    # ------------------------------------------------------------------

    def _build_vec(self, opponent_specs: list[dict] | None) -> None:
        """Create (or re-create) the vec env. Each sub-env gets its own seed.

        If `opponent_specs` is None, every env uses `self._initial_opponent_name`
        (stateless). Otherwise, each entry is a dict passed to `make_env` as
        `opponent_kwargs` with `opponent_name="neural"`.
        """
        # Tear down any previous vec env — AsyncVectorEnv leaks subprocs.
        if hasattr(self, "vec"):
            try:
                self.vec.close()
            except Exception:
                pass

        N = self.cfg.n_envs
        factories = []
        for i in range(N):
            if opponent_specs is None:
                factories.append(make_env(
                    seed=self.seed + i,
                    level_name=self.cfg.level_name,
                    opponent_name=self._initial_opponent_name,
                ))
            else:
                factories.append(make_env(
                    seed=self.seed + i,
                    level_name=self.cfg.level_name,
                    opponent_name="neural",
                    opponent_kwargs=opponent_specs[i],
                ))

        if self.cfg.vec_mode == "async":
            # Context choice: Linux's default is `fork`. Fork is fast (~0.5s
            # to spawn 16 workers) but breaks after CUDA init in the parent —
            # forked children inherit broken CUDA state and deadlock when
            # they import torch. That only matters if subprocs touch torch,
            # which they do exactly when self-play's neural opponent is loaded.
            # So: use `spawn` for self-play (safe, +15s startup), fork
            # elsewhere (fast).
            ctx = "spawn" if self.cfg.self_play else None
            self.vec = gym.vector.AsyncVectorEnv(
                factories, shared_memory=False, context=ctx,
            )
        elif self.cfg.vec_mode == "sync":
            self.vec = gym.vector.SyncVectorEnv(factories)
        else:
            raise ValueError(f"unknown vec_mode: {self.cfg.vec_mode!r}")

        obs_batch, _ = self.vec.reset(seed=self.seed)
        self._obs_raw, self._masks = self._encode_batch(obs_batch)
        if self.obs_norm is not None:
            self.obs_norm.update(self._obs_raw)
        self._obs = self._apply_norm(self._obs_raw)

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

        obs_buf   = np.zeros((T, N, OBS_DIM), dtype=np.float32)
        mask_buf  = np.zeros((T, N, A),       dtype=bool)
        # Chained heads store three sub-action indices; the flat `action` is
        # only used for env.step() and isn't needed in the PPO update.
        src_buf   = np.zeros((T, N),          dtype=np.int64)
        type_buf  = np.zeros((T, N),          dtype=np.int64)
        tgt_buf   = np.zeros((T, N),          dtype=np.int64)
        logp_buf  = np.zeros((T, N),          dtype=np.float32)
        val_buf   = np.zeros((T, N),          dtype=np.float32)
        rew_buf   = np.zeros((T, N),          dtype=np.float32)
        done_buf  = np.zeros((T, N),          dtype=np.float32)

        for t in range(T):
            actions, srcs, types, tgts, logps, values = self.agent.act_batch(
                self._obs, self._masks,
            )
            obs_buf[t]  = self._obs
            mask_buf[t] = self._masks
            src_buf[t]  = srcs
            type_buf[t] = types
            tgt_buf[t]  = tgts
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
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            body_t = self.agent.net.forward_body(obs_t)
            bootstrap = self.agent.net.value(body_t).cpu().numpy()

        adv, ret = self._compute_gae(rew_buf, val_buf, done_buf, bootstrap)

        flat = T * N
        return {
            "obs":       obs_buf.reshape(flat, OBS_DIM),
            "mask":      mask_buf.reshape(flat, A),
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
        src_t     = torch.as_tensor(batch["src"],       device=self.device)
        type_t    = torch.as_tensor(batch["type"],      device=self.device)
        tgt_t     = torch.as_tensor(batch["tgt"],       device=self.device)
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
                    obs_t[mb_t], src_t[mb_t], type_t[mb_t], tgt_t[mb_t], mask_t[mb_t]
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

        self._update_count += 1

        # Self-play refresh: register a snapshot at the cadence and rebuild
        # the vec env so sub-envs load the fresh opponents from disk.
        if self.cfg.self_play and self.pool is not None:
            if self._update_count % self.cfg.snapshot_every == 0:
                self._refresh_opponents()
                metrics["pool_size"] = len(self.pool)

        return metrics

    def _refresh_opponents(self) -> None:
        """Snapshot learner + rebuild vec env with fresh per-env opponents.

        Sampling per env:
          1. With prob `leaderboard_bias` → pick a leaderboard opponent
             (cross-lineage: top-K Elo from other training runs).
          2. Otherwise → self-play pool (latest_bias rules within).

        leaderboard_bias=0 + leaderboard empty → original pure-self-play.
        """
        assert self.pool is not None
        w, n = self.pool.register(
            self.agent.net, self.obs_norm, tag=f"u{self._update_count:05d}"
        )
        lb = self._leaderboard
        use_lb = self.cfg.leaderboard_bias > 0 and len(lb) > 0
        specs: list[dict] = []
        for _ in range(self.cfg.n_envs):
            if use_lb and self._rng.random() < self.cfg.leaderboard_bias:
                entry = lb[int(self._rng.integers(0, len(lb)))]
            else:
                entry = self.pool.sample(self._rng, latest_bias=self.cfg.latest_bias)
                if entry is None:
                    entry = (w, n)
            w_path, n_path = entry
            specs.append({
                "weights_path":  str(w_path),
                "obs_norm_path": str(n_path) if n_path else None,
                # Opponents run on CPU inside sub-processes — MPS/CUDA across
                # 32-128 processes is more trouble than it's worth for
                # single-step forwards.
                "device":        "cpu",
            })
        self._build_vec(opponent_specs=specs)

    def close(self):
        """Explicit teardown. AsyncVectorEnv leaks subprocesses if not closed."""
        try:
            self.vec.close()
        except Exception:
            pass
