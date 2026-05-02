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
    # Source for cross-lineage opponents. 'pfsp' (default) draws from the
    # champion archive weighted by PFSP score (peak at 50% win-rate);
    # 'elo' is the legacy top-K-Elo path kept for compatibility.
    leaderboard_source:   str   = "pfsp"
    # Recency bias on top of PFSP weights. After ordering champions by
    # archived_at DESC, multiply each champion's weight by
    # (recency_decay ** position). 1.0 = no recency bias (uniform PFSP),
    # 0.5 = newest dominates with older still sampled, 0.0 = newest only.
    # Rationale: in a self-play loop, newer champions are usually stronger;
    # facing them most often gives the highest training signal.
    leaderboard_recency_decay: float = 0.5

    # ---- KL early-stopping + KL-adaptive lr (PPO stability) ---------------
    # Without these, an unlucky update can blow up the policy in a regime
    # where the clip alone isn't enough — see 843d5a0a's collapse around
    # update 100 (win rate held 90%+ for 95 updates → dropped to 4-20%).
    # target_kl: typical PPO default; how far we expect the policy to move per update.
    target_kl:           float = 0.01
    # kl_early_stop: bail on the inner-loop epochs early if the running mean
    # KL exceeds target_kl × kl_early_stop_mult. Saves compute on bad updates.
    kl_early_stop:       bool  = True
    kl_early_stop_mult:  float = 1.5
    # kl_adaptive_lr: track an EMA of approx_kl across updates and rescale
    # the optimizer's lr each update — high KL trend → cool down, low KL
    # trend → warm up (capped at kl_lr_max_mult × base lr).
    kl_adaptive_lr:      bool  = True
    kl_ema_alpha:        float = 0.1     # EMA: ema = (1-α)·ema + α·kl
    kl_lr_decay:         float = 0.7     # mul on lr when KL too high
    kl_lr_warmup:        float = 1.05    # mul on lr when KL too low
    kl_lr_min:           float = 3e-5    # absolute floor
    kl_lr_max_mult:      float = 3.0     # cap relative to base lr
    # Level
    # Static name (e.g. "crossroads_6") or dynamic "random_<min>_<max>".
    # Dynamic levels regenerate per reset so training sees varied geometry.
    level_name:           str   = "crossroads_6"
    # Fused rollout (FUSED_ROLLOUT_PLAN). Off by default; opt in when
    # SIM_BACKEND=jax and not self_play. With action_repeat=1 it's
    # byte-identical to the per-tick path under the same seed.
    fused_rollout:        bool  = False
    action_repeat:        int   = 1     # K: env ticks per agent decision under fused
    # Reward scheme. Two ways to set this:
    #   - reward_version: int   — canonical. Supports v1.2(0)/v1.3(1)/v1.4(2)/v1.5(3).
    #   - reward_v13: bool      — back-compat shim. True → v1.3, False → v1.2.
    # If both are set, reward_version wins.
    #   v1.4 = v1.3 + per-tick shaping (buildings/units delta).
    #   v1.5 = v1.4 + asymmetric capture/loss (enemy 4× neutral). Designed to
    #          break the 37% timeout_rate observed under v1.4 on big maps —
    #          agent dominates territory but never finishes games. See
    #          KARPATHY_LOG.md fire 99 (Stage 1 of the timeout-fix plan).
    reward_v13:           bool  = False
    reward_version:       int   = -1     # -1 = unset (use reward_v13)
    # Opponent pool mode (2026-04-29 fire 65). When set, the trainer rotates
    # through leaderboard_paths each PPO update instead of pinning to a single
    # opponent for the whole run. Only relevant when leaderboard_paths is
    # populated (worker downloads them via _download_pfsp_champions).
    #   ""                — disabled (default; single opponent for the run)
    #   "rotate_per_update" — pick a random archive member each update,
    #                          swap the opponent callable in-place. ~50ms per
    #                          swap (state_dict load + small net construct).
    opponent_pool_mode:   str   = ""
    # Optional per-env level distribution. None → all envs use cfg.level_name
    # (back-compat). Otherwise: list of (name, weight) pairs; each env (re)reset
    # samples from this mix. Only honoured by SIM_BACKEND=jax — the numpy
    # AsyncVectorEnv path uses level_name per factory.
    level_mix: list | None = None
    # Archive eval — periodic measurement of current policy vs top-N champions.
    # Cheap signal that actually moves with strength (unlike vs random_legal which
    # saturates at ~95% in <100 updates). Runs `archive_eval_games` games against
    # each of the top-`archive_eval_top_n` champions every `archive_eval_every`
    # updates. Skipped if the archive has fewer than `archive_eval_min_pool` entries.
    archive_eval_every:    int  = 5
    archive_eval_top_n:    int  = 10
    archive_eval_games:    int  = 10
    # None → use the training level (cfg.level_name) so eval stays apples-to-
    # apples with rollouts. Set explicitly to a different level (e.g.
    # "random_8_16") if you want held-out generalization signal.
    archive_eval_level:    str | None = None
    archive_eval_min_pool: int  = 3
    archive_eval_max_ticks: int = 200

    # Replay capture — record N games after each PPO update using the
    # current policy on a fresh numpy env. Output is the JSON event log
    # defined by sim/envs/replay.py:Recorder. Buffered in memory; the
    # worker uploads them as artifacts at end of training.
    # Cost: ~50-200ms per game on small maps; ~1-5% wall-time overhead per game.
    replay_per_update:        bool = False
    # How many games to capture per PPO update. Each game uses a different
    # seed so map layouts vary. Files saved as upd_NNNN_gN.json.
    replay_games_per_update:  int  = 1
    # Random seed offset for replay capture so the same training run can
    # produce reproducible but varied replays.
    replay_seed_offset:       int  = 1_000_003


def _extract_label_from_weights_path(weights_path) -> str:
    """Extract the champion run_id prefix from a PFSP-downloaded weights_path.

    Files written by `_download_pfsp_champions` follow the layout:
        /tmp/mw2-pfsp-XXXX/{champ_id[:8]}-weights.pt
    where the *parent dir* is shared across all archive members. Extract the
    label from the FILENAME (not the dirname), or all rotations look identical
    on the dashboard.

    Test coverage: tests/test_opponent_rotation.py.
    """
    import os
    if not weights_path:
        return "?"
    fname = os.path.basename(str(weights_path))
    for suffix in ("-weights.pt", ".pt"):
        if fname.endswith(suffix):
            return fname[: -len(suffix)]
    return fname


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        config: PPOConfig | None = None,
        seed: int = 0,
        opponent_name: str = "random_legal",
        opponent_kwargs: dict | None = None,
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

        # Replay buffer — list of {update, replay_json} dicts. Populated by
        # _capture_replay() after each PPO update when cfg.replay_per_update
        # is on. Worker drains this at end of run via get_replays().
        self._replay_buffer: list[dict] = []

        # KL-adaptive lr state. Tracks an EMA of approx_kl across updates and
        # rescales the optimizer's lr each step. Initialised to the target so
        # the first update's adaptation is neutral.
        self._base_lr = float(self.cfg.lr)
        self._kl_ema  = float(self.cfg.target_kl)

        # Cumulative wall-time spent in each high-level phase. Surfaced via
        # sim_phase_breakdown() so the run result JSON can record where
        # compute went (CPU-bound sim vs GPU-bound learn, etc).
        self._phase_ns: dict[str, int] = {
            "act_batch_ns":   0,   # GPU inference during rollout
            "env_step_ns":    0,   # async vec env step (returns across IPC)
            "rollout_ns":     0,   # full rollout loop incl. act + step
            "learn_ns":       0,   # PPO minibatch forward + backward + optim
            "update_total_ns": 0,  # everything from update() entry to return
        }

        # Self-play pool. Only populated when cfg.self_play is True.
        self.pool: OpponentPool | None = None
        if self.cfg.self_play:
            self.pool = OpponentPool(root=pool_root, max_size=self.cfg.pool_max_size)

        # Cross-lineage opponents. Accepts both shapes:
        #   (weights_path, obs_norm_path|None)            — uniform weight 1.0
        #   (weights_path, obs_norm_path|None, weight)    — PFSP-weighted draw
        # Normalised to (path, norm, weight) tuples; `_lb_weights` is a
        # cached numpy probability vector matching `self._leaderboard`.
        normalised: list[tuple] = []
        for entry in (leaderboard_paths or []):
            if len(entry) == 3:
                normalised.append(entry)
            else:
                normalised.append((entry[0], entry[1], 1.0))
        self._leaderboard: list[tuple] = normalised
        if self._leaderboard:
            raw = np.asarray([e[2] for e in self._leaderboard], dtype=np.float64)
            total = raw.sum()
            self._lb_weights = (raw / total) if total > 0 else None
        else:
            self._lb_weights = None

        # 2026-04-29 fire 65: pre-load every leaderboard archive member's
        # state_dict + obs_norm into RAM at init. Lets _rotate_opponent_for_update
        # avoid disk I/O on every swap (~10-30ms saving). RAM cost: ~5MB per
        # archive member × 10-20 members = trivial.
        # Each archive member: (state_dict, encoder_version) so the loader can
        # dispatch the right encoder per swap. v9.0 archive members default
        # to "v9.0" via the unstamped-checkpoint fallback.
        self._lb_state_dicts: list[tuple[dict, str]] = []
        self._lb_obs_norms: list = []
        if self._leaderboard and self.cfg.opponent_pool_mode == "rotate_per_update":
            from sim.envs.opponents import preload_state_dict, preload_obs_norm
            from training.encoders import get_encoder
            opp_device = (opponent_kwargs or {}).get("device", "cpu")
            for w_path, n_path, _w in self._leaderboard:
                sd, enc_v = preload_state_dict(str(w_path), device=opp_device)
                self._lb_state_dicts.append((sd, enc_v))
                # Size obs_norm to match the encoder version that produced
                # the saved obs_norm file (the file shape wins anyway).
                self._lb_obs_norms.append(preload_obs_norm(
                    str(n_path) if n_path else None,
                    obs_dim=get_encoder(enc_v).obs_dim,
                ))
            print(f"[trainer] pre-loaded {len(self._lb_state_dicts)} archive members "
                  f"into RAM for per-update rotation (device={opp_device})")
        # Track which leaderboard indices the agent rotated through so we can
        # rematch each one at end-of-run (fire 67).
        self._rotation_history: set[int] = set()

        # Initial opponent: simple name for the very first rollout. After
        # the first snapshot registers (every cfg.snapshot_every updates),
        # the vec env gets rebuilt with neural opponents drawn from the pool.
        self._initial_opponent_name = (
            opponent_name if not self.cfg.self_play else self.cfg.initial_opponent
        )
        # `opponent_kwargs` is only consumed when opponent_name == "neural"
        # (single fixed opponent across all envs — not the self-play pool path).
        self._initial_opponent_kwargs = dict(opponent_kwargs or {})

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

        Vec-env backend is chosen by `SIM_BACKEND` (see sim/backend.py):
          numpy → existing gymnasium.vector.{Sync,Async}VectorEnv path;
          jax   → JaxVecEnv adapter (one process, XLA-batched sim).
        Trainer logic below is backend-agnostic — both paths return the same
        gym-style (obs_dict, rewards, term, trunc, infos).
        """
        from sim.backend import get_backend_name

        # Tear down any previous vec env — AsyncVectorEnv leaks subprocs.
        if hasattr(self, "vec"):
            try:
                self.vec.close()
            except Exception:
                pass

        N = self.cfg.n_envs
        backend = get_backend_name()

        rv = self.cfg.reward_version if self.cfg.reward_version >= 0 else (1 if self.cfg.reward_v13 else 0)
        # Normalise level_mix from a possibly-jsonified dict into a list of
        # (name, weight) tuples. Accept dict {name: weight} or list[[name, w]].
        level_mix = None
        if self.cfg.level_mix:
            raw = self.cfg.level_mix
            if isinstance(raw, dict):
                level_mix = [(str(k), float(v)) for k, v in raw.items()]
            else:
                level_mix = [(str(item[0]), float(item[1])) for item in raw]
        if backend == "jax":
            # JaxVecEnv is a single-process, one-opponent env. Per-env neural
            # opponent specs (self-play pool) aren't supported on this path
            # yet — fall back to the initial opponent name for now.
            if opponent_specs is not None:
                raise NotImplementedError(
                    "SIM_BACKEND=jax doesn't yet support per-env neural "
                    "opponents. Set self_play=False, or run numpy backend."
                )
            from sim.backend import make_vec_env
            self.vec = make_vec_env(
                n_envs=N,
                seed=self.seed,
                level_name=self.cfg.level_name,
                opponent_name=self._initial_opponent_name,
                opponent_kwargs=self._initial_opponent_kwargs or None,
                reward_version=rv,
                level_mix=level_mix,
            )
        else:
            factories = []
            for i in range(N):
                if opponent_specs is None:
                    factories.append(make_env(
                        seed=self.seed + i,
                        level_name=self.cfg.level_name,
                        opponent_name=self._initial_opponent_name,
                        reward_version=rv,
                    ))
                else:
                    factories.append(make_env(
                        seed=self.seed + i,
                        level_name=self.cfg.level_name,
                        opponent_name="neural",
                        opponent_kwargs=opponent_specs[i],
                        reward_version=rv,
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
        if self.cfg.fused_rollout:
            return self._collect_rollout_fused()
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

        import time as _time
        for t in range(T):
            _t_act = _time.perf_counter_ns()
            actions, srcs, types, tgts, logps, values = self.agent.act_batch(
                self._obs, self._masks,
            )
            self._phase_ns["act_batch_ns"] += _time.perf_counter_ns() - _t_act
            obs_buf[t]  = self._obs
            mask_buf[t] = self._masks
            src_buf[t]  = srcs
            type_buf[t] = types
            tgt_buf[t]  = tgts
            logp_buf[t] = logps
            val_buf[t]  = values

            _t_env = _time.perf_counter_ns()
            next_obs_batch, rewards, terminated, truncated, infos = self.vec.step(actions)
            self._phase_ns["env_step_ns"] += _time.perf_counter_ns() - _t_env
            rewards = np.asarray(rewards, dtype=np.float32)
            done = np.logical_or(terminated, truncated)
            rew_buf[t]  = rewards
            done_buf[t] = done.astype(np.float32)

            # Per-env episode bookkeeping. We prefer the literal terminal_phase
            # signal (1 = P1_WINS, 2 = P2_WINS, 3 = DRAW) when the env exposes
            # it; the JaxVecEnv path threads it through `infos`. Falls back to
            # the reward-sum proxy (return > 0.5) for envs that don't.
            terminal_phase = None
            if isinstance(infos, dict):
                terminal_phase = infos.get("terminal_phase")
            self._ep_return += rewards
            self._ep_length += 1
            for i in range(N):
                if done[i]:
                    if terminal_phase is not None:
                        # PHASE_PLAYING=0, P1_WINS=1, P2_WINS=2, DRAW=3.
                        phase = int(terminal_phase[i])
                    else:
                        # Reward-sum proxy: episode return >0.5 → win, <-0.5
                        # → loss, else draw. Matches the v1.3+ reward shape
                        # where only terminal events drive |reward| past 0.5.
                        r = float(self._ep_return[i])
                        phase = 1 if r > 0.5 else (2 if r < -0.5 else 3)
                    self._completed_episodes.append((
                        float(self._ep_return[i]),
                        int(self._ep_length[i]),
                        phase,
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

    def _collect_rollout_fused(self) -> dict:
        """Rollout via `training/fused_rollout.py`. Requires SIM_BACKEND=jax
        and a `_JaxVecAdapter` wrapping a `JaxVecEnv` underneath."""
        from sim.backend import get_backend_name
        if get_backend_name() != "jax":
            raise RuntimeError("cfg.fused_rollout=True requires SIM_BACKEND=jax")
        if self.cfg.self_play:
            raise NotImplementedError(
                "fused rollout doesn't support self_play yet; "
                "set cfg.fused_rollout=False or self_play=False"
            )
        if not hasattr(self.vec, "_inner"):
            raise RuntimeError(
                "fused rollout expected _JaxVecAdapter; got "
                f"{type(self.vec).__name__}. Check SIM_BACKEND."
            )

        # Lazy-init the device-side carry; per-call re-bind the trainer's
        # current `_completed_episodes` list since `update()` resets it
        # to `[]` between rollouts (so we can't stash a stale reference).
        if not hasattr(self, "_fused_bookkeeping"):
            self._fused_bookkeeping = {
                "ep_return":           self._ep_return,
                "ep_length":           self._ep_length,
                "last_obs_dev":        None,
                "last_p1_mask":        None,
                "last_p2_mask":        None,
                # Pre-loaded neural opponent callable (from the adapter), used
                # by the fused-rollout collector when opponent_name=='neural'.
                # JAX-native opponents (random_legal, noop, greedy_capacity_aware)
                # don't need a host callable.
                "opponent_fn":         getattr(self.vec, "_opponent", None)
                                          if self._initial_opponent_name not in (
                                              "random_legal", "noop", "greedy_capacity_aware",
                                          )
                                          else None,
            }
        self._fused_bookkeeping["completed_episodes"] = self._completed_episodes

        # `update()` already wraps `collect_rollout()` with rollout_ns
        # timing, so we don't add another counter here. act_batch_ns and
        # env_step_ns aren't broken out under fused (would need to plumb
        # them through `collect_rollout_fused`); they show as 0 in the
        # phase breakdown for fused runs.
        from training.fused_rollout import collect_rollout_fused
        return collect_rollout_fused(
            agent=self.agent,
            vec_env=self.vec._inner,
            cfg=self.cfg,
            obs_norm=self.obs_norm,
            obs_clip=self.cfg.obs_clip,
            bookkeeping=self._fused_bookkeeping,
            rng=self._rng,
            opponent_name=self._initial_opponent_name,
        )

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
        import time as _time
        _t_update = _time.perf_counter_ns()

        # 2026-04-29 fire 65: per-update opponent rotation. When enabled,
        # pick a random archive member and swap the opponent callable BEFORE
        # collecting this update's rollout. ~50ms per swap (state_dict load).
        if self.cfg.opponent_pool_mode == "rotate_per_update" and self._leaderboard:
            self._rotate_opponent_for_update()

        _t_rollout = _time.perf_counter_ns()
        batch = self.collect_rollout()
        self._phase_ns["rollout_ns"] += _time.perf_counter_ns() - _t_rollout
        B = len(batch["obs"])
        _t_learn = _time.perf_counter_ns()

        obs_t     = torch.as_tensor(batch["obs"],       device=self.device)
        mask_t    = torch.as_tensor(batch["mask"],      device=self.device)
        src_t     = torch.as_tensor(batch["src"],       device=self.device)
        type_t    = torch.as_tensor(batch["type"],      device=self.device)
        tgt_t     = torch.as_tensor(batch["tgt"],       device=self.device)
        oldlogp_t = torch.as_tensor(batch["logprob"],   device=self.device)
        adv_t     = torch.as_tensor(batch["advantage"], device=self.device)
        ret_t     = torch.as_tensor(batch["return"],    device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        pol_losses, val_losses, ent_losses, approx_kls, clip_fracs, grad_norms = [], [], [], [], [], []
        idx = np.arange(B)
        mb_size = min(self.cfg.minibatch_size, B)
        early_stop_kl = self.cfg.target_kl * self.cfg.kl_early_stop_mult
        kl_early_stopped = False
        for _ in range(self.cfg.update_epochs):
            if kl_early_stopped:
                break
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
                # clip_grad_norm_ returns the pre-clip total norm — capture it
                # so we can chart instability spikes that a clipped post-norm hides.
                gn = torch.nn.utils.clip_grad_norm_(
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
                grad_norms.append(float(gn))

                # KL early-stop: if running mean kl across this update so
                # far exceeds target × early_stop_mult, bail on remaining
                # minibatches + epochs. Saves compute on bad updates.
                if self.cfg.kl_early_stop:
                    if float(np.mean(approx_kls)) > early_stop_kl:
                        kl_early_stopped = True
                        break

        # Critic honesty: 1 - Var[returns - values] / Var[returns].
        # Computed once per update from the rollout's pre-update predictions
        # vs the GAE returns. ~1.0 = perfect critic; ~0 = no better than mean;
        # <0 = worse than mean (silent fail mode worth catching).
        ret_arr = np.asarray(batch["return"], dtype=np.float64)
        val_arr = np.asarray(batch["value"],  dtype=np.float64)
        ret_var = float(ret_arr.var())
        ev = float(1.0 - ((ret_arr - val_arr).var() / ret_var)) if ret_var > 1e-12 else 0.0

        # KL-adaptive lr: update an EMA of approx_kl, scale the optimizer's
        # lr based on whether the trend is too high (cool down) or too low
        # (warm up). Logged so the dashboard chart can show the lr curve.
        kl_mean = float(np.mean(approx_kls)) if approx_kls else float(self.cfg.target_kl)
        self._kl_ema = (1.0 - self.cfg.kl_ema_alpha) * self._kl_ema + self.cfg.kl_ema_alpha * kl_mean
        if self.cfg.kl_adaptive_lr:
            cur_lr = float(self.optimizer.param_groups[0]["lr"])
            target = float(self.cfg.target_kl)
            if self._kl_ema > target * 1.5:
                cur_lr *= float(self.cfg.kl_lr_decay)
            elif self._kl_ema < target * 0.5:
                cur_lr *= float(self.cfg.kl_lr_warmup)
            cur_lr = max(float(self.cfg.kl_lr_min),
                         min(self._base_lr * float(self.cfg.kl_lr_max_mult), cur_lr))
            for g in self.optimizer.param_groups:
                g["lr"] = cur_lr

        metrics = {
            "policy_loss":        float(np.mean(pol_losses)),
            "value_loss":         float(np.mean(val_losses)),
            "entropy_loss":       float(np.mean(ent_losses)),
            "approx_kl":          float(np.mean(approx_kls)),
            "kl_ema":             float(self._kl_ema),
            "kl_early_stopped":   bool(kl_early_stopped),
            "lr":                 float(self.optimizer.param_groups[0]["lr"]),
            "clip_fraction":      float(np.mean(clip_fracs)),
            "grad_norm":          float(np.mean(grad_norms)),
            "explained_variance": ev,
            "mean_reward":        float(batch["reward"].mean()),
            # Run-state breadcrumbs: who the agent played + on which level mix.
            # Logged every update so a chart can show how training conditions
            # shifted over time (opponent swaps, level-mix changes).
            "training_opp_label":  self._current_training_opp_label(),
            "level_mix_resolved":  self._current_level_mix_resolved(),
        }
        ep = self._completed_episodes
        if ep:
            returns = np.asarray([e[0] for e in ep], dtype=np.float64)
            metrics["episodes_completed"]  = len(ep)
            metrics["mean_episode_return"] = float(returns.mean())
            metrics["mean_episode_length"] = float(np.mean([e[1] for e in ep]))
            phases = np.asarray([e[2] for e in ep], dtype=np.int32)
            metrics["win_rate"]            = float((phases == 1).mean())
            metrics["loss_rate"]           = float((phases == 2).mean())
            metrics["draw_rate"]           = float((phases == 3).mean())
            # phase==0 (PHASE_PLAYING) means the episode ended via the
            # max_ticks cap without resolution. Tracking this so
            # W+L+D+T = 100% on the chart — a high timeout_rate means the
            # agent isn't finishing games (common on big maps where the
            # agent dominates but doesn't eliminate the opponent before
            # max_ticks). Added 2026-04-30.
            metrics["timeout_rate"]        = float((phases == 0).mean())
            # Min/max/p10/p90 bands. Mean alone hides variance collapse —
            # AlphaStar Nature paper plots min/max bands for this reason.
            metrics["episode_return_min"]  = float(returns.min())
            metrics["episode_return_max"]  = float(returns.max())
            metrics["episode_return_p10"]  = float(np.percentile(returns, 10))
            metrics["episode_return_p50"]  = float(np.percentile(returns, 50))
            metrics["episode_return_p90"]  = float(np.percentile(returns, 90))
            self._completed_episodes = []
            # Per-level win rate is meaningful only when all envs share one
            # level (level_mix=None). Multi-level training collapses every
            # episode under cfg.level_name, which would mislabel mixed runs;
            # plumbing per-env level requires touching jax_vec_env state and
            # is left for a follow-up. The dashboard hides this card when
            # the key is absent or has only one bucket.
            if not self.cfg.level_mix:
                lvl = str(self.cfg.level_name)
                metrics["win_rate_by_level"] = {lvl: metrics["win_rate"]}

        self._phase_ns["learn_ns"] += _time.perf_counter_ns() - _t_learn
        self._update_count += 1

        # Self-play refresh: register a snapshot at the cadence and rebuild
        # the vec env so sub-envs load the fresh opponents from disk.
        if self.cfg.self_play and self.pool is not None:
            if self._update_count % self.cfg.snapshot_every == 0:
                self._refresh_opponents()
                metrics["pool_size"] = len(self.pool)

        # Archive eval: every N updates, play current policy vs top-K champions.
        # Returns a metrics dict (may be None if skipped). Time is folded into
        # update_total_ns since it's part of one PPO step from the user's view.
        eval_metrics = self._archive_eval()
        if eval_metrics is not None:
            metrics.update(eval_metrics)

        # Replay capture: N numpy-env games with the current policy after
        # each update. Each game gets a different seed so layouts vary.
        # ~50-200ms per game on small maps. Stored in self._replay_buffer
        # for the worker to upload at end of training.
        if self.cfg.replay_per_update:
            try:
                games_n = max(1, int(self.cfg.replay_games_per_update))
                for game_idx in range(games_n):
                    rep = self._capture_replay(game_idx=game_idx)
                    if rep is not None:
                        self._replay_buffer.append({
                            "update": self._update_count,
                            "game":   game_idx,
                            "replay": rep,
                        })
            except Exception as exc:
                if not getattr(self, "_replay_disabled", False):
                    print(f"[replay] disabled — capture failed: {type(exc).__name__}: {exc}", flush=True)
                    self._replay_disabled = True

        self._phase_ns["update_total_ns"] += _time.perf_counter_ns() - _t_update
        return metrics

    def get_replays(self) -> list[dict]:
        """Return captured replays. Each entry: {update: int, replay: dict}."""
        return list(self._replay_buffer)

    def _capture_replay(self, game_idx: int = 0) -> dict | None:
        """Run one full game on a fresh numpy env using current policy as P1
        and random_legal as P2. Returns the Recorder JSON dict or None if
        disabled mid-run. Uses the trainer's level_name (or first member of
        level_mix if set). Deterministic per (update_count, game_idx,
        replay_seed_offset) so replays are reproducible across re-runs but
        each captured game varies.
        """
        if getattr(self, "_replay_disabled", False):
            return None
        # Lazy imports — replay capture isn't on the hot path of most runs,
        # and we don't want to pay these imports during smoke tests.
        from sim import config as C
        from sim import levels as sim_levels
        from sim.actions import compute_mask, decode, NOOP_INDEX
        from sim.engine import step_tick
        from sim.envs.opponents import random_legal_opponent
        from sim.envs.replay import Recorder
        import torch

        # Per-game seed offset so each captured game in an update uses a
        # different layout. Mix in a large prime so game_idx=1 isn't just
        # one off from game_idx=0.
        per_game_off = int(game_idx) * 7919

        # Pick a level. With level_mix, sample by weight (deterministic).
        if self.cfg.level_mix:
            mix = self.cfg.level_mix
            if isinstance(mix, dict):
                pairs = [(str(k), float(v)) for k, v in mix.items()]
            else:
                pairs = [(str(p[0]), float(p[1])) for p in mix]
            weights = np.array([w for _, w in pairs], dtype=np.float64)
            weights = weights / weights.sum() if weights.sum() > 0 else None
            rng_pick = np.random.default_rng(
                self.seed + self._update_count + self.cfg.replay_seed_offset + per_game_off
            )
            idx = int(rng_pick.choice(len(pairs), p=weights))
            level_name = pairs[idx][0]
        else:
            level_name = str(self.cfg.level_name)

        seed = int(self.seed + self._update_count + self.cfg.replay_seed_offset + per_game_off)
        rv = self.cfg.reward_version if self.cfg.reward_version >= 0 else (1 if self.cfg.reward_v13 else 0)
        state = sim_levels.reset(level_name=level_name, seed=seed, reward_version=rv)

        # Recorder uses the engine's per-tick event buffer to produce the
        # public replay schema. capture_map snapshots the initial layout.
        recorder = Recorder(
            game_id=f"upd{self._update_count:04d}_g{game_idx}",
            sim_version="v12",
            level_name=level_name,
            seed=seed,
        )
        recorder.capture_map(state)

        # P2 random rng — separate from level-pick rng so opponent variability
        # doesn't depend on the level mix.
        opp_rng = np.random.default_rng(seed ^ 0x5A5A5A5A)

        # _state_to_obs_dict_for_player from tournament.py builds the dict the
        # encoder expects (with v10+ event fields). Importing here keeps the
        # cli/tournament import out of the trainer's hot path.
        from scripts.tournament import _state_to_obs_dict_for_player

        max_ticks = int(C.GAME_TIMEOUT_TICKS)
        tick = 0
        done = False
        while not done and tick < max_ticks:
            # Decision-interval: env actions only fire on the env's decision boundary.
            is_decision_tick = (tick % C.DECISION_INTERVAL_TICKS) == 0
            a1_idx = NOOP_INDEX
            a2_idx = NOOP_INDEX
            if is_decision_tick:
                # P1 — neural agent (deterministic to keep replays clean).
                mask_p1 = compute_mask(state, C.OWNER_P1)
                obs_p1  = _state_to_obs_dict_for_player(state, mask_p1, C.OWNER_P1)
                enc_p1  = encode_obs(obs_p1)
                if self.obs_norm is not None:
                    enc_p1 = self.obs_norm.normalize(enc_p1[None, :], clip=self.cfg.obs_clip)[0]
                action_arr, *_ = self.agent.act_batch(
                    enc_p1[None, :], mask_p1[None, :], deterministic=True,
                )
                a1_idx = int(action_arr[0])
                # P2 — random legal.
                a2_idx = random_legal_opponent(state, opp_rng)

            a1 = decode(a1_idx)
            a2 = decode(a2_idx)
            buf = recorder.get_tick_events_buffer()
            _r1, _r2, done = step_tick(state, a1, a2, events=buf)
            recorder.absorb_tick(state)
            tick += 1

        return recorder.to_dict()

    def _current_training_opp_label(self) -> str:
        """Best-effort label for who the agent is currently training against.

        Used by the dashboard to show "training opponent over time" — so a
        reader can spot opponent swaps mid-run. Three regimes:
          - Self-play (cfg.self_play=True, pool populated): label is the
            number of pool snapshots. Once the leaderboard is wired in,
            we'd surface the dominant champion instead.
          - Single neural opponent (opponent_name='neural', kwargs has run_id):
            'champion:<run_id_short>'.
          - Stateless name (random_legal/noop/latest_champion-not-yet-resolved):
            return that name verbatim.
        """
        if self.cfg.self_play and self.pool is not None and len(self.pool) > 0:
            return f"self_play:pool_{len(self.pool)}"
        name = self._initial_opponent_name or "unknown"
        if name == "neural":
            kw = self._initial_opponent_kwargs or {}
            # Worker stashes the source run_id under a _label_ sentinel key so
            # the dashboard can render "champion:abc12345" rather than just
            # "neural" or its weights_path. See workers/worker.py.
            run_id = kw.get("_label_opponent_run_id") or kw.get("opponent_run_id", "")
            return f"champion:{str(run_id)[:8]}" if run_id else "neural:?"
        return name

    def _current_level_mix_resolved(self) -> list:
        """Resolve the trainer's level config to a [(name, weight), ...] list.

        Static level → [(name, 1.0)]. level_mix dict/list → normalised list.
        Logged each update so a chart can show level-distribution changes.
        """
        if self.cfg.level_mix:
            raw = self.cfg.level_mix
            if isinstance(raw, dict):
                return [[str(k), float(v)] for k, v in raw.items()]
            return [[str(item[0]), float(item[1])] for item in raw]
        return [[str(self.cfg.level_name), 1.0]]

    def _rotate_opponent_for_update(self) -> None:
        """Pick a random archive member and swap the opponent callable in-place.

        Called at the start of update() when cfg.opponent_pool_mode ==
        'rotate_per_update'. Uses self._lb_state_dicts (pre-loaded into RAM
        at trainer init) so the per-swap cost is just net-construct +
        state_dict copy (~10-30ms; no disk I/O).

        Updates:
          - self.vec._opponent (used by the legacy per-tick path in backend.py)
          - self._fused_bookkeeping['opponent_fn'] (used by fused_rollout)
          - self._initial_opponent_kwargs (so _current_opponent_label reflects
            the swap on dashboard)
        """
        import os
        from sim.envs.opponents import make_neural_opponent_cached

        if not self._leaderboard or not self._lb_state_dicts:
            return  # nothing to rotate to (or pre-load skipped)

        # Weighted pick using the cached PFSP weights (or uniform if missing).
        if self._lb_weights is not None:
            idx = int(np.random.choice(len(self._leaderboard), p=self._lb_weights))
        else:
            idx = int(np.random.randint(0, len(self._leaderboard)))

        # Cached (state_dict, encoder_version) + obs_norm — already in RAM from init.
        state_dict, encoder_version = self._lb_state_dicts[idx]
        obs_norm = self._lb_obs_norms[idx]
        # Track which archive members the agent has faced so end_of_run_rematch
        # can replay each one at the end of training.
        self._rotation_history.add(idx)
        device = (self._initial_opponent_kwargs or {}).get("device", "cpu")
        new_opponent = make_neural_opponent_cached(
            state_dict=state_dict,
            obs_norm=obs_norm,
            device=device,
            encoder_version=encoder_version,
        )

        # Swap in both the vec env (legacy path) and the fused_rollout
        # bookkeeping (the path actually used under sim_backend=jax).
        if hasattr(self.vec, "_opponent"):
            self.vec._opponent = new_opponent
        if hasattr(self, "_fused_bookkeeping"):
            self._fused_bookkeeping["opponent_fn"] = new_opponent

        # Update the dashboard label so the run page shows which opponent
        # the agent is currently training against. Pure helper — see
        # _extract_label_from_weights_path docstring + tests/test_opponent_rotation.py.
        weights_path = self._leaderboard[idx][0]
        opp_id = _extract_label_from_weights_path(weights_path)
        if self._initial_opponent_kwargs is None:
            self._initial_opponent_kwargs = {}
        self._initial_opponent_kwargs["_label_opponent_run_id"] = opp_id

    def _archive_eval(self) -> dict | None:
        """Periodic eval of current policy vs top-N champions in the archive.

        Returns metrics dict or None if skipped (cadence not hit, archive too
        thin, or runtime error). Errors are caught here so a transient DB or
        download failure never breaks training.

        Output keys when populated:
            win_rate_vs_leaderboard:   float in [0,1], P1 wins / total over all matches
            archive_eval_n_opponents:  int, number of champions matched against
            archive_eval_n_games:      int, total games played this eval
            archive_eval_wall_s:       float, wall-clock for the eval
        """
        if self._update_count == 0 or self._update_count % self.cfg.archive_eval_every != 0:
            return None

        import time as _time
        import tempfile
        import importlib
        from pathlib import Path
        import urllib.request

        t0 = _time.perf_counter()
        try:
            # Lazy import: cli.db / scripts.tournament pull in Supabase, which
            # most smoke tests don't have. We tolerate ImportError at first eval
            # and disable archive eval for the rest of the run.
            try:
                cli_db = importlib.import_module("cli.db")
                tournament = importlib.import_module("scripts.tournament")
            except Exception as exc:
                if not getattr(self, "_archive_eval_disabled", False):
                    print(f"[archive_eval] disabled — import failed: {type(exc).__name__}: {exc}", flush=True)
                    self._archive_eval_disabled = True
                return None
            if getattr(self, "_archive_eval_disabled", False):
                return None

            # Top-N champions by their source run's Elo score. NULL elo (unrated)
            # sorts last via NULLS LAST. We pull a few extra rows and let the
            # download loop drop entries with missing weights_url.
            with cli_db.connect() as c, c.cursor() as cur:
                cur.execute("""
                    SELECT ch.id, ch.label, ch.weights_url, ch.obs_norm_url,
                           r.elo_score
                      FROM champions ch
                      LEFT JOIN runs r ON r.id = ch.source_run_id
                     ORDER BY r.elo_score DESC NULLS LAST, ch.archived_at DESC
                     LIMIT %s
                """, (self.cfg.archive_eval_top_n + 5,))
                cols = ("id", "label", "weights_url", "obs_norm_url", "elo_score")
                arch = [dict(zip(cols, row)) for row in cur.fetchall()]

            if len(arch) < self.cfg.archive_eval_min_pool:
                return {"archive_eval_n_opponents": 0, "archive_eval_n_games": 0}

            # Cache: download top-N champion checkpoints once per run, reuse
            # across evals. Refreshed if archive_eval_top_n changes.
            top_n = min(self.cfg.archive_eval_top_n, len(arch))
            cache = getattr(self, "_archive_eval_cache", None)
            if cache is None or cache.get("top_n") != top_n:
                cache_dir = Path(tempfile.mkdtemp(prefix=f"mw2-eval-{self._update_count}-"))
                # Cache entry: (path, label, elo, id_short) — id_short is the
                # 8-char champion id prefix used in dashboards and logs.
                paths: list[tuple[str, str, float | None, str]] = []
                supabase_url = __import__('os').environ.get('SUPABASE_URL')
                if not supabase_url:
                    print("[archive_eval] SUPABASE_URL not set — disabling archive eval", flush=True)
                    self._archive_eval_disabled = True
                    return None
                for champ in arch:
                    if len(paths) >= top_n:
                        break
                    if not champ.get("weights_url"):
                        continue
                    short = str(champ["id"])[:8]
                    label = champ.get("label") or short
                    elo = champ.get("elo_score")
                    cdir = cache_dir / short
                    cdir.mkdir(exist_ok=True)
                    w_path = cdir / "weights.pt"
                    n_path = cdir / "obs_norm.pt"
                    try:
                        urllib.request.urlretrieve(
                            f"{supabase_url}/storage/v1/object/public/{champ['weights_url']}",
                            w_path,
                        )
                        if champ.get("obs_norm_url"):
                            urllib.request.urlretrieve(
                                f"{supabase_url}/storage/v1/object/public/{champ['obs_norm_url']}",
                                n_path,
                            )
                    except Exception as exc:
                        print(f"[archive_eval] skip {label}: download failed ({exc})", flush=True)
                        continue
                    paths.append((str(cdir), label, elo, short))
                cache = {"top_n": top_n, "paths": paths, "dir": cache_dir}
                self._archive_eval_cache = cache
                elo_strs = [f"{e:.0f}" if e is not None else "—" for _, _, e, _ in paths]
                print(f"[archive_eval] cached {len(paths)} champion checkpoints in {cache_dir}; "
                      f"elos: {elo_strs}", flush=True)

            opponents = cache["paths"]
            if not opponents:
                return {"archive_eval_n_opponents": 0, "archive_eval_n_games": 0}

            # Save current policy weights to a stable eval-checkpoint dir so
            # tournament._load_policy can read them. Overwritten every eval.
            eval_ckpt_dir = getattr(self, "_archive_eval_ckpt_dir", None)
            if eval_ckpt_dir is None:
                eval_ckpt_dir = Path(tempfile.mkdtemp(prefix="mw2-eval-self-"))
                self._archive_eval_ckpt_dir = eval_ckpt_dir
            # v10: stamp encoder_version so the eval loader knows what the
            # weights expect. See training.checkpoint.
            from training.checkpoint import save_state_dict
            save_state_dict(
                {k: v.detach().cpu() for k, v in self.agent.net.state_dict().items()},
                eval_ckpt_dir / "weights.pt",
            )
            if self.obs_norm is not None:
                self.obs_norm.save(str(eval_ckpt_dir / "obs_norm.pt"))

            # Resolve eval level: explicit override wins; otherwise fall back to
            # the training level so eval and rollouts stay apples-to-apples.
            # When training uses a level_mix, we pick the most-weighted level
            # (run_match takes a single level string). Set archive_eval_level
            # explicitly if you want a held-out generalization split.
            eval_level = self.cfg.archive_eval_level
            if eval_level is None:
                if self.cfg.level_mix:
                    raw = self.cfg.level_mix
                    pairs = (list(raw.items()) if isinstance(raw, dict)
                             else [(item[0], item[1]) for item in raw])
                    pairs.sort(key=lambda x: float(x[1]), reverse=True)
                    eval_level = str(pairs[0][0]) if pairs else self.cfg.level_name
                else:
                    eval_level = self.cfg.level_name

            # Sequentially run N-game batches against each champion. We capture
            # per-opponent records so the dashboard can show "which champions
            # did we beat / lose to" rather than a single averaged number.
            total_p1_wins = 0.0
            total_games = 0
            matches: list[dict] = []
            for opp_dir, opp_label, opp_elo, opp_id_short in opponents:
                m_t0 = _time.perf_counter()
                try:
                    res = tournament.run_match(
                        p1=str(eval_ckpt_dir),
                        p2=opp_dir,
                        games=self.cfg.archive_eval_games,
                        level=eval_level,
                        max_ticks=self.cfg.archive_eval_max_ticks,
                        seed=int(self._rng.integers(0, 2**31 - 1)),
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"[archive_eval] match vs {opp_label} failed: {exc}", flush=True)
                    matches.append({
                        "update":       self._update_count,
                        "opp_label":    opp_label,
                        "opp_id_short": opp_id_short,
                        "opp_elo":      float(opp_elo) if opp_elo is not None else None,
                        "level":        eval_level,
                        "error":        f"{type(exc).__name__}: {exc}",
                    })
                    continue
                m_wall = _time.perf_counter() - m_t0
                p1_wins = int(res["p1_wins"])
                p2_wins = int(res["p2_wins"])
                draws   = int(res["draws"])
                total   = int(res["total"])
                total_p1_wins += p1_wins + 0.5 * draws
                total_games += total
                wr = (p1_wins + 0.5 * draws) / max(total, 1)
                matches.append({
                    "update":       self._update_count,
                    "opp_label":    opp_label,
                    "opp_id_short": opp_id_short,
                    "opp_elo":      float(opp_elo) if opp_elo is not None else None,
                    "level":        eval_level,
                    "p1_wins":      p1_wins,
                    "p2_wins":      p2_wins,
                    "draws":        draws,
                    "total":        total,
                    "win_rate":     round(wr, 3),
                    "wall_s":       round(m_wall, 3),
                })

            wall = _time.perf_counter() - t0
            if total_games == 0:
                return {"archive_eval_n_opponents": len(opponents),
                        "archive_eval_n_games": 0,
                        "archive_eval_wall_s": round(wall, 3),
                        "eval_matches": matches}

            wr = total_p1_wins / total_games
            print(f"[archive_eval] u{self._update_count}: win_rate={wr:.3f} "
                  f"({int(total_p1_wins)}/{total_games} vs {len(opponents)} champs, {wall:.2f}s)",
                  flush=True)
            return {
                "win_rate_vs_leaderboard": float(wr),
                "archive_eval_n_opponents": len(opponents),
                "archive_eval_n_games":     int(total_games),
                "archive_eval_wall_s":      round(wall, 3),
                "eval_matches":             matches,
            }
        except Exception as exc:
            print(f"[archive_eval] unexpected failure: {type(exc).__name__}: {exc}", flush=True)
            return None

    def sim_phase_breakdown(self) -> dict:
        """Return cumulative wall-time breakdown of training phases.

        Keys are percentages of `update_total_ns` so the result is
        dimensionless and easy to compare across machines. Also returns
        raw ms totals for debugging.
        """
        total = self._phase_ns.get("update_total_ns", 0) or 1
        pct = {k: round(100.0 * v / total, 1) for k, v in self._phase_ns.items()}
        ms = {k: round(v / 1e6, 1) for k, v in self._phase_ns.items()}
        return {
            "pct_of_update": pct,
            "ms_cumulative": ms,
            "updates": self._update_count,
        }

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
                if self._lb_weights is not None:
                    idx = int(self._rng.choice(len(lb), p=self._lb_weights))
                else:
                    idx = int(self._rng.integers(0, len(lb)))
                w_path, n_path, _wt = lb[idx]
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
