"""
Sim backend selector — reads `SIM_BACKEND` and builds a gym-style vec env.

Routes trainer/bench code through a single `make_vec_env` factory so the
actual backend (numpy `AsyncVectorEnv` vs `JaxVecEnv`) is one env-var flip
away.

Contract: whatever `make_vec_env` returns must present the gymnasium
`VectorEnv` surface that the trainer uses:

    vec.reset(seed=int) -> (obs_dict, info)
    vec.step(actions)   -> (obs_dict, rewards, terminated, truncated, infos)
    vec.close()

where `obs_dict` is a dict of stacked numpy arrays (one per env) matching
the keys that `training.encoder.encode_obs` consumes.

Backend choice (env var `SIM_BACKEND`):
- `numpy` (default): existing `gymnasium.vector.{Sync,Async}VectorEnv`
  built from `sim.envs.make_env` factories.
- `jax`: `_JaxVecAdapter` — wraps `sim.envs.jax_vec_env.JaxVecEnv` and
  applies the opponent policy on the numpy host side so the trainer sees
  the same gym-style interface.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional


# Default backend. Flipped to "jax" at the end of JAX_PORT_PLAN Phase 6.
# To roll back: set `SIM_BACKEND=numpy` in the worker's environment (e.g.
# `~/.config/systemd/user/mushroom-worker.service` `Environment=` line on
# PaulLinux, or export it in the shell for a one-off run).
_DEFAULT_BACKEND = "jax"


def get_backend_name() -> str:
    """Resolve the active backend. Respects the `SIM_BACKEND` env var;
    falls back to `_DEFAULT_BACKEND` (currently "jax")."""
    name = os.environ.get("SIM_BACKEND", _DEFAULT_BACKEND).strip().lower()
    if name not in ("numpy", "jax"):
        raise ValueError(f"SIM_BACKEND must be 'numpy' or 'jax'; got {name!r}")
    return name


def make_vec_env(
    n_envs: int,
    *,
    seed: int = 0,
    level_name: str = "crossroads_6",
    opponent_name: str = "random_legal",
    opponent_kwargs: Optional[dict] = None,
    vec_mode: str = "async",
    context: Optional[str] = None,
    reward_version: int = 0,
):
    """Return a gym-style vector env on the active backend.

    `vec_mode` and `context` only apply to the numpy backend; JAX ignores
    them (one-process by design). `reward_version` selects the reward scheme
    (0=v1.2, 1=v1.3) — only honoured by the JAX backend at the moment;
    numpy ignores it (random_legal training only).
    """
    backend = get_backend_name()
    if backend == "numpy":
        return _build_numpy_vec_env(
            n_envs=n_envs, seed=seed, level_name=level_name,
            opponent_name=opponent_name, opponent_kwargs=opponent_kwargs,
            vec_mode=vec_mode, context=context,
        )
    return _build_jax_vec_env(
        n_envs=n_envs, seed=seed, level_name=level_name,
        opponent_name=opponent_name, opponent_kwargs=opponent_kwargs,
        reward_version=reward_version,
    )


def make_factories(
    n_envs: int,
    *,
    seed: int = 0,
    level_name: str = "crossroads_6",
    opponent_name: str = "random_legal",
    opponent_kwargs_list: Optional[list[dict]] = None,
) -> list[Callable]:
    """For the numpy backend: build per-env `make_env` factories.

    The trainer's `_build_vec` path sometimes constructs per-env opponent
    specs (self-play pool); expose this helper so that path stays unchanged.
    """
    from sim.envs import make_env
    factories = []
    for i in range(n_envs):
        kw = (opponent_kwargs_list[i] if opponent_kwargs_list else None)
        factories.append(make_env(
            seed=seed + i, level_name=level_name,
            opponent_name=opponent_name, opponent_kwargs=kw,
        ))
    return factories


# ---------------------------------------------------------------------------
# Numpy backend
# ---------------------------------------------------------------------------

def _build_numpy_vec_env(
    n_envs: int,
    seed: int,
    level_name: str,
    opponent_name: str,
    opponent_kwargs: Optional[dict],
    vec_mode: str,
    context: Optional[str],
):
    import gymnasium as gym

    factories = make_factories(
        n_envs=n_envs, seed=seed, level_name=level_name,
        opponent_name=opponent_name,
        opponent_kwargs_list=[dict(opponent_kwargs or {})] * n_envs
        if opponent_kwargs else None,
    )
    if vec_mode == "async":
        return gym.vector.AsyncVectorEnv(factories, shared_memory=False, context=context)
    if vec_mode == "sync":
        return gym.vector.SyncVectorEnv(factories)
    raise ValueError(f"unknown vec_mode: {vec_mode!r}")


# ---------------------------------------------------------------------------
# JAX backend adapter — presents the gym VectorEnv surface over JaxVecEnv.
# ---------------------------------------------------------------------------

def _build_jax_vec_env(
    n_envs: int,
    seed: int,
    level_name: str,
    opponent_name: str,
    opponent_kwargs: Optional[dict],
    reward_version: int = 0,
):
    # Lazy-import: don't pay the JAX startup tax when the numpy backend is active.
    from sim.envs.jax_vec_env import JaxVecEnv
    return _JaxVecAdapter(
        n_envs=n_envs, seed=seed, level_name=level_name,
        opponent_name=opponent_name, opponent_kwargs=opponent_kwargs,
        inner_factory=lambda: JaxVecEnv(
            n_envs=n_envs, level_name=level_name, base_seed=seed,
            reward_version=reward_version,
        ),
    )


class _JaxVecAdapter:
    """Gym VectorEnv surface over `sim.envs.jax_vec_env.JaxVecEnv`.

    Responsibilities:
    - Present `(obs_dict, rewards, terminated, truncated, infos)` per step.
    - Pick P2's action each tick using the configured opponent (batched on
      numpy: compute mask per env, then random choice or neural forward).
    - Bundle P1/P2 actions into the (n_envs, 2, 4) int32 tensor JaxVecEnv expects.
    """

    def __init__(
        self,
        n_envs: int,
        seed: int,
        level_name: str,
        opponent_name: str,
        opponent_kwargs: Optional[dict],
        inner_factory: Callable,
    ):
        import numpy as np

        self.n_envs = int(n_envs)
        self.level_name = level_name
        self._rng = np.random.default_rng(seed)
        self._opponent_name = opponent_name
        self._opponent_kwargs = dict(opponent_kwargs or {})
        self._inner = inner_factory()

        # Build the opponent callable once. Matches the per-env opponent
        # plumbing used by `sim.envs.make_env`.
        from sim.envs.opponents import noop_opponent, random_legal_opponent, make_neural_opponent
        if opponent_name == "random_legal":
            self._opponent = random_legal_opponent
        elif opponent_name == "noop":
            self._opponent = noop_opponent
        elif opponent_name == "neural":
            self._opponent = make_neural_opponent(**self._opponent_kwargs)
        else:
            raise ValueError(f"unknown opponent_name: {opponent_name!r}")

        # Mirror the `gymnasium.VectorEnv.single_observation_space` / etc used
        # by the trainer's `_encode_batch` path. The trainer actually uses
        # action_space only to know the action-space size (known-constant), so
        # we keep things light.
        from sim import config as C
        from sim.actions import ACTION_SPACE_SIZE
        import gymnasium as gym
        self.num_envs = self.n_envs  # gym-compat
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # gym VectorEnv surface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            import numpy as np
            self._rng = np.random.default_rng(seed)
            # Reseed inner env from the provided seed too.
            import numpy as _np
            base = seed
            seeds = _np.arange(self.n_envs, dtype=_np.int64) + base
            self._inner.reset(seeds=seeds)
        else:
            self._inner.reset()
        obs = self._make_obs_batch_from_host(self._bulk_state_copy())
        info: dict = {}
        return obs, info

    def step(self, actions):
        """Accepts per-env P1 action indices. Builds P2 actions from opponent;
        steps inner; returns gym-style 5-tuple.

        Single device->host sync per step: `_bulk_state_copy` pulls every
        batched field once, reused for (a) the opponent policy, (b) the mask
        computation, (c) the returned obs_batch. Previous version did two
        separate `snapshot_numpy_states()` calls per step (~N Python-State
        wrappers each time); that's the chunk of env_step_ns Phase 4 spent.
        """
        import numpy as np
        from sim.actions import decode
        from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND

        actions_arr = np.asarray(actions, dtype=np.int64)
        if actions_arr.shape != (self.n_envs,):
            raise ValueError(f"actions shape {actions_arr.shape} != ({self.n_envs},)")

        # Pull every batched field to host in one shot.
        host = self._bulk_state_copy()

        # Compute P2 actions + P1/P2 masks in a single Python loop over envs.
        # Opponent still takes a `State`-shaped view; build a cheap per-env
        # view that reuses the `host` arrays without copying.
        a_batch, obs_batch = self._pack_step_inputs_and_obs(
            host, actions_arr, decode, ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND,
        )

        result = self._inner.step(a_batch)

        rewards   = result.rewards                       # (n_envs,) float32
        terminated = result.terminated                   # (n_envs,) bool
        truncated  = result.truncated                    # (n_envs,) bool
        infos: dict[str, Any] = {}
        return obs_batch, rewards, terminated, truncated, infos

    def close(self) -> None:
        self._inner.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bulk_state_copy(self) -> dict:
        """Pull every batched StateJax field to host in a single sync.

        Returns a dict of (n_envs, …) numpy arrays keyed the same as a
        StateJax. Used to feed both the opponent policy and the obs_batch
        construction — no second pull.
        """
        import numpy as np
        state = self._inner.state
        return {
            "buildings_alive":    np.asarray(state.buildings_alive),
            "buildings_owner":    np.asarray(state.buildings_owner),
            "buildings_type":     np.asarray(state.buildings_type),
            "buildings_garrison": np.asarray(state.buildings_garrison),
            "buildings_capacity": np.asarray(state.buildings_capacity),
            "buildings_x":        np.asarray(state.buildings_x),
            "buildings_y":        np.asarray(state.buildings_y),
            "groups_alive":       np.asarray(state.groups_alive),
            "groups_owner":       np.asarray(state.groups_owner),
            "groups_src":         np.asarray(state.groups_src),
            "groups_tgt":         np.asarray(state.groups_tgt),
            "groups_count":       np.asarray(state.groups_count),
            "groups_progress":    np.asarray(state.groups_progress),
            "groups_travel":      np.asarray(state.groups_travel),
            "travel_matrix":      np.asarray(state.travel_matrix),
            "tick":               np.asarray(state.tick),
            "phase":              np.asarray(state.phase),
        }

    def _make_obs_batch_from_host(self, host: dict) -> dict:
        """Build the gym-style obs dict from an already-pulled `host` snapshot.

        Reset path uses this directly (no actions to pack). Step path uses
        `_pack_step_inputs_and_obs` which does the same thing plus the
        opponent/action loop.
        """
        import numpy as np
        from sim import config as C
        from sim.actions import compute_mask_batched

        N = self.n_envs
        p1_mask = compute_mask_batched(
            host["buildings_alive"], host["buildings_owner"],
            host["buildings_garrison"], host["groups_alive"],
            C.OWNER_P1,
        )
        return {
            "buildings_alive":    host["buildings_alive"].astype(np.int8, copy=True),
            "buildings_owner":    host["buildings_owner"].astype(np.int8, copy=True),
            "buildings_type":     host["buildings_type"].astype(np.int8, copy=True),
            "buildings_garrison": host["buildings_garrison"].astype(np.int16, copy=True),
            "buildings_capacity": host["buildings_capacity"].astype(np.int16, copy=True),
            "buildings_x":        host["buildings_x"].astype(np.int16, copy=True),
            "buildings_y":        host["buildings_y"].astype(np.int16, copy=True),
            "groups_alive":       host["groups_alive"].astype(np.int8, copy=True),
            "groups_owner":       host["groups_owner"].astype(np.int8, copy=True),
            "groups_src":         host["groups_src"].astype(np.int8, copy=True),
            "groups_tgt":         host["groups_tgt"].astype(np.int8, copy=True),
            "groups_count":       host["groups_count"].astype(np.int16, copy=True),
            "groups_progress":    host["groups_progress"].astype(np.int16, copy=True),
            "groups_travel":      host["groups_travel"].astype(np.int16, copy=True),
            "travel_matrix":      host["travel_matrix"].astype(np.int16, copy=True),
            "tick":               host["tick"].astype(np.int32, copy=True)
                                     if host["tick"].ndim else np.full((N,), int(host["tick"]), dtype=np.int32),
            "action_mask":        p1_mask,
        }

    def _pack_step_inputs_and_obs(
        self,
        host: dict,
        actions_arr,
        decode_fn,
        action_dim: int,
        kind_noop: int,
        kind_send: int,
    ):
        """Single pass over envs: builds the (n_envs, 2, 4) action batch AND
        the gym-style obs_batch dict.

        Fast path (simple opponents: random_legal, noop): mask computation
        and opponent action selection are fully batched numpy — no Python
        per-env loop. This path is the hot one for non-self-play training.

        Fallback (neural opponent): per-env loop, one scratch State reused,
        same shape as the original adapter.
        """
        import numpy as np
        from sim import config as C
        from sim.state import empty_state
        from sim.actions import (
            ACTION_SPACE_SIZE, SLOTS_SQ, NOOP_INDEX,
            compute_mask, compute_mask_batched,
        )
        from sim.envs.opponents import random_legal_opponent_batched

        N = self.n_envs

        # Batched P1 mask — hot path needs this regardless.
        p1_mask = compute_mask_batched(
            host["buildings_alive"], host["buildings_owner"],
            host["buildings_garrison"], host["groups_alive"],
            C.OWNER_P1,
        )

        obs = {
            "buildings_alive":    host["buildings_alive"].astype(np.int8, copy=True),
            "buildings_owner":    host["buildings_owner"].astype(np.int8, copy=True),
            "buildings_type":     host["buildings_type"].astype(np.int8, copy=True),
            "buildings_garrison": host["buildings_garrison"].astype(np.int16, copy=True),
            "buildings_capacity": host["buildings_capacity"].astype(np.int16, copy=True),
            "buildings_x":        host["buildings_x"].astype(np.int16, copy=True),
            "buildings_y":        host["buildings_y"].astype(np.int16, copy=True),
            "groups_alive":       host["groups_alive"].astype(np.int8, copy=True),
            "groups_owner":       host["groups_owner"].astype(np.int8, copy=True),
            "groups_src":         host["groups_src"].astype(np.int8, copy=True),
            "groups_tgt":         host["groups_tgt"].astype(np.int8, copy=True),
            "groups_count":       host["groups_count"].astype(np.int16, copy=True),
            "groups_progress":    host["groups_progress"].astype(np.int16, copy=True),
            "groups_travel":      host["groups_travel"].astype(np.int16, copy=True),
            "travel_matrix":      host["travel_matrix"].astype(np.int16, copy=True),
            "tick":               host["tick"].astype(np.int32, copy=True)
                                     if host["tick"].ndim else np.full((N,), int(host["tick"]), dtype=np.int32),
            "action_mask":        p1_mask,
        }

        a_batch = np.zeros((N, 2, action_dim), dtype=np.int32)

        # Pack P1 actions: decode each index to (kind, type_idx, src, tgt).
        # This is a (n_envs,) Python loop but each iteration is cheap (no
        # state/mask work) — ~50 µs total at n_envs=64.
        for i in range(N):
            a1 = decode_fn(int(actions_arr[i]))
            if a1.kind == "noop":
                a_batch[i, 0] = [kind_noop, 0, 0, 0]
            else:
                a_batch[i, 0] = [kind_send, a1.type_idx, a1.src, a1.tgt]

        if self._opponent_name in ("random_legal", "noop"):
            # Batched P2 mask + random-legal pick + decode. No per-env state view.
            if self._opponent_name == "noop":
                p2_actions = np.full(N, NOOP_INDEX, dtype=np.int64)
            else:
                p2_mask = compute_mask_batched(
                    host["buildings_alive"], host["buildings_owner"],
                    host["buildings_garrison"], host["groups_alive"],
                    C.OWNER_P2,
                )
                p2_actions = random_legal_opponent_batched(p2_mask, self._rng)

            # Decode batched: NOOP = NOOP_INDEX; else (type, src, tgt) from the
            # packing formula.
            noop_mask = (p2_actions == NOOP_INDEX)
            type_idx = (p2_actions // SLOTS_SQ).astype(np.int32)
            rem      = (p2_actions %  SLOTS_SQ).astype(np.int32)
            src_idx  = (rem // C.MAX_BUILDING_SLOTS).astype(np.int32)
            tgt_idx  = (rem %  C.MAX_BUILDING_SLOTS).astype(np.int32)

            a_batch[:, 1, 0] = np.where(noop_mask, kind_noop, kind_send)
            a_batch[:, 1, 1] = np.where(noop_mask, 0, type_idx)
            a_batch[:, 1, 2] = np.where(noop_mask, 0, src_idx)
            a_batch[:, 1, 3] = np.where(noop_mask, 0, tgt_idx)

            return a_batch, obs

        # Neural / unknown opponent: slow path via per-env scratch State.
        scratch = empty_state()
        for i in range(N):
            scratch.buildings_alive    = host["buildings_alive"][i]
            scratch.buildings_owner    = host["buildings_owner"][i]
            scratch.buildings_type     = host["buildings_type"][i]
            scratch.buildings_garrison = host["buildings_garrison"][i]
            scratch.buildings_capacity = host["buildings_capacity"][i]
            scratch.buildings_x        = host["buildings_x"][i]
            scratch.buildings_y        = host["buildings_y"][i]
            scratch.groups_alive       = host["groups_alive"][i]
            scratch.groups_owner       = host["groups_owner"][i]
            scratch.groups_src         = host["groups_src"][i]
            scratch.groups_tgt         = host["groups_tgt"][i]
            scratch.groups_count       = host["groups_count"][i]
            scratch.groups_progress    = host["groups_progress"][i]
            scratch.groups_travel      = host["groups_travel"][i]
            scratch.travel_matrix      = host["travel_matrix"][i]
            scratch.tick  = int(host["tick"][i])  if host["tick"].ndim  else int(host["tick"])
            scratch.phase = int(host["phase"][i]) if host["phase"].ndim else int(host["phase"])
            scratch._refresh_proxies()  # type: ignore[attr-defined]

            a2_idx = int(self._opponent(scratch, self._rng))
            a2 = decode_fn(a2_idx)
            if a2.kind == "noop":
                a_batch[i, 1] = [kind_noop, 0, 0, 0]
            else:
                a_batch[i, 1] = [kind_send, a2.type_idx, a2.src, a2.tgt]

        return a_batch, obs
