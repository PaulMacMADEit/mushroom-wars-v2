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
):
    """Return a gym-style vector env on the active backend.

    `vec_mode` and `context` only apply to the numpy backend; JAX ignores
    them (one-process by design).
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
):
    # Lazy-import: don't pay the JAX startup tax when the numpy backend is active.
    from sim.envs.jax_vec_env import JaxVecEnv
    return _JaxVecAdapter(
        n_envs=n_envs, seed=seed, level_name=level_name,
        opponent_name=opponent_name, opponent_kwargs=opponent_kwargs,
        inner_factory=lambda: JaxVecEnv(n_envs=n_envs, level_name=level_name, base_seed=seed),
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
        obs = self._make_obs_batch()
        info: dict = {}
        return obs, info

    def step(self, actions):
        """Accepts per-env P1 action indices. Builds P2 actions from opponent;
        steps inner; returns gym-style 5-tuple."""
        import numpy as np
        from sim.actions import decode
        from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND

        actions_arr = np.asarray(actions, dtype=np.int64)
        if actions_arr.shape != (self.n_envs,):
            raise ValueError(f"actions shape {actions_arr.shape} != ({self.n_envs},)")

        # Snapshot numpy states for opponent-policy + mask computation.
        states = self._inner.snapshot_numpy_states()

        a_batch = np.zeros((self.n_envs, 2, ACTION_DIM), dtype=np.int32)
        for i, s in enumerate(states):
            # P1 from caller.
            a1 = decode(int(actions_arr[i]))
            # P2 from opponent policy.
            a2_idx = int(self._opponent(s, self._rng))
            a2 = decode(a2_idx)
            for k, a in enumerate((a1, a2)):
                if a.kind == "noop":
                    a_batch[i, k] = [ACTION_KIND_NOOP, 0, 0, 0]
                else:
                    a_batch[i, k] = [ACTION_KIND_SEND, a.type_idx, a.src, a.tgt]

        result = self._inner.step(a_batch)

        obs_batch = self._make_obs_batch()
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

    def _make_obs_batch(self) -> dict:
        """Materialise per-env numpy obs dicts, stack into (n_envs, …) arrays.

        Keys mirror `MushroomEnv._make_obs` so `training.encoder.encode_obs`
        consumes each env's slice unchanged.
        """
        import numpy as np
        from sim import config as C
        from sim.actions import compute_mask, ACTION_SPACE_SIZE

        states = self._inner.snapshot_numpy_states()
        N = self.n_envs
        MAX_B = C.MAX_BUILDING_SLOTS
        MAX_G = C.MAX_UNIT_GROUP_SLOTS

        out = {
            "buildings_alive":    np.empty((N, MAX_B), dtype=np.int8),
            "buildings_owner":    np.empty((N, MAX_B), dtype=np.int8),
            "buildings_type":     np.empty((N, MAX_B), dtype=np.int8),
            "buildings_garrison": np.empty((N, MAX_B), dtype=np.int16),
            "buildings_capacity": np.empty((N, MAX_B), dtype=np.int16),
            "buildings_x":        np.empty((N, MAX_B), dtype=np.int16),
            "buildings_y":        np.empty((N, MAX_B), dtype=np.int16),
            "groups_alive":       np.empty((N, MAX_G), dtype=np.int8),
            "groups_owner":       np.empty((N, MAX_G), dtype=np.int8),
            "groups_src":         np.empty((N, MAX_G), dtype=np.int8),
            "groups_tgt":         np.empty((N, MAX_G), dtype=np.int8),
            "groups_count":       np.empty((N, MAX_G), dtype=np.int16),
            "groups_progress":    np.empty((N, MAX_G), dtype=np.int16),
            "groups_travel":      np.empty((N, MAX_G), dtype=np.int16),
            "travel_matrix":      np.empty((N, MAX_B, MAX_B), dtype=np.int16),
            "tick":               np.empty((N,),        dtype=np.int32),
            "action_mask":        np.empty((N, ACTION_SPACE_SIZE), dtype=bool),
        }
        for i, s in enumerate(states):
            out["buildings_alive"][i]    = s.buildings_alive
            out["buildings_owner"][i]    = s.buildings_owner
            out["buildings_type"][i]     = s.buildings_type
            out["buildings_garrison"][i] = s.buildings_garrison
            out["buildings_capacity"][i] = s.buildings_capacity
            out["buildings_x"][i]        = s.buildings_x
            out["buildings_y"][i]        = s.buildings_y
            out["groups_alive"][i]    = s.groups_alive
            out["groups_owner"][i]    = s.groups_owner
            out["groups_src"][i]      = s.groups_src
            out["groups_tgt"][i]      = s.groups_tgt
            out["groups_count"][i]    = s.groups_count
            out["groups_progress"][i] = s.groups_progress
            out["groups_travel"][i]   = s.groups_travel
            out["travel_matrix"][i]   = s.travel_matrix
            out["tick"][i]            = s.tick
            out["action_mask"][i]     = compute_mask(s, C.OWNER_P1)
        return out
