"""
JAX-backed vectorised environment.

Holds `n_envs` games as a single batched `StateJax` pytree and steps them
through `jax.vmap(step_tick_single)` — one XLA kernel per tick instead of
N Python-per-env dispatches. This is where the JAX port earns its keep:
the whole rollout phase becomes one fused GPU call (on PaulLinux; on Mac
CPU it's still useful for correctness checks but won't beat multiprocess).

Boundary contract:
- `reset(seeds)`: levels are generated on numpy (see JAX_PORT_PLAN §3.4) then
  lifted into a batched StateJax via `from_numpy_state` + stacking.
- `step(actions)`: takes (n_envs, 4) int32 actions, returns numpy arrays for
  rewards/dones/infos at the vec-env boundary so downstream trainer code
  stays unchanged (JAX_PORT_PLAN §3.5).
- Auto-reset-on-done is vectorised: regenerate fresh states on CPU, then
  overwrite done-env slots via jnp.where.

Not a subclass of `gymnasium.vector.VectorEnv` (which would force each env
into a subprocess). The trainer integration in Phase 4 swaps in this class
behind the same `make_vec_env` factory used for AsyncVectorEnv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from sim import config as C
from sim.engine_jax import (
    ACTION_DIM,
    ACTION_KIND_NOOP,
    ACTION_KIND_SEND,
    step_many_single,
    step_tick_single,
)
from sim.levels import reset as level_reset
from sim.state import State
from sim.state_jax import StateJax, from_numpy_state


# ---------------------------------------------------------------------------
# Helpers: stack numpy States into a batched StateJax
# ---------------------------------------------------------------------------

def _stack_states(states: list[State]) -> StateJax:
    """Stack a list of numpy States into a batched StateJax.

    Leading dim = len(states). Each field gets shape (n_envs, …).
    """
    leaves = [from_numpy_state(s) for s in states]
    # tree_map over the per-env StateJax list → batched StateJax.
    return jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *leaves)


def _gen_state_batch(
    level_name: str,
    seeds: np.ndarray,
) -> list[State]:
    """Regenerate numpy States for a batch of seeds. CPU-only."""
    return [level_reset(level_name, seed=int(s)) for s in seeds]


# vmap over the first axis for every StateJax leaf + the actions.
_step_batched = jax.jit(jax.vmap(step_tick_single, in_axes=(0, 0, 0)))

# Multi-tick fused step over N envs. Actions come in as (T, n_envs, 4); scan
# lives inside the jit, so T ticks resolve in ONE XLA dispatch — collapses
# the per-launch overhead that caps single-step throughput at ~2k tick/s on
# CUDA. vmap axis for actions is 1 (the per-env axis); scan walks axis 0 (T).
_step_many_batched = jax.jit(
    jax.vmap(step_many_single, in_axes=(0, 1, 1))
)


# ---------------------------------------------------------------------------
# JaxVecEnv
# ---------------------------------------------------------------------------

class _LazyInfos:
    """Defers per-env info dict construction until the caller actually
    indexes into the list. Materialising all N infos forces N device-to-host
    syncs which is the whole `vmap` win back through the drain — trainer code
    that ignores `infos` pays nothing."""

    __slots__ = ("_state", "_n", "_materialised")

    def __init__(self, state, n_envs: int):
        self._state = state
        self._n = n_envs
        self._materialised: list[dict] | None = None

    def _materialise(self) -> list[dict]:
        if self._materialised is None:
            phase = np.asarray(self._state.phase)
            tick  = np.asarray(self._state.tick)
            self._materialised = [
                {"phase": int(phase[i]), "tick": int(tick[i])}
                for i in range(self._n)
            ]
        return self._materialised

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        return self._materialise()[i]

    def __iter__(self):
        return iter(self._materialise())


@dataclass
class JaxVecStepResult:
    """Mirror of gymnasium's (obs, reward, terminated, truncated, info) — but
    with numpy arrays at the boundary. `obs` is not built by the env itself
    (encoder-as-trainer rule in ARCHITECTURE §9); callers that need the
    gym-dict obs can grab it from `env.snapshot_numpy_states()`.
    """
    rewards:    np.ndarray     # (n_envs,) float32 — P1 rewards
    rewards_p2: np.ndarray     # (n_envs,) float32 — P2 rewards
    terminated: np.ndarray     # (n_envs,) bool
    truncated:  np.ndarray     # (n_envs,) bool — always False (no time limit beyond done)
    infos:      Any            # list-like of per-env {phase, tick} — lazily materialised


class JaxVecEnv:
    """`n_envs` games in a single batched StateJax."""

    def __init__(
        self,
        n_envs: int,
        level_name: str = "crossroads_6",
        base_seed: int = 0,
    ):
        self.n_envs = int(n_envs)
        self.level_name = str(level_name)
        self._rng = np.random.default_rng(base_seed)
        self._next_reset_seed = int(base_seed)

        # Build the first batch.
        initial_seeds = np.arange(self.n_envs, dtype=np.int64) + int(base_seed)
        self.state: StateJax = _stack_states(
            _gen_state_batch(self.level_name, initial_seeds)
        )
        self._next_reset_seed += self.n_envs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seeds: Optional[Iterable[int]] = None) -> None:
        """Reset all envs. Pass a seed list for reproducible benches."""
        if seeds is None:
            arr = np.arange(self.n_envs, dtype=np.int64) + self._next_reset_seed
            self._next_reset_seed += self.n_envs
        else:
            arr = np.asarray(list(seeds), dtype=np.int64)
            if arr.shape != (self.n_envs,):
                raise ValueError(f"seeds must have length {self.n_envs}, got {arr.shape}")
        self.state = _stack_states(_gen_state_batch(self.level_name, arr))

    def step(self, actions: np.ndarray) -> JaxVecStepResult:
        """Step every env by one tick. `actions` shape (n_envs, 2, 4) int32:
        per env [action_p1, action_p2] each a (4,) [kind, type_idx, src, tgt].
        """
        if actions.shape != (self.n_envs, 2, ACTION_DIM):
            raise ValueError(
                f"actions must be shape ({self.n_envs}, 2, {ACTION_DIM}); got {actions.shape}"
            )

        a1 = jnp.asarray(actions[:, 0, :], dtype=jnp.int32)
        a2 = jnp.asarray(actions[:, 1, :], dtype=jnp.int32)

        self.state, r1, r2, done = _step_batched(self.state, a1, a2)

        # Bring reward/done to host for the trainer boundary. One sync per
        # call each — bulk copies, not per-env scalar pulls.
        r1_np   = np.asarray(r1)
        r2_np   = np.asarray(r2)
        done_np = np.asarray(done)

        # Vectorised auto-reset.
        if done_np.any():
            self._auto_reset(done_np)

        # Defer the per-env info dicts: each `int(self.state.phase[i])` forces
        # a device-to-host sync, which at n_envs=1024 burns the whole benefit
        # of vmap. Build the list lazily so callers that don't read `infos`
        # (our bench, PPO rollout) pay nothing.
        return JaxVecStepResult(
            rewards    = r1_np.astype(np.float32),
            rewards_p2 = r2_np.astype(np.float32),
            terminated = done_np,
            truncated  = np.zeros(self.n_envs, dtype=bool),
            infos      = _LazyInfos(self.state, self.n_envs),
        )

    def step_many(self, actions: np.ndarray) -> dict:
        """Run T ticks in one fused XLA dispatch (scan inside the jit).

        `actions` shape: (T, n_envs, 2, ACTION_DIM) int32. Skips auto-reset
        entirely — the caller is responsible for resetting after the batch.
        Use when you want maximum throughput (bench / synthetic rollouts);
        for the trainer's PPO rollout with per-tick decisions, keep using
        `.step()`.

        Returns {"rewards": (T, n_envs), "rewards_p2": (T, n_envs),
                 "dones": (T, n_envs)} as numpy arrays. Final state replaces
        `self.state`.
        """
        if actions.ndim != 4 or actions.shape[1] != self.n_envs or actions.shape[2] != 2 or actions.shape[3] != ACTION_DIM:
            raise ValueError(
                f"step_many actions must be (T, {self.n_envs}, 2, {ACTION_DIM}); got {actions.shape}"
            )
        T = actions.shape[0]

        actions_jx = jnp.asarray(actions, dtype=jnp.int32)
        a1 = actions_jx[:, :, 0, :]   # (T, n_envs, 4)
        a2 = actions_jx[:, :, 1, :]

        self.state, r1s, r2s, dones = _step_many_batched(self.state, a1, a2)
        # vmap(..., in_axes=(0, 1, 1)) pushes the batch axis to the OUTPUT
        # front by default → r1s has shape (n_envs, T). Transpose so caller
        # sees (T, n_envs) like .step() would.
        r1_np   = np.asarray(r1s).T
        r2_np   = np.asarray(r2s).T
        done_np = np.asarray(dones).T
        return {
            "rewards":    r1_np.astype(np.float32),
            "rewards_p2": r2_np.astype(np.float32),
            "dones":      done_np,
            "ticks":      T,
        }

    def close(self) -> None:
        """No subprocess resources to release; here for gym parity."""
        pass

    def snapshot_numpy_states(self) -> list[State]:
        """Materialise the current batched state into a list of numpy States.
        Used by parity tests and by trainer code that still speaks numpy."""
        host = jax.tree_util.tree_map(np.asarray, self.state)
        out = []
        from sim.state import empty_state
        for i in range(self.n_envs):
            s = empty_state()
            s.buildings_alive[:]    = host.buildings_alive[i]
            s.buildings_owner[:]    = host.buildings_owner[i]
            s.buildings_type[:]     = host.buildings_type[i]
            s.buildings_garrison[:] = host.buildings_garrison[i]
            s.buildings_capacity[:] = host.buildings_capacity[i]
            s.buildings_x[:]        = host.buildings_x[i]
            s.buildings_y[:]        = host.buildings_y[i]
            s.groups_alive[:]    = host.groups_alive[i]
            s.groups_owner[:]    = host.groups_owner[i]
            s.groups_src[:]      = host.groups_src[i]
            s.groups_tgt[:]      = host.groups_tgt[i]
            s.groups_count[:]    = host.groups_count[i]
            s.groups_progress[:] = host.groups_progress[i]
            s.groups_travel[:]   = host.groups_travel[i]
            s.travel_matrix[:]   = host.travel_matrix[i]
            s.tick  = int(host.tick[i])
            s.phase = int(host.phase[i])
            out.append(s)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auto_reset(self, done_mask: np.ndarray) -> None:
        """For every env with done=True, generate a fresh level (on CPU), lift
        to JAX, and overwrite that slot in the batched StateJax.
        """
        idx = np.where(done_mask)[0]
        n = len(idx)
        if n == 0:
            return

        seeds = np.arange(n, dtype=np.int64) + self._next_reset_seed
        self._next_reset_seed += n
        fresh = _stack_states(_gen_state_batch(self.level_name, seeds))  # (n, …)

        # Build a (n_envs,) bool mask + scatter each field.
        mask = jnp.asarray(done_mask)  # (n_envs,)

        def _splice(batched_field, fresh_field):
            # fresh_field is (n, …); need to scatter-assign into batched_field
            # at `idx`. Use jnp.where on a full-size expansion.
            scratch = batched_field
            scratch = scratch.at[jnp.asarray(idx)].set(fresh_field)
            # Above already wrote only the done slots; mask isn't needed — but
            # the scatter might silently clobber if shape mismatch. Guard with
            # where as belt-and-braces.
            rank = batched_field.ndim
            mask_b = mask.reshape((self.n_envs,) + (1,) * (rank - 1))
            return jnp.where(mask_b, scratch, batched_field)

        self.state = jax.tree_util.tree_map(_splice, self.state, fresh)
