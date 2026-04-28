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
    reward_version: int = C.REWARD_VERSION_V12,
    level_mix: list[tuple[str, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> list[State]:
    """Regenerate numpy States for a batch of seeds. CPU-only.

    `level_mix`: optional list of (level_name, weight) pairs. When provided,
    each generated state samples its level independently from this mix
    (reproducible under the supplied `rng`). `level_name` is ignored in this
    case but kept as a fallback for callers that pass mix=None.
    """
    if level_mix:
        if rng is None:
            rng = np.random.default_rng(int(seeds[0]) if len(seeds) else 0)
        names  = [n for n, _ in level_mix]
        probs  = np.asarray([w for _, w in level_mix], dtype=np.float64)
        probs  = probs / probs.sum()
        choices = rng.choice(len(names), size=len(seeds), p=probs)
        return [
            level_reset(names[int(c)], seed=int(s), reward_version=reward_version)
            for c, s in zip(choices, seeds)
        ]
    return [
        level_reset(level_name, seed=int(s), reward_version=reward_version)
        for s in seeds
    ]


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
# Chunked step (action-repetition) — Phase B of FUSED_ROLLOUT_PLAN
# ---------------------------------------------------------------------------
#
# `_step_chunk_impl` takes a SINGLE per-env action pair (a1, a2) and runs K
# env ticks: tick 0 uses the real actions; ticks 1..K-1 use NOOP for both.
# Returns the post-K state, summed P1/P2 rewards, OR-folded done flag.
# K is a Python int (static argument — bakes into the JIT cache key).

def _step_chunk_single(state, action_p1, action_p2, K: int):
    """Run K env ticks for one game with action-repetition.

    Tick 0: (action_p1, action_p2). Ticks 1..K-1: (NOOP, NOOP).
    Returns (final_state, reward_p1_total, reward_p2_total, done_any).
    """
    # Build the (K, 4) action stack with real action at index 0 and NOOP fill
    # for the rest. Use scatter-on-init via where-against-arange so the whole
    # thing stays inside the trace.
    noop = jnp.zeros((4,), dtype=jnp.int32)  # ACTION_KIND_NOOP=0, rest don't care
    idx = jnp.arange(K)
    # tick_mask: (K,) — True at tick 0, False elsewhere.
    tick_mask = (idx == 0)[:, None]
    a1_stack = jnp.where(tick_mask, action_p1[None, :], noop[None, :])  # (K, 4)
    a2_stack = jnp.where(tick_mask, action_p2[None, :], noop[None, :])  # (K, 4)

    final, r1s, r2s, dones = step_many_single(state, a1_stack, a2_stack)
    return final, r1s.sum(), r2s.sum(), dones.any()


def _make_step_chunk_batched(K: int):
    """Build a (state, a1, a2) -> (state, r1, r2, done) function for a fixed K.

    `K` is captured in the closure so jax.jit treats different K's as
    distinct compiled functions — same convention as `step_many_batched` but
    keyed on the chunk size.
    """
    def _chunk(state, a1, a2):
        return _step_chunk_single(state, a1, a2, K)

    return jax.jit(jax.vmap(_chunk, in_axes=(0, 0, 0)))


# Cache compiled chunk functions by K so the trainer can swap K without
# recompiling on every call.
_step_chunk_cache: dict[int, "jax.tree_util.Partial"] = {}


def _step_chunk_batched(state, action_p1, action_p2, K: int):
    """Vmap'd, JIT'd chunked step. K caches across calls."""
    fn = _step_chunk_cache.get(K)
    if fn is None:
        fn = _make_step_chunk_batched(K)
        _step_chunk_cache[K] = fn
    return fn(state, action_p1, action_p2)


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
    rewards:        np.ndarray     # (n_envs,) float32 — P1 rewards
    rewards_p2:     np.ndarray     # (n_envs,) float32 — P2 rewards
    terminated:     np.ndarray     # (n_envs,) bool
    truncated:      np.ndarray     # (n_envs,) bool — always False (no time limit beyond done)
    terminal_phase: np.ndarray     # (n_envs,) int8 — 0=ongoing, 1=P1_WINS, 2=P2_WINS, 3=DRAW
    infos:          Any            # list-like of per-env {phase, tick} — lazily materialised


class JaxVecEnv:
    """`n_envs` games in a single batched StateJax."""

    def __init__(
        self,
        n_envs: int,
        level_name: str = "crossroads_6",
        base_seed: int = 0,
        reward_version: int = C.REWARD_VERSION_V12,
        level_mix: list[tuple[str, float]] | None = None,
    ):
        self.n_envs = int(n_envs)
        self.level_name = str(level_name)
        self.reward_version = int(reward_version)
        # When level_mix is set, each (re)generated env picks a level from the
        # distribution. Each tuple is (level_name, weight); weights normalised.
        self.level_mix = list(level_mix) if level_mix else None
        self._rng = np.random.default_rng(base_seed)
        self._next_reset_seed = int(base_seed)

        # Build the first batch.
        initial_seeds = np.arange(self.n_envs, dtype=np.int64) + int(base_seed)
        self.state: StateJax = _stack_states(
            _gen_state_batch(
                self.level_name, initial_seeds, self.reward_version,
                level_mix=self.level_mix, rng=self._rng,
            )
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
        self.state = _stack_states(
            _gen_state_batch(
                self.level_name, arr, self.reward_version,
                level_mix=self.level_mix, rng=self._rng,
            )
        )

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
        # call each — bulk copies, not per-env scalar pulls. Capture the
        # terminal phase BEFORE auto-reset overwrites it so callers can
        # tell winner from loser without reading the noisy reward proxy.
        r1_np      = np.asarray(r1)
        r2_np      = np.asarray(r2)
        done_np    = np.asarray(done)
        phase_np   = np.asarray(self.state.phase)  # 1=P1_WINS, 2=P2_WINS, 3=DRAW, 0=mid-game

        # Vectorised auto-reset.
        if done_np.any():
            self._auto_reset(done_np)

        # Defer the per-env info dicts: each `int(self.state.phase[i])` forces
        # a device-to-host sync, which at n_envs=1024 burns the whole benefit
        # of vmap. Build the list lazily so callers that don't read `infos`
        # (our bench, PPO rollout) pay nothing.
        return JaxVecStepResult(
            rewards         = r1_np.astype(np.float32),
            rewards_p2      = r2_np.astype(np.float32),
            terminated      = done_np,
            truncated       = np.zeros(self.n_envs, dtype=bool),
            terminal_phase  = phase_np.astype(np.int8),
            infos           = _LazyInfos(self.state, self.n_envs),
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

    def step_chunk(self, actions: np.ndarray, K: int) -> dict:
        """Run K env ticks per env in one fused XLA dispatch with action-
        repetition. Returns chunk-summed rewards + OR-folded done.

        `actions` shape: (n_envs, 2, ACTION_DIM) int32 — one P1+P2 action
        pair per env. Tick 0 of the chunk uses these; ticks 1..K-1 use NOOP.
        After the chunk, done envs auto-reset (CPU level-gen + splice).

        Designed for the fused-rollout PPO collector: one agent decision
        per K env ticks, summed reward per chunk per env. K is a Python int
        (static — JIT cache key); reusing K across calls hits the cache.

        Returns:
          {
            "rewards":    (n_envs,) float32 — sum of P1 reward over K ticks,
            "rewards_p2": (n_envs,) float32 — same for P2,
            "dones":      (n_envs,) bool    — any tick of the chunk hit done,
            "K":          K,
          }
        """
        if actions.shape != (self.n_envs, 2, ACTION_DIM):
            raise ValueError(
                f"step_chunk actions must be ({self.n_envs}, 2, {ACTION_DIM}); "
                f"got {actions.shape}"
            )
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")

        a1 = jnp.asarray(actions[:, 0, :], dtype=jnp.int32)
        a2 = jnp.asarray(actions[:, 1, :], dtype=jnp.int32)

        self.state, r1, r2, done = _step_chunk_batched(
            self.state, a1, a2, K,
        )
        r1_np    = np.asarray(r1)
        r2_np    = np.asarray(r2)
        done_np  = np.asarray(done)
        # Capture terminal phase BEFORE auto-reset (parity with .step()).
        phase_np = np.asarray(self.state.phase).astype(np.int8)

        # Auto-reset on done (same path as `.step()`).
        if done_np.any():
            self._auto_reset(done_np)

        return {
            "rewards":        r1_np.astype(np.float32),
            "rewards_p2":     r2_np.astype(np.float32),
            "dones":          done_np,
            "terminal_phase": phase_np,
            "K":              K,
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
            s.reward_version = int(host.reward_version[i])
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
        fresh = _stack_states(
            _gen_state_batch(
                self.level_name, seeds, self.reward_version,
                level_mix=self.level_mix, rng=self._rng,
            )
        )  # (n, …)

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
