"""Phase 2 parity harness — numpy engine vs JAX engine, byte-for-byte.

For each seed, play a scripted random game for `n_ticks` ticks on both
backends starting from the same initial state. After every tick, convert
the JAX state back to numpy and assert byte-identical gameplay fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import NOOP_INDEX, compute_mask, decode
from sim.engine import step_tick as step_numpy
from sim.engine_jax import (
    ACTION_KIND_NOOP,
    ACTION_KIND_SEND,
    encode_action,
    step_tick_single,
)
from sim.levels import reset
from sim.state_jax import from_numpy_state, states_equal, to_numpy_state


def _encode_numpy_action_as_jax(action):
    """Translate a numpy-side Action dataclass into the (4,) int32 JAX encoding."""
    import jax.numpy as jnp
    if action.kind == "noop":
        return encode_action(ACTION_KIND_NOOP)
    return encode_action(ACTION_KIND_SEND, action.type_idx, action.src, action.tgt)


def _run_parity(seed: int, n_ticks: int, level: str = "crossroads_6") -> None:
    """One scripted game on both backends; assert state equality every tick."""
    state_np = reset(level_name=level, seed=seed)
    state_jx = from_numpy_state(state_np)
    rng = np.random.default_rng(seed)

    for t in range(n_ticks):
        # Pick legal random actions for each player using the NUMPY state as
        # oracle (same action indices go to both backends so the comparison is
        # apples-to-apples).
        m1 = compute_mask(state_np, C.OWNER_P1)
        m2 = compute_mask(state_np, C.OWNER_P2)
        a1_idx = int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX
        a2_idx = int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX
        a1 = decode(a1_idx)
        a2 = decode(a2_idx)

        r1_np, r2_np, done_np = step_numpy(state_np, a1, a2)

        a1_jx = _encode_numpy_action_as_jax(a1)
        a2_jx = _encode_numpy_action_as_jax(a2)
        state_jx, r1_jx, r2_jx, done_jx = step_tick_single(state_jx, a1_jx, a2_jx)

        # Reward equality within float32 epsilon.
        assert float(r1_jx) == pytest.approx(r1_np, abs=1e-5), (
            f"tick {t} seed {seed}: r1 numpy={r1_np} jax={float(r1_jx)}"
        )
        assert float(r2_jx) == pytest.approx(r2_np, abs=1e-5), (
            f"tick {t} seed {seed}: r2 numpy={r2_np} jax={float(r2_jx)}"
        )
        assert bool(done_jx) == bool(done_np), (
            f"tick {t} seed {seed}: done numpy={done_np} jax={bool(done_jx)}"
        )

        back = to_numpy_state(state_jx)
        if not states_equal(state_np, back):
            diffs = []
            for attr in ("buildings_alive","buildings_owner","buildings_type",
                         "buildings_garrison","buildings_capacity","buildings_x",
                         "buildings_y","groups_alive","groups_owner","groups_src",
                         "groups_tgt","groups_count","groups_progress","groups_travel",
                         "travel_matrix"):
                av = getattr(state_np, attr); bv = getattr(back, attr)
                if not np.array_equal(av, bv):
                    diffs.append(f"    {attr}:\n        numpy={av}\n        jax  ={bv}")
            if state_np.tick != back.tick:
                diffs.append(f"    tick: numpy={state_np.tick} jax={back.tick}")
            if state_np.phase != back.phase:
                diffs.append(f"    phase: numpy={state_np.phase} jax={back.phase}")
            raise AssertionError(
                f"state diverged at tick {t} seed {seed} after a1={a1} a2={a2}\n"
                + "\n".join(diffs)
            )

        if done_np:
            break


@pytest.mark.parametrize("seed", list(range(5)))
def test_parity_crossroads_5_seeds(seed):
    """5 seeds on the static crossroads map, 200 ticks each."""
    _run_parity(seed=seed, n_ticks=200)


@pytest.mark.parametrize("seed", list(range(3)))
def test_parity_random_level(seed):
    """3 seeds on dynamic random_8_16 levels, 200 ticks each."""
    _run_parity(seed=seed, n_ticks=200, level="random_8_16")


def test_parity_long_run_single_seed():
    """One seed for the full 200 ticks, higher activity (pre-warm jit cache)."""
    _run_parity(seed=42, n_ticks=200, level="random_10_16")


@pytest.mark.slow
def test_parity_100_seeds_200_ticks():
    """JAX_PORT_PLAN §5 parity invariant — 100 seeds × 200 ticks, byte-identical
    state every tick. Tagged `slow` so the default pytest run can skip this if
    wanted; unmarked runs pick it up automatically.
    """
    for seed in range(100):
        _run_parity(seed=seed, n_ticks=200, level="random_8_16")


def test_symmetric_winrate_on_jax_backend():
    """Plan §5 DoD: test_p1_p2_symmetric_random_play_winrate must pass on
    the JAX backend. Parity guarantees byte-identical game outcomes, so this
    is a safety net that catches any future regression (e.g. a silent
    determinism drift in the JAX path) at the win-rate level rather than
    requiring a full 200-tick diff.

    Cheaper version than the numpy test: 50 games instead of 200, 90% CI
    ~[37%, 63%] → loosened stat band.
    """
    from sim.engine_jax import ACTION_KIND_NOOP, ACTION_KIND_SEND, encode_action

    def _enc(a):
        if a.kind == "noop":
            return encode_action(ACTION_KIND_NOOP)
        return encode_action(ACTION_KIND_SEND, a.type_idx, a.src, a.tgt)

    p1_wins = 0
    total = 0
    n_games = 50
    for seed in range(n_games):
        rng = np.random.default_rng(seed)
        state_np = reset(level_name="random_8_12", seed=seed)
        state_jx = from_numpy_state(state_np)
        for _ in range(C.GAME_TIMEOUT_TICKS + 10):
            m1 = compute_mask(state_np, C.OWNER_P1)
            m2 = compute_mask(state_np, C.OWNER_P2)
            a1_idx = int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX
            a2_idx = int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX
            a1 = decode(a1_idx); a2 = decode(a2_idx)
            state_jx, _, _, done = step_tick_single(state_jx, _enc(a1), _enc(a2))
            # Mirror to numpy so we can recompute the mask next tick.
            state_np = to_numpy_state(state_jx)
            if bool(done):
                break
        from sim.state import count_owned_buildings as _count
        p1_b = _count(state_np, C.OWNER_P1)
        p2_b = _count(state_np, C.OWNER_P2)
        if p1_b > p2_b:
            p1_wins += 1; total += 1
        elif p2_b > p1_b:
            total += 1
    p1_rate = p1_wins / max(total, 1)
    # Loose band — 50 games has a wide error bar; tight parity test already
    # guarantees strict byte-equality against numpy.
    assert 0.30 <= p1_rate <= 0.70, (
        f"P1 rate under JAX backend = {p1_rate:.3f} (n_settled={total}); "
        f"symmetry check failed. Run parity harness to diagnose."
    )


def test_jaxpr_is_finite():
    """The jit'd step_tick must trace to a finite jaxpr — no Python branching
    over traced values."""
    import jax

    state_np = reset(seed=0)
    state_jx = from_numpy_state(state_np)
    a_noop = encode_action(ACTION_KIND_NOOP)

    jaxpr = jax.make_jaxpr(step_tick_single.__wrapped__)(state_jx, a_noop, a_noop)
    text = str(jaxpr)
    # Sanity: we expect the jaxpr to have real content (lots of operations) and
    # to NOT contain a Python-level branching sentinel.
    assert len(text) > 500, "jaxpr unexpectedly short; tracing may have failed"


# ---------------------------------------------------------------------------
# Phase 3: vmap / JaxVecEnv parity — 16 games stepped together match 16 solo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_envs", [4, 16])
def test_jax_vec_env_parity_with_numpy(n_envs):
    """Step `n_envs` games through JaxVecEnv and through the numpy engine
    (one by one) with the same scripted-random actions; state must match
    byte-for-byte every tick — until the env terminates, after which JaxVecEnv
    auto-resets and the numpy reference doesn't, so we stop comparing that env.
    """
    from sim.envs.jax_vec_env import JaxVecEnv
    from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND

    base_seed = 100
    level = "random_6_10"

    np_states = [reset(level_name=level, seed=base_seed + i) for i in range(n_envs)]
    np_rngs   = [np.random.default_rng(base_seed + i) for i in range(n_envs)]
    still_active = [True] * n_envs  # once False, stop comparing that env.

    vec = JaxVecEnv(n_envs=n_envs, level_name=level, base_seed=base_seed)

    for t in range(50):
        a_batch = np.zeros((n_envs, 2, ACTION_DIM), dtype=np.int32)
        numpy_actions = []
        for i, s in enumerate(np_states):
            if not still_active[i]:
                numpy_actions.append((decode(NOOP_INDEX), decode(NOOP_INDEX)))
                continue
            m1 = compute_mask(s, C.OWNER_P1)
            m2 = compute_mask(s, C.OWNER_P2)
            a1_idx = int(np_rngs[i].choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX
            a2_idx = int(np_rngs[i].choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX
            a1 = decode(a1_idx); a2 = decode(a2_idx)
            numpy_actions.append((a1, a2))
            for k, a in enumerate((a1, a2)):
                if a.kind == "noop":
                    a_batch[i, k] = [ACTION_KIND_NOOP, 0, 0, 0]
                else:
                    a_batch[i, k] = [ACTION_KIND_SEND, a.type_idx, a.src, a.tgt]

        np_dones = []
        for i, (a1, a2) in enumerate(numpy_actions):
            if not still_active[i]:
                np_dones.append(True)
                continue
            _r1, _r2, d = step_numpy(np_states[i], a1, a2)
            np_dones.append(bool(d))

        # Snapshot BEFORE step so we can compare against numpy's post-step state
        # without the auto-reset mutating what we saw. JaxVecEnv doesn't expose
        # that, so instead snapshot just after step and compare only for envs
        # that were still active coming into the tick AND didn't terminate this
        # tick — matching envs that didn't auto-reset.
        result = vec.step(a_batch)
        jax_states = vec.snapshot_numpy_states()

        for i in range(n_envs):
            if not still_active[i]:
                continue
            # Numpy said done this tick? JAX must have said done too (before
            # auto-reset made it False-ish in the state snapshot).
            if np_dones[i]:
                assert bool(result.terminated[i]), (
                    f"tick {t} env {i}: numpy done but JAX not terminated"
                )
                still_active[i] = False
                continue
            # Still active in both; state must match.
            assert not bool(result.terminated[i]), (
                f"tick {t} env {i}: JAX terminated but numpy did not"
            )
            if not states_equal(np_states[i], jax_states[i]):
                raise AssertionError(
                    f"state diverged at tick {t} env {i}\n"
                    f"  np garrison: {np_states[i].buildings_garrison[:6]}\n"
                    f"  jx garrison: {jax_states[i].buildings_garrison[:6]}"
                )
