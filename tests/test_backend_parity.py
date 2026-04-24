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
