"""Tests for sim.envs.MushroomEnv.

Focus: gymnasium contract (reset/step return shapes and dtypes), the
decision-interval batching rule, opponent plumbing, and termination.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import config as C
from sim.actions import (
    ACTION_SPACE_SIZE,
    NOOP_INDEX,
    Action,
    decode,
    encode,
    is_valid,
)
from sim.envs import MushroomEnv, noop_opponent, random_legal_opponent


P1_BASE = 0
P2_BASE = 1
N1_TOP = 2


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_reset_returns_obs_and_info():
    env = MushroomEnv(seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert info["tick"] == 0
    assert info["phase"] == C.PHASE_PLAYING


def test_action_space_size():
    env = MushroomEnv()
    assert env.action_space.n == ACTION_SPACE_SIZE


def test_obs_has_action_mask_and_noop_always_legal():
    env = MushroomEnv(seed=1)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_SPACE_SIZE,)
    assert obs["action_mask"][NOOP_INDEX]


def test_obs_mirrors_underlying_state_after_reset():
    env = MushroomEnv(seed=2)
    obs, _ = env.reset()
    # crossroads_6 seeds: P1 and P2 bases at 10 real units each.
    assert int(obs["buildings_owner"][P1_BASE]) == C.OWNER_P1
    assert int(obs["buildings_owner"][P2_BASE]) == C.OWNER_P2
    assert int(obs["buildings_garrison"][P1_BASE]) == 10 * C.SCALE
    assert int(obs["buildings_garrison"][P2_BASE]) == 10 * C.SCALE


def test_step_returns_five_tuple_with_expected_types():
    env = MushroomEnv(seed=3)
    env.reset()
    out = env.step(NOOP_INDEX)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Decision interval batching
# ---------------------------------------------------------------------------

def test_step_advances_decision_interval_sim_ticks():
    env = MushroomEnv(seed=4, opponent=noop_opponent)
    env.reset()
    tick0 = env.state.tick
    env.step(NOOP_INDEX)
    assert env.state.tick == tick0 + C.DECISION_INTERVAL_TICKS


def test_custom_decision_interval_respected():
    env = MushroomEnv(seed=5, opponent=noop_opponent, decision_interval=1)
    env.reset()
    env.step(NOOP_INDEX)
    assert env.state.tick == 1


def test_action_applied_on_first_inner_tick_only():
    """The send action should spawn exactly one group on the first inner tick,
    and subsequent inner ticks should not spawn additional groups."""
    env = MushroomEnv(seed=6, opponent=noop_opponent, decision_interval=3)
    env.reset()
    # Send to P2 base (travel = 4 ticks > decision_interval), so the group is
    # still in flight after the step — one group if the action applied exactly
    # once, more than one if it duplicated across inner ticks.
    env.step(encode(type_idx=3, src=P1_BASE, tgt=P2_BASE))
    alive = env.state.unit_groups[env.state.unit_groups["alive"] == 1]
    assert len(alive) == 1
    assert int(alive[0]["owner"]) == C.OWNER_P1


# ---------------------------------------------------------------------------
# Opponent plumbing
# ---------------------------------------------------------------------------

def test_opponent_is_invoked_each_step():
    calls: list[int] = []

    def counting_opponent(state, rng):
        calls.append(int(state.tick))
        return NOOP_INDEX

    env = MushroomEnv(seed=7, opponent=counting_opponent)
    env.reset()
    env.step(NOOP_INDEX)
    env.step(NOOP_INDEX)
    assert len(calls) == 2


def test_random_legal_opponent_produces_legal_actions():
    rng = np.random.default_rng(8)
    env = MushroomEnv(seed=8, opponent=random_legal_opponent)
    env.reset()
    # Force a situation where P2 can send: run a few ticks to build garrison.
    for _ in range(5):
        env.step(NOOP_INDEX)
    # Sample directly from the opponent to verify legality.
    action_idx = random_legal_opponent(env.state, rng)
    action = decode(action_idx)
    assert is_valid(env.state, C.OWNER_P2, action)


def test_default_opponent_is_random_legal():
    env = MushroomEnv(seed=9)
    # Smoke: stepping with random-legal P2 should not crash and should
    # eventually produce enemy activity.
    env.reset()
    saw_p2_group = False
    for _ in range(60):
        _, _, terminated, _, _ = env.step(NOOP_INDEX)
        g = env.state.unit_groups
        if np.any((g["alive"] == 1) & (g["owner"] == C.OWNER_P2)):
            saw_p2_group = True
            break
        if terminated:
            break
    assert saw_p2_group


# ---------------------------------------------------------------------------
# Reward + termination
# ---------------------------------------------------------------------------

def test_reward_captured_in_step_return():
    """When a capture resolves inside the decision interval, the env should
    emit REWARD_CAPTURE as P1's reward for that step."""
    env = MushroomEnv(seed=10, opponent=noop_opponent)
    env.reset()
    # Force a fast arrival: synthetic travel time = 1.
    env.state.travel_matrix[P1_BASE, N1_TOP] = 1
    total = 0.0
    _, r, _, _, _ = env.step(encode(type_idx=3, src=P1_BASE, tgt=N1_TOP))
    total += r
    # Action lands in the same env step because travel=1 and action applies
    # before movement on the first inner tick.
    assert int(env.state.buildings["owner"][N1_TOP]) == C.OWNER_P1
    assert total == pytest.approx(C.REWARD_CAPTURE)


def test_env_terminates_on_elimination():
    env = MushroomEnv(seed=11, opponent=noop_opponent)
    env.reset()
    # Wipe P2 directly.
    env.state.buildings["owner"][P2_BASE] = C.OWNER_NEUTRAL
    env.state.buildings["garrison"][P2_BASE] = 0
    env.state.unit_groups[:] = 0
    _, reward, terminated, truncated, info = env.step(NOOP_INDEX)
    assert terminated
    assert not truncated
    assert reward == pytest.approx(C.REWARD_WIN)
    assert info["phase"] == C.PHASE_P1_WINS


def test_step_after_terminal_is_safe():
    env = MushroomEnv(seed=12, opponent=noop_opponent)
    env.reset()
    env.state.buildings["owner"][P2_BASE] = C.OWNER_NEUTRAL
    env.state.buildings["garrison"][P2_BASE] = 0
    env.state.unit_groups[:] = 0
    env.step(NOOP_INDEX)
    # Another step should not crash and should report terminated.
    _, reward, terminated, _, _ = env.step(NOOP_INDEX)
    assert terminated
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Determinism via seed
# ---------------------------------------------------------------------------

def test_random_opponent_is_deterministic_under_seed():
    def run():
        env = MushroomEnv(seed=99, opponent=random_legal_opponent)
        env.reset()
        total = 0.0
        for _ in range(50):
            _, r, terminated, _, _ = env.step(NOOP_INDEX)
            total += r
            if terminated:
                break
        return total, env.state.tick, env.state.phase

    assert run() == run()


def test_reset_with_seed_resets_opponent_rng():
    env = MushroomEnv(opponent=random_legal_opponent)
    env.reset(seed=42)
    trace_a = [env.step(NOOP_INDEX)[1] for _ in range(20)]
    env.reset(seed=42)
    trace_b = [env.step(NOOP_INDEX)[1] for _ in range(20)]
    assert trace_a == trace_b


def test_invalid_action_does_not_crash():
    """Sim silently drops invalid sends; env should pass-through cleanly."""
    env = MushroomEnv(seed=13, opponent=noop_opponent)
    env.reset()
    # Send from P2 base (not owned by P1).
    bogus = encode(type_idx=3, src=P2_BASE, tgt=N1_TOP)
    _, reward, terminated, _, _ = env.step(bogus)
    # Nothing captured this step.
    assert reward == 0.0
    assert not terminated
