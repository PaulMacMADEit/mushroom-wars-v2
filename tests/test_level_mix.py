"""Tests for `JaxVecEnv` level_mix — per-env level distribution at reset."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Force JAX backend before any sim/env import.
os.environ.setdefault("SIM_BACKEND", "jax")


def _building_count(state) -> int:
    return int((state.buildings_alive == 1).sum())


def test_level_mix_distribution_matches_weights():
    """Sampling 256 envs from a 50/50 mix should give ~50/50 building counts."""
    from sim.envs.jax_vec_env import JaxVecEnv

    mix = [("random_4_8", 0.5), ("random_16_24", 0.5)]
    vec = JaxVecEnv(n_envs=256, base_seed=42, level_mix=mix)
    states = vec.snapshot_numpy_states()
    counts = [_building_count(s) for s in states]
    small = sum(1 for c in counts if c <= 8)
    large = sum(1 for c in counts if c >= 16)
    # Allow ±25% slack on a 256-sample binomial.
    assert 90 <= small <= 160, f"small={small} small_count_distribution off: {Counter(counts)}"
    assert 90 <= large <= 160, f"large={large} large_count_distribution off: {Counter(counts)}"


def test_level_mix_auto_reset_preserves_distribution():
    """After many auto-resets, env still sees mixed-distribution levels."""
    from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP
    from sim.envs.jax_vec_env import JaxVecEnv

    mix = [("random_4_8", 0.7), ("random_16_24", 0.3)]
    vec = JaxVecEnv(n_envs=64, base_seed=7, level_mix=mix)

    # Force every env to terminate via NOOP-only play. random_4_8 ends fast
    # by timeout-tiebreak; random_16_24 takes longer. Run plenty of ticks.
    a_batch = np.zeros((64, 2, ACTION_DIM), dtype=np.int32)  # all NOOPs
    for _ in range(C_GAME_TIMEOUT := 250):
        result = vec.step(a_batch)
        if result.terminated.any():
            # Continue stepping so auto-reset keeps cycling.
            continue
    # Re-sample after lots of auto-resets.
    counts = [_building_count(s) for s in vec.snapshot_numpy_states()]
    small = sum(1 for c in counts if c <= 8)
    large = sum(1 for c in counts if c >= 16)
    # Loose check — main goal: still see BOTH categories represented.
    assert small >= 5, f"small bucket nearly empty after auto-reset: {Counter(counts)}"
    assert large >= 5, f"large bucket nearly empty after auto-reset: {Counter(counts)}"


def test_level_mix_none_uses_level_name():
    """level_mix=None preserves single-level behaviour (back-compat)."""
    from sim.envs.jax_vec_env import JaxVecEnv

    vec = JaxVecEnv(n_envs=16, level_name="random_4_8", base_seed=0, level_mix=None)
    counts = [_building_count(s) for s in vec.snapshot_numpy_states()]
    for c in counts:
        assert 4 <= c <= 8, f"level_name=random_4_8 produced count {c}"


def test_level_mix_via_ppoconfig():
    """PPOConfig.level_mix accepts a dict and passes it through correctly."""
    import torch
    from training.agent import PPOAgent
    from training.net import ActorCritic
    from training.trainer import PPOConfig, PPOTrainer

    cfg = PPOConfig(
        n_envs=8, vec_mode="sync", rollout_steps=8,
        fused_rollout=False,
        level_mix={"random_4_8": 0.5, "random_16_24": 0.5},
        normalize_obs=False, self_play=False,
    )
    net = ActorCritic()
    agent = PPOAgent(net, device=torch.device("cpu"))
    trainer = PPOTrainer(agent, cfg, seed=0)
    try:
        # Sanity: vec env constructed; collect one rollout doesn't crash.
        batch = trainer.collect_rollout()
        assert batch["obs"].shape[0] == 8 * 8
    finally:
        trainer.close()
