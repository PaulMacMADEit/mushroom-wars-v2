"""
Bench: numpy `encode_obs` per-env loop vs JAX `encode_obs_batched`.

Measures encoder cost in isolation — no env step, no policy, no rollout.
Useful for sizing the win the fused rollout will pull from moving the
encoder onto the device.

Usage:
    python scripts/bench_encoder.py
    python scripts/bench_encoder.py --n-envs 1024 --reps 200
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import jax
import jax.numpy as jnp
import numpy as np

from sim.actions import NOOP_INDEX, compute_mask, decode
from sim.engine import step_tick
from sim.levels import reset
from sim.state_jax import StateJax, from_numpy_state
from training.encoder import encode_obs
from training.encoder_jax import encode_obs_batched_jit


def _build_states(n_envs: int, seed: int):
    rng = np.random.default_rng(seed)
    states = []
    for i in range(n_envs):
        s = reset(level_name="random_8_16", seed=seed + i)
        # Step 30 ticks of random play so the state has groups in flight.
        from sim import config as C
        for _ in range(30):
            m1 = compute_mask(s, C.OWNER_P1)
            m2 = compute_mask(s, C.OWNER_P2)
            a1 = decode(int(rng.choice(np.where(m1)[0])) if m1.any() else NOOP_INDEX)
            a2 = decode(int(rng.choice(np.where(m2)[0])) if m2.any() else NOOP_INDEX)
            _, _, done = step_tick(s, a1, a2)
            if done:
                break
        states.append(s)
    return states


def _obs_dict_from_state(state):
    return {
        "buildings_alive":    state.buildings_alive,
        "buildings_owner":    state.buildings_owner,
        "buildings_type":     state.buildings_type,
        "buildings_garrison": state.buildings_garrison,
        "buildings_capacity": state.buildings_capacity,
        "buildings_x":        state.buildings_x,
        "buildings_y":        state.buildings_y,
        "groups_alive":       state.groups_alive,
        "groups_owner":       state.groups_owner,
        "groups_src":         state.groups_src,
        "groups_tgt":         state.groups_tgt,
        "groups_count":       state.groups_count,
        "groups_progress":    state.groups_progress,
        "groups_travel":      state.groups_travel,
        "tick":               np.int32(state.tick),
    }


def _stack_states_to_jax(states):
    """Stack numpy States into a batched StateJax via tree_map — picks up
    any new StateJax field without needing enumeration."""
    import jax
    leaves = [from_numpy_state(s) for s in states]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *leaves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=1024)
    ap.add_argument("--reps",   type=int, default=200)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    print(f"host: {platform.node()} | platform: {platform.platform()}")
    print(f"jax devices: {[str(d) for d in jax.devices()]}")
    print(f"n_envs={args.n_envs}  reps={args.reps}\n")

    states = _build_states(args.n_envs, args.seed)
    obs_dicts = [_obs_dict_from_state(s) for s in states]

    # numpy: per-env loop
    # warmup
    _ = np.stack([encode_obs(o) for o in obs_dicts], axis=0)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        np.stack([encode_obs(o) for o in obs_dicts], axis=0)
    np_wall = time.perf_counter() - t0

    # JAX: batched
    sj = _stack_states_to_jax(states)
    _ = encode_obs_batched_jit(sj).block_until_ready()  # compile
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = encode_obs_batched_jit(sj)
    out.block_until_ready()
    jx_wall = time.perf_counter() - t0

    np_per_call = np_wall / args.reps * 1000.0
    jx_per_call = jx_wall / args.reps * 1000.0
    print(f"  numpy per-env loop: {np_wall:6.3f}s total  ({np_per_call:7.2f} ms/call)")
    print(f"  jax  vmap'd batch:  {jx_wall:6.3f}s total  ({jx_per_call:7.2f} ms/call)")
    if jx_per_call > 0:
        print(f"  speedup: {np_per_call / jx_per_call:.1f}x")


if __name__ == "__main__":
    main()
