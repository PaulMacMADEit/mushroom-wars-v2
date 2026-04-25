"""
Phase G perf profile — PaulLinux RTX 3070.

Reproduces the Phase E measurement shape under the post-G1/G2 code path:
  1. K-sweep us/tick at n_envs=1024, rollout_steps=64.
  2. Per-rollout-step phase attribution (step_chunk / pack_actions /
     encode_mask / act_batch).
  3. Wall-clock training session for an external `nvidia-smi dmon` capture.

Usage:
  SIM_BACKEND=jax python scripts/phase_g_bench.py k-sweep
  SIM_BACKEND=jax python scripts/phase_g_bench.py phase-attr
  SIM_BACKEND=jax python scripts/phase_g_bench.py train --seconds 300

The phase-attr mode wraps key functions via monkey-patching to record per-call
timings without modifying production code.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import jax
import numpy as np
import torch

import training.fused_rollout as fr
from sim.envs.jax_vec_env import JaxVecEnv
from training.agent import PPOAgent
from training.net import ActorCritic
from training.trainer import PPOConfig, PPOTrainer


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cmd_k_sweep(args):
    """K-sweep us/tick. Mirrors Phase E table."""
    device = _device()
    print(f"[k-sweep] device={device}  n_envs={args.envs}  rollout={args.rollout}")
    print(f"          backend={os.environ.get('SIM_BACKEND', 'numpy')}")

    results = []

    # Per-tick (no fused) — uses cfg.fused_rollout=False.
    for label, fused, K in [
        ("per-tick",   False, 1),
        ("fused K=1",  True,  1),
        ("fused K=4",  True,  4),
        ("fused K=8",  True,  8),
        ("fused K=16", True,  16),
    ]:
        net = ActorCritic()
        agent = PPOAgent(net, device=device)
        cfg = PPOConfig(
            n_envs=args.envs,
            rollout_steps=args.rollout,
            fused_rollout=fused,
            action_repeat=K,
        )
        trainer = PPOTrainer(agent, cfg, seed=0)

        # Warmup (JIT compile + torch kernel warmup).
        trainer.update()
        trainer.update()
        _sync()

        # Measure 2 updates.
        t0 = time.perf_counter()
        for _ in range(2):
            trainer.update()
        _sync()
        wall = time.perf_counter() - t0

        ticks = 2 * args.rollout * args.envs * (K if fused else 1)
        us_per_tick = wall * 1e6 / ticks
        results.append((label, K if fused else 1, us_per_tick))

        trainer.close()
        del trainer, agent, net

    print()
    print("Results:")
    print(f"  {'config':<12} {'us/tick':>10}    {'speedup':>8}")
    base = results[0][2]
    for label, _, us in results:
        print(f"  {label:<12} {us:>9.2f}      {base/us:>6.2f}x")


def cmd_phase_attr(args):
    """Per-rollout-step phase attribution by wrapping the four hot calls."""
    device = _device()
    print(f"[phase-attr] device={device}  n_envs={args.envs}  rollout={args.rollout}  K={args.K}")

    times = {"step_chunk": [], "pack_actions": [], "encode_mask": [], "act_batch": []}

    orig_encode_and_masks = fr._encode_and_masks
    orig_pack_action_batch = fr.pack_action_batch_jax
    orig_act_batch = PPOAgent.act_batch

    def timed_encode_and_masks(vec_env):
        t0 = time.perf_counter()
        out = orig_encode_and_masks(vec_env)
        jax.block_until_ready(out[0])
        times["encode_mask"].append(time.perf_counter() - t0)
        return out

    def timed_pack(p1, p2_mask, key, opp):
        t0 = time.perf_counter()
        out = orig_pack_action_batch(p1, p2_mask, key, opp)
        jax.block_until_ready(out)
        times["pack_actions"].append(time.perf_counter() - t0)
        return out

    def timed_act_batch(self, obs, mask):
        t0 = time.perf_counter()
        out = orig_act_batch(self, obs, mask)
        _sync()
        times["act_batch"].append(time.perf_counter() - t0)
        return out

    # Wrap step_chunk on the live JaxVecEnv after construction.
    fr._encode_and_masks = timed_encode_and_masks
    fr.pack_action_batch_jax = timed_pack
    PPOAgent.act_batch = timed_act_batch

    try:
        net = ActorCritic()
        agent = PPOAgent(net, device=device)
        cfg = PPOConfig(
            n_envs=args.envs,
            rollout_steps=args.rollout,
            fused_rollout=True,
            action_repeat=args.K,
        )
        trainer = PPOTrainer(agent, cfg, seed=0)

        # Wrap step_chunk after the trainer constructs the env.
        vec_env = trainer.vec._inner
        orig_step_chunk = vec_env.step_chunk

        def timed_step_chunk(actions, K):
            t0 = time.perf_counter()
            out = orig_step_chunk(actions, K)
            _sync()
            times["step_chunk"].append(time.perf_counter() - t0)
            return out

        vec_env.step_chunk = timed_step_chunk

        # Warmup, then clear timers.
        trainer.update()
        trainer.update()
        _sync()
        for k in times:
            times[k].clear()

        # Measure.
        t0 = time.perf_counter()
        for _ in range(args.updates):
            trainer.update()
        _sync()
        wall = time.perf_counter() - t0

        trainer.close()
    finally:
        fr._encode_and_masks = orig_encode_and_masks
        fr.pack_action_batch_jax = orig_pack_action_batch
        PPOAgent.act_batch = orig_act_batch

    n_steps = sum(len(v) for v in times.values()) / 4
    print(f"\n[phase-attr] {args.updates} updates in {wall:.2f}s "
          f"({args.updates/wall:.2f} upd/s, ~{int(n_steps)} rollout steps)")

    total_per_step = sum(np.mean(v) * 1000 for v in times.values()) if all(times.values()) else 0
    print(f"\nPer-rollout-step phase attribution:")
    print(f"  {'phase':<14} {'ms/step':>10}    {'%':>6}")
    for k in ("step_chunk", "pack_actions", "encode_mask", "act_batch"):
        if not times[k]:
            print(f"  {k:<14} (no samples)")
            continue
        ms = np.mean(times[k]) * 1000
        pct = 100 * ms / total_per_step if total_per_step else 0
        print(f"  {k:<14} {ms:>9.2f}     {pct:>5.1f}%")


def cmd_train(args):
    """Wall-clock training run for `nvidia-smi dmon -s u` to capture in parallel."""
    device = _device()
    print(f"[train] device={device}  n_envs={args.envs}  rollout={args.rollout}  K={args.K}  seconds={args.seconds}")

    net = ActorCritic()
    agent = PPOAgent(net, device=device)
    cfg = PPOConfig(
        n_envs=args.envs,
        rollout_steps=args.rollout,
        fused_rollout=True,
        action_repeat=args.K,
    )
    trainer = PPOTrainer(agent, cfg, seed=args.seed)

    trainer.update()  # warmup
    _sync()

    t0 = time.perf_counter()
    deadline = t0 + args.seconds
    n_updates = 0
    last_print = t0
    while time.perf_counter() < deadline:
        log = trainer.update()
        n_updates += 1
        now = time.perf_counter()
        if now - last_print > 30:
            wr = log.get("win_rate") or log.get("win_rate_p1") or 0.0
            print(f"  t={now-t0:6.1f}s upd={n_updates}  win={wr:.2f}")
            last_print = now
    wall = time.perf_counter() - t0
    final_log = trainer.update()
    n_updates += 1
    final_win = final_log.get("win_rate") or final_log.get("win_rate_p1") or 0.0
    print(f"\n[train] {n_updates} updates in {wall:.1f}s ({n_updates/wall:.2f} upd/s)")
    print(f"        final win-rate: {final_win:.2f}")
    trainer.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("k-sweep")
    a1.add_argument("--envs", type=int, default=1024)
    a1.add_argument("--rollout", type=int, default=64)
    a1.set_defaults(func=cmd_k_sweep)

    a2 = sub.add_parser("phase-attr")
    a2.add_argument("--envs", type=int, default=1024)
    a2.add_argument("--rollout", type=int, default=64)
    a2.add_argument("--K", type=int, default=8)
    a2.add_argument("--updates", type=int, default=4)
    a2.set_defaults(func=cmd_phase_attr)

    a3 = sub.add_parser("train")
    a3.add_argument("--envs", type=int, default=1024)
    a3.add_argument("--rollout", type=int, default=64)
    a3.add_argument("--K", type=int, default=8)
    a3.add_argument("--seconds", type=int, default=300)
    a3.add_argument("--seed", type=int, default=0)
    a3.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
