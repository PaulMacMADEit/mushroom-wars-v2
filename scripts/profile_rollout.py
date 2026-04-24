"""
Micro-profiler for one PPO rollout under each backend.

Prints phase-ns breakdown (act_batch / env_step / rollout / learn / update)
after a short warmup. Lets us see exactly which part of the training loop
is bottlenecked on the JAX-backend path vs numpy.

Usage:
  SIM_BACKEND=numpy python scripts/profile_rollout.py --envs 64 --updates 3
  SIM_BACKEND=jax   python scripts/profile_rollout.py --envs 64 --updates 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep JAX mem modest so we don't fight torch on shared VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import torch

from training.agent import PPOAgent
from training.net import ActorCritic
from training.trainer import PPOConfig, PPOTrainer


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs",    type=int, default=64)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--vec-mode", default="sync", choices=["sync", "async"])
    args = ap.parse_args()

    device = _device()
    backend = os.environ.get("SIM_BACKEND", "numpy")
    print(f"[profile] backend={backend}  device={device}  envs={args.envs}  rollout={args.rollout}")

    net = ActorCritic()
    agent = PPOAgent(net, device=device)
    cfg = PPOConfig(n_envs=args.envs, rollout_steps=args.rollout, vec_mode=args.vec_mode)
    trainer = PPOTrainer(agent, cfg, seed=0)

    # Warmup: one update so JIT / torch kernels compile.
    t0 = time.perf_counter()
    trainer.update()
    t_warmup = time.perf_counter() - t0
    print(f"[profile] warmup update: {t_warmup:.2f}s")

    # Reset phase counters so only the timed updates show.
    for k in trainer._phase_ns:
        trainer._phase_ns[k] = 0

    t0 = time.perf_counter()
    for _ in range(args.updates):
        trainer.update()
    wall = time.perf_counter() - t0

    breakdown = trainer.sim_phase_breakdown()
    print(f"\n[profile] {args.updates} updates in {wall:.2f}s "
          f"({args.updates/wall:.2f} upd/s)")
    print("\nPhase breakdown (ms, % of update_total):")
    ms = breakdown["ms"]
    pct = breakdown["pct"]
    for k in ("rollout_ns", "act_batch_ns", "env_step_ns", "learn_ns", "update_total_ns"):
        short = k.replace("_ns", "")
        print(f"  {short:20s} {ms.get(k, 0):>10.1f} ms   {pct.get(k, 0):>5.1f}%")

    trainer.close()


if __name__ == "__main__":
    main()
