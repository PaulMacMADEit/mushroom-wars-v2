"""Phase 2 smoke training: 1-minute PPO run end-to-end.

Goal: prove the loop closes — env collects rollouts, trainer updates weights,
metrics print. Not a champion. If this runs without errors and win rate moves
off the random-baseline, Phase 2 is kicked off.

Usage:
    python scripts/smoke_train.py
    python scripts/smoke_train.py --seconds 60 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.agent import PPOAgent
from training.net import ActorCritic
from training.trainer import PPOConfig, PPOTrainer


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60, help="wall-clock budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=128,
                    help="per-env steps per rollout (total samples = n_envs × rollout)")
    ap.add_argument("--envs", type=int, default=32,
                    help="parallel envs (1 = single-env path; GPU wins at ≥32)")
    ap.add_argument("--vec-mode", default="async", choices=["async", "sync"])
    ap.add_argument("--fused-rollout", action="store_true",
                    help="use the fused (chunked) rollout collector "
                         "(requires SIM_BACKEND=jax)")
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="K: env ticks per agent decision under --fused-rollout")
    args = ap.parse_args()

    device = _device()
    print(f"[smoke] device={device}")

    net = ActorCritic()
    agent = PPOAgent(net, device=device)
    cfg = PPOConfig(
        n_envs=args.envs,
        vec_mode=args.vec_mode,
        rollout_steps=args.rollout,
        fused_rollout=args.fused_rollout,
        action_repeat=args.action_repeat,
    )
    trainer = PPOTrainer(agent, cfg, seed=args.seed)

    start = time.time()
    updates = 0
    print(
        f"[smoke] budget={args.seconds}s  envs={cfg.n_envs} ({cfg.vec_mode})  "
        f"rollout_steps={cfg.rollout_steps}  "
        f"lr={cfg.lr} gamma={cfg.gamma} lam={cfg.gae_lambda}"
    )

    while time.time() - start < args.seconds:
        t0 = time.time()
        metrics = trainer.update()
        dt = time.time() - t0
        updates += 1
        win = metrics.get("win_rate")
        win_str = f"{win:.2f}" if win is not None else "—"
        eps = metrics.get("episodes_completed", 0)
        print(
            f"[{updates:03d}] wall={time.time()-start:5.1f}s "
            f"upd={dt:4.2f}s "
            f"reward={metrics['mean_reward']:+.4f} "
            f"pol={metrics['policy_loss']:+.4f} "
            f"val={metrics['value_loss']:.4f} "
            f"ent={metrics['entropy_loss']:+.4f} "
            f"kl={metrics['approx_kl']:+.4f} "
            f"eps={eps:3d} "
            f"win={win_str}"
        )

    trainer.close()
    total = time.time() - start
    print(f"[smoke] done in {total:.1f}s, {updates} updates.")


if __name__ == "__main__":
    main()
