"""Rollout throughput benchmark: games-completed-per-second.

Runs the new training stack (sim + encoder + PPO forward + env step) for a
wall-clock budget and reports games/sec. Configurable device so the same
script answers "Mac CPU vs Mac MPS" and "PC CPU vs PC CUDA".

This is training-time throughput, not pure sim and not pure NN. Both the sim
and the NN are on whatever device you picked (numpy always CPU; torch on the
chosen device). Opponent = random-legal (same cost on every config).

Usage:
    python scripts/bench_games.py --device cpu  --seconds 30
    python scripts/bench_games.py --device mps  --seconds 30
    python scripts/bench_games.py --device cuda --seconds 30
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from sim.envs import MushroomEnv, random_legal_opponent
from training.agent import PPOAgent
from training.encoder import encode_obs
from training.net import ActorCritic


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(flag)


def bench(seconds: float, device: torch.device, seed: int) -> dict:
    env = MushroomEnv(seed=seed, opponent=random_legal_opponent)
    net = ActorCritic()
    agent = PPOAgent(net, device=device)

    obs_dict, _ = env.reset()
    obs = encode_obs(obs_dict)
    mask = obs_dict["action_mask"].copy()

    games = 0
    steps = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds

    while time.perf_counter() < deadline:
        action, _logprob, _value = agent.act(obs, mask)
        obs_dict, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated or truncated:
            games += 1
            obs_dict, _ = env.reset()
        obs = encode_obs(obs_dict)
        mask = obs_dict["action_mask"].copy()

    wall = time.perf_counter() - t0
    return {
        "device":    str(device),
        "wall":      wall,
        "games":     games,
        "steps":     steps,
        "games/sec": games / wall if wall > 0 else 0.0,
        "steps/sec": steps / wall if wall > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"host: {platform.node()} | platform: {platform.platform()}")
    print(f"torch: {torch.__version__} | device: {device}")
    print(f"budget: {args.seconds:.0f}s  seed: {args.seed}\n")

    result = bench(args.seconds, device, args.seed)
    print(
        f"device={result['device']:<4}  "
        f"wall={result['wall']:6.2f}s  "
        f"games={result['games']:>5}  "
        f"steps={result['steps']:>7}  "
        f"games/sec={result['games/sec']:>6.2f}  "
        f"steps/sec={result['steps/sec']:>8.1f}"
    )


if __name__ == "__main__":
    main()
