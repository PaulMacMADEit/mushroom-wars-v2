"""Vectorized games/sec benchmark.

Runs N parallel MushroomEnv instances under gymnasium's Sync or Async vector
wrapper and a single batched NN forward per decision step. Answers the key
question from the single-env bench: does batching NN forwards across envs
flip the GPU win on?

Usage:
    python scripts/bench_vec_games.py --device cpu  --envs 64
    python scripts/bench_vec_games.py --device mps  --envs 64
    python scripts/bench_vec_games.py --device cuda --envs 64 --mode async
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch

from sim.envs import make_env
from training.agent import PPOAgent
from training.encoder import OBS_DIM, encode_obs
from training.net import ActorCritic


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(flag)


def build_vec_env(n_envs: int, mode: str, base_seed: int):
    factories = [make_env(seed=base_seed + i) for i in range(n_envs)]
    if mode == "sync":
        return gym.vector.SyncVectorEnv(factories)
    if mode == "async":
        return gym.vector.AsyncVectorEnv(factories, shared_memory=False)
    raise ValueError(f"unknown mode: {mode}")


def encode_batch(obs_batch: dict, n_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a vector-env dict-of-stacked-arrays into (N, OBS_DIM) + (N, A)
    mask. Loop-based; encoder cost is ~10% of rollout at these sizes."""
    obs_out = np.empty((n_envs, OBS_DIM), dtype=np.float32)
    masks_out = np.empty((n_envs, obs_batch["action_mask"].shape[1]), dtype=bool)
    for i in range(n_envs):
        single = {k: v[i] for k, v in obs_batch.items()}
        obs_out[i] = encode_obs(single)
        masks_out[i] = single["action_mask"]
    return obs_out, masks_out


def bench(seconds: float, n_envs: int, mode: str, device: torch.device, seed: int) -> dict:
    vec = build_vec_env(n_envs, mode, seed)
    net = ActorCritic()
    agent = PPOAgent(net, device=device)

    obs_batch, _ = vec.reset(seed=seed)
    obs, masks = encode_batch(obs_batch, n_envs)

    games = 0
    steps = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds

    while time.perf_counter() < deadline:
        # act_batch now returns 6 values (chained heads); we only need actions for stepping.
        actions = agent.act_batch(obs, masks)[0]
        obs_batch, _rewards, terminated, truncated, _info = vec.step(actions)
        done_mask = terminated | truncated
        games += int(done_mask.sum())
        steps += n_envs
        obs, masks = encode_batch(obs_batch, n_envs)

    wall = time.perf_counter() - t0
    vec.close()
    return {
        "device":    str(device),
        "mode":      mode,
        "envs":      n_envs,
        "wall":      wall,
        "games":     games,
        "steps":     steps,
        "games/sec": games / wall if wall > 0 else 0.0,
        "steps/sec": steps / wall if wall > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--mode", default="sync", choices=["sync", "async"])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"host: {platform.node()} | platform: {platform.platform()}")
    print(f"torch: {torch.__version__} | device: {device}")
    print(f"envs: {args.envs}  mode: {args.mode}  budget: {args.seconds:.0f}s  seed: {args.seed}\n")

    result = bench(args.seconds, args.envs, args.mode, device, args.seed)
    print(
        f"device={result['device']:<4}  "
        f"mode={result['mode']:<5}  "
        f"envs={result['envs']:>3}  "
        f"wall={result['wall']:6.2f}s  "
        f"games={result['games']:>6}  "
        f"steps={result['steps']:>8}  "
        f"games/sec={result['games/sec']:>7.2f}  "
        f"steps/sec={result['steps/sec']:>9.1f}"
    )


if __name__ == "__main__":
    main()
