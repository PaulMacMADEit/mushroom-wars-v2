"""Forward-pass throughput benchmark: old-approach net vs new-approach net.

Compares the two PyTorch architectures used in training:

  old — Games/mushroom-wars/training/model.py
        230 → 256 → 256 → 256 → (549 policy + 1 value), masked logits
  new — training/net.py
        289 → 128 → 128 → (4097 policy + 1 value), masked logits

Runs each on CPU and (where available) MPS, across batch sizes that cover both
single-env rollout (B=1) and vec-env rollout (B=64, 256). Prints forwards/sec.

Not meant as a scientific benchmark — just an honest head-to-head on the same
Mac, warmed up, timed with perf_counter.

Usage:
    python scripts/bench_nn.py
    python scripts/bench_nn.py --iters 500 --batches 1,16,64,256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Old architecture (copy-pasted from Games/mushroom-wars/training/model.py)
# ---------------------------------------------------------------------------

class OldNet(nn.Module):
    def __init__(self, obs_size=230, action_size=549, hidden_size=256, num_layers=3):
        super().__init__()
        layers = []
        in_size = obs_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs, mask):
        features = self.trunk(obs)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(features)
        return logits, value


# ---------------------------------------------------------------------------
# New architecture (from training/net.py)
# ---------------------------------------------------------------------------

from training.net import ActorCritic as NewNet


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def bench_forward(
    net: nn.Module,
    obs_dim: int,
    action_dim: int,
    device: str,
    batch_size: int,
    iters: int,
    warmup: int,
    is_old: bool,
) -> float:
    """Return forwards/sec, averaged over `iters` forward passes after warmup."""
    net = net.to(device).eval()
    dev = torch.device(device)

    # Pre-generate inputs once to avoid measuring RNG cost in the inner loop.
    obs = torch.randn(batch_size, obs_dim, device=dev)
    mask = torch.ones(batch_size, action_dim, dtype=torch.bool, device=dev)

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def _step():
        if is_old:
            net(obs, mask)
        else:
            logits, _ = net(obs)
            _ = net.masked_logits(logits, mask)

    # Warmup (lazy kernel compile, esp. on MPS)
    for _ in range(warmup):
        _step()
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _step()
    _sync()
    dt = time.perf_counter() - t0
    return iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batches", type=str, default="1,16,64,256")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    devices = available_devices()

    print(f"devices available: {devices}")
    print(f"iters={args.iters}, warmup={args.warmup}, batches={batches}\n")

    old_net = OldNet()
    new_net = NewNet()
    print(f"old net params: {count_params(old_net):,}   (230→256×3 + 549-way + value)")
    print(f"new net params: {count_params(new_net):,}   (289→128×2 + 4097-way + value)\n")

    header = f"{'net':<5} {'device':<6} {'batch':>6} {'fwd/s':>12} {'fwd·samples/s':>16}"
    print(header)
    print("-" * len(header))

    for device in devices:
        for batch in batches:
            rate = bench_forward(
                net=OldNet(),
                obs_dim=230,
                action_dim=549,
                device=device,
                batch_size=batch,
                iters=args.iters,
                warmup=args.warmup,
                is_old=True,
            )
            print(f"{'old':<5} {device:<6} {batch:>6} {rate:>12,.0f} {rate * batch:>16,.0f}")

            rate = bench_forward(
                net=NewNet(),
                obs_dim=289,
                action_dim=4097,
                device=device,
                batch_size=batch,
                iters=args.iters,
                warmup=args.warmup,
                is_old=False,
            )
            print(f"{'new':<5} {device:<6} {batch:>6} {rate:>12,.0f} {rate * batch:>16,.0f}")
        print()


if __name__ == "__main__":
    main()
