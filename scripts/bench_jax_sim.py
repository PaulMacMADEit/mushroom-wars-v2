"""
Benchmark the JAX sim backend via JaxVecEnv.

Measures games/sec and ticks/sec at a given batch size. Meant to pair with
the numpy-side `scripts/bench_sim.py` baseline:

  # Baseline (numpy, single-process):
  python scripts/bench_sim.py single --games 50 --ticks 200

  # JAX, 1024 envs, 200 ticks:
  python scripts/bench_jax_sim.py --n-envs 1024 --ticks 200

On PaulLinux (RTX 3070) this is the JAX_PORT_PLAN §1 success gate:
  - ≥10× games/sec vs the numpy baseline
  - ≥40% GPU SM utilisation during the run

On Mac CPU the 10× target does not apply — JAX on CPU doesn't vectorise
onto SIMD the way a CUDA kernel does. Mac results are correctness-level
sanity checks.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Importing engine_jax flips x64 on before jax.numpy is imported — keep this
# import order even if the linter gripes.
from sim import engine_jax  # noqa: F401  — side-effect import (x64 toggle)

import jax

from sim import config as C
from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND
from sim.envs.jax_vec_env import JaxVecEnv


def _random_actions(n_envs: int, rng: np.random.Generator) -> np.ndarray:
    """Pick random (but cheaply-valid-looking) actions for each env.

    This is a throughput bench, not a correctness test — we don't compute the
    mask (that would defeat the whole point of measuring XLA throughput). We
    emit plausible send/noop actions and let the engine's internal validity
    check drop the invalid ones.
    """
    a = np.zeros((n_envs, 2, ACTION_DIM), dtype=np.int32)
    # 80% send / 20% noop roughly — matches what an actual policy produces.
    kinds = rng.integers(0, 5, size=(n_envs, 2))      # 0..4
    a[:, :, 0] = np.where(kinds == 0, ACTION_KIND_NOOP, ACTION_KIND_SEND)
    a[:, :, 1] = rng.integers(0, 4, size=(n_envs, 2))            # type_idx in [0,4)
    a[:, :, 2] = rng.integers(0, C.MAX_BUILDING_SLOTS, size=(n_envs, 2))
    a[:, :, 3] = rng.integers(0, C.MAX_BUILDING_SLOTS, size=(n_envs, 2))
    return a


def bench(
    n_envs: int,
    ticks: int,
    level: str,
    seed: int,
    warmup_ticks: int = 20,
    fused: bool = False,
    fused_chunk: int = 50,
) -> dict:
    """Run a fixed-length sweep. Returns a result dict.

    `fused=True` uses `JaxVecEnv.step_many` (T ticks in one XLA dispatch)
    which is how we hit >10× on CUDA. `fused_chunk` is the T per call.
    """
    vec = JaxVecEnv(n_envs=n_envs, level_name=level, base_seed=seed)
    rng = np.random.default_rng(seed)

    # Warmup: the first call triggers XLA compilation; don't count that time.
    if fused:
        a = np.stack([_random_actions(n_envs, rng) for _ in range(fused_chunk)], axis=0)
        vec.step_many(a)
    else:
        for _ in range(warmup_ticks):
            vec.step(_random_actions(n_envs, rng))
    jax.block_until_ready(vec.state.tick)
    vec.reset()

    total_games = 0
    t0 = time.perf_counter()
    if fused:
        chunks, rem = divmod(ticks, fused_chunk)
        for _ in range(chunks):
            a = np.stack([_random_actions(n_envs, rng) for _ in range(fused_chunk)], axis=0)
            r = vec.step_many(a)
            total_games += int(r["dones"].sum())
        if rem > 0:
            a = np.stack([_random_actions(n_envs, rng) for _ in range(rem)], axis=0)
            r = vec.step_many(a)
            total_games += int(r["dones"].sum())
    else:
        for _ in range(ticks):
            r = vec.step(_random_actions(n_envs, rng))
            total_games += int(r.terminated.sum())
    jax.block_until_ready(vec.state.tick)
    wall = time.perf_counter() - t0

    total_ticks = ticks * n_envs
    return {
        "n_envs":    n_envs,
        "ticks":     ticks,
        "wall_s":    wall,
        "games":     total_games,
        "games/sec": total_games / wall if wall > 0 else 0.0,
        "ticks/sec": total_ticks / wall if wall > 0 else 0.0,
        "devices":   [str(d) for d in jax.devices()],
        "fused":     fused,
        "fused_chunk": fused_chunk if fused else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs",    type=int, default=1024)
    ap.add_argument("--ticks",     type=int, default=200)
    ap.add_argument("--level",     type=str, default="random_8_16")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--warmup",    type=int, default=20)
    ap.add_argument("--sweep",     action="store_true",
                    help="sweep 1,16,64,256,1024 envs instead of a single point")
    ap.add_argument("--fused",     action="store_true",
                    help="use step_many (scan inside jit); one dispatch per --fused-chunk ticks")
    ap.add_argument("--fused-chunk", type=int, default=50,
                    help="T ticks per fused dispatch (only used with --fused)")
    args = ap.parse_args()

    print(f"host: {platform.node()} | platform: {platform.platform()}")
    print(f"jax devices: {[str(d) for d in jax.devices()]}")
    print(f"ticks per env: {args.ticks}  level: {args.level}  seed: {args.seed}\n")

    envs_list = [1, 16, 64, 256, 1024] if args.sweep else [args.n_envs]
    mode = f"fused (chunk={args.fused_chunk})" if args.fused else "per-tick"
    print(f"mode: {mode}\n")
    print(f"{'n_envs':>8s}  {'wall_s':>8s}  {'ticks/sec':>12s}  {'games/sec':>12s}  {'games':>6s}")
    for ne in envs_list:
        r = bench(
            ne, args.ticks, args.level, args.seed,
            warmup_ticks=args.warmup,
            fused=args.fused, fused_chunk=args.fused_chunk,
        )
        print(
            f"{r['n_envs']:>8d}  {r['wall_s']:>8.3f}  "
            f"{r['ticks/sec']:>12,.0f}  {r['games/sec']:>12.2f}  {r['games']:>6d}"
        )


if __name__ == "__main__":
    main()
