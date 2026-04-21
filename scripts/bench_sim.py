"""
Benchmark the v0.1 sim.

Two modes:
  single   — single-process, N games sequentially. Shows per-tick subsystem breakdown.
  parallel — ProcessPoolExecutor, sweeps worker counts to find the machine's sweet spot.

Run:
    python scripts/bench_sim.py               # both modes, default settings
    python scripts/bench_sim.py --ticks 5000  # more ticks per game
    python scripts/bench_sim.py single        # just single-process breakdown
    python scripts/bench_sim.py parallel      # just the parallelism sweep
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Make `sim` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sim import config as C
from sim.actions import compute_mask, decode
from sim.engine import step_tick
from sim.levels import reset


# ---------------------------------------------------------------------------
# One-game workload: random self-play
# ---------------------------------------------------------------------------

def _run_random_game(seed: int, max_ticks: int) -> dict:
    """Play a full random game. Returns final perf dict + ticks simulated."""
    rng = np.random.default_rng(seed)
    state = reset(seed=seed)
    ticks = 0
    while ticks < max_ticks:
        m1 = compute_mask(state, C.OWNER_P1)
        m2 = compute_mask(state, C.OWNER_P2)
        a1 = decode(int(rng.choice(np.where(m1)[0])))
        a2 = decode(int(rng.choice(np.where(m2)[0])))
        _, _, done = step_tick(state, a1, a2)
        ticks += 1
        if done:
            break
    out = dict(state.perf)
    out["ticks"] = ticks
    out["phase"] = state.phase
    return out


def _run_batch(args: tuple[int, int, int]) -> dict:
    """Run N games in a worker; aggregate. Used by the parallel sweep."""
    start_seed, n_games, max_ticks = args
    agg = {
        "production_ns": 0, "movement_ns": 0, "combat_ns": 0,
        "actions_ns": 0, "victory_ns": 0, "total_ns": 0,
        "n_ticks": 0, "games": 0,
    }
    for i in range(n_games):
        r = _run_random_game(start_seed + i, max_ticks)
        for k in agg:
            if k == "games":
                agg[k] += 1
            elif k in r:
                agg[k] += r[k]
    return agg


# ---------------------------------------------------------------------------
# Single-process mode
# ---------------------------------------------------------------------------

def bench_single(n_games: int, max_ticks: int) -> None:
    print(f"\n=== single-process: {n_games} games, max {max_ticks} ticks each ===")
    t0 = time.perf_counter()
    total = _run_batch((0, n_games, max_ticks))
    wall = time.perf_counter() - t0

    ticks = total["n_ticks"]
    print(f"  wall time:        {wall:.3f} s")
    print(f"  total ticks:      {ticks:,}")
    print(f"  ticks/sec:        {ticks/wall:,.0f}")
    print(f"  games/sec:        {n_games/wall:.1f}")
    print(f"  game-sec simulated/wall-sec: {ticks/wall:,.0f}   (target ≥10k per §18)")
    print()
    print("  Per-tick breakdown (avg ns):")
    for k in ("actions_ns", "production_ns", "movement_ns", "combat_ns", "victory_ns", "total_ns"):
        avg = total[k] / ticks if ticks else 0
        frac = total[k] / total["total_ns"] if total["total_ns"] else 0
        label = k.replace("_ns", "")
        print(f"    {label:12s}: {avg:8.0f} ns  ({frac*100:5.1f}%)")


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------

def bench_parallel(n_games_per_worker: int, max_ticks: int, workers_list: list[int]) -> None:
    cpu_count = mp.cpu_count()
    print(f"\n=== parallel sweep: {cpu_count} logical CPUs on this machine ===")
    print(f"    {n_games_per_worker} games/worker, max {max_ticks} ticks each\n")
    print(f"    {'workers':>8s}  {'ticks/sec':>14s}  {'games/sec':>12s}  {'scaling':>10s}")

    baseline_tps = None
    for n_workers in workers_list:
        if n_workers > cpu_count * 2:
            continue
        args_list = [(i * 100_000, n_games_per_worker, max_ticks) for i in range(n_workers)]

        t0 = time.perf_counter()
        if n_workers == 1:
            results = [_run_batch(args_list[0])]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_run_batch, args_list))
        wall = time.perf_counter() - t0

        total_ticks = sum(r["n_ticks"] for r in results)
        total_games = sum(r["games"]   for r in results)
        tps = total_ticks / wall
        gps = total_games / wall

        if baseline_tps is None:
            baseline_tps = tps
            scale_str = "1.0×"
        else:
            scale_str = f"{tps/baseline_tps:.1f}×"

        print(f"    {n_workers:>8d}  {tps:>14,.0f}  {gps:>12.1f}  {scale_str:>10s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("single", "parallel", "both"), default="both")
    ap.add_argument("--games", type=int, default=50, help="games in single mode")
    ap.add_argument("--games-per-worker", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=C.GAME_TIMEOUT_TICKS, help="max ticks per game")
    args = ap.parse_args()

    if args.mode in ("single", "both"):
        bench_single(args.games, args.ticks)

    if args.mode in ("parallel", "both"):
        cpu = mp.cpu_count()
        workers = sorted({1, 2, 4, 8, cpu, cpu * 2})
        workers = [w for w in workers if w >= 1]
        bench_parallel(args.games_per_worker, args.ticks, workers)


if __name__ == "__main__":
    main()
