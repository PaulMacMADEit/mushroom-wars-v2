"""Finish-speed monitor (CURRICULUM_PLAN.md success criterion 5).

Runs 100 games of the current Elo champion vs random_legal and reports:
  * mean ticks-to-end (across all settled games)
  * % of P1 wins that ended before tick 150 (the "quick-win" criterion)
  * histogram of game lengths

Logs JSON to monitoring/finish_speed.jsonl (append mode) so we can plot
the trend over time.

Usage:
    python scripts/eval_finish_speed.py --p1 <run_id_or_path>
    python scripts/eval_finish_speed.py --p1 elo-champ   # auto-resolve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("SIM_BACKEND", "jax")

import numpy as np


def _resolve_elo_champ() -> str:
    from cli.db import PROJECT, connect
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, label, elo_score, elo_n_matches FROM runs
                 WHERE project=%s AND status='done' AND weights_url IS NOT NULL
                   AND elo_n_matches >= 3
                 ORDER BY elo_score DESC
                 LIMIT 1
                """,
                (PROJECT,),
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("no Elo-rated champion in Supabase (need ≥3 matches)")
    print(f"[eval_finish_speed] Elo champion: {row[1]} score={float(row[2]):.1f} matches={int(row[3])}")
    return str(row[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True,
                    help="Supabase run id, local experiment dir, or 'elo-champ' "
                         "to auto-pick the top-Elo run")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--level", default="random_8_16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick-win-threshold", type=int, default=150,
                    help="tick threshold for the 'quick win' criterion (default 150)")
    ap.add_argument("--no-log", action="store_true",
                    help="don't append to monitoring/finish_speed.jsonl")
    args = ap.parse_args()

    p1 = args.p1
    if p1 == "elo-champ":
        p1 = _resolve_elo_champ()

    # Run the match per-tick so we can record settle-tick per game.
    import torch
    import jax.numpy as jnp
    from sim import config as C
    from sim.engine_jax import ACTION_DIM
    from sim.envs.jax_vec_env import JaxVecEnv, _step_batched
    from scripts.tournament import (
        _decode_action_to_packed, _load_policy, _pick_actions,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"P1: {p1}")
    print(f"{args.games} games × max {args.max_ticks} ticks on {args.level}")

    p1_kind, p1_agent, p1_norm = _load_policy(p1, device)
    p2_kind, p2_agent, p2_norm = _load_policy("random_legal", device)
    vec = JaxVecEnv(n_envs=args.games, level_name=args.level, base_seed=args.seed)
    rng = np.random.default_rng(args.seed)

    settle_tick = np.full(args.games, -1, dtype=np.int32)
    finished    = np.zeros(args.games, dtype=bool)
    p1_wins = p2_wins = draws = 0

    t0 = time.perf_counter()
    for tick in range(args.max_ticks):
        states = vec.snapshot_numpy_states()
        a1_flat = _pick_actions(p1_kind, p1_agent, p1_norm, states, C.OWNER_P1, rng)
        a2_flat = _pick_actions(p2_kind, p2_agent, p2_norm, states, C.OWNER_P2, rng)
        a_batch = np.zeros((args.games, 2, ACTION_DIM), dtype=np.int32)
        for i in range(args.games):
            _decode_action_to_packed(int(a1_flat[i]), a_batch[i, 0])
            _decode_action_to_packed(int(a2_flat[i]), a_batch[i, 1])
        a1 = jnp.asarray(a_batch[:, 0, :], dtype=jnp.int32)
        a2 = jnp.asarray(a_batch[:, 1, :], dtype=jnp.int32)
        vec.state, _r1, _r2, dones = _step_batched(vec.state, a1, a2)
        terminated = np.asarray(dones)
        new_done = terminated & ~finished
        if new_done.any():
            phase_arr = np.asarray(vec.state.phase)
            for i in np.where(new_done)[0]:
                ph = int(phase_arr[i])
                if ph == C.PHASE_P1_WINS:
                    p1_wins += 1
                elif ph == C.PHASE_P2_WINS:
                    p2_wins += 1
                else:
                    draws += 1
                finished[i] = True
                settle_tick[i] = tick + 1  # post-step state.tick is tick+1
        if finished.all():
            break

    wall = time.perf_counter() - t0
    not_settled = (~finished).sum()
    if not_settled > 0:
        draws += int(not_settled)

    settled_ticks = settle_tick[settle_tick > 0]
    # Indices of P1 wins (re-derive from final state).
    phase_final = np.asarray(vec.state.phase)
    p1_win_idx = np.where(phase_final == C.PHASE_P1_WINS)[0]
    p1_win_ticks = settle_tick[p1_win_idx]
    p1_win_ticks = p1_win_ticks[p1_win_ticks > 0]
    quick_wins = int((p1_win_ticks < args.quick_win_threshold).sum())
    pct_quick = quick_wins / max(len(p1_win_ticks), 1)

    # Histogram in 25-tick bins.
    bins = list(range(0, args.max_ticks + 26, 25))
    hist, edges = np.histogram(settled_ticks, bins=bins)
    total = p1_wins + p2_wins + draws

    out = {
        "ts":                 datetime.now(timezone.utc).isoformat(),
        "p1":                 p1,
        "level":              args.level,
        "games":              args.games,
        "wall_s":             round(wall, 2),
        "p1_wins":            int(p1_wins),
        "p2_wins":            int(p2_wins),
        "draws":              int(draws),
        "win_rate":           round(p1_wins / max(total, 1), 3),
        "mean_ticks_settled": round(float(settled_ticks.mean()), 1) if settled_ticks.size else None,
        "median_ticks_settled": int(np.median(settled_ticks)) if settled_ticks.size else None,
        "max_ticks_settled":  int(settled_ticks.max()) if settled_ticks.size else None,
        "min_ticks_settled":  int(settled_ticks.min()) if settled_ticks.size else None,
        "quick_win_threshold": args.quick_win_threshold,
        "pct_quick_wins":     round(pct_quick, 3),
        "p1_win_count":       int(len(p1_win_ticks)),
        "histogram_bins":     [int(x) for x in edges],
        "histogram_counts":   [int(x) for x in hist],
    }

    print(f"\n=== finish-speed report ({wall:.1f}s wall) ===")
    print(f"  P1 wins: {p1_wins:>5d} ({100*p1_wins/total:5.1f}%)")
    print(f"  P2 wins: {p2_wins:>5d} ({100*p2_wins/total:5.1f}%)")
    print(f"  draws:   {draws:>5d} ({100*draws/total:5.1f}%)")
    if settled_ticks.size:
        print(f"  mean tick:   {out['mean_ticks_settled']}")
        print(f"  median tick: {out['median_ticks_settled']}")
    print(f"  P1 wins before tick {args.quick_win_threshold}: "
          f"{quick_wins}/{len(p1_win_ticks)} ({100*pct_quick:.1f}%)")

    if not args.no_log:
        log_dir = ROOT / "monitoring"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "finish_speed.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        print(f"  logged to {log_path}")

    print(json.dumps(out))


if __name__ == "__main__":
    main()
