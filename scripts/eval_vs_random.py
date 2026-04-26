"""Eval an Elo champion vs random_legal over N games.

Used by the cron-agent's graduation gate (CURRICULUM_PLAN.md §3.2): once
the Elo champion's win-rate vs random_legal hits ≥95% over 100 games, we
flip the curriculum from phase1_close to phase2_wild.

Usage:
    python scripts/eval_vs_random.py --p1 <run_id_or_path> --games 100 --level random_8_16

Returns JSON to stdout (single line) plus a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("SIM_BACKEND", "jax")

from scripts.tournament import run_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True, help="Supabase run id or local experiment dir")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--level", default="random_8_16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[eval_vs_random] p1={args.p1} games={args.games} level={args.level}")
    res = run_match(
        p1=args.p1, p2="random_legal",
        games=args.games, max_ticks=args.max_ticks,
        level=args.level, seed=args.seed,
        verbose=True,
    )
    rate = res["p1_wins"] / max(res["total"], 1)
    res["win_rate"] = rate
    res["p1"] = args.p1
    res["level"] = args.level

    print(f"[eval_vs_random] p1 win rate: {rate:.3f} ({res['p1_wins']}/{res['total']})")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
