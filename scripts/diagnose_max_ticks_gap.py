"""Test whether max_ticks=200 truncation is the source of the train/eval gap.

The train PPO win_rate has no max_ticks limit — episodes run until phase !=
PHASE_PLAYING. The auto_rate eval truncates at max_ticks=200 and counts
unresolved games as timeouts (not wins). If the agent's policy produces
slow-to-finish wins, the auto_rate rate will read low.

Hypothesis: bumping max_ticks from 200 → 500 should close most of the gap.

Run:
  python scripts/diagnose_max_ticks_gap.py --run-id 1757a025  # lr-lo
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _resolve_run(run_id_prefix: str) -> tuple[str, str, str | None]:
    """Pick a run by prefix; return (run_id, weights_url, obs_norm_url)."""
    from cli.db import connect

    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, weights_url, obs_norm_url
              FROM runs
             WHERE id::text LIKE %s
             ORDER BY queued_at DESC LIMIT 1
            """,
            (run_id_prefix + "%",),
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"No run matches prefix {run_id_prefix!r}")
    return row[0], row[1], row[2]


def _download(rel: str, dst: Path) -> None:
    import urllib.request
    from workers.worker import _public_url

    url = _public_url(rel)
    if url is None:
        raise SystemExit(f"Could not resolve URL for {rel!r}")
    print(f"[diag] download: {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dst.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="1757a025",
                    help="run_id prefix (default: lr-lo)")
    ap.add_argument("--games", type=int, default=192,
                    help="games per match (auto_rate runs 192 = 3×64)")
    ap.add_argument("--level", default="random_close_4_5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_id, w_url, n_url = _resolve_run(args.run_id)
    print(f"[diag] run: {run_id[:8]}")

    work = Path(tempfile.mkdtemp(prefix="mw2-maxticks-"))
    w_path = work / "weights.pt"
    _download(w_url, w_path)
    n_path: Path | None = None
    if n_url:
        n_path = work / "obs_norm.pt"
        _download(n_url, n_path)

    # Build the checkpoint as a "p1" loadable for tournament.run_match.
    # `_load_policy` accepts a directory path with weights.pt + obs_norm.pt.
    p1_dir = work / "p1"
    p1_dir.mkdir()
    (p1_dir / "weights.pt").write_bytes(w_path.read_bytes())
    if n_path is not None:
        (p1_dir / "obs_norm.pt").write_bytes(n_path.read_bytes())

    from scripts import tournament

    # Run at multiple max_ticks values to see how the rate moves.
    print()
    print(f"[diag] running {args.games} games on {args.level} vs random_legal at varying max_ticks")
    print(f"[diag] {'max_ticks':>10s}  {'p1_wins':>8s}  {'p2_wins':>8s}  {'draws':>6s}  {'timeouts':>9s}  {'rate':>6s}  {'rate_decided':>12s}")
    for mt in (200, 300, 500, 1000):
        res = tournament.run_match(
            p1=str(p1_dir), p2="random_legal",
            games=args.games, level=args.level,
            max_ticks=mt, seed=args.seed, verbose=False,
        )
        rate = res["p1_wins"] / max(res["total"], 1)
        decided = res["p1_wins"] + res["p2_wins"] + res["draws"]
        rate_decided = res["p1_wins"] / max(decided, 1)
        print(f"[diag] {mt:>10d}  {res['p1_wins']:>8d}  {res['p2_wins']:>8d}  "
              f"{res['draws']:>6d}  {res['timeouts']:>9d}  {rate:>6.3f}  {rate_decided:>12.3f}")


if __name__ == "__main__":
    main()
