"""Play N games of a fresh v10 random-init agent against a v9.0 archive
champion — proves the encoder adapter end-to-end on real archive weights.

Without the adapter (training/encoders + training/checkpoint shipped
2026-04-29), this would crash with `size mismatch for trunk.0.weight`
the moment we tried to load the v9.0 state_dict against current code.

Usage:
  python scripts/play_vs_archive_champion.py
  python scripts/play_vs_archive_champion.py --champion-run-id <uuid>
  python scripts/play_vs_archive_champion.py --n-games 10 --level random_4_8

Default behaviour: pick the oldest champion in the archive (most likely
to be a true v9.0 unstamped checkpoint) and play 5 games on
`crossroads_6` with a fresh randomly-initialised v10 net as P1.

Output: per-game winner + tick count, and a final summary line of
P1 (v10-fresh) wins / P2 (v9.0-archive) wins / draws.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _resolve_champion(champion_run_id: str | None) -> tuple[str, str, str | None]:
    """Pick a champion + return (run_id, weights_url, obs_norm_url).

    None champion_run_id → oldest champion in the archive.
    """
    from cli.db import connect

    with connect() as c, c.cursor() as cur:
        if champion_run_id:
            cur.execute(
                """
                SELECT c.source_run_id::text, c.label, r.weights_url, r.obs_norm_url
                  FROM champions c
                  LEFT JOIN runs r ON r.id = c.source_run_id
                 WHERE c.source_run_id::text = %s
                """,
                (champion_run_id,),
            )
        else:
            cur.execute(
                """
                SELECT c.source_run_id::text, c.label, r.weights_url, r.obs_norm_url
                  FROM champions c
                  LEFT JOIN runs r ON r.id = c.source_run_id
                 ORDER BY c.archived_at ASC
                 LIMIT 1
                """
            )
        row = cur.fetchone()

    if row is None:
        raise SystemExit(
            "No champion found. Either the archive is empty, or the requested "
            "run_id isn't in the champions table."
        )
    run_id, label, w_url, n_url = row
    print(f"[demo] champion: {label}  run_id={run_id[:8]}")
    print(f"[demo]   weights_url: {w_url}")
    print(f"[demo]   obs_norm_url: {n_url}")
    return run_id, w_url, n_url


def _download(url_relpath: str, dst: Path) -> None:
    """Pull a Storage-relative path to local disk via the project's
    public-URL helper."""
    import urllib.request
    from workers.worker import _public_url

    url = _public_url(url_relpath)
    if url is None:
        raise SystemExit(f"Could not resolve public URL for {url_relpath!r}")
    print(f"[demo] downloading: {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dst.write_bytes(data)


def _play_one(
    p1_agent,
    p1_norm,
    p2_opponent,
    level: str,
    seed: int,
) -> dict:
    """Run one game. Returns {winner, ticks, phase}."""
    import numpy as np

    from sim import config as C
    from sim.envs.mushroom_env import MushroomEnv
    from training.encoder import encode_obs

    env = MushroomEnv(level_name=level, opponent=p2_opponent, seed=seed)
    obs, _ = env.reset(seed=seed)

    while True:
        # P1 (fresh v10) action.
        x = encode_obs(obs)
        if p1_norm is not None:
            x = p1_norm.normalize(x)
        action_arr, *_ = p1_agent.act_batch(x[None, :], obs["action_mask"][None, :])
        action = int(action_arr[0])

        obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    phase = int(info["phase"])
    if phase == C.PHASE_P1_WINS:
        winner = "p1"
    elif phase == C.PHASE_P2_WINS:
        winner = "p2"
    else:
        winner = "draw"
    return {"winner": winner, "ticks": int(info["tick"]), "phase": phase}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--champion-run-id", default=None,
                    help="Specific champion to load (default: oldest archive entry).")
    ap.add_argument("--n-games", type=int, default=5)
    ap.add_argument("--level", default="crossroads_6")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Force CPU before any torch import — sim envs and PPOAgent both touch torch
    # and if CUDA initialises here, neural-opponent subprocs we don't even start
    # would still inherit that init.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    run_id, w_url, n_url = _resolve_champion(args.champion_run_id)

    # Download the archive weights + obs_norm to local files.
    work = Path(tempfile.mkdtemp(prefix="mw2-demo-"))
    w_path = work / "weights.pt"
    _download(w_url, w_path)
    n_path: Path | None = None
    if n_url:
        n_path = work / "obs_norm.pt"
        _download(n_url, n_path)

    # Build the v9.0 opponent via the (versioned) loader.
    from sim.envs.opponents import make_neural_opponent
    p2_opponent = make_neural_opponent(
        weights_path=str(w_path),
        obs_norm_path=str(n_path) if n_path else None,
        device="cpu",
    )
    print("[demo] v9.0 opponent loaded successfully (adapter routed through v9.0 encoder)")

    # Build a fresh v10 random-init agent for P1.
    from training.agent import PPOAgent
    from training.net import ActorCritic
    p1_net = ActorCritic()
    p1_agent = PPOAgent(p1_net, device="cpu")
    p1_norm = None  # unnormalised; the demo just shows the wires hold

    # Play N games.
    print(f"[demo] playing {args.n_games} games on {args.level} "
          f"(P1 = fresh v10 random-init; P2 = v9.0 champion {run_id[:8]})")
    results = {"p1": 0, "p2": 0, "draw": 0}
    for i in range(args.n_games):
        out = _play_one(p1_agent, p1_norm, p2_opponent, args.level, args.seed + i)
        results[out["winner"]] += 1
        print(f"  game {i:2d}  winner={out['winner']:5s}  ticks={out['ticks']}")

    print()
    print(f"[demo] summary  p1 (v10 fresh): {results['p1']}/{args.n_games}  "
          f"p2 (v9.0 champion): {results['p2']}/{args.n_games}  "
          f"draws: {results['draw']}/{args.n_games}")
    print(f"[demo] PASS — v9.0 archive checkpoint plays cleanly under v10 code.")


if __name__ == "__main__":
    main()
