"""Head-to-head match runner.

Plays N games between two trained runs' weights on a given level spec.
Half the games flip sides so P1-advantage bias cancels out. Each game's
result is written as a row in `games`; aggregate stats are packed into
`matches.summary` at the end.

Module-level design:
  - `download_run_state(run_id)` → {weights, obs_norm}
  - `run_match(...)` → list of game result dicts (caller writes to DB)

The worker imports + calls these; they don't touch the DB directly.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from sim import config as C
from sim.envs import MushroomEnv, make_neural_opponent
from training.agent import PPOAgent
from training.encoder import OBS_DIM, encode_obs
from training.net import ActorCritic
from training.obs_norm import RunningNorm


def _public_url(path: str | None) -> str | None:
    if not path:
        return None
    base = os.environ.get("SUPABASE_URL")
    if not base:
        raise RuntimeError("SUPABASE_URL not set")
    return f"{base}/storage/v1/object/public/{path}"


def _fetch_load(url_path: str | None):
    url = _public_url(url_path)
    if url is None:
        return None
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def download_run_state(weights_url: str, obs_norm_url: str | None) -> dict:
    """Fetch a run's trained state from Storage. Missing obs_norm is tolerated."""
    return {
        "weights":  _fetch_load(weights_url),
        "obs_norm": _fetch_load(obs_norm_url),
    }


def _load_agent(state: dict, device: torch.device) -> tuple[PPOAgent, RunningNorm | None]:
    net = ActorCritic()
    net.load_state_dict(state["weights"])
    agent = PPOAgent(net, device=device)
    obs_norm: RunningNorm | None = None
    if state["obs_norm"] is not None:
        obs_norm = RunningNorm(OBS_DIM)
        obs_norm.load_state_dict(state["obs_norm"])
    return agent, obs_norm


def _play_one_game(
    p1_agent: PPOAgent,
    p1_norm: RunningNorm | None,
    p2_weights_path: str,
    p2_obs_norm_path: str | None,
    level_name: str,
    seed: int,
) -> dict:
    """Run one game. Returns {winner, ticks, wall_ms, phase} plus diagnostics.

    P1 is driven in-process by `p1_agent`. P2 runs as MushroomEnv's opponent
    — the same neural-opponent factory vec-env self-play already uses, just
    invoked synchronously in this process instead of a subproc.
    """
    import time

    opponent = make_neural_opponent(
        weights_path=p2_weights_path,
        obs_norm_path=p2_obs_norm_path,
        device="cpu",
    )
    env = MushroomEnv(level_name=level_name, opponent=opponent, seed=seed)
    obs, info = env.reset(seed=seed)

    def _encode(o):
        x = encode_obs(o)
        return x if p1_norm is None else p1_norm.normalize(x)

    x = _encode(obs)
    mask = obs["action_mask"]

    t0 = time.time()
    while True:
        action, *_ = p1_agent.act_batch(x[None, :], mask[None, :])
        obs, _r, terminated, truncated, info = env.step(int(action[0]))
        if terminated or truncated:
            break
        x = _encode(obs)
        mask = obs["action_mask"]
    wall_ms = int((time.time() - t0) * 1000)

    phase = info["phase"]
    if phase == C.PHASE_P1_WINS:
        winner = "p1"
    elif phase == C.PHASE_P2_WINS:
        winner = "p2"
    else:
        winner = "draw"
    return {
        "winner":  winner,
        "phase":   int(phase),
        "ticks":   int(info["tick"]),
        "wall_ms": wall_ms,
    }


def _materialize(state: dict, dir_path: Path, stem: str) -> tuple[str, str | None]:
    """Write (weights, obs_norm) to local files and return their paths."""
    w_path = dir_path / f"{stem}-weights.pt"
    torch.save(state["weights"], w_path)
    n_path: str | None = None
    if state["obs_norm"] is not None:
        n_file = dir_path / f"{stem}-obs_norm.pt"
        torch.save(state["obs_norm"], n_file)
        n_path = str(n_file)
    return str(w_path), n_path


def run_match(
    run_a_id,
    run_b_id,
    state_a: dict,
    state_b: dict,
    n_games: int,
    level_name: str,
    seed_base: int,
    device: torch.device,
) -> list[dict]:
    """Play `n_games` games between A and B on `level_name`, alternating sides.

    Returns a list of per-game dicts ready to insert into `games`. Each dict:
        {game_index, seed, map_name, player_1_run_id, player_2_run_id,
         winner (UUID str or None for draw), duration_ms, stats}
    """
    import tempfile

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="mw2-match-state-") as tmp:
        tmp_path = Path(tmp)
        a_w, a_n = _materialize(state_a, tmp_path, "a")
        b_w, b_n = _materialize(state_b, tmp_path, "b")

        agent_a, norm_a = _load_agent(state_a, device)
        agent_b, norm_b = _load_agent(state_b, device)

        for i in range(n_games):
            swap = (i % 2 == 1)  # odd games: B is P1, A is P2
            seed = seed_base + i

            if swap:
                # B plays as P1 in-process, A plays as P2 (opponent).
                p1_agent, p1_norm = agent_b, norm_b
                p2_w, p2_n = a_w, a_n
                p1_run, p2_run = str(run_b_id), str(run_a_id)
            else:
                p1_agent, p1_norm = agent_a, norm_a
                p2_w, p2_n = b_w, b_n
                p1_run, p2_run = str(run_a_id), str(run_b_id)

            g = _play_one_game(p1_agent, p1_norm, p2_w, p2_n, level_name, seed)

            # Resolve winner from side → run id
            winner_run: str | None = None
            if g["winner"] == "p1":
                winner_run = p1_run
            elif g["winner"] == "p2":
                winner_run = p2_run
            # draws: winner_run stays None

            results.append({
                "game_index":      i,
                "seed":             int(seed),
                "map_name":         level_name,
                "player_1_run_id":  p1_run,
                "player_2_run_id":  p2_run,
                "winner":           winner_run,
                "duration_ms":      g["wall_ms"],
                "stats": {
                    "phase":  g["phase"],
                    "ticks":  g["ticks"],
                    "swapped": swap,
                },
            })
    return results


def summarize(results: list[dict], run_a_id, run_b_id) -> dict:
    """Aggregate per-game results into a match summary."""
    a_id, b_id = str(run_a_id), str(run_b_id)
    wins_a = sum(1 for g in results if g["winner"] == a_id)
    wins_b = sum(1 for g in results if g["winner"] == b_id)
    draws  = sum(1 for g in results if g["winner"] is None)
    return {
        "wins_a":     wins_a,
        "wins_b":     wins_b,
        "draws":      draws,
        "rate_a":     wins_a / len(results) if results else 0.0,
        "rate_b":     wins_b / len(results) if results else 0.0,
        "draw_rate":  draws  / len(results) if results else 0.0,
        "games":      len(results),
    }
