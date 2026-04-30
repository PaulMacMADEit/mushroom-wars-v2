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
import json
import os
import tempfile
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import torch

from sim import config as C
from sim.envs import MushroomEnv, make_neural_opponent
from sim.envs.replay import Recorder
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


def download_run_state(weights_url: str | None, obs_norm_url: str | None) -> dict:
    """Fetch a run's trained state from Storage.

    A None weights_url marks the run as a baseline/scripted opponent
    (random_legal). The returned dict has `weights=None`; downstream code
    treats that as "use random_legal_opponent for this side".
    """
    if not weights_url:
        return {"weights": None, "obs_norm": None}
    return {
        "weights":  _fetch_load(weights_url),
        "obs_norm": _fetch_load(obs_norm_url),
    }


def _load_agent(state: dict, device: torch.device):
    """Load an ActorCritic + obs_norm + encoder fn from state dicts.

    Returns (None, None, None) for a baseline state (weights is None) — the
    caller must substitute `random_legal_opponent` when playing that side.

    Otherwise returns (agent, obs_norm, encode_fn) where `encode_fn` is the
    encoder version that produced the weights — i.e. v9.0 for unstamped
    archive checkpoints, v10 for new ones. Loaders that ignored the
    encoder version pre-v10 silently used the current encoder, which
    breaks once OBS_DIM diverges across versions.
    """
    if state.get("weights") is None:
        return None, None, None
    from training.encoders import DEFAULT_ENCODER_VERSION, get_encoder
    from training.net import infer_body_dim, infer_obs_dim

    # `state["weights"]` is either a raw state_dict (legacy) or a wrapped
    # {state_dict, encoder_version} dict (v10+).
    raw = state["weights"]
    if isinstance(raw, dict) and "state_dict" in raw and "encoder_version" in raw:
        weights        = raw["state_dict"]
        encoder_version = raw["encoder_version"]
    else:
        weights        = raw
        encoder_version = DEFAULT_ENCODER_VERSION

    encoder_entry = get_encoder(encoder_version)

    # Size the net to the trunk's actual obs_dim, not the current OBS_DIM
    # constant. ActorCritic(obs_dim=…) wires the trunk's first Linear layer.
    body_dim = infer_body_dim(weights)
    obs_dim  = infer_obs_dim(weights)
    if obs_dim != encoder_entry.obs_dim:
        raise ValueError(
            f"checkpoint trunk obs_dim={obs_dim} but encoder_version="
            f"{encoder_version!r} expects {encoder_entry.obs_dim}"
        )
    net = ActorCritic(obs_dim=obs_dim, body_dim=body_dim)
    net.load_state_dict(weights)
    agent = PPOAgent(net, device=device)
    obs_norm: RunningNorm | None = None
    if state["obs_norm"] is not None:
        obs_norm = RunningNorm(obs_dim)  # file shape wins on load anyway
        obs_norm.load_state_dict(state["obs_norm"])
    return agent, obs_norm, encoder_entry.encode


def _play_one_game(
    p1_agent: PPOAgent | None,
    p1_norm: RunningNorm | None,
    p2_weights_path: str | None,
    p2_obs_norm_path: str | None,
    level_name: str,
    seed: int,
    game_id: str | None = None,
    record: bool = True,
    p1_encode=None,
) -> dict:
    """Run one game. Returns {winner, ticks, wall_ms, phase} plus diagnostics.

    - If `p1_agent` is None, P1 plays as `random_legal_opponent` (baseline).
    - If `p2_weights_path` is None, P2 plays as `random_legal_opponent`.
    - Otherwise both sides are neural, with P1 driven in-process and P2 via
      the existing `make_neural_opponent` factory.

    When `record=True` and `game_id` is set, the env's replay recorder is
    attached; the returned dict includes `replay_data` (dict) ready to
    upload to the replays bucket.
    """
    import time

    from sim.envs.opponents import random_legal_opponent

    recorder: Recorder | None = None
    if record and game_id is not None:
        recorder = Recorder(game_id=str(game_id), level_name=level_name, seed=int(seed))

    # Choose P2: neural net (default) or random_legal baseline. The neural
    # opponent gets the recorder so P2's decisions are captured too.
    if p2_weights_path is None:
        opponent = random_legal_opponent
    else:
        opponent = make_neural_opponent(
            weights_path=p2_weights_path,
            obs_norm_path=p2_obs_norm_path,
            device="cpu",
            recorder=recorder,
        )

    env = MushroomEnv(level_name=level_name, opponent=opponent, seed=seed, recorder=recorder)
    obs, info = env.reset(seed=seed)

    # P1 may have been trained against a different encoder version than
    # the current one; the caller passes that encoder fn in. Default to
    # the current (v10) encoder for back-compat.
    _p1_encode_fn = p1_encode if p1_encode is not None else encode_obs

    def _encode(o):
        x = _p1_encode_fn(o)
        return x if p1_norm is None else p1_norm.normalize(x)

    # P1 action chooser: neural agent if we have one, else random-legal over
    # the P1 mask (which is what the env embeds in obs).
    def _pick_p1_action(o, m):
        if p1_agent is not None:
            x = _encode(o)
            # When recording, use the diag variant so we can capture the
            # policy breakdown (value, top-k per head, entropy).
            if recorder is not None:
                action, diag = p1_agent.act_one_with_diag(x, m)
                # Decision stamped at post-tick time (matches event timestamps
                # — env.step increments state.tick by decision_interval).
                recorder.record_decision(
                    tick=int(env.state.tick + 1),
                    player=C.OWNER_P1,
                    diag=diag,
                )
                return int(action)
            # Eval is deterministic — sampling injects noise that costs win
            # rate vs weak opponents and creates unstable Elo against strong
            # ones. Training still samples (entropy_coef handles exploration).
            return int(p1_agent.act_batch(x[None, :], m[None, :], deterministic=True)[0][0])
        # Baseline: sample from valid actions for P1.
        legal_idx = np.where(m)[0]
        return int(env._rng.choice(legal_idx)) if legal_idx.size else 0

    t0 = time.time()
    while True:
        action = _pick_p1_action(obs, obs["action_mask"])
        obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    wall_ms = int((time.time() - t0) * 1000)

    phase = info["phase"]
    if phase == C.PHASE_P1_WINS:
        winner = "p1"
    elif phase == C.PHASE_P2_WINS:
        winner = "p2"
    else:
        winner = "draw"
    out: dict = {
        "winner":  winner,
        "phase":   int(phase),
        "ticks":   int(info["tick"]),
        "wall_ms": wall_ms,
    }
    if recorder is not None:
        out["replay_data"] = recorder.to_dict()
    return out


def _upload_replay(game_id: str, data: dict) -> str | None:
    """Write replay JSON to a temp file and upload to the `replays` bucket.

    Returns the storage path (like `replays/<id>/events.json`) on success,
    None on failure (we don't want a flaky upload to fail the whole match).
    """
    from workers import storage
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, separators=(",", ":"))
            path = f.name
        try:
            return storage.upload(
                "replays",
                f"{game_id}/events.json",
                path,
                content_type="application/json",
            )
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"[match_runner] replay upload failed game={game_id}: {exc}")
        return None


def _materialize(state: dict, dir_path: Path, stem: str) -> tuple[str | None, str | None]:
    """Write (weights, obs_norm) to local files and return their paths.

    Returns (None, None) for a baseline state (weights is None) — the caller
    passes None through to `_play_one_game` to trigger random_legal_opponent.
    """
    if state.get("weights") is None:
        return None, None
    w_path = dir_path / f"{stem}-weights.pt"
    # `state["weights"]` may be either a raw state_dict (legacy S3 payload
    # produced before the v10 stamp) or already a wrapped {state_dict,
    # encoder_version} dict (new payloads). Re-wrap-or-passthrough so the
    # downstream loader (`load_state_dict_with_version`) reads the right
    # encoder either way. We do NOT default-stamp v10 on legacy payloads —
    # those are v9.0 by `DEFAULT_ENCODER_VERSION` semantics.
    weights = state["weights"]
    if isinstance(weights, dict) and "state_dict" in weights and "encoder_version" in weights:
        torch.save(weights, w_path)
    else:
        torch.save(weights, w_path)  # raw passthrough → loader treats as v9.0
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
    record: bool = True,
) -> list[dict]:
    """Play `n_games` games between A and B on `level_name`, alternating sides.

    Returns a list of per-game dicts ready to insert into `games`. Each dict:
        {id, game_index, seed, map_name, player_1_run_id, player_2_run_id,
         winner (UUID str or None for draw), duration_ms, stats, actions_url}

    When `record=True`, each game is replayed-captured and the resulting
    event log is uploaded to the `replays` bucket; the storage path lands
    in `actions_url`. An upload failure degrades gracefully (actions_url=None).
    """
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="mw2-match-state-") as tmp:
        tmp_path = Path(tmp)
        a_w, a_n = _materialize(state_a, tmp_path, "a")
        b_w, b_n = _materialize(state_b, tmp_path, "b")

        agent_a, norm_a, encode_a = _load_agent(state_a, device)
        agent_b, norm_b, encode_b = _load_agent(state_b, device)

        for i in range(n_games):
            swap = (i % 2 == 1)  # odd games: B is P1, A is P2
            seed = seed_base + i
            game_id = str(uuid.uuid4())

            if swap:
                # B plays as P1 in-process, A plays as P2 (opponent).
                p1_agent, p1_norm, p1_encode = agent_b, norm_b, encode_b
                p2_w, p2_n = a_w, a_n
                p1_run, p2_run = str(run_b_id), str(run_a_id)
            else:
                p1_agent, p1_norm, p1_encode = agent_a, norm_a, encode_a
                p2_w, p2_n = b_w, b_n
                p1_run, p2_run = str(run_a_id), str(run_b_id)

            g = _play_one_game(
                p1_agent, p1_norm, p2_w, p2_n, level_name, seed,
                game_id=game_id, record=record,
                p1_encode=p1_encode,
            )

            # Resolve winner from side → run id
            winner_run: str | None = None
            if g["winner"] == "p1":
                winner_run = p1_run
            elif g["winner"] == "p2":
                winner_run = p2_run
            # draws: winner_run stays None

            actions_url: str | None = None
            if record and "replay_data" in g:
                actions_url = _upload_replay(game_id, g["replay_data"])

            results.append({
                "id":              game_id,
                "game_index":      i,
                "seed":            int(seed),
                "map_name":        level_name,
                "player_1_run_id": p1_run,
                "player_2_run_id": p2_run,
                "winner":          winner_run,
                "duration_ms":     g["wall_ms"],
                "stats": {
                    "phase":   g["phase"],
                    "ticks":   g["ticks"],
                    "swapped": swap,
                },
                "actions_url":     actions_url,
            })
    return results


def summarize(results: list[dict], run_a_id, run_b_id) -> dict:
    """Aggregate per-game results into a match summary.

    For true self-play (a_id == b_id), `winner == a_id` and `winner == b_id`
    are both true on every win, which makes naive wins_a / wins_b counts
    overlap. We also track engine-side counts (wins_p1 / wins_p2) drawn from
    `stats.phase`; these are unambiguous and useful for measuring map
    asymmetry. Self-play counts wins_a as the side-A-was-on victories
    (i.e. half from P1, half from P2 across the swap rotation).
    """
    a_id, b_id = str(run_a_id), str(run_b_id)
    is_self_play = (a_id == b_id)

    draws = sum(1 for g in results if g["winner"] is None)
    n = len(results)

    # Engine-side counts (1=P1, 2=P2, 3=draw). Stable across self-play /
    # head-to-head; reflects pure spatial asymmetry of the level.
    wins_p1 = sum(1 for g in results if (g.get("stats") or {}).get("phase") == 1)
    wins_p2 = sum(1 for g in results if (g.get("stats") or {}).get("phase") == 2)

    if is_self_play:
        # Run-id-keyed counts are meaningless in self-play. Report the
        # side-rotated view: A's wins = the wins it got while playing whichever
        # side it was assigned (alternates every game via stats.swapped).
        wins_a = sum(
            1 for g in results
            if g["winner"] is not None and (
                ((g.get("stats") or {}).get("swapped") is False and (g.get("stats") or {}).get("phase") == 1) or
                ((g.get("stats") or {}).get("swapped") is True  and (g.get("stats") or {}).get("phase") == 2)
            )
        )
        wins_b = sum(
            1 for g in results
            if g["winner"] is not None and (
                ((g.get("stats") or {}).get("swapped") is False and (g.get("stats") or {}).get("phase") == 2) or
                ((g.get("stats") or {}).get("swapped") is True  and (g.get("stats") or {}).get("phase") == 1)
            )
        )
    else:
        wins_a = sum(1 for g in results if g["winner"] == a_id)
        wins_b = sum(1 for g in results if g["winner"] == b_id)

    return {
        "wins_a":     wins_a,
        "wins_b":     wins_b,
        "wins_p1":    wins_p1,
        "wins_p2":    wins_p2,
        "draws":      draws,
        "rate_a":     wins_a / n if n else 0.0,
        "rate_b":     wins_b / n if n else 0.0,
        "draw_rate":  draws  / n if n else 0.0,
        "games":      n,
        "self_play":  is_self_play,
    }
