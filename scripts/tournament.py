"""
Head-to-head tournament between saved checkpoints.

Loads two policies (or one and `random_legal`), runs N games on JaxVecEnv,
reports win/loss/draw for P1.

Usage:
    python scripts/tournament.py \\
        --p1 experiments/b3_001_endurance_2h \\
        --p2 experiments/b5_002_vs_b3_001_2h \\
        --games 1024 --level random_8_16

    # vs random_legal:
    python scripts/tournament.py --p1 experiments/b3_001_endurance_2h --p2 random_legal

Both `--p1` and `--p2` can be either:
  - A path to an experiment directory containing weights.pt / obs_norm.pt
  - The literal string `random_legal` (uses the batched mask sampler)
  - The literal string `noop`
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("SIM_BACKEND", "jax")

import numpy as np
import torch

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX, SLOTS_SQ, compute_mask_batched
from sim.engine_jax import ACTION_DIM, ACTION_KIND_NOOP, ACTION_KIND_SEND
from sim.envs.jax_vec_env import JaxVecEnv, _step_batched
from sim.envs.opponents import random_legal_opponent_batched
from sim.state import State, empty_state
from training.agent import PPOAgent
from training.encoder import OBS_DIM, encode_obs
from training.net import ActorCritic, infer_body_dim
from training.obs_norm import RunningNorm


def _resolve_supabase_run(run_id: str, device: torch.device):
    """Download weights+obs_norm for a Supabase run id; return local paths."""
    import tempfile
    import urllib.request
    from cli.db import connect
    from workers.worker import _public_url  # type: ignore

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT weights_url, obs_norm_url FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"run {run_id} not found in Supabase")
    w_url, n_url = row
    if not w_url:
        raise RuntimeError(f"run {run_id} has no weights_url (status not done?)")

    out_dir = Path(tempfile.mkdtemp(prefix=f"mw2-tour-{run_id[:8]}-"))
    w_path = out_dir / "weights.pt"
    urllib.request.urlretrieve(_public_url(w_url), w_path)
    n_path = None
    if n_url:
        n_path = out_dir / "obs_norm.pt"
        urllib.request.urlretrieve(_public_url(n_url), n_path)
    return w_path, n_path


def _load_policy(path: str | Path, device: torch.device):
    """Returns ('neural', agent, obs_norm) or ('random_legal', None, None) or
    ('noop', None, None). For neural, agent is a PPOAgent; obs_norm is a
    RunningNorm (or None).

    `path` accepts:
      - 'random_legal' or 'noop' (literal opponent names)
      - a Supabase run id (UUID or short prefix matching one row)
      - an experiment dir path (containing weights.pt + obs_norm.pt)
    """
    if path == "random_legal":
        return ("random_legal", None, None)
    if path == "noop":
        return ("noop", None, None)

    # If it looks like a UUID or short hex prefix, try Supabase first.
    is_uuid_like = (
        len(str(path)) >= 8 and
        all(c in "0123456789abcdefABCDEF-" for c in str(path))
    )
    weights_path = None
    obs_norm_p = None
    if is_uuid_like:
        try:
            weights_path, obs_norm_p = _resolve_supabase_run(str(path), device)
        except Exception as exc:
            print(f"[tournament] Supabase lookup failed for {path}: {exc}; trying local path")
            weights_path = None

    if weights_path is None:
        p = Path(path)
        weights_path = p / "weights.pt"
        if (p / "obs_norm.pt").exists():
            obs_norm_p = p / "obs_norm.pt"

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights.pt not found at {weights_path}")
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    body_dim = infer_body_dim(state_dict)
    net = ActorCritic(body_dim=body_dim)
    net.load_state_dict(state_dict)
    agent = PPOAgent(net, device=device)
    obs_norm = None
    if obs_norm_p and Path(obs_norm_p).exists():
        obs_norm = RunningNorm(OBS_DIM)
        obs_norm.load(str(obs_norm_p))
    return ("neural", agent, obs_norm)


def _state_to_obs_dict_for_player(state: State, mask: np.ndarray, player: int) -> dict:
    """Build the obs dict for a single player. For P2 we mirror ownership so
    a P1-trained policy sees itself as P1."""
    out = {
        "buildings_alive":    state.buildings_alive.copy(),
        "buildings_owner":    state.buildings_owner.copy(),
        "buildings_type":     state.buildings_type.copy(),
        "buildings_garrison": state.buildings_garrison.copy(),
        "buildings_capacity": state.buildings_capacity.copy(),
        "buildings_x":        state.buildings_x.copy(),
        "buildings_y":        state.buildings_y.copy(),
        "groups_alive":       state.groups_alive.copy(),
        "groups_owner":       state.groups_owner.copy(),
        "groups_src":         state.groups_src.copy(),
        "groups_tgt":         state.groups_tgt.copy(),
        "groups_count":       state.groups_count.copy(),
        "groups_progress":    state.groups_progress.copy(),
        "groups_travel":      state.groups_travel.copy(),
        "travel_matrix":      state.travel_matrix.copy(),
        "tick":               np.int32(state.tick),
        "action_mask":        mask,
    }
    if player == C.OWNER_P2:
        # Mirror P1 <-> P2 in ownership so P2's policy sees itself as P1.
        for k in ("buildings_owner", "groups_owner"):
            o = out[k]
            swapped = o.copy()
            swapped = np.where(o == C.OWNER_P1, C.OWNER_P2, swapped)
            swapped = np.where(o == C.OWNER_P2, C.OWNER_P1, swapped)
            out[k] = swapped.astype(o.dtype)
    return out


def _decode_action_to_packed(idx: int, out: np.ndarray):
    """Turn a flat action index into [kind, type, src, tgt]."""
    if idx == NOOP_INDEX:
        out[:] = [ACTION_KIND_NOOP, 0, 0, 0]
    else:
        type_i = idx // SLOTS_SQ
        rem = idx % SLOTS_SQ
        src_i = rem // C.MAX_BUILDING_SLOTS
        tgt_i = rem % C.MAX_BUILDING_SLOTS
        out[:] = [ACTION_KIND_SEND, type_i, src_i, tgt_i]


def _pick_actions(kind: str, agent, obs_norm, states: list[State],
                  player: int, rng: np.random.Generator) -> np.ndarray:
    """Returns (n_envs,) flat action indices for the given player."""
    n = len(states)
    actions = np.zeros(n, dtype=np.int64)

    # Compute masks for this player. Batched numpy.
    bulk_alive    = np.stack([s.buildings_alive    for s in states])
    bulk_owner    = np.stack([s.buildings_owner    for s in states])
    bulk_garrison = np.stack([s.buildings_garrison for s in states])
    bulk_galive   = np.stack([s.groups_alive       for s in states])
    masks = compute_mask_batched(bulk_alive, bulk_owner, bulk_garrison, bulk_galive, player)

    if kind == "random_legal":
        return random_legal_opponent_batched(masks, rng)

    if kind == "noop":
        return np.full(n, NOOP_INDEX, dtype=np.int64)

    # Neural — encode each state with mirroring if P2, run agent.
    obs_array = np.zeros((n, len(encode_obs(_state_to_obs_dict_for_player(states[0], masks[0], player)))),
                         dtype=np.float32)
    for i, s in enumerate(states):
        d = _state_to_obs_dict_for_player(s, masks[i], player)
        obs_array[i] = encode_obs(d)
    if obs_norm is not None:
        obs_array = obs_norm.normalize(obs_array)
    flat_actions, *_ = agent.act_batch(obs_array, masks)
    return flat_actions.astype(np.int64)


def run_match(
    p1: str,
    p2: str,
    games: int = 100,
    max_ticks: int = 200,
    level: str = "random_8_16",
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    """Run one head-to-head match and return {p1_wins, p2_wins, draws, total, settled, wall_s}.

    Reusable from cron-agent / Elo-update path. `p1`/`p2` accept the same
    forms as the CLI: experiment dir path, Supabase run id, or
    'random_legal'/'noop'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p1_kind, p1_agent, p1_norm = _load_policy(p1, device)
    p2_kind, p2_agent, p2_norm = _load_policy(p2, device)

    vec = JaxVecEnv(n_envs=games, level_name=level, base_seed=seed)
    rng = np.random.default_rng(seed)

    p1_wins = p2_wins = draws = settled = 0
    finished = np.zeros(games, dtype=bool)

    import jax.numpy as jnp
    t0 = time.perf_counter()
    for tick in range(max_ticks):
        states = vec.snapshot_numpy_states()
        a1_flat = _pick_actions(p1_kind, p1_agent, p1_norm, states, C.OWNER_P1, rng)
        a2_flat = _pick_actions(p2_kind, p2_agent, p2_norm, states, C.OWNER_P2, rng)
        a_batch = np.zeros((games, 2, ACTION_DIM), dtype=np.int32)
        for i in range(games):
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
                settled += 1
                finished[i] = True
        if finished.all():
            break

    wall = time.perf_counter() - t0
    not_settled = games - settled
    if not_settled > 0:
        draws += not_settled
    total = p1_wins + p2_wins + draws

    if verbose:
        print(f"  {games} games × max {max_ticks} ticks on {level}: "
              f"P1 {p1_wins} / P2 {p2_wins} / draw {draws} ({wall:.1f}s)")

    return {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws":   draws,
        "total":   total,
        "settled": settled,
        "wall_s":  wall,
    }


# ---------------------------------------------------------------------------
# Elo update helpers
# ---------------------------------------------------------------------------

def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32) -> tuple[float, float]:
    """Standard Elo. score_a in [0, 1] (1 win, 0.5 draw, 0 loss).

    Returns (new_rating_a, new_rating_b).
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def fetch_run_elo(conn, run_id: str) -> tuple[float, int]:
    """Returns (elo_score, elo_n_matches) for a Supabase run."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT elo_score, elo_n_matches FROM runs WHERE id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"run {run_id} not found")
    score = float(row[0]) if row[0] is not None else 1200.0
    n     = int(row[1])   if row[1] is not None else 0
    return score, n


def write_run_elo(conn, run_id: str, new_score: float, new_n_matches: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE runs SET elo_score = %s, elo_n_matches = %s WHERE id = %s",
            (float(new_score), int(new_n_matches), run_id),
        )
    conn.commit()


def update_elo_from_match(
    conn,
    p1_run_id: str,
    p2_run_id: str | None,
    result: dict,
    k: float = 32,
) -> tuple[float, float]:
    """Apply Elo update to two Supabase runs based on a match result dict.

    `p2_run_id=None` is supported when p2 was random_legal/noop (no Elo to
    update on the right side); only p1 gets a delta against a fixed 1200
    baseline.

    Score for p1 = (p1_wins + 0.5*draws) / total, weighted by total games.
    For simplicity we apply ONE Elo step using the aggregate score (matches
    practical Elo on tournament outcomes; high game count keeps variance low).
    """
    total = result["total"]
    p1_score = (result["p1_wins"] + 0.5 * result["draws"]) / max(total, 1)

    p1_rating, p1_n = fetch_run_elo(conn, p1_run_id)
    if p2_run_id is None:
        # Treat opponent as fixed at baseline rating.
        p2_rating = 1200.0
        p1_new, _ = elo_update(p1_rating, p2_rating, p1_score, k=k)
        write_run_elo(conn, p1_run_id, p1_new, p1_n + 1)
        return p1_new, p2_rating
    p2_rating, p2_n = fetch_run_elo(conn, p2_run_id)
    p1_new, p2_new = elo_update(p1_rating, p2_rating, p1_score, k=k)
    write_run_elo(conn, p1_run_id, p1_new, p1_n + 1)
    write_run_elo(conn, p2_run_id, p2_new, p2_n + 1)
    return p1_new, p2_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True, help="path to experiment dir, or 'random_legal'/'noop'")
    ap.add_argument("--p2", required=True, help="path to experiment dir, or 'random_legal'/'noop'")
    ap.add_argument("--games", type=int, default=1024,
                    help="number of parallel games (= n_envs). One eval round.")
    ap.add_argument("--max-ticks", type=int, default=200,
                    help="hard cap on game length (timeout).")
    ap.add_argument("--level", default="random_8_16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--update-elo", action="store_true",
                    help="write Elo deltas back to Supabase for p1 and p2 (when "
                         "they are run UUIDs). Random/noop opponents are treated "
                         "as fixed 1200 baseline.")
    ap.add_argument("--elo-k", type=float, default=32.0,
                    help="Elo K-factor for --update-elo (default 32)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"P1: {args.p1}")
    print(f"P2: {args.p2}")
    print(f"{args.games} games × max {args.max_ticks} ticks on {args.level}")

    p1_kind, p1_agent, p1_norm = _load_policy(args.p1, device)
    p2_kind, p2_agent, p2_norm = _load_policy(args.p2, device)

    vec = JaxVecEnv(n_envs=args.games, level_name=args.level, base_seed=args.seed)
    rng = np.random.default_rng(args.seed)

    p1_wins = 0
    p2_wins = 0
    draws   = 0
    settled = 0
    finished = np.zeros(args.games, dtype=bool)

    # Bypass JaxVecEnv.step_chunk because it auto-resets done envs (which
    # corrupts post-step phase reads). Use _step_batched directly so each
    # env stops at its terminal phase and we can read it.
    import jax.numpy as jnp
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

        # Score newly terminated envs by reading their phase from the
        # post-step state (no auto-reset — phase reflects winner).
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
                settled += 1
                finished[i] = True

        if finished.all():
            break

    wall = time.perf_counter() - t0

    # Any envs that never terminated count as draws (timeout reached or post-
    # auto-reset; latter shouldn't happen with single-tick K=1 but defensive).
    not_settled = args.games - settled
    if not_settled > 0:
        draws += not_settled

    total = p1_wins + p2_wins + draws
    print(f"\n=== results ({wall:.1f}s wall) ===")
    print(f"  P1 wins: {p1_wins:>5d} ({100*p1_wins/total:5.1f}%)")
    print(f"  P2 wins: {p2_wins:>5d} ({100*p2_wins/total:5.1f}%)")
    print(f"  draws:   {draws:>5d} ({100*draws/total:5.1f}%)")
    print(f"  settled: {settled:>5d}/{total}")

    if args.update_elo:
        # Resolve which sides are real Supabase run ids (not random_legal/noop).
        def _is_uuid_like(x):
            x = str(x)
            return len(x) >= 8 and all(c in "0123456789abcdefABCDEF-" for c in x)

        p1_id = args.p1 if _is_uuid_like(args.p1) and args.p1 not in ("random_legal", "noop") else None
        p2_id = args.p2 if _is_uuid_like(args.p2) and args.p2 not in ("random_legal", "noop") else None
        if p1_id is None and p2_id is None:
            print("  (no run ids on either side — nothing to update)")
        else:
            from cli.db import connect
            result = {"p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws, "total": total}
            with connect() as conn:
                if p1_id is not None and p2_id is not None:
                    p1_new, p2_new = update_elo_from_match(conn, p1_id, p2_id, result, k=args.elo_k)
                    print(f"  Elo: p1 -> {p1_new:.1f}, p2 -> {p2_new:.1f}")
                elif p1_id is not None:
                    p1_new, _ = update_elo_from_match(conn, p1_id, None, result, k=args.elo_k)
                    print(f"  Elo: p1 -> {p1_new:.1f} (vs baseline 1200)")
                else:
                    # Only p2 is real; flip the result and update p2 against baseline.
                    flipped = {"p1_wins": p2_wins, "p2_wins": p1_wins, "draws": draws, "total": total}
                    p2_new, _ = update_elo_from_match(conn, p2_id, None, flipped, k=args.elo_k)
                    print(f"  Elo: p2 -> {p2_new:.1f} (vs baseline 1200)")


if __name__ == "__main__":
    main()
