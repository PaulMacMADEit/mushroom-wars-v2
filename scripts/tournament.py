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
from training.net import ActorCritic, infer_body_dim, infer_obs_dim
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

    out_dir = Path(tempfile.mkdtemp(prefix=f"mw2-tour-{str(run_id)[:8]}-"))
    w_path = out_dir / "weights.pt"
    urllib.request.urlretrieve(_public_url(w_url), w_path)
    n_path = None
    if n_url:
        n_path = out_dir / "obs_norm.pt"
        urllib.request.urlretrieve(_public_url(n_url), n_path)
    return w_path, n_path


def _load_policy(path: str | Path, device: torch.device):
    """Returns ('neural', agent, obs_norm, encode_fn) or ('random_legal',
    None, None, None) or ('noop', None, None, None). For neural, agent is a
    PPOAgent; obs_norm is a RunningNorm (or None); encode_fn is the encoder
    that was current when this checkpoint was saved (v9.0 for legacy, v10
    for new) — caller uses it instead of the bare `encode_obs` so cross-
    version matches don't feed v10-shape obs into a v9.0 net.

    `path` accepts:
      - 'random_legal' or 'noop' (literal opponent names)
      - a Supabase run id (UUID or short prefix matching one row)
      - an experiment dir path (containing weights.pt + obs_norm.pt)
    """
    if path == "random_legal":
        return ("random_legal", None, None, None)
    if path == "noop":
        return ("noop", None, None, None)

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
        # Support three layouts:
        #   1. Direct .pt file path: weights_path = p, sibling obs_norm guessed.
        #      e.g. /tmp/mw2-pfsp-XXX/abc12345-weights.pt → abc12345-obs_norm.pt
        #   2. Experiment dir with weights.pt + obs_norm.pt (legacy).
        #   3. Direct .pt with no sibling — weights only, no obs_norm.
        if str(p).endswith(".pt") and p.is_file():
            weights_path = p
            # Try same-prefix-different-suffix sibling for obs_norm.
            for sibling_name in (
                p.name.replace("-weights.pt", "-obs_norm.pt"),
                "obs_norm.pt",
            ):
                cand = p.parent / sibling_name
                if cand.exists() and cand != p:
                    obs_norm_p = cand
                    break
        else:
            weights_path = p / "weights.pt"
            if (p / "obs_norm.pt").exists():
                obs_norm_p = p / "obs_norm.pt"

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights.pt not found at {weights_path}")
    raw = torch.load(str(weights_path), map_location=device, weights_only=True)
    # 2026-04-29 fire 80: v10 trainer wraps weights as {state_dict, encoder_version}.
    # v9 saved a flat state_dict (no version stamp).
    if isinstance(raw, dict) and "state_dict" in raw and "encoder_version" in raw:
        state_dict      = raw["state_dict"]
        encoder_version = raw["encoder_version"]
    else:
        from training.encoders import DEFAULT_ENCODER_VERSION
        state_dict      = raw
        encoder_version = DEFAULT_ENCODER_VERSION
    from training.encoders import get_encoder
    encoder_entry = get_encoder(encoder_version)
    body_dim = infer_body_dim(state_dict)
    obs_dim  = infer_obs_dim(state_dict)
    if obs_dim != encoder_entry.obs_dim:
        raise ValueError(
            f"checkpoint trunk obs_dim={obs_dim} but encoder {encoder_version!r} "
            f"expects {encoder_entry.obs_dim} — version mismatch"
        )
    # 2026-04-29: must size the net to the checkpoint's actual obs_dim — not
    # the current OBS_DIM constant — or v9.0 (1002) ↔ v10 (1008) cross-version
    # matches crash with `size mismatch for trunk.0.weight`. The encoder is
    # dispatched per-checkpoint so a v9 net gets fed v9-shape obs.
    net = ActorCritic(obs_dim=obs_dim, body_dim=body_dim)
    net.load_state_dict(state_dict)
    agent = PPOAgent(net, device=device)
    obs_norm = None
    if obs_norm_p and Path(obs_norm_p).exists():
        # File shape wins on load; the constant just sets a placeholder.
        obs_norm = RunningNorm(obs_dim)
        obs_norm.load(str(obs_norm_p))
    return ("neural", agent, obs_norm, encoder_entry.encode)


def _state_to_obs_dict_for_player(state: State, mask: np.ndarray, player: int) -> dict:
    """Build the obs dict for a single player. For P2 we mirror ownership so
    a P1-trained policy sees itself as P1.

    Must include the v10 decision-interval fields (arrivals_*, prev_*,
    last_actions_*) so encode_obs doesn't KeyError on a v10 checkpoint.
    """
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
        # v10 decision-interval fields. Must mirror sim.envs.mushroom_env._make_obs.
        "arrivals_p1":          state.arrivals_p1.copy(),
        "arrivals_p2":          state.arrivals_p2.copy(),
        "prev_buildings_owner": state.prev_buildings_owner.copy(),
        "prev_p1_units_total":  np.int32(state.prev_p1_units_total),
        "prev_p2_units_total":  np.int32(state.prev_p2_units_total),
        "last_actions_p1":      state.last_actions_p1.copy(),
        "last_actions_p2":      state.last_actions_p2.copy(),
    }
    if player == C.OWNER_P2:
        # Mirror P1 <-> P2 in every owner-keyed field. Same semantics as
        # sim.envs.opponents._mirror_ownership — keep the two in sync.
        for k in ("buildings_owner", "groups_owner", "prev_buildings_owner"):
            o = out[k]
            swapped = o.copy()
            swapped = np.where(o == C.OWNER_P1, C.OWNER_P2, swapped)
            swapped = np.where(o == C.OWNER_P2, C.OWNER_P1, swapped)
            out[k] = swapped.astype(o.dtype)
        # Per-player arrays just swap labels.
        for key_p1, key_p2 in (
            ("arrivals_p1",         "arrivals_p2"),
            ("last_actions_p1",     "last_actions_p2"),
            ("prev_p1_units_total", "prev_p2_units_total"),
        ):
            out[key_p1], out[key_p2] = out[key_p2], out[key_p1]
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
                  player: int, rng: np.random.Generator,
                  encode_fn=None, deterministic: bool = True) -> np.ndarray:
    """Returns (n_envs,) flat action indices for the given player.

    `encode_fn` is the encoder matching this checkpoint's version (v9.0 vs
    v10). Falls back to the current encoder for back-compat with callers
    that don't thread the version (random_legal/noop don't care).
    """
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
    enc = encode_fn if encode_fn is not None else encode_obs
    obs_array = np.zeros((n, len(enc(_state_to_obs_dict_for_player(states[0], masks[0], player)))),
                         dtype=np.float32)
    for i, s in enumerate(states):
        d = _state_to_obs_dict_for_player(s, masks[i], player)
        obs_array[i] = enc(d)
    if obs_norm is not None:
        obs_array = obs_norm.normalize(obs_array)
    # Default eval is deterministic — argmax keeps Elo stable. Pass
    # deterministic=False from paths that want apples-to-apples with the
    # stochastic training rollouts (e.g. End-of-run rematch).
    flat_actions, *_ = agent.act_batch(obs_array, masks, deterministic=deterministic)
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
    level_mix: list | dict | None = None,
    deterministic: bool = True,
) -> dict:
    """Run one head-to-head match and return {p1_wins, p2_wins, draws, total, settled, wall_s}.

    Reusable from cron-agent / Elo-update path. `p1`/`p2` accept the same
    forms as the CLI: experiment dir path, Supabase run id, or
    'random_legal'/'noop'.

    `level_mix`: optional dict {name: weight} or list of (name, weight). When
    provided, each env samples a level on reset; `level` becomes a label only.
    Required when `level` is a label like "phase1_full_mix_4_8" that the
    static level loader doesn't recognise.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p1_kind, p1_agent, p1_norm, p1_encode = _load_policy(p1, device)
    p2_kind, p2_agent, p2_norm, p2_encode = _load_policy(p2, device)

    # Normalise level_mix to list-of-tuples (matches trainer.py:326-334).
    mix = None
    if level_mix:
        if isinstance(level_mix, dict):
            mix = [(str(k), float(v)) for k, v in level_mix.items()]
        else:
            mix = [(str(item[0]), float(item[1])) for item in level_mix]

    vec = JaxVecEnv(n_envs=games, level_name=level, base_seed=seed, level_mix=mix)
    rng = np.random.default_rng(seed)

    p1_wins = p2_wins = draws = settled = 0
    finished = np.zeros(games, dtype=bool)

    import jax.numpy as jnp
    t0 = time.perf_counter()
    for tick in range(max_ticks):
        states = vec.snapshot_numpy_states()
        a1_flat = _pick_actions(p1_kind, p1_agent, p1_norm, states, C.OWNER_P1, rng, p1_encode, deterministic=deterministic)
        a2_flat = _pick_actions(p2_kind, p2_agent, p2_norm, states, C.OWNER_P2, rng, p2_encode, deterministic=deterministic)
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
    timeouts = games - settled  # 2026-04-29 fire 65: tracked separately from resolved draws
    total = p1_wins + p2_wins + draws + timeouts

    if verbose:
        print(f"  {games} games × max {max_ticks} ticks on {level}: "
              f"P1 {p1_wins} / P2 {p2_wins} / draw {draws} / timeout {timeouts} ({wall:.1f}s)")

    return {
        "p1_wins":  p1_wins,
        "p2_wins":  p2_wins,
        "draws":    draws,     # resolved draws only (mutual elimination)
        "timeouts": timeouts,  # neither player resolved by max_ticks
        "total":    total,
        "settled":  settled,
        "wall_s":   wall,
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
    score = float(row[0]) if row[0] is not None else 1000.0
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

    Score for p1 = (p1_wins + 0.5*draws) / decided_games.

    2026-04-29 fire 65: TIMEOUTS NO LONGER COUNT IN ELO. Previously,
    timeouts (games that hit MAX_TICKS without either player winning)
    were lumped into draws at 0.5 each, which let mutual-noop policies
    maintain Elo against each other indefinitely. Now timeouts are
    excluded from the Elo calculation entirely — they're treated as
    "games not played" and don't contribute to either player's rating.
    Resolved draws (mutual elimination) still score 0.5 each.

    Edge case: if ALL games timed out, decided_games == 0 and we skip
    the Elo update (no signal to extract).
    """
    total = result["total"]
    timeouts = result.get("timeouts", 0)
    decided = total - timeouts
    if decided <= 0:
        # No games resolved — nothing to learn. Skip the Elo update.
        p1_rating, _ = fetch_run_elo(conn, p1_run_id)
        if p2_run_id is None:
            return p1_rating, 1000.0
        p2_rating, _ = fetch_run_elo(conn, p2_run_id)
        return p1_rating, p2_rating
    p1_score = (result["p1_wins"] + 0.5 * result["draws"]) / decided

    p1_rating, p1_n = fetch_run_elo(conn, p1_run_id)
    if p2_run_id is None:
        # Treat opponent as fixed at baseline rating.
        p2_rating = 1000.0
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

    p1_kind, p1_agent, p1_norm, p1_encode = _load_policy(args.p1, device)
    p2_kind, p2_agent, p2_norm, p2_encode = _load_policy(args.p2, device)

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

        a1_flat = _pick_actions(p1_kind, p1_agent, p1_norm, states, C.OWNER_P1, rng, p1_encode)
        a2_flat = _pick_actions(p2_kind, p2_agent, p2_norm, states, C.OWNER_P2, rng, p2_encode)

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

    # 2026-04-29 fire 65: timeouts tracked separately from resolved draws.
    timeouts = args.games - settled

    total = p1_wins + p2_wins + draws + timeouts
    print(f"\n=== results ({wall:.1f}s wall) ===")
    print(f"  P1 wins:  {p1_wins:>5d} ({100*p1_wins/total:5.1f}%)")
    print(f"  P2 wins:  {p2_wins:>5d} ({100*p2_wins/total:5.1f}%)")
    print(f"  draws:    {draws:>5d} ({100*draws/total:5.1f}%)")
    print(f"  timeouts: {timeouts:>5d} ({100*timeouts/total:5.1f}%)  ← both score 0 in Elo")
    print(f"  settled:  {settled:>5d}/{total}")

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
                    print(f"  Elo: p1 -> {p1_new:.1f} (vs baseline 1000)")
                else:
                    # Only p2 is real; flip the result and update p2 against baseline.
                    flipped = {"p1_wins": p2_wins, "p2_wins": p1_wins, "draws": draws, "timeouts": timeouts, "total": total}
                    p2_new, _ = update_elo_from_match(conn, p2_id, None, flipped, k=args.elo_k)
                    print(f"  Elo: p2 -> {p2_new:.1f} (vs baseline 1000)")


if __name__ == "__main__":
    main()
