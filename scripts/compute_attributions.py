"""Per-run feature attribution via Integrated Gradients on the value head.

Loads a run's trained weights, plays the policy vs random_legal on its
training level to collect a sample of decision states, computes IG of the
value head w.r.t. the (normalized) observation, then aggregates the raw
1002-dim attributions to the 41 named features defined in the v9 encoder
(10 globals + 22 per-building summed across 32 slots + 9 per-group summed
across 32 slots). Writes to run_feature_importance.

Usage:
    python scripts/compute_attributions.py --run-id <uuid>
    python scripts/compute_attributions.py --run-id <uuid> --games 32 --ig-steps 50
"""
from __future__ import annotations

import argparse
import json
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

from cli.db import connect
from sim import config as C
from sim.actions import compute_mask_batched
from sim.engine_jax import ACTION_DIM
from sim.envs.jax_vec_env import JaxVecEnv, _step_batched
from training.encoder import (
    BUILDING_FEATS,
    GLOBAL_FEATS,
    GROUP_FEATS,
    N_BUILDINGS,
    N_GROUPS,
    encode_obs,
)
from training.net import ActorCritic
from scripts.tournament import (
    _decode_action_to_packed,
    _load_policy,
    _pick_actions,
    _state_to_obs_dict_for_player,
)


# Named features matching training/encoder.py — keep in sync if the encoder
# changes. The unit tests at the bottom of this file assert the counts match
# the constants imported above.
GLOBAL_FEATURES = [
    "tick", "time_remaining",
    "p1_buildings_share", "p2_buildings_share", "neutral_buildings_share",
    "p1_total_force", "p2_total_force",
    "p1_share", "building_margin", "unit_margin",
]

BUILDING_FEATURES = [
    "alive", "is_p1", "is_p2", "is_neutral",
    "garrison_raw", "garrison_ratio", "capacity", "over_cap",
    "x", "y",
    "type_oh_0", "type_oh_1", "type_oh_2", "type_oh_3", "type_oh_4",
    "incoming_p1", "incoming_p2", "incoming_friendly", "incoming_hostile",
    "threat_capped", "will_fall", "near_cap",
]

GROUP_FEATURES = [
    "alive", "is_p1", "is_p2", "progress_fraction", "count",
    "src_x", "src_y", "tgt_x", "tgt_y",
]

assert len(GLOBAL_FEATURES) == GLOBAL_FEATS == 10
assert len(BUILDING_FEATURES) == BUILDING_FEATS == 22
assert len(GROUP_FEATURES) == GROUP_FEATS == 9


def collect_states(agent, obs_norm, level: str, n_games: int,
                   max_ticks: int, seed: int) -> np.ndarray:
    """Roll out policy vs random_legal; capture every P1 decision-tick obs.
    Returns (M, OBS_DIM) of *normalized* observations — what the net sees."""
    import jax.numpy as jnp

    vec = JaxVecEnv(n_envs=n_games, level_name=level, base_seed=seed)
    rng = np.random.default_rng(seed)
    finished = np.zeros(n_games, dtype=bool)
    obs_buffer: list[np.ndarray] = []

    for _ in range(max_ticks):
        states = vec.snapshot_numpy_states()
        bulk_alive    = np.stack([s.buildings_alive    for s in states])
        bulk_owner    = np.stack([s.buildings_owner    for s in states])
        bulk_garrison = np.stack([s.buildings_garrison for s in states])
        bulk_galive   = np.stack([s.groups_alive       for s in states])
        masks_p1 = compute_mask_batched(bulk_alive, bulk_owner, bulk_garrison,
                                        bulk_galive, C.OWNER_P1)

        for i, s in enumerate(states):
            if finished[i]:
                continue
            d = _state_to_obs_dict_for_player(s, masks_p1[i], C.OWNER_P1)
            obs_buffer.append(encode_obs(d))

        a1 = _pick_actions("neural",       agent, obs_norm, states, C.OWNER_P1, rng)
        a2 = _pick_actions("random_legal", None,  None,     states, C.OWNER_P2, rng)
        a_batch = np.zeros((n_games, 2, ACTION_DIM), dtype=np.int32)
        for i in range(n_games):
            _decode_action_to_packed(int(a1[i]), a_batch[i, 0])
            _decode_action_to_packed(int(a2[i]), a_batch[i, 1])
        a1j = jnp.asarray(a_batch[:, 0, :], dtype=jnp.int32)
        a2j = jnp.asarray(a_batch[:, 1, :], dtype=jnp.int32)
        vec.state, _r1, _r2, dones = _step_batched(vec.state, a1j, a2j)
        finished |= np.asarray(dones)
        if finished.all():
            break

    obs_arr = np.stack(obs_buffer, axis=0).astype(np.float32)
    if obs_norm is not None:
        obs_arr = obs_norm.normalize(obs_arr)
    return obs_arr


def integrated_gradients(net: ActorCritic, obs_normalized: np.ndarray,
                         steps: int = 50, batch: int = 64,
                         device: torch.device | None = None) -> np.ndarray:
    """Integrated Gradients of value head w.r.t. obs.
    Baseline = zeros (reasonable for normalized obs). Returns (M, OBS_DIM)."""
    if device is None:
        device = next(net.parameters()).device
    net.eval()
    obs = torch.as_tensor(obs_normalized, dtype=torch.float32, device=device)
    M, D = obs.shape
    out = torch.zeros_like(obs)
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(steps, 1, 1)

    for start in range(0, M, batch):
        end = min(start + batch, M)
        x = obs[start:end]
        baseline = torch.zeros_like(x)
        interp = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)
        interp = interp.reshape(-1, D).requires_grad_(True)

        body = net.forward_body(interp)
        value = net.value(body)
        grads = torch.autograd.grad(value.sum(), interp)[0]
        avg_grad = grads.view(steps, end - start, D).mean(dim=0)
        out[start:end] = (x - baseline) * avg_grad

    return out.detach().cpu().numpy()


def aggregate_to_named(ig: np.ndarray) -> dict:
    """Aggregate raw (M, 1002) IG to per-named-feature stats.

    Globals: direct per-index stats.
    Buildings: indices 10..713 reshape to (M, 32, 22); sum |attrib| across
        the 32 slots per state, then mean+std over states. Same for signed.
    Groups: same as buildings but (M, 32, 9).
    """
    M = ig.shape[0]
    out: dict[str, list] = {}

    g = ig[:, :GLOBAL_FEATS]
    out["globals"] = list(zip(
        range(GLOBAL_FEATS), GLOBAL_FEATURES,
        np.abs(g).mean(axis=0).tolist(), g.mean(axis=0).tolist(),
        np.abs(g).std(axis=0).tolist(),
    ))

    b_end = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS
    b = ig[:, GLOBAL_FEATS:b_end].reshape(M, N_BUILDINGS, BUILDING_FEATS)
    b_abs    = np.abs(b).sum(axis=1)
    b_signed = b.sum(axis=1)
    out["building"] = list(zip(
        range(BUILDING_FEATS), BUILDING_FEATURES,
        b_abs.mean(axis=0).tolist(), b_signed.mean(axis=0).tolist(),
        b_abs.std(axis=0).tolist(),
    ))

    g_block = ig[:, b_end:].reshape(M, N_GROUPS, GROUP_FEATS)
    g_abs    = np.abs(g_block).sum(axis=1)
    g_signed = g_block.sum(axis=1)
    out["group"] = list(zip(
        range(GROUP_FEATS), GROUP_FEATURES,
        g_abs.mean(axis=0).tolist(), g_signed.mean(axis=0).tolist(),
        g_abs.std(axis=0).tolist(),
    ))
    return out


def compute_weight_l2(net: ActorCritic) -> dict:
    """Per-named-feature L2 norm of trunk.0 input columns, aggregated to
    match the IG aggregation (sum across 32 slots for building/group blocks)."""
    W = net.trunk[0].weight.detach().cpu().numpy()
    col_l2 = np.linalg.norm(W, ord=2, axis=0)

    b_end = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS
    return {
        "globals":  col_l2[:GLOBAL_FEATS].tolist(),
        "building": col_l2[GLOBAL_FEATS:b_end].reshape(N_BUILDINGS, BUILDING_FEATS).sum(axis=0).tolist(),
        "group":    col_l2[b_end:].reshape(N_GROUPS, GROUP_FEATS).sum(axis=0).tolist(),
    }


def upsert(run_id: str, agg: dict, wl2: dict, n_states: int) -> int:
    rows = []
    for block in ("globals", "building", "group"):
        col = wl2[block]
        for idx, name, mean_abs, mean_signed, std in agg[block]:
            rows.append((run_id, block, idx, name,
                         float(mean_abs), float(mean_signed), float(std),
                         float(col[idx]), int(n_states)))
    sql = """
        INSERT INTO run_feature_importance
            (run_id, feature_block, feature_index, feature_name,
             ig_mean_abs, ig_mean_signed, ig_std, weight_l2, n_states)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, feature_block, feature_index) DO UPDATE
        SET feature_name = EXCLUDED.feature_name,
            ig_mean_abs    = EXCLUDED.ig_mean_abs,
            ig_mean_signed = EXCLUDED.ig_mean_signed,
            ig_std         = EXCLUDED.ig_std,
            weight_l2      = EXCLUDED.weight_l2,
            n_states       = EXCLUDED.n_states,
            computed_at    = now()
    """
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--games", type=int, default=16,
                    help="Parallel games rolled out for state collection.")
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--max-states", type=int, default=2000,
                    help="Cap on states used for IG (random subsample).")
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT hyperparams::text FROM runs WHERE id = %s", (args.run_id,))
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"run {args.run_id} not found")
    hp = json.loads(row[0])
    level = hp.get("level_name") or "random_8_16"
    print(f"[attributions] run={args.run_id} level={level} device={device}")

    t0 = time.perf_counter()
    kind, agent, obs_norm = _load_policy(args.run_id, device)
    if kind != "neural":
        raise SystemExit(f"run {args.run_id} resolved to {kind!r}, not neural")
    print(f"[attributions] policy loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    obs_arr = collect_states(agent, obs_norm, level, args.games, args.max_ticks, args.seed)
    print(f"[attributions] collected {obs_arr.shape[0]} states in {time.perf_counter()-t0:.1f}s")

    if obs_arr.shape[0] > args.max_states:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(obs_arr.shape[0], size=args.max_states, replace=False)
        obs_arr = obs_arr[idx]
        print(f"[attributions] subsampled to {obs_arr.shape[0]} states")

    t0 = time.perf_counter()
    ig = integrated_gradients(agent.net, obs_arr, steps=args.ig_steps, device=device)
    print(f"[attributions] IG ({args.ig_steps} steps) in {time.perf_counter()-t0:.1f}s")

    agg = aggregate_to_named(ig)
    wl2 = compute_weight_l2(agent.net)

    n_rows = upsert(args.run_id, agg, wl2, n_states=int(obs_arr.shape[0]))
    print(f"[attributions] wrote {n_rows} rows for run {args.run_id}")

    flat = []
    for block in ("globals", "building", "group"):
        for idx, name, mean_abs, mean_signed, _std in agg[block]:
            flat.append((mean_abs, block, name, mean_signed, wl2[block][idx]))
    flat.sort(reverse=True)
    print("\n[attributions] top 15 by mean |IG|:")
    for ma, block, name, ms, w in flat[:15]:
        print(f"  {block:9s} {name:24s} |IG|={ma:.4f}  signed={ms:+.4f}  wL2={w:.3f}")


if __name__ == "__main__":
    main()
