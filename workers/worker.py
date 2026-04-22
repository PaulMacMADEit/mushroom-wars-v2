"""Mushroom-wars-v2 training worker.

Polls Supabase for queued runs, trains each one, writes results back.

Usage:
    # Single-shot: claim one run, run it, exit.
    python workers/worker.py --one

    # Poll forever until told to stop.
    python workers/worker.py

    # Poll until N consecutive empty polls, then exit (useful for CI/cron).
    python workers/worker.py --exit-after-idle 6 --poll-interval 5

Scope notes:
  - No Storage upload yet — weights/optimizer stay local. `weights_url` is
    NULL on the completed row.
  - No champion-pool eval — `result.rate` is the final rollout win rate
    against random-legal opponent (same signal Phase-2 smoke used).
  - Single-env trainer. Phase-2 vec-trainer lands later.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from cli.db import PROJECT, connect
from sim.envs import MushroomEnv, random_legal_opponent
from training.agent import PPOAgent
from training.net import ActorCritic
from training.trainer import PPOConfig, PPOTrainer


# ---------------------------------------------------------------------------
# Device + net construction
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_net_for_model(model_id: str, obs_size: int, num_actions: int) -> ActorCritic:
    """Phase-3 minimum: only the v9.0-smoke net is supported.

    When more model variants exist, this dispatch grows — or better, moves to
    a registry keyed on model_id. For now, fail loudly on unknown models
    instead of silently falling through.
    """
    from training.net import ActorCritic as SmokeNet
    from training.encoder import OBS_DIM
    from sim.actions import ACTION_SPACE_SIZE

    if model_id == "v9.0-smoke":
        if obs_size != OBS_DIM or num_actions != ACTION_SPACE_SIZE:
            raise ValueError(
                f"model row {model_id!r} specifies obs={obs_size}, "
                f"actions={num_actions}, but code has obs={OBS_DIM}, actions={ACTION_SPACE_SIZE}. "
                "Did the encoder/action space change without a new model id?"
            )
        return SmokeNet()
    raise ValueError(f"unknown model_id: {model_id!r} — add a case in build_net_for_model")


# ---------------------------------------------------------------------------
# Claim + finalize
# ---------------------------------------------------------------------------

def claim_one(conn, machine: str):
    """Call claim_next_run; return dict or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, model_id, simulator_id, label, budget_ms, seed, hyperparams::text "
            "FROM claim_next_run(%s, %s)",
            (PROJECT, machine),
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    id_, model_id, sim_id, label, budget_ms, seed, hp_text = row
    return {
        "id":         id_,
        "model_id":   model_id,
        "sim_id":     sim_id,
        "label":      label,
        "budget_ms":  budget_ms,
        "seed":       seed,
        "hyperparams": json.loads(hp_text) if hp_text else {},
    }


def fetch_model_meta(conn, model_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT obs_size, num_actions FROM models WHERE id = %s",
            (model_id,),
        )
        row = cur.fetchone()
    if not row:
        raise ValueError(f"model {model_id!r} not registered")
    return {"obs_size": row[0], "num_actions": row[1]}


def mark_done(conn, run_id, result: dict, games_played: int, wall_ms: int):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE runs
               SET status       = 'done',
                   result       = %s::jsonb,
                   games_played = %s,
                   wall_ms      = %s,
                   finished_at  = now()
             WHERE id = %s
        """, (json.dumps(result), games_played, wall_ms, run_id))
    conn.commit()


def mark_failed(conn, run_id, error: str, wall_ms: int):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE runs
               SET status      = 'failed',
                   error       = %s,
                   wall_ms     = %s,
                   finished_at = now()
             WHERE id = %s
        """, (error[:8000], wall_ms, run_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Train one run
# ---------------------------------------------------------------------------

def run_training(job: dict, model_meta: dict, device: torch.device) -> tuple[dict, int]:
    """Do the actual PPO training for the claimed job. Return (result, games_played)."""
    hp = job["hyperparams"] or {}
    seed_int = _seed_to_int(job["seed"])

    # Build config: start from defaults, overlay any hyperparams the caller provided.
    cfg_kwargs = {k: v for k, v in hp.items() if k in PPOConfig.__dataclass_fields__}
    cfg = PPOConfig(**cfg_kwargs)

    # Build env + agent + trainer.
    env = MushroomEnv(seed=seed_int, opponent=random_legal_opponent)
    net = build_net_for_model(job["model_id"], model_meta["obs_size"], model_meta["num_actions"])
    agent = PPOAgent(net, device=device)
    trainer = PPOTrainer(env, agent, cfg)

    # Budget loop: update until we hit budget_ms.
    deadline = time.time() + job["budget_ms"] / 1000.0
    last_metrics: dict = {}
    total_eps = 0
    total_wins = 0
    updates = 0

    while time.time() < deadline:
        metrics = trainer.update()
        updates += 1
        if "episodes_completed" in metrics:
            eps_count = metrics["episodes_completed"]
            total_eps += eps_count
            total_wins += int(round(eps_count * metrics.get("win_rate", 0.0)))
        last_metrics = metrics

    overall_win_rate = total_wins / total_eps if total_eps else 0.0
    result = {
        "rate":                 overall_win_rate,
        "updates":              updates,
        "training_episodes":    total_eps,
        "training_wins":        total_wins,
        "final_metrics":        last_metrics,
        "config":               asdict(cfg),
        "device":               str(device),
    }
    return result, total_eps


def _seed_to_int(seed: str) -> int:
    """Map a seed string to an int for numpy/torch RNG. 'a' -> 97, 'abc' -> hash."""
    if seed is None:
        return 0
    if seed.isdigit() or (seed.startswith("-") and seed[1:].isdigit()):
        return int(seed)
    # Simple deterministic mapping.
    h = 0
    for ch in seed:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true",
                    help="claim + run one job, then exit")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--exit-after-idle", type=int, default=0,
                    help="exit after N consecutive empty polls (0 = forever)")
    ap.add_argument("--machine", default=socket.gethostname())
    args = ap.parse_args()

    device = pick_device()
    print(f"[worker] machine={args.machine}  device={device}")

    idle_streak = 0
    while True:
        claimed = None
        try:
            with connect() as conn:
                job = claim_one(conn, args.machine)
                if job is None:
                    idle_streak += 1
                    if args.one:
                        print("[worker] no queued run, exiting (--one).")
                        return
                    if args.exit_after_idle and idle_streak >= args.exit_after_idle:
                        print(f"[worker] {idle_streak} idle polls, exiting.")
                        return
                    print(f"[worker] idle ({idle_streak}) — sleeping {args.poll_interval}s")
                    time.sleep(args.poll_interval)
                    continue

                idle_streak = 0
                claimed = job
                print(f"[worker] claimed run id={job['id']} label={job['label']!r} "
                      f"model={job['model_id']} sim={job['sim_id']} "
                      f"budget={job['budget_ms']}ms seed={job['seed']!r}")

                model_meta = fetch_model_meta(conn, job["model_id"])

            # Training happens outside the `with connect()` block so we don't
            # hold a DB connection for the whole training run.
            t0 = time.time()
            result, games = run_training(claimed, model_meta, device)
            wall_ms = int((time.time() - t0) * 1000)

            with connect() as conn:
                mark_done(conn, claimed["id"], result, games, wall_ms)
            print(f"[worker] done id={claimed['id']} games={games} "
                  f"wall={wall_ms}ms rate={result['rate']:.3f}")

        except KeyboardInterrupt:
            if claimed is not None:
                wall_ms = int((time.time() - t0) * 1000) if "t0" in dir() else 0
                with connect() as conn:
                    mark_failed(conn, claimed["id"], "interrupted (SIGINT)", wall_ms)
                print(f"[worker] marked run {claimed['id']} failed (interrupted)")
            raise
        except Exception as exc:
            err = f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}"
            print(f"[worker] error: {err}")
            if claimed is not None:
                wall_ms = int((time.time() - t0) * 1000) if "t0" in dir() else 0
                try:
                    with connect() as conn:
                        mark_failed(conn, claimed["id"], err, wall_ms)
                except Exception as exc2:
                    print(f"[worker] failed to mark run as failed: {exc2}")
            if args.one:
                return
            time.sleep(args.poll_interval)

        if args.one:
            return


if __name__ == "__main__":
    main()
