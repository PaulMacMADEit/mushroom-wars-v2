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
import os
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
    """Dispatch model_id → nn.Module.

    Keyed on the model_id string so we can swap architectures without touching
    existing rows. The obs/action checks catch "model row vs code drift" —
    i.e. someone changed encoder/action-space dims without bumping model_id.
    """
    from sim.actions import ACTION_SPACE_SIZE
    from training.encoder import OBS_DIM
    from training.net import ActorCritic

    KNOWN = {
        # v9.0-enc-full was the interim commit-1 model (full encoder + old
        # flat 4097 head). Code has since moved to chained heads — running
        # that model against this code will fail at inference, which is
        # correct: the net topology doesn't match.
        "v9.0-enc-full": (OBS_DIM, ACTION_SPACE_SIZE, ActorCritic),
        # v9.0-full: full encoder + chained source/type/target heads
        # (ARCHITECTURE §9.4). This is the current production model.
        "v9.0-full":     (OBS_DIM, ACTION_SPACE_SIZE, ActorCritic),
    }
    entry = KNOWN.get(model_id)
    if entry is None:
        raise ValueError(
            f"unknown model_id: {model_id!r} — add a case in build_net_for_model. "
            f"Known: {sorted(KNOWN)}"
        )
    expected_obs, expected_actions, cls = entry
    if obs_size != expected_obs or num_actions != expected_actions:
        raise ValueError(
            f"model row {model_id!r} specifies obs={obs_size}, actions={num_actions}; "
            f"code expects obs={expected_obs}, actions={expected_actions}. "
            "Did the encoder/action space change without a new model id?"
        )
    return cls()


# ---------------------------------------------------------------------------
# Claim + finalize
# ---------------------------------------------------------------------------

def claim_one(conn, machine: str):
    """Call claim_next_run; return dict or None.

    Also fetches parent artifact URLs when parent_run_id is set, so the
    caller can init the trainer from the parent's weights + optimizer +
    obs_norm.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, model_id, simulator_id, label, budget_ms, seed, "
            "hyperparams::text, parent_run_id "
            "FROM claim_next_run(%s, %s)",
            (PROJECT, machine),
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    id_, model_id, sim_id, label, budget_ms, seed, hp_text, parent_id = row
    job = {
        "id":           id_,
        "model_id":     model_id,
        "sim_id":       sim_id,
        "label":        label,
        "budget_ms":    budget_ms,
        "seed":         seed,
        "hyperparams":  json.loads(hp_text) if hp_text else {},
        "parent_run_id": parent_id,
        "parent":       None,
    }
    if parent_id is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT weights_url, optimizer_url, obs_norm_url "
                "FROM runs WHERE id = %s",
                (parent_id,),
            )
            prow = cur.fetchone()
        if prow is not None:
            job["parent"] = {
                "weights_url":   prow[0],
                "optimizer_url": prow[1],
                "obs_norm_url":  prow[2],
            }
    return job


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


def mark_done(
    conn,
    run_id,
    result: dict,
    games_played: int,
    wall_ms: int,
    weights_url: str | None = None,
    optimizer_url: str | None = None,
    obs_norm_url: str | None = None,
    log_url: str | None = None,
):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE runs
               SET status        = 'done',
                   result        = %s::jsonb,
                   games_played  = %s,
                   wall_ms       = %s,
                   weights_url   = %s,
                   optimizer_url = %s,
                   obs_norm_url  = %s,
                   log_url       = %s,
                   finished_at   = now()
             WHERE id = %s
        """, (json.dumps(result), games_played, wall_ms,
              weights_url, optimizer_url, obs_norm_url, log_url, run_id))
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
# Parent-run artifact download (continuation training)
# ---------------------------------------------------------------------------

def _public_url(path: str | None) -> str | None:
    if not path:
        return None
    base = os.environ.get("SUPABASE_URL")
    if not base:
        raise RuntimeError("SUPABASE_URL not set")
    return f"{base}/storage/v1/object/public/{path}"


def _download_parent_state(parent: dict) -> dict:
    """Download parent's weights.pt / optimizer.pt / obs_norm.pt from Storage.

    Returns a dict with three torch-state values (any may be None if the parent
    didn't upload that artifact).
    """
    import io
    import urllib.request

    def fetch_load(path: str | None):
        url = _public_url(path)
        if url is None:
            return None
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    return {
        "weights":   fetch_load(parent.get("weights_url")),
        "optimizer": fetch_load(parent.get("optimizer_url")),
        "obs_norm":  fetch_load(parent.get("obs_norm_url")),
    }


# ---------------------------------------------------------------------------
# Train one run
# ---------------------------------------------------------------------------

def run_training(
    job: dict,
    model_meta: dict,
    device: torch.device,
) -> tuple[dict, int, PPOTrainer, list[dict]]:
    """Run PPO for the claimed job.

    Returns (result dict, games_played, trainer instance so the caller can save
    the trained state — weights/optimizer/metrics).
    """
    hp = job["hyperparams"] or {}
    seed_int = _seed_to_int(job["seed"])

    # Build config: start from defaults, overlay any hyperparams the caller provided.
    cfg_kwargs = {k: v for k, v in hp.items() if k in PPOConfig.__dataclass_fields__}
    cfg = PPOConfig(**cfg_kwargs)

    # Build agent + trainer. Trainer owns its own vec env.
    net = build_net_for_model(job["model_id"], model_meta["obs_size"], model_meta["num_actions"])

    # Continuation: download the parent's weights/optimizer/obs_norm and
    # load them into the freshly-built net before training. Artifacts live
    # in public Storage buckets, so we just HTTP-GET them.
    parent_state = None
    if job.get("parent") is not None:
        parent_state = _download_parent_state(job["parent"])
        if parent_state["weights"] is not None:
            net.load_state_dict(parent_state["weights"])
            print(f"[worker] loaded parent weights "
                  f"(params={sum(p.numel() for p in net.parameters()):,})")

    agent = PPOAgent(net, device=device)
    trainer = PPOTrainer(agent, cfg, seed=seed_int)

    if parent_state is not None:
        if parent_state["optimizer"] is not None:
            trainer.optimizer.load_state_dict(parent_state["optimizer"])
            print("[worker] loaded parent optimizer state")
        if parent_state["obs_norm"] is not None and trainer.obs_norm is not None:
            trainer.obs_norm.load_state_dict(parent_state["obs_norm"])
            print("[worker] loaded parent obs_norm state")

    # Budget loop: update until we hit budget_ms. Keep full metrics history
    # for the log artifact; keep overall win-rate stats for `result`.
    deadline = time.time() + job["budget_ms"] / 1000.0
    metrics_history: list[dict] = []
    total_eps = 0
    total_wins = 0

    try:
        while time.time() < deadline:
            m = trainer.update()
            metrics_history.append(m)
            if "episodes_completed" in m:
                eps_count = m["episodes_completed"]
                total_eps += eps_count
                total_wins += int(round(eps_count * m.get("win_rate", 0.0)))
    finally:
        # AsyncVectorEnv subprocesses need explicit cleanup; don't leak them
        # across claimed runs in the polling worker.
        pass  # trainer.close() happens in the caller after save_and_upload

    overall_win_rate = total_wins / total_eps if total_eps else 0.0
    result = {
        "rate":                 overall_win_rate,
        "updates":              len(metrics_history),
        "training_episodes":    total_eps,
        "training_wins":        total_wins,
        "final_metrics":        metrics_history[-1] if metrics_history else {},
        "config":               asdict(cfg),
        "device":               str(device),
    }
    return result, total_eps, trainer, metrics_history


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
# Save + upload artifacts
# ---------------------------------------------------------------------------

def save_and_upload(
    run_id,
    trainer: PPOTrainer,
    metrics_history: list[dict],
    result: dict,
) -> dict[str, str | None]:
    """Write weights/optimizer/metrics locally, then upload to Storage.

    Returns dict with bucket/key paths ready for the runs.*_url columns.
    """
    import tempfile

    from workers import storage

    run_id_str = str(run_id)
    urls = {
        "weights_url":   None,
        "optimizer_url": None,
        "obs_norm_url":  None,
        "log_url":       None,
    }

    with tempfile.TemporaryDirectory(prefix=f"mw2-run-{run_id_str}-") as tmp:
        tmp_path = Path(tmp)
        weights_file   = tmp_path / "weights.pt"
        optimizer_file = tmp_path / "optimizer.pt"
        obs_norm_file  = tmp_path / "obs_norm.pt"
        log_file       = tmp_path / "metrics.json"

        torch.save(trainer.agent.net.state_dict(), weights_file)
        torch.save(trainer.optimizer.state_dict(), optimizer_file)
        if trainer.obs_norm is not None:
            trainer.obs_norm.save(obs_norm_file)
        log_file.write_text(json.dumps({
            "metrics": metrics_history,
            "result":  result,
        }))

        urls["weights_url"]   = storage.upload("models", f"{run_id_str}/weights.pt",   weights_file)
        urls["optimizer_url"] = storage.upload("models", f"{run_id_str}/optimizer.pt", optimizer_file)
        if obs_norm_file.exists():
            urls["obs_norm_url"] = storage.upload("models", f"{run_id_str}/obs_norm.pt", obs_norm_file)
        urls["log_url"]       = storage.upload("logs",   f"{run_id_str}/metrics.json", log_file,
                                               content_type="application/json")

    return urls


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
            result, games, trainer, metrics_history = run_training(
                claimed, model_meta, device
            )
            wall_ms = int((time.time() - t0) * 1000)

            try:
                urls = save_and_upload(claimed["id"], trainer, metrics_history, result)
            finally:
                trainer.close()  # tear down AsyncVectorEnv subprocesses

            with connect() as conn:
                mark_done(
                    conn, claimed["id"], result, games, wall_ms,
                    weights_url=urls["weights_url"],
                    optimizer_url=urls["optimizer_url"],
                    obs_norm_url=urls["obs_norm_url"],
                    log_url=urls["log_url"],
                )
            print(f"[worker] done id={claimed['id']} games={games} "
                  f"wall={wall_ms}ms rate={result['rate']:.3f} "
                  f"artifacts: {urls['weights_url']}, {urls['log_url']}")

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
