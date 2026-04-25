"""
Run a JSON-defined batch of training experiments sequentially.

Designed for the supervisor loop: caller writes a list of experiment specs to
`experiments/queue.json`, this script picks the next not-yet-run one,
trains it, writes metrics to `experiments/<run_id>/metrics.json`, and
moves on to the next.

Each experiment is one fresh PPOTrainer; weights are NOT carried across
runs. The point is to compare configs, not chain training.

Usage:
    python scripts/run_experiment_batch.py
    python scripts/run_experiment_batch.py --queue path/to/queue.json
    python scripts/run_experiment_batch.py --max-runs 1   # one then exit

Queue file format:
    {
      "runs": [
        {
          "id": "exp_001_k1_baseline",
          "minutes": 30,
          "config": {
            "n_envs": 1024, "rollout_steps": 64, "fused_rollout": true,
            "action_repeat": 1, "level_name": "random_8_16",
            "lr": 0.0003, "vec_mode": "sync"
          },
          "seed": 42,
          "notes": "K=1 baseline"
        },
        ...
      ]
    }

Each finished run produces `experiments/<id>/metrics.json` with:
    {
      "id": ..., "config": ..., "seed": ..., "wall_s": ..., "updates": ...,
      "env_ticks": ..., "episodes": ..., "win_rate_final": ..., "metrics": [...]
    }
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

import torch  # noqa: E402  — env vars must be set before torch imports CUDA

EXPERIMENTS_DIR = ROOT / "experiments"


def _resolve_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_one(spec: dict) -> dict:
    """Run a single experiment per spec. Returns the result dict."""
    from training.agent import PPOAgent
    from training.net import ActorCritic
    from training.trainer import PPOConfig, PPOTrainer

    run_id = spec["id"]
    minutes = float(spec["minutes"])
    cfg_kwargs = dict(spec.get("config", {}))
    seed = int(spec.get("seed", 0))

    out_dir = EXPERIMENTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PPOConfig(**cfg_kwargs)

    device = _resolve_device()
    print(f"\n=== {run_id} === device={device} budget={minutes}min seed={seed}", flush=True)
    print(f"  config: n_envs={cfg.n_envs} rollout={cfg.rollout_steps} "
          f"K={cfg.action_repeat} fused={cfg.fused_rollout} level={cfg.level_name} lr={cfg.lr}",
          flush=True)

    net = ActorCritic()
    agent = PPOAgent(net, device=device)
    trainer = PPOTrainer(agent, cfg, seed=seed)

    metrics_history: list[dict] = []
    total_eps = 0
    total_wins = 0
    total_env_ticks = 0
    deadline = time.time() + minutes * 60.0
    start = time.time()
    update_idx = 0
    last_print = start

    try:
        while time.time() < deadline:
            t0 = time.time()
            m = trainer.update()
            dt = time.time() - t0
            update_idx += 1
            eps_count = m.get("episodes_completed", 0)
            total_eps += eps_count
            total_wins += int(round(eps_count * m.get("win_rate", 0.0)))
            total_env_ticks += cfg.rollout_steps * cfg.n_envs * max(1, cfg.action_repeat)

            row = {
                "update":         update_idx,
                "wall_s":         round(time.time() - start, 2),
                "upd_s":          round(dt, 3),
                "mean_reward":    round(m["mean_reward"], 5),
                "policy_loss":    round(m["policy_loss"], 5),
                "value_loss":     round(m["value_loss"], 5),
                "entropy_loss":   round(m["entropy_loss"], 5),
                "approx_kl":      round(m["approx_kl"], 5),
                "episodes":       eps_count,
                "win_rate":       m.get("win_rate"),
                "env_ticks":      total_env_ticks,
            }
            metrics_history.append(row)

            # Print at most every ~5s.
            now = time.time()
            if now - last_print >= 5.0 or update_idx == 1:
                wr = "—" if m.get("win_rate") is None else f"{m.get('win_rate'):.2f}"
                print(f"  upd {update_idx:>4d}  wall={row['wall_s']:>6.1f}s  "
                      f"r={row['mean_reward']:+.4f}  ent={row['entropy_loss']:+.3f}  "
                      f"eps={eps_count:>4d}  win={wr}  ticks={total_env_ticks:,}",
                      flush=True)
                last_print = now
    except Exception as exc:
        # Don't crash the batch — record the failure and move on.
        result = {
            "id":        run_id,
            "config":    cfg_kwargs,
            "seed":      seed,
            "wall_s":    round(time.time() - start, 2),
            "updates":   update_idx,
            "env_ticks": total_env_ticks,
            "episodes":  total_eps,
            "wins":      total_wins,
            "win_rate_final": (
                metrics_history[-1].get("win_rate") if metrics_history else None
            ),
            "error":     f"{type(exc).__name__}: {exc}",
            "metrics":   metrics_history,
        }
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)
        try:
            trainer.close()
        except Exception:
            pass
        return result

    wall = time.time() - start
    win_rate_final = metrics_history[-1].get("win_rate") if metrics_history else None

    # Save final weights + obs_norm so other runs can use this checkpoint as
    # an opponent. Wrapped in try/except so a save failure doesn't crash the
    # whole batch.
    weights_path = out_dir / "weights.pt"
    obs_norm_path = out_dir / "obs_norm.pt"
    try:
        torch.save(trainer.agent.net.state_dict(), weights_path)
        if trainer.obs_norm is not None:
            trainer.obs_norm.save(obs_norm_path)
        print(f"  saved weights → {weights_path}", flush=True)
    except Exception as exc:
        print(f"  weight save failed: {type(exc).__name__}: {exc}", flush=True)

    result = {
        "id":            run_id,
        "config":        cfg_kwargs,
        "seed":          seed,
        "wall_s":        round(wall, 2),
        "updates":       update_idx,
        "env_ticks":     total_env_ticks,
        "episodes":      total_eps,
        "wins":          total_wins,
        "win_rate_final": win_rate_final,
        "ticks_per_sec": round(total_env_ticks / wall, 0) if wall else 0,
        "eps_per_sec":   round(total_eps / wall, 2) if wall else 0,
        "notes":         spec.get("notes", ""),
        "weights_path":  str(weights_path) if weights_path.exists() else None,
        "obs_norm_path": str(obs_norm_path) if obs_norm_path.exists() else None,
        "metrics":       metrics_history,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  DONE {run_id}: {update_idx} upd in {wall:.0f}s  "
          f"win_rate_final={win_rate_final}  ticks/s={result['ticks_per_sec']:,.0f}",
          flush=True)
    try:
        trainer.close()
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(EXPERIMENTS_DIR / "queue.json"))
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Stop after this many runs (0 = run all).")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"queue file not found: {queue_path}", file=sys.stderr)
        return 1

    with queue_path.open() as f:
        queue = json.load(f)
    runs = queue.get("runs", [])
    if not runs:
        print("queue is empty — nothing to run.")
        return 0

    completed = 0
    for spec in runs:
        run_id = spec["id"]
        marker = EXPERIMENTS_DIR / run_id / "metrics.json"
        if marker.exists():
            print(f"skipping {run_id} — metrics.json already exists.", flush=True)
            continue
        _run_one(spec)
        completed += 1
        if args.max_runs and completed >= args.max_runs:
            break

    print(f"\nbatch complete: ran {completed} of {len(runs)} queued runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
