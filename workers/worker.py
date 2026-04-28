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
import io
import json
import os
import socket
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# PyTorch and JAX share the RTX 3070's 8 GiB of VRAM. JAX pre-allocates 75%
# of device memory on first import, which causes PyTorch OOMs in self-play.
# Cap JAX to 40% (3.2 GiB) BEFORE any JAX import happens elsewhere in the
# process. See JAX_PORT_PLAN §3.6. Only applies when SIM_BACKEND=jax is
# requested; numpy runs fine without the cap, but setting the env var early
# is harmless.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

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

    # (obs_dim, action_dim, builder) — builder takes no args; produces the net.
    def _mk(body_dim: int):
        return lambda: ActorCritic(body_dim=body_dim)
    KNOWN = {
        # v9.0-enc-full was the interim commit-1 model (full encoder + old
        # flat 4097 head). Code has since moved to chained heads — running
        # that model against this code will fail at inference, which is
        # correct: the net topology doesn't match.
        "v9.0-enc-full": (OBS_DIM, ACTION_SPACE_SIZE, _mk(128)),
        # v9.0-full: full encoder + chained source/type/target heads
        # (ARCHITECTURE §9.4). Default 128-wide body (production baseline).
        "v9.0-full":     (OBS_DIM, ACTION_SPACE_SIZE, _mk(128)),
        # Capacity-sweep variants (same architecture, wider trunk).
        "v9.0-256":      (OBS_DIM, ACTION_SPACE_SIZE, _mk(256)),
        "v9.0-512":      (OBS_DIM, ACTION_SPACE_SIZE, _mk(512)),
        "v9.0-1024":     (OBS_DIM, ACTION_SPACE_SIZE, _mk(1024)),
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


# ---------------------------------------------------------------------------
# Auto-admission: on new run completion, queue a small set of ranking
# matches so its Elo emerges without manual intervention.
# ---------------------------------------------------------------------------

ADMISSION_TOP_K           = 10  # play vs current top-K Elo runs
ADMISSION_GAMES_PER_MATCH = 12  # ±12% SE on 30% win-rate; 4-5 min total drain
                                  # (was 30, dropped 2026-04-24 to cap drain time
                                  # at larger model sizes; ranking signal is
                                  # still strong enough at 12 games × 10 opps).
# Baseline auto-admission dropped 2026-04-24: trained models saturate vs
# random_legal (85-100%), so the match was signal-free. Random pseudo-run
# stays in the DB for historical queries; nothing new gets queued against it.
BASELINE_RUN_ID = "00000000-0000-0000-0000-000000000001"
ADMISSION_LEVEL = "random_8_12"


def _current_top_elo_runs(conn, k: int) -> list[str]:
    """Compute current Elo client-side from `games` and return top-k run IDs.

    Skips the baseline pseudo-run — we always play it separately so it
    doesn't count as a "top-K" opponent.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT g.player_1_run_id, g.player_2_run_id, g.winner
              FROM games g
              JOIN matches m ON g.match_id = m.id
             WHERE m.project = %s AND m.status = 'done'
             ORDER BY m.created_at, g.game_index
        """, (PROJECT,))
        games = cur.fetchall()

    K, INIT = 32, 1200
    elo: dict[str, float] = {}
    for p1, p2, winner in games:
        p1s, p2s = str(p1), str(p2)
        winner_s = str(winner) if winner else None
        ra = elo.get(p1s, INIT); rb = elo.get(p2s, INIT)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = 1.0 if winner_s == p1s else (0.0 if winner_s == p2s else 0.5)
        elo[p1s] = ra + K * (sa - ea)
        elo[p2s] = rb + K * ((1 - sa) - (1 - ea))

    ranked = [rid for rid, _ in sorted(elo.items(), key=lambda x: -x[1])
              if rid != BASELINE_RUN_ID]
    return ranked[:k]


def _queue_admission_matches(conn, new_run_id, level_name: str = ADMISSION_LEVEL):
    """Insert matches: new_run vs current top-K Elo runs, N games each.

    Baseline (random_legal) matches were dropped because all trained models
    now saturate vs random; the match was pure compute with no signal.
    """
    top = _current_top_elo_runs(conn, ADMISSION_TOP_K)
    # Don't self-match; if the new run is already in top-K (possible after
    # chain continuation), skip itself.
    top = [rid for rid in top if rid != str(new_run_id)]

    with conn.cursor() as cur:
        for opp_id in top:
            cur.execute("""
                INSERT INTO matches (project, description, model_a_run_id, model_b_run_id,
                                     simulator_id, games_planned, status, summary)
                VALUES (%s, 'auto-admission', %s, %s,
                        (SELECT simulator_id FROM runs WHERE id = %s),
                        %s, 'queued', %s::jsonb)
            """, (PROJECT, new_run_id, opp_id, new_run_id,
                  ADMISSION_GAMES_PER_MATCH,
                  json.dumps({"level_name": level_name})))
    conn.commit()
    print(f"[worker] auto-admission: queued {len(top)} top-{ADMISSION_TOP_K} matches "
          f"× {ADMISSION_GAMES_PER_MATCH} games for run {new_run_id}")


# Auto-rate config (CURRICULUM_PLAN.md §3.3 — fast Elo seeding so runs land
# on the dashboard with a real score within minutes of finishing).
AUTO_RATE_GAMES   = 64    # per match — same as cron-agent's Elo review
AUTO_RATE_LEVEL   = "random_8_16"
AUTO_RATE_K       = 32    # standard Elo K
AUTO_RATE_OPPONENTS_VS_BASELINE = 3   # 3 matches vs random_legal

def _auto_rate_run(run_id: str, label: str) -> None:
    """Run a quick Elo benchmarking pass for a freshly-finished run.

    Runs N matches vs random_legal (the stable absolute baseline anchored at
    1000), each updating runs.elo_score via the standard tournament helpers.
    Reuses the same code path as scripts/tournament.py --update-elo so Elo
    updates are consistent across all paths (worker / cron / manual).

    Failures here are non-fatal (the cron's Elo review will catch up later).
    """
    # Lazy import — keeps the worker startup time low when the function isn't
    # called (e.g. failed runs).
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    tournament = importlib.import_module("scripts.tournament")

    print(f"[worker] auto-rate: {label} ({AUTO_RATE_OPPONENTS_VS_BASELINE} matches "
          f"vs random_legal on {AUTO_RATE_LEVEL})", flush=True)
    for i in range(AUTO_RATE_OPPONENTS_VS_BASELINE):
        res = tournament.run_match(
            p1=run_id, p2="random_legal",
            games=AUTO_RATE_GAMES, level=AUTO_RATE_LEVEL,
            max_ticks=200, seed=i, verbose=False,
        )
        with connect() as c:
            new_elo, _ = tournament.update_elo_from_match(
                c, p1_run_id=run_id, p2_run_id=None,
                result=res, k=AUTO_RATE_K,
            )
        rate = res["p1_wins"] / max(res["total"], 1)
        print(f"[worker] auto-rate [{i+1}/{AUTO_RATE_OPPONENTS_VS_BASELINE}] "
              f"vs random_legal: rate={rate:.3f}  Elo→{new_elo:.0f}", flush=True)


# ---------------------------------------------------------------------------
# Match claim + finalize (eval jobs)
# ---------------------------------------------------------------------------

def _worker_mode(conn, machine: str) -> tuple[bool, bool]:
    """Dashboard-controlled flags for this machine. Returns (paused, matches_only).

    - paused=True → skip everything, just sleep.
    - matches_only=True → skip training runs, still claim eval matches.
    - both False → normal full operation.

    Upserts the row if missing so a new machine defaults to "full on" but
    can be flipped from the dashboard.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO worker_state (machine, paused) VALUES (%s, false) "
            "ON CONFLICT (machine) DO NOTHING",
            (machine,),
        )
        cur.execute(
            "SELECT paused, matches_only FROM worker_state WHERE machine = %s",
            (machine,),
        )
        row = cur.fetchone()
    conn.commit()
    if not row:
        return False, False
    return bool(row[0]), bool(row[1])


def claim_one_match(conn, machine: str | None = None):
    """Atomically claim the oldest queued match row. Returns dict or None.

    `machine`: when set, only claims matches that EITHER have no
    `summary.target_machine` (open to any worker) OR have it set to
    exactly this hostname. This keeps interactive-play matches pinned to
    Mac so the dashboard's Play button doesn't steal GPU time from
    PaulLinux's training loop.

    Priority: matches with an explicit `target_machine` (= user-initiated,
    e.g. dashboard Play) jump the queue ahead of background admission
    matches. Tie-break by created_at within each priority class.
    """
    with conn.cursor() as cur:
        if machine is not None:
            cur.execute("""
                UPDATE matches SET status='running'
                 WHERE id IN (
                   SELECT id FROM matches
                    WHERE project=%s AND status='queued'
                      AND (summary->>'target_machine' IS NULL
                           OR summary->>'target_machine' = %s)
                    ORDER BY (summary->>'target_machine' IS NOT NULL) DESC,
                             created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                 )
                 RETURNING id, model_a_run_id, model_b_run_id, games_planned,
                           simulator_id, description, summary::text
            """, (PROJECT, machine))
        else:
            cur.execute("""
                UPDATE matches SET status='running'
                 WHERE id IN (
                   SELECT id FROM matches
                    WHERE project=%s AND status='queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                 )
                 RETURNING id, model_a_run_id, model_b_run_id, games_planned,
                           simulator_id, description, summary::text
            """, (PROJECT,))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    id_, a, b, games_planned, sim_id, description, summary_text = row
    summary = json.loads(summary_text) if summary_text else {}
    return {
        "id":           id_,
        "run_a":        a,
        "run_b":        b,
        "games_planned": games_planned,
        "sim_id":       sim_id,
        "description":  description,
        "level_name":   summary.get("level_name", "crossroads_6"),
    }


def fetch_run_artifacts(conn, run_id):
    """Grab weights_url + obs_norm_url for a completed training run.

    The baseline pseudo-run (BASELINE_RUN_ID) has no weights — returning
    (None, None) is the signal match_runner uses to substitute random_legal.
    """
    if str(run_id) == BASELINE_RUN_ID:
        return {"weights_url": None, "obs_norm_url": None}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT weights_url, obs_norm_url, status FROM runs WHERE id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    if not row:
        raise ValueError(f"run {run_id!r} not found")
    w, n, status = row
    if status != "done" or not w:
        raise ValueError(f"run {run_id!r} status={status!r} weights_url={w!r} — can't eval")
    return {"weights_url": w, "obs_norm_url": n}


def mark_match_done(conn, match_id, summary: dict):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE matches SET status='done', summary=%s::jsonb, finished_at=now()
             WHERE id = %s
        """, (json.dumps(summary), match_id))
    conn.commit()


def mark_match_failed(conn, match_id, error: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE matches SET status='failed',
                               summary=jsonb_set(COALESCE(summary,'{}'::jsonb), '{error}', to_jsonb(%s::text)),
                               finished_at=now()
             WHERE id = %s
        """, (error[:4000], match_id))
    conn.commit()


def insert_games(conn, match_id, games: list[dict]):
    with conn.cursor() as cur:
        for g in games:
            cur.execute("""
                INSERT INTO games (
                    id, match_id, game_index, seed, map_name,
                    player_1_run_id, player_2_run_id, winner,
                    duration_ms, stats, actions_url
                )
                VALUES (
                    COALESCE(%s::uuid, gen_random_uuid()),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                )
            """, (
                g.get("id"), match_id, g["game_index"], g["seed"], g["map_name"],
                g["player_1_run_id"], g["player_2_run_id"], g["winner"],
                g["duration_ms"], json.dumps(g["stats"]), g.get("actions_url"),
            ))
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


def _resolve_opponent_kwargs(kw: dict) -> dict:
    """Translate cloud-friendly opponent specs into local file paths.

    Accepts (in order of preference):
      1. {"opponent_run_id": "<uuid>"}  — looks up runs.weights_url/obs_norm_url
         and downloads them.
      2. {"weights_url": ..., "obs_norm_url": ...}  — Storage relative paths
         (the same shape stored in `runs.weights_url`).
      3. {"weights_path": ..., "obs_norm_path": ...}  — already-local paths;
         used as-is.

    Always returns a dict with `weights_path` (and optionally `obs_norm_path`)
    pointing at local filesystem paths suitable for `make_neural_opponent`.
    Other keys (e.g. "device") pass through unchanged.
    """
    import tempfile

    out = {k: v for k, v in kw.items() if k not in (
        "opponent_run_id", "weights_url", "obs_norm_url",
        "weights_path", "obs_norm_path",
    )}

    # Path 1: by run id.
    if "opponent_run_id" in kw:
        opp_id = str(kw["opponent_run_id"])
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT weights_url, obs_norm_url FROM runs WHERE id = %s",
                    (opp_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"opponent_run_id={opp_id} not found in runs")
        w_url, n_url = row
        if not w_url:
            raise RuntimeError(f"opponent_run_id={opp_id} has no weights_url")
        kw = {**kw, "weights_url": w_url, "obs_norm_url": n_url}

    # Path 2: by storage URL → download to a temp file.
    if "weights_url" in kw and kw["weights_url"]:
        out_dir = Path(tempfile.mkdtemp(prefix="mw2-opp-"))
        w_path = out_dir / "weights.pt"
        urllib.request.urlretrieve(_public_url(kw["weights_url"]), w_path)
        out["weights_path"] = str(w_path)
        if kw.get("obs_norm_url"):
            n_path = out_dir / "obs_norm.pt"
            urllib.request.urlretrieve(_public_url(kw["obs_norm_url"]), n_path)
            out["obs_norm_path"] = str(n_path)
        return out

    # Path 3: already-local paths — pass through.
    if "weights_path" in kw:
        out["weights_path"] = kw["weights_path"]
        if "obs_norm_path" in kw:
            out["obs_norm_path"] = kw["obs_norm_path"]
        return out

    raise RuntimeError(
        "opponent_kwargs needs one of: opponent_run_id, weights_url, weights_path"
    )


def _download_leaderboard_opponents(run_id, top_k: int) -> list[tuple]:
    """LEGACY top-K Elo opponent downloader. Kept as a fallback for runs that
    request it explicitly (cfg.leaderboard_source='elo'). Champions-archive
    PFSP path is the new default — see _download_pfsp_champions.

    Returns [(weights_path, obs_norm_path|None)] — uniform-sample shape.
    """
    import tempfile

    with connect() as conn:
        top_ids = _current_top_elo_runs(conn, top_k)
        top_ids = [rid for rid in top_ids if rid != str(run_id) and rid != BASELINE_RUN_ID]
        if not top_ids:
            return []
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, weights_url, obs_norm_url FROM runs WHERE id = ANY(%s)",
                (top_ids,),
            )
            rows = cur.fetchall()

    out_dir = Path(tempfile.mkdtemp(prefix="mw2-leaderboard-"))
    results: list[tuple] = []
    for rid, w_url, n_url in rows:
        if not w_url:
            continue
        w_path = out_dir / f"{rid}-weights.pt"
        n_path: Path | None = None
        try:
            urllib.request.urlretrieve(_public_url(w_url), w_path)
            if n_url:
                n_path = out_dir / f"{rid}-obs_norm.pt"
                urllib.request.urlretrieve(_public_url(n_url), n_path)
        except Exception as exc:
            print(f"[worker] leaderboard: skip {rid} — download failed: {exc}")
            continue
        results.append((w_path, n_path))
    return results


def _download_pfsp_champions(run_id) -> list[tuple]:
    """Download every champion in the archive with PFSP sampling weights.

    Returns [(weights_path, obs_norm_path|None, weight)]. `weight` is an
    unnormalised PFSP score in [0, 1]: how informative is this opponent
    based on the most-recent run that bench-eval'd it.

    Heuristic: for each champion, pull the most-recent rated run that has a
    bench_vector entry for it. weight = 1 - |wr - 0.5|. Defaults to 0.5
    (mid-info) if no prior bench data exists yet.

    Returns [] if the champions table is empty — caller falls back to pure
    self-play.
    """
    import tempfile

    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT id, source_run_id, weights_url, obs_norm_url, label
                  FROM champions
                 WHERE source_run_id <> %s
                 ORDER BY archived_at DESC
            """, (str(run_id),))
            rows = cur.fetchall()

    if not rows:
        return []

    # For each champion, find the most-recent bench_vector entry that scored it.
    weights_by_champ: dict[str, float] = {}
    with connect() as c:
        with c.cursor() as cur:
            for champ_id, _, _, _, _ in rows:
                cur.execute("""
                    SELECT (bench_vector ->> %s)::float AS wr
                      FROM runs
                     WHERE elo_status = 'rated'
                       AND bench_vector ? %s
                     ORDER BY finished_at DESC NULLS LAST
                     LIMIT 1
                """, (str(champ_id), str(champ_id)))
                r = cur.fetchone()
                if r and r[0] is not None:
                    wr = float(r[0])
                    weights_by_champ[str(champ_id)] = max(1.0 - abs(wr - 0.5), 1e-3)
                else:
                    weights_by_champ[str(champ_id)] = 0.5  # no data → mid-info default

    out_dir = Path(tempfile.mkdtemp(prefix="mw2-pfsp-"))
    results: list[tuple] = []
    for champ_id, source_rid, w_url, n_url, label in rows:
        if not w_url:
            continue
        w_path = out_dir / f"{str(champ_id)[:8]}-weights.pt"
        n_path: Path | None = None
        try:
            urllib.request.urlretrieve(_public_url(w_url), w_path)
            if n_url:
                n_path = out_dir / f"{str(champ_id)[:8]}-obs_norm.pt"
                urllib.request.urlretrieve(_public_url(n_url), n_path)
        except Exception as exc:
            print(f"[worker] pfsp: skip champion {label} — download failed: {exc}")
            continue
        weight = weights_by_champ[str(champ_id)]
        results.append((w_path, n_path, weight))
        print(f"[worker] pfsp opp '{label[:40]}' weight={weight:.3f}")
    return results


def _public_url(path: str) -> str:
    base = os.environ.get("SUPABASE_URL")
    if not base:
        raise RuntimeError("SUPABASE_URL not set")
    return f"{base}/storage/v1/object/public/{path}"


# ---------------------------------------------------------------------------
# Train one run
# ---------------------------------------------------------------------------

METRICS_UPLOAD_EVERY    = 5    # updates; ~every 15-30s depending on rollout size
SNAPSHOT_INTERVAL_S     = 600  # 10 min between mid-run weight snapshots


def _upload_snapshot(run_id, snap_n: int, trainer) -> str | None:
    """Upload current weights to storage as a mid-run snapshot.

    Returns the storage path, or None on failure (non-fatal).
    path: snapshots/<run_id>/<n>/weights.pt
    Also records the snapshot in the DB (snapshots table if it exists).
    """
    import tempfile
    from workers import storage

    run_id_str = str(run_id)
    storage_path = f"snapshots/{run_id_str}/{snap_n}/weights.pt"
    obs_storage_path = f"snapshots/{run_id_str}/{snap_n}/obs_norm.pt"

    try:
        with tempfile.TemporaryDirectory(prefix="mw2-snap-") as tmp:
            tmp_path = Path(tmp)
            w_file = tmp_path / "weights.pt"
            torch.save(trainer.agent.net.state_dict(), w_file)
            w_url = storage.upload("models", storage_path, w_file)

            n_url = None
            if trainer.obs_norm is not None:
                n_file = tmp_path / "obs_norm.pt"
                trainer.obs_norm.save(n_file)
                n_url = storage.upload("models", obs_storage_path, n_file)

        with connect() as c:
            with c.cursor() as cur:
                cur.execute("""
                    INSERT INTO run_snapshots
                        (run_id, snap_n, weights_url, obs_norm_url)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (run_id_str, snap_n, w_url, n_url))
            c.commit()

        print(f"[worker] snapshot {snap_n} uploaded: {w_url}", flush=True)
        return w_url
    except Exception as exc:
        print(f"[worker] snapshot {snap_n} failed (non-fatal): {exc}", flush=True)
        return None


def _handle_match(conn, match: dict, device: torch.device) -> None:
    """Play a claimed head-to-head match. Writes to games + matches tables."""
    from workers import match_runner

    match_id = match["id"]
    print(f"[worker] claimed match id={match_id} "
          f"A={match['run_a']} B={match['run_b']} "
          f"games={match['games_planned']} level={match['level_name']!r}")

    try:
        art_a = fetch_run_artifacts(conn, match["run_a"])
        art_b = fetch_run_artifacts(conn, match["run_b"])
        state_a = match_runner.download_run_state(art_a["weights_url"], art_a["obs_norm_url"])
        state_b = match_runner.download_run_state(art_b["weights_url"], art_b["obs_norm_url"])

        results = match_runner.run_match(
            run_a_id=match["run_a"],
            run_b_id=match["run_b"],
            state_a=state_a,
            state_b=state_b,
            n_games=match["games_planned"],
            level_name=match["level_name"],
            # Seed the match-level rng from the match id so each match is
            # reproducible given the same inputs. hash() isn't stable
            # across runs, but sum of bytes works.
            seed_base=int(sum(str(match_id).encode())) & 0x7FFFFFFF,
            device=device,
        )
        summary = match_runner.summarize(results, match["run_a"], match["run_b"])
        summary["level_name"] = match["level_name"]

        with connect() as c2:
            insert_games(c2, match_id, results)
            mark_match_done(c2, match_id, summary)

        print(f"[worker] match done id={match_id} "
              f"A={summary['wins_a']} B={summary['wins_b']} draws={summary['draws']} "
              f"rate_a={summary['rate_a']:.2f}")
    except Exception as exc:
        err = f"{exc.__class__.__name__}: {exc}"
        print(f"[worker] match failed id={match_id}: {err}")
        with connect() as c2:
            mark_match_failed(c2, match_id, err)


def _upload_live_metrics(run_id, metrics_history: list[dict], snapshot: dict) -> str:
    """Upload current (partial) metrics.json so the dashboard can chart mid-run.

    Returns the storage path it wrote (same as the final log_url — the final
    upload overwrites this).
    """
    import tempfile

    from workers import storage

    run_id_str = str(run_id)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"metrics": metrics_history, "partial": snapshot}, f)
        path = f.name
    try:
        return storage.upload("logs", f"{run_id_str}/metrics.json", path,
                              content_type="application/json")
    finally:
        Path(path).unlink(missing_ok=True)


def _set_log_url(run_id, log_url: str) -> None:
    """Point runs.log_url at the live metrics file so the dashboard picks it up
    even before the run finishes."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET log_url = %s WHERE id = %s AND log_url IS NULL",
                (log_url, run_id),
            )
        conn.commit()


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

    # Optional per-run backend pin. hyperparams.sim_backend overrides any
    # ambient SIM_BACKEND env var for this run only. sim/backend.py reads the
    # env var each call so setting it here is enough.
    if "sim_backend" in hp:
        os.environ["SIM_BACKEND"] = str(hp["sim_backend"])

    # Build config: start from defaults, overlay any hyperparams the caller provided.
    cfg_kwargs = {k: v for k, v in hp.items() if k in PPOConfig.__dataclass_fields__}
    cfg = PPOConfig(**cfg_kwargs)

    # Optional per-run opponent override. `hyperparams.opponent_name` (str) +
    # `hyperparams.opponent_kwargs` (dict) get passed to PPOTrainer's
    # constructor — used for non-self-play training against a fixed neural
    # opponent (e.g. an earlier checkpoint).
    #
    # `opponent_kwargs` may carry storage paths instead of local paths:
    #   {"opponent_run_id": "<uuid>"}                   — resolve via runs table
    #   {"weights_url": "models/<id>/weights.pt", ...}   — direct Storage path
    #   {"weights_path": "/abs/local/path", ...}         — already local
    # We normalise to local paths via _resolve_opponent_kwargs so the trainer
    # never has to know about Storage.
    opponent_name = hp.get("opponent_name", "random_legal")
    opponent_kwargs_raw = hp.get("opponent_kwargs") or None
    opponent_kwargs = (
        _resolve_opponent_kwargs(opponent_kwargs_raw)
        if opponent_kwargs_raw is not None
        else None
    )

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

    # Fetch cross-lineage opponents. Default path is the champion archive with
    # PFSP weights; legacy top-K-Elo path is available via
    # cfg.leaderboard_source='elo' for back-compat / experiments.
    leaderboard_paths: list[tuple] = []
    if cfg.leaderboard_bias > 0:
        source = getattr(cfg, "leaderboard_source", "pfsp")
        try:
            if source == "elo":
                lb_raw = _download_leaderboard_opponents(
                    run_id=job["id"], top_k=cfg.leaderboard_top_k,
                )
                # Legacy path returns (w, n) — promote to (w, n, 1.0) so the trainer
                # can treat both shapes uniformly.
                leaderboard_paths = [(w, n, 1.0) for (w, n) in lb_raw]
                print(f"[worker] downloaded {len(leaderboard_paths)} top-Elo opponents "
                      f"(source=elo, bias={cfg.leaderboard_bias:.2f})")
            else:
                leaderboard_paths = _download_pfsp_champions(run_id=job["id"])
                print(f"[worker] downloaded {len(leaderboard_paths)} PFSP-weighted "
                      f"champions (source=pfsp, bias={cfg.leaderboard_bias:.2f})")
        except Exception as exc:
            print(f"[worker] opponent download failed ({source}); pure self-play: {exc}")

    agent = PPOAgent(net, device=device)
    trainer = PPOTrainer(
        agent, cfg, seed=seed_int,
        opponent_name=opponent_name,
        opponent_kwargs=opponent_kwargs,
        leaderboard_paths=leaderboard_paths,
    )

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
    training_started_at = time.time()
    metrics_history: list[dict] = []
    total_eps = 0
    total_wins = 0

    live_log_url: str | None = None
    last_snapshot_at = training_started_at
    snap_count = 0

    # Per-run telemetry: CPU%, GPU%, VRAM, RAM. Summarised into result JSON.
    from workers.telemetry import ResourceSampler
    sampler = ResourceSampler(interval_s=2.0)
    sampler.start()

    # Per-update phase timings: how much wall time goes to rollouts vs
    # optimisation. Filled in by trainer.update() if the trainer exposes
    # `last_phase_ms`; otherwise stays empty.
    phase_times_ms: dict[str, float] = {}

    try:
        while time.time() < deadline:
            m = trainer.update()
            metrics_history.append(m)
            if "episodes_completed" in m:
                eps_count = m["episodes_completed"]
                total_eps += eps_count
                total_wins += int(round(eps_count * m.get("win_rate", 0.0)))

            # Periodic live upload so the dashboard can show a mid-run chart.
            # First upload also sets runs.log_url so the dashboard knows where
            # to look; subsequent uploads just overwrite the same path.
            if len(metrics_history) % METRICS_UPLOAD_EVERY == 0:
                snapshot = {
                    "updates":             len(metrics_history),
                    "training_episodes":   total_eps,
                    "training_wins":       total_wins,
                    "rate_so_far":         (total_wins / total_eps) if total_eps else 0.0,
                    "device":              str(device),
                    "wall_ms_so_far":      int((time.time() - (deadline - job["budget_ms"] / 1000.0)) * 1000),
                }
                try:
                    url = _upload_live_metrics(job["id"], metrics_history, snapshot)
                    if live_log_url is None:
                        _set_log_url(job["id"], url)
                        live_log_url = url
                except Exception as upload_exc:
                    # Don't crash training on transient upload failure.
                    print(f"[worker] live-upload failed: {upload_exc}")

            # Periodic mid-run weight snapshot every SNAPSHOT_INTERVAL_S.
            now = time.time()
            if now - last_snapshot_at >= SNAPSHOT_INTERVAL_S:
                snap_count += 1
                _upload_snapshot(job["id"], snap_count, trainer)
                last_snapshot_at = now
    finally:
        # AsyncVectorEnv subprocesses need explicit cleanup; don't leak them
        # across claimed runs in the polling worker.
        pass  # trainer.close() happens in the caller after save_and_upload

    training_wall_s = max(time.time() - training_started_at, 1e-3)
    resource_usage = sampler.stop()

    # Pull the trainer's in-training sim-phase breakdown if available.
    sim_phase_breakdown = None
    try:
        sim_phase_breakdown = getattr(trainer, "sim_phase_breakdown", None)
        if callable(sim_phase_breakdown):
            sim_phase_breakdown = sim_phase_breakdown()
    except Exception:
        sim_phase_breakdown = None

    param_count = sum(p.numel() for p in net.parameters())
    trunk_width = getattr(net, "body_width", None) or getattr(cfg, "body_width", None)

    from sim.backend import get_backend_name
    overall_win_rate = total_wins / total_eps if total_eps else 0.0
    result = {
        "rate":                 overall_win_rate,
        "updates":              len(metrics_history),
        "training_episodes":    total_eps,
        "training_wins":        total_wins,
        "final_metrics":        metrics_history[-1] if metrics_history else {},
        "config":               asdict(cfg),
        "device":               str(device),
        "sim_backend":          get_backend_name(),
        "param_count":          int(param_count),
        "trunk_width":          trunk_width,
        "games_per_sec":        round(total_eps / training_wall_s, 2),
        "steps_per_sec":        round(len(metrics_history) * cfg.n_envs * cfg.rollout_steps /
                                       training_wall_s, 1) if cfg.rollout_steps else 0,
        "training_wall_s":      round(training_wall_s, 1),
        "resource_usage":       resource_usage,
        "sim_phase_breakdown":  sim_phase_breakdown,
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
    last_mode: str | None = None  # "paused" | "matches_only" | "full" — log only on change
    while True:
        claimed = None
        try:
            with connect() as conn:
                paused, matches_only = _worker_mode(conn, args.machine)
                if paused:
                    mode = "paused"
                elif matches_only:
                    mode = "matches_only"
                else:
                    mode = "full"
                if mode != last_mode:
                    print(f"[worker] mode={mode} for machine={args.machine}")
                    last_mode = mode

                if paused:
                    time.sleep(args.poll_interval)
                    continue

                # 1) Try training runs first (longer, higher priority) — unless
                # this machine is in matches_only mode (dashboard-controlled).
                job = None if matches_only else claim_one(conn, args.machine)
                if job is None:
                    # 2) Fall back to queued eval matches. Filter by
                    # `summary.target_machine` so interactive-play matches
                    # routed to Mac stay on Mac.
                    match = claim_one_match(conn, machine=args.machine)
                    if match is not None:
                        idle_streak = 0
                        _handle_match(conn, match, device)
                        continue

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
                # Auto-admission: queue matches so the new run's Elo emerges
                # automatically. Failure is non-fatal (admission is a nicety,
                # not part of the training contract).
                try:
                    _queue_admission_matches(conn, claimed["id"])
                except Exception as admit_exc:
                    print(f"[worker] auto-admission failed for {claimed['id']}: {admit_exc}")

            # Archive sweep + PFSP weight update. See workers/bench_eval.py.
            try:
                from workers.bench_eval import run_bench_eval
                run_bench_eval(claimed["id"], claimed["label"])
            except Exception as rate_exc:
                print(f"[worker] bench-eval failed for {claimed['id']}: {rate_exc}")

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
