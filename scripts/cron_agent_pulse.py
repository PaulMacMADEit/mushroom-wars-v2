"""
Cron-fired training scheduler.

Runs on a 3-hour cadence (via systemd timer on PaulLinux). Each fire:

  1. Reads recent runs' results from Supabase (last `--review-window-hours`).
  2. Cancels any queued backlog (status='queued') so we always queue the
     next batch fresh — runs that are 'running' are NOT touched.
  3. Decides curriculum phase from the kv table (CURRICULUM_PLAN.md §3.4).
     Graduation P1 → P2 happens automatically once the Elo champion's
     win-rate vs random_legal hits ≥ 0.95 over a 100-game eval (run lazily
     once per cron fire when in P1 with a candidate champion).
  4. Decides next batch using the active phase's opponent_mix.
  5. Pushes runs into Supabase (`status='queued'`).

Curriculum phases (CURRICULUM_PLAN.md §3.2):
  - phase1_close: random_close_4_6 / random_close_6_10, K=4, opponent =
    random_legal only. Goal: hit ≥ 0.95 win rate vs random_legal.
  - phase2_wild:  full random_*_* mix, K=2, opponent_mix = 80% self-play
    Elo champion / 15% leaderboard top-3 / 5% random_legal floor.

Strength (CURRICULUM_PLAN.md §3.3):
  Picked by `runs.elo_score` (with `elo_n_matches >= 3`). Prior to this,
  cron used `updates × win_rate` which never refreshed the champion.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect
from cli.loop_config import load_levels


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

# Default training cfg. Each phase overrides level/opponent/K/reward_v13.
DEFAULT_CFG = {
    "n_envs": 1024,
    "rollout_steps": 64,
    "fused_rollout": True,
    "vec_mode": "sync",
    "sim_backend": "jax",
}

DEFAULT_MODEL_ID = "v9.0-1024"
DEFAULT_SIM_ID   = "sim-v1.3"

# Curriculum (CURRICULUM_PLAN.md §3.2, revised post-b5).
#
# Original phase1_close was dropped after b5 evidence (2026-04-26 21:00 PDT):
# building-count, NOT map size, is the binding constraint on transfer. A
# random_close_6_10 specialist hit 91% on close maps but only 5% on
# random_16_24. So phase1 now trains directly on the FULL distribution and
# graduation requires generalist skill (≥80% on each of 4_8 / 8_16 / 16_24).
# Level config (level_name + level_mix) is read from configs/training_levels.yaml
# at queue time via cli.loop_config.load_levels(). Both cron and karp share that
# file so the two schedulers always train on the same distribution.
CURRICULUM = {
    "phase1_full_mix": {
        # K = action_repeat. opponent_mix drives _pick_opponent_for_run.
        "K":            4,
        "opponent_mix": {"random_legal": 1.0},
        "reward_v13":   True,
        "gamma":        0.97,
    },
    "phase2_selfplay": {
        # Opponent flips to Elo champion; level config still comes from the
        # canonical YAML.
        "K":            2,
        "opponent_mix": {"self_play_elo_champ": 0.80,
                         "leaderboard_top3":    0.15,
                         "random_legal":        0.05},
        "reward_v13":   True,
        "gamma":        0.97,
    },
}

# Graduation: champion must clear THIS threshold on EACH check level vs
# random_legal. Multi-level gate replaces the single-level 0.95 of the
# original plan — generalist skill is what we need, not point-on-curve.
GRADUATION_THRESHOLD = 0.80
GRADUATION_LEVELS    = ("random_4_8", "random_8_16", "random_16_24")
GRADUATION_GAMES     = 100
ELO_MIN_MATCHES      = 3      # need at least this many head-to-heads to pick champion
ELO_REVIEW_PER_FIRE  = 5      # max NEW unrated runs to score per cron fire (cost cap)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_recent_runs(conn, since: datetime) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, label, hyperparams::text, result::text,
                   weights_url, obs_norm_url,
                   launch_at, started_at, finished_at,
                   elo_score, elo_n_matches
              FROM runs
             WHERE launch_at >= %s
               AND project = %s
             ORDER BY launch_at DESC
            """,
            (int(since.timestamp() * 1000), PROJECT),
        )
        rows = cur.fetchall()
    out = []
    for row in rows:
        out.append({
            "id":            str(row[0]),
            "status":        row[1],
            "label":         row[2],
            "hyperparams":   json.loads(row[3]) if row[3] else {},
            "result":        json.loads(row[4]) if row[4] else None,
            "weights_url":   row[5],
            "obs_norm_url":  row[6],
            "launch_at":     row[7],
            "started_at":    row[8],
            "finished_at":   row[9],
            "elo_score":     float(row[10]) if row[10] is not None else 1000.0,
            "elo_n_matches": int(row[11])   if row[11] is not None else 0,
        })
    return out


def _cancel_queued_backlog(conn, dry_run: bool) -> int:
    with conn.cursor() as cur:
        if dry_run:
            cur.execute(
                "SELECT COUNT(*) FROM runs WHERE status='queued' AND project=%s",
                (PROJECT,),
            )
            n = cur.fetchone()[0]
            print(f"[dry-run] would discard {n} queued runs")
            return n
        cur.execute(
            """
            UPDATE runs
               SET status = 'discarded', finished_at = NOW()
             WHERE status = 'queued' AND project = %s
            RETURNING id
            """,
            (PROJECT,),
        )
        ids = cur.fetchall()
    conn.commit()
    print(f"  discarded {len(ids)} queued runs from previous batch")
    return len(ids)


def _kv_get(conn, key: str, default: str | None = None) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM kv WHERE key = %s", (key,))
        row = cur.fetchone()
    if row is None:
        return default
    return str(row[0])


def _kv_set(conn, key: str, value: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kv (key, value, updated_at) VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, value),
        )
    conn.commit()


def _summarise_recent_runs(runs: list[dict]) -> dict:
    done    = [r for r in runs if r["status"] == "done"]
    failed  = [r for r in runs if r["status"] == "failed"]
    running = [r for r in runs if r["status"] == "running"]
    queued  = [r for r in runs if r["status"] == "queued"]

    win_rates = [
        (r["result"].get("rate") or 0)
        for r in done
        if r.get("result")
    ]
    return {
        "total":    len(runs),
        "done":     len(done),
        "failed":   len(failed),
        "running":  len(running),
        "queued":   len(queued),
        "max_win":  max(win_rates) if win_rates else None,
        "mean_win": (sum(win_rates) / len(win_rates)) if win_rates else None,
    }


# ---------------------------------------------------------------------------
# Elo champion selection + bootstrap
# ---------------------------------------------------------------------------

def _read_top_elo_runs(conn, limit: int = 5) -> list[dict]:
    """Pull the top-`limit` done runs by elo_score (only those with enough matches)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, elo_score, elo_n_matches, weights_url, obs_norm_url,
                   result::text
              FROM runs
             WHERE project=%s AND status='done' AND weights_url IS NOT NULL
               AND elo_status = 'rated'
               AND elo_n_matches >= %s
             ORDER BY elo_score DESC
             LIMIT %s
            """,
            (PROJECT, ELO_MIN_MATCHES, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id":            str(r[0]),
            "label":         r[1],
            "elo_score":     float(r[2]),
            "elo_n_matches": int(r[3]),
            "weights_url":   r[4],
            "obs_norm_url":  r[5],
            "result":        json.loads(r[6]) if r[6] else None,
        }
        for r in rows
    ]


def _read_unrated_done_runs(conn, limit: int) -> list[dict]:
    """Done runs that have weights but haven't been Elo-rated yet (or below
    minimum match count). Capped at `limit` to bound cron-fire cost."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label
              FROM runs
             WHERE project=%s AND status='done' AND weights_url IS NOT NULL
               AND elo_status = 'unrated'
               AND (elo_n_matches IS NULL OR elo_n_matches < %s)
             ORDER BY finished_at DESC NULLS LAST
             LIMIT %s
            """,
            (PROJECT, ELO_MIN_MATCHES, limit),
        )
        return [{"id": str(r[0]), "label": r[1]} for r in cur.fetchall()]


def _elo_review_pass(conn, dry_run: bool, max_runs: int = ELO_REVIEW_PER_FIRE) -> None:
    """For up to `max_runs` recently-finished but unrated runs, run a quick
    head-to-head tournament vs random_legal (and vs current top Elo if any)
    to assign them an Elo score. Uses `scripts/tournament.py --update-elo`.

    Runs SUBPROCESSES because tournament.py needs JAX initialised — running
    inline in the cron python process leaks GPU state across cron fires.
    """
    unrated = _read_unrated_done_runs(conn, max_runs)
    if not unrated:
        print("  Elo review: no unrated done runs")
        return

    # Pick opponents: top-3 Elo runs (if available) plus random_legal.
    top = _read_top_elo_runs(conn, limit=3)
    opponents = [r["id"] for r in top] + ["random_legal"]
    opponents = list(dict.fromkeys(opponents))  # dedupe, preserve order

    print(f"  Elo review: {len(unrated)} unrated, opponents={[o[:8] if len(o)==36 else o for o in opponents]}")
    if dry_run:
        print("  [dry-run] would run elo review tournaments; skipping")
        return

    for u in unrated:
        for opp in opponents[:2]:  # cap matches/run to avoid runaway
            cmd = [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "tournament.py"),
                "--p1", u["id"],
                "--p2", opp,
                "--games", "64",
                "--level", "random_8_16",
                "--update-elo",
            ]
            print(f"    {u['label']}  vs  {opp[:8] if len(opp)==36 else opp} ...", flush=True)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if proc.returncode != 0:
                    print(f"      (tournament failed; rc={proc.returncode})")
                    print(proc.stderr[-500:])
            except subprocess.TimeoutExpired:
                print("      (tournament timeout)")


# ---------------------------------------------------------------------------
# Curriculum decision
# ---------------------------------------------------------------------------

def _check_graduation_to_phase2(
    conn,
    champion: dict | None,
    dry_run: bool,
) -> bool:
    """If we have an Elo champion, run a 100-game eval vs random_legal at each
    graduation level. Returns True only if rate ≥ GRADUATION_THRESHOLD on EVERY
    level (multi-level generalist gate, post-b5 design)."""
    if champion is None:
        print("  graduation check: no Elo champion yet — staying in phase1")
        return False
    if dry_run:
        print(f"  [dry-run] graduation would eval {champion['label']} vs random_legal "
              f"on {GRADUATION_LEVELS} ({GRADUATION_GAMES} games each)")
        return False

    rates = {}
    for level in GRADUATION_LEVELS:
        print(f"  graduation check: eval {champion['label']} vs random_legal "
              f"on {level} ({GRADUATION_GAMES} games)")
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "eval_vs_random.py"),
            "--p1",   champion["id"],
            "--games", str(GRADUATION_GAMES),
            "--level", level,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            print("    (eval timeout — keeping current phase)")
            return False
        if proc.returncode != 0:
            print(f"    (eval failed rc={proc.returncode})")
            print(proc.stderr[-500:])
            return False
        last_line = proc.stdout.strip().splitlines()[-1]
        try:
            result = json.loads(last_line)
        except Exception:
            print(f"    (couldn't parse eval JSON; stdout tail: {last_line[:200]})")
            return False
        rate = result.get("win_rate", 0.0)
        rates[level] = rate
        print(f"    win_rate vs random_legal on {level} = {rate:.3f}")

    pass_all = all(r >= GRADUATION_THRESHOLD for r in rates.values())
    print(f"  graduation: {'PASS' if pass_all else 'FAIL'} "
          f"(threshold {GRADUATION_THRESHOLD:.2f}, rates={rates})")
    return pass_all


def _decide_curriculum_phase(conn, dry_run: bool) -> str:
    """Resolve current curriculum phase from the kv table; promote if eligible."""
    current = _kv_get(conn, "curriculum_phase", default="phase1_full_mix")
    # Migrate stale phase names from earlier designs.
    if current in ("phase1_small", "phase1_close"):
        current = "phase1_full_mix"
        if not dry_run:
            _kv_set(conn, "curriculum_phase", current)
    if current == "phase2_wild":
        # Old name → new name.
        current = "phase2_selfplay"
        if not dry_run:
            _kv_set(conn, "curriculum_phase", current)
    if current not in CURRICULUM:
        current = "phase1_full_mix"
        if not dry_run:
            _kv_set(conn, "curriculum_phase", current)

    if current == "phase2_selfplay":
        return current

    # We're in phase1 — check if we can graduate.
    champ = _read_top_elo_runs(conn, limit=1)
    champion = champ[0] if champ else None
    if _check_graduation_to_phase2(conn, champion, dry_run):
        if not dry_run:
            _kv_set(conn, "curriculum_phase", "phase2_selfplay")
            _kv_set(conn, "curriculum_phase_advanced_at", _utc_now().isoformat())
        print("  *** graduating to phase2_selfplay ***")
        return "phase2_selfplay"
    return "phase1_full_mix"


def _pick_opponent_for_run(
    phase: str,
    rng: random.Random,
    elo_champion: dict | None,
    elo_top: list[dict],
) -> tuple[str, dict | None]:
    """Returns (opponent_name, opponent_kwargs) for a single run.

    Resolves the abstract opponent_mix entries:
      'self_play_elo_champ' → opponent_name='neural', opponent_run_id=champion.id
      'leaderboard_top3'    → opponent_name='neural', opponent_run_id=random top-3
      'random_legal'        → opponent_name='random_legal'
    """
    mix = CURRICULUM[phase]["opponent_mix"]
    keys = list(mix.keys())
    pick = rng.choices(keys, weights=list(mix.values()), k=1)[0]

    if pick == "random_legal":
        return ("random_legal", None)
    if pick == "self_play_elo_champ":
        if elo_champion is None:
            return ("random_legal", None)
        return ("neural", {
            "opponent_run_id": elo_champion["id"],
            "device":          "cuda",
        })
    if pick == "leaderboard_top3":
        candidates = [r for r in elo_top[:3]]
        if not candidates:
            return ("random_legal", None)
        sel = rng.choice(candidates)
        return ("neural", {
            "opponent_run_id": sel["id"],
            "device":          "cuda",
        })
    raise ValueError(f"unknown opponent_mix key: {pick!r}")


# ---------------------------------------------------------------------------
# Batch design
# ---------------------------------------------------------------------------

def _decide_next_batch(
    phase: str,
    elo_champion: dict | None,
    elo_top: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Build the next ~6h batch (14 runs).

    Each run uses the phase's `level_mix` so a single rollout sees the full
    building-count distribution (4 → 24). cfg.level_name is a label only.
    """
    cfg_phase = CURRICULUM[phase]
    levels = load_levels()
    base_cfg = {
        **DEFAULT_CFG,
        "action_repeat": cfg_phase["K"],
        "reward_v13":    cfg_phase["reward_v13"],
        "gamma":         cfg_phase["gamma"],
        "level_name":    levels["level_name"],
        "level_mix":     levels["level_mix"],
    }

    runs = []
    epoch = _utc_now().strftime("%y%m%d-%H%M")

    def _make_run(label_suffix: str, minutes: int, notes: str):
        opp_name, opp_kwargs = _pick_opponent_for_run(phase, rng, elo_champion, elo_top)
        return {
            "label":   f"cron-{epoch}-{phase}-{label_suffix}",
            "minutes": minutes,
            "seed":    str(rng.randint(0, 2**31 - 1)),
            "config":  dict(base_cfg),
            "opponent_name":   opp_name,
            "opponent_kwargs": opp_kwargs,
            "notes":   notes,
        }

    # 6 short probes (15 min × 6 = 90 min)
    for i in range(6):
        runs.append(_make_run(f"short-{i:02d}", 15, f"phase={phase} short probe"))
    # 4 medium runs (30 min × 4 = 120 min)
    for i in range(4):
        runs.append(_make_run(f"med-{i:02d}", 30, f"phase={phase} medium"))
    # 3 long runs (60 min × 3 = 180 min). In phase1 these are extra random_legal
    # endurance probes; in phase2 they're forced self-play vs the Elo champion.
    if elo_champion is not None and phase == "phase2_selfplay":
        for i in range(3):
            runs.append({
                "label":   f"cron-{epoch}-{phase}-selfplay-{i:02d}",
                "minutes": 60,
                "seed":    str(rng.randint(0, 2**31 - 1)),
                "config":  dict(base_cfg),
                "opponent_name":   "neural",
                "opponent_kwargs": {
                    "opponent_run_id": elo_champion["id"],
                    "device":          "cuda",
                },
                "notes":  f"phase={phase} self-play vs Elo champion",
            })
    else:
        for i in range(3):
            runs.append(_make_run(f"long-{i:02d}", 60, f"phase={phase} 1h long"))

    # 1 endurance run (60 min) — same opponent_mix sampling.
    runs.append(_make_run("endurance", 60, f"phase={phase} 1h endurance"))

    total_min = sum(r["minutes"] for r in runs)
    print(f"  decided {len(runs)} runs, total wall budget = {total_min} min "
          f"({total_min/60:.1f}h)")
    return runs


def _push_runs(conn, runs: list[dict], model_id: str, sim_id: str, dry_run: bool) -> list[str]:
    inserted = []
    launch_at = int(time.time() * 1000)
    with conn.cursor() as cur:
        for r in runs:
            hp = dict(r["config"])
            if "opponent_name" in r:
                hp["opponent_name"] = r["opponent_name"]
            if r.get("opponent_kwargs"):
                hp["opponent_kwargs"] = r["opponent_kwargs"]
            row = (
                model_id, sim_id, PROJECT,
                r["label"], r.get("notes"),
                "queued", r["minutes"] * 60 * 1000,
                r["seed"], json.dumps(hp),
                "unassigned", launch_at,
            )
            if dry_run:
                opp = r.get("opponent_name", "random_legal")
                opp_id = (r.get("opponent_kwargs") or {}).get("opponent_run_id", "")
                opp_id_short = opp_id[:8] if opp_id else ""
                print(f"  [dry-run] queue: {r['label']:55s} {r['minutes']}m "
                      f"L={r['config']['level_name']:22s} K={r['config'].get('action_repeat')} "
                      f"opp={opp}{('('+opp_id_short+')') if opp_id_short else ''}")
                continue
            cur.execute(
                """
                INSERT INTO runs (model_id, simulator_id, project, label, description,
                                  status, budget_ms, seed, hyperparams, machine, launch_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                row,
            )
            inserted.append(str(cur.fetchone()[0]))
    if not dry_run:
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log without writing to Supabase")
    ap.add_argument("--no-cancel-backlog", action="store_true")
    ap.add_argument("--review-window-hours", type=int, default=6)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--sim-id",   default=DEFAULT_SIM_ID)
    ap.add_argument("--seed",     type=int, default=None)
    ap.add_argument("--skip-elo-review", action="store_true",
                    help="skip the per-fire Elo tournament pass (faster dry-run)")
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    now = _utc_now()
    print(f"[{now.isoformat()}] cron_agent_pulse fire")
    print(f"  model={args.model_id} sim={args.sim_id} seed={args.seed}")

    since = now - timedelta(hours=args.review_window_hours)
    with connect() as conn:
        recent = _read_recent_runs(conn, since)
        summary = _summarise_recent_runs(recent)
        print(f"  reviewed {summary['total']} runs since {since.isoformat()}: "
              f"done={summary['done']} running={summary['running']} "
              f"failed={summary['failed']} queued={summary['queued']}")
        if summary["max_win"] is not None:
            print(f"  max win_rate observed: {summary['max_win']:.3f}  "
                  f"(mean over done runs: {summary['mean_win']:.3f})")

        # Cancel pre-existing queued backlog (running runs preserved).
        if not args.no_cancel_backlog:
            _cancel_queued_backlog(conn, args.dry_run)

        # ---- queue FIRST so the worker isn't idle while we run Elo / eval. ----
        # Read the current Elo top + curriculum state without doing any
        # tournament work.
        elo_top = _read_top_elo_runs(conn, limit=5)
        elo_champion = elo_top[0] if elo_top else None
        if elo_champion is not None:
            print(f"  Elo champion (pre-review): {elo_champion['label']} "
                  f"score={elo_champion['elo_score']:.1f} "
                  f"matches={elo_champion['elo_n_matches']}")
        else:
            print("  no Elo-rated champion yet (need ≥3 matches)")

        kv_phase = _kv_get(conn, "curriculum_phase", default="phase1_full_mix")
        # Map any legacy phase names → current.
        if kv_phase in ("phase1_small", "phase1_close"):
            kv_phase = "phase1_full_mix"
        elif kv_phase == "phase2_wild":
            kv_phase = "phase2_selfplay"
        if kv_phase not in CURRICULUM:
            kv_phase = "phase1_full_mix"
        print(f"  curriculum phase (pre-review): {kv_phase}")

        next_runs = _decide_next_batch(kv_phase, elo_champion, elo_top, rng)
        ids = _push_runs(conn, next_runs, args.model_id, args.sim_id, args.dry_run)
        if not args.dry_run:
            print(f"  queued {len(ids)} new runs (worker can start immediately)")

        # ---- THEN run Elo review + graduation eval (worker training in parallel). ----
        if not args.skip_elo_review:
            _elo_review_pass(conn, args.dry_run)

        # Version-bump trigger: register_model.py --trigger-rerate sets
        # kv['needs_rerate']='1' to ask for a top-30 re-rating against new
        # weights. Honour it once, then clear the flag.
        rerate_flag = _kv_get(conn, "needs_rerate")
        if rerate_flag == '1':
            print("  [kv] needs_rerate=1 — kicking off rate_all_runs --include-rated --max-runs 30")
            if args.dry_run:
                print("  [dry-run] would call scripts/rate_all_runs.py and clear flag")
            else:
                cmd = [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "rate_all_runs.py"),
                    "--include-rated", "--max-runs", "30",
                    "--matches", "4", "--games", "64",
                ]
                try:
                    subprocess.run(cmd, timeout=7200)
                except subprocess.TimeoutExpired:
                    print("  rate_all_runs timeout (2h cap)")
                _kv_set(conn, "needs_rerate", "0")
                print("  cleared kv['needs_rerate']")

        # Now check graduation against the freshly-rated champion. If a
        # graduation event fires, the kv flips so the NEXT cron fire queues
        # the new phase. Don't re-queue this fire (workers already busy).
        new_phase = _decide_curriculum_phase(conn, args.dry_run)
        if new_phase != kv_phase:
            print(f"  curriculum phase changed: {kv_phase} → {new_phase} "
                  f"(applies to next cron fire)")

    if args.dry_run:
        print("[dry-run] no rows inserted")
    else:
        print(f"  queued {len(ids)} new runs")
        for i, rid in enumerate(ids[:5]):
            print(f"    {i:02d} id={rid}  label={next_runs[i]['label']}")
        if len(ids) > 5:
            print(f"    ... +{len(ids)-5} more")
    print(f"[{_utc_now().isoformat()}] done")


if __name__ == "__main__":
    sys.exit(main())
