"""Post-run evaluation: archive sweep + PFSP weight update.

Replaces elo_gate.py's random_legal gate with champion-archive-based
measurement. No hard-coded AIs — we measure everything relative to past
frozen selves.

Flow after a run finishes
--------------------------
1. Bootstrap check — if archive has fewer than MIN_ARCHIVE_FOR_GATE entries,
   do a quick random_legal gate (30 games, 70%) to confirm the model is not
   broken before entering the archive pool. Once bootstrapped this step is
   skipped entirely.

2. Archive sweep — play 8 games vs each champion in the archive.  Write
   win-rates into runs.bench_vector (JSON: {champ_id: win_rate}).
   Update Elo from each match result.

3. PFSP weight update — write `runs.pfsp_weight` = harmonic mean of the
   win-rates in bench_vector (so it's high when we beat some opponents but
   not others — exactly the range where learning signal is richest).

4. Champion promotion — if the run beats the most-recent champion at ≥60%
   over 16 games, snapshot the old champion into the archive before evicting.
   Archive cap = MAX_ARCHIVE_SIZE; oldest within the same arch_era pruned first
   when count exceeds ERA_SOFT_CAP.

Tunables (edit here, not in worker.py):
"""
from __future__ import annotations

import importlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cli.db import connect, PROJECT
from cli.bench_config import load as _load_bench_config


# ---------------------------------------------------------------------------
# Tunables — loaded from configs/bench_eval.yaml at module import.
# Edit the YAML, not this file. Reload requires worker restart (same as
# karpathy_loop.yaml — config is read once per process).
# ---------------------------------------------------------------------------
_BENCH_CFG = _load_bench_config()

LEVEL           = _BENCH_CFG.match["level_name"]
MAX_TICKS       = int(_BENCH_CFG.match["max_ticks"])
ELO_K           = int(_BENCH_CFG.match["elo_k"])

SWEEP_GAMES     = int(_BENCH_CFG.sweep["games_per_champion"])
PROMO_GAMES     = int(_BENCH_CFG.promotion["games"])
PROMO_THRESHOLD = float(_BENCH_CFG.promotion["threshold"])

BOOTSTRAP_GATE_GAMES     = int(_BENCH_CFG.bootstrap_gate["games"])
BOOTSTRAP_GATE_THRESHOLD = float(_BENCH_CFG.bootstrap_gate["threshold"])
MIN_ARCHIVE_FOR_GATE     = int(_BENCH_CFG.bootstrap_gate["min_archive_for_gate"])

MAX_ARCHIVE_SIZE = int(_BENCH_CFG.archive["max_size"])
ERA_SOFT_CAP     = int(_BENCH_CFG.archive["era_soft_cap"])


# ---------------------------------------------------------------------------
# Arch era tag
# ---------------------------------------------------------------------------

def _current_arch_era() -> str:
    """Derive era tag from the net module docstring major version."""
    try:
        import training.net as net_mod
        doc = (net_mod.__doc__ or "").strip()
        m = re.match(r"v(\d+)", doc)
        if m:
            return f"v{m.group(1)}"
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _import_tournament():
    return importlib.import_module("scripts.tournament")


def _run_match(tournament, run_id: str, opp: str | None, games: int, seed: int) -> dict:
    return tournament.run_match(
        p1=run_id,
        p2=opp if opp is not None else "random_legal",
        games=games, level=LEVEL, max_ticks=MAX_TICKS, seed=seed, verbose=False,
    )


def _apply_elo(tournament, run_id: str, opp_run_id: str | None, result: dict) -> float:
    with connect() as c:
        new_elo, _ = tournament.update_elo_from_match(
            c, p1_run_id=run_id, p2_run_id=opp_run_id, result=result, k=ELO_K,
        )
    return new_elo


def _set_elo_status(run_id: str, status: str) -> None:
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("UPDATE runs SET elo_status = %s WHERE id = %s", (status, run_id))
        c.commit()


def _write_bench_vector(run_id: str, vec: dict) -> None:
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE runs SET bench_vector = %s::jsonb WHERE id = %s",
                (json.dumps(vec), run_id),
            )
        c.commit()


def _write_pfsp_weight(run_id: str, weight: float) -> None:
    with connect() as c:
        with c.cursor() as cur:
            # Column may not exist yet on older schemas — graceful skip
            try:
                cur.execute(
                    "UPDATE runs SET pfsp_weight = %s WHERE id = %s",
                    (weight, run_id),
                )
                c.commit()
            except Exception:
                pass  # column optional


def _archive_size() -> int:
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM champions")
            return cur.fetchone()[0]


def _get_archive(exclude_source_run_id: str | None = None) -> list[dict]:
    """Return all champions ordered by archived_at DESC.  Optionally exclude
    rows where source_run_id = exclude_source_run_id."""
    with connect() as c:
        with c.cursor() as cur:
            if exclude_source_run_id:
                cur.execute("""
                    SELECT id, source_run_id, arch_era, weights_url, obs_norm_url, label
                      FROM champions
                     WHERE source_run_id <> %s
                     ORDER BY archived_at DESC
                """, (exclude_source_run_id,))
            else:
                cur.execute("""
                    SELECT id, source_run_id, arch_era, weights_url, obs_norm_url, label
                      FROM champions
                     ORDER BY archived_at DESC
                """)
            cols = ("id", "source_run_id", "arch_era", "weights_url", "obs_norm_url", "label")
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _most_recent_champion() -> Optional[dict]:
    champs = _get_archive()
    return champs[0] if champs else None


def _snapshot_run_into_archive(run_id: str, label: str, notes: str = "") -> str:
    """Insert a run into the champion archive. Returns new champion.id."""
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT weights_url, obs_norm_url FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if not row or not row[0]:
            raise ValueError(f"run {run_id} has no weights_url — cannot archive")
        weights_url, obs_norm_url = row
        era = _current_arch_era()
        with c.cursor() as cur:
            cur.execute("""
                INSERT INTO champions
                    (source_run_id, arch_era, weights_url, obs_norm_url, label, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (run_id, era, weights_url, obs_norm_url, label, notes))
            new_id = str(cur.fetchone()[0])
        c.commit()
    print(f"[bench] archived run {run_id[:8]} as champion {new_id[:8]} (era={era})", flush=True)
    return new_id


def _prune_archive_if_needed() -> None:
    """Enforce MAX_ARCHIVE_SIZE. Prune oldest within the most-populated era first."""
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM champions")
            total = cur.fetchone()[0]
        if total <= MAX_ARCHIVE_SIZE:
            return
        # Find era with the most entries
        with c.cursor() as cur:
            cur.execute("""
                SELECT arch_era, COUNT(*) AS n
                  FROM champions
                 GROUP BY arch_era
                 ORDER BY n DESC
                 LIMIT 1
            """)
            row = cur.fetchone()
        if not row:
            return
        busiest_era = row[0]
        n_to_delete = total - MAX_ARCHIVE_SIZE
        with c.cursor() as cur:
            cur.execute("""
                DELETE FROM champions
                 WHERE id IN (
                     SELECT id FROM champions
                      WHERE arch_era = %s
                      ORDER BY archived_at ASC
                      LIMIT %s
                 )
            """, (busiest_era, n_to_delete))
        c.commit()
        print(f"[bench] pruned {n_to_delete} oldest champions from era '{busiest_era}'", flush=True)


# ---------------------------------------------------------------------------
# PFSP weight calculation
# ---------------------------------------------------------------------------

def _pfsp_weight(win_rates: list[float]) -> float:
    """Harmonic mean of (1 - |wr - 0.5|) — peaks at 50% win-rate (maximum
    information) and falls off toward 0 or 1 (already solved / hopeless)."""
    if not win_rates:
        return 0.5
    vals = [1.0 - abs(wr - 0.5) for wr in win_rates]
    denom = sum(1.0 / max(v, 1e-6) for v in vals)
    return len(vals) / denom


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_bench_eval(run_id: str, label: str) -> dict:
    """Evaluate a freshly-finished run against the champion archive.

    Returns a summary dict. Exceptions are caught at the worker call-site so
    the training pipeline never blocks.
    """
    t0 = time.perf_counter()
    tournament = _import_tournament()

    summary: dict = {
        "run_id": run_id, "label": label,
        "bootstrap_rate": None, "passed_bootstrap": None,
        "archive_size": 0, "bench_vector": {},
        "pfsp_weight": None, "promoted": False,
        "final_status": "unrated", "wall_s": None,
    }

    # ------------------------------------------------------------------ #
    # Step 1: Bootstrap gate (only when archive is thin)                  #
    # ------------------------------------------------------------------ #
    arch = _get_archive(exclude_source_run_id=run_id)
    summary["archive_size"] = len(arch)

    if len(arch) < MIN_ARCHIVE_FOR_GATE:
        print(f"[bench] {label}: archive thin ({len(arch)} champs) — "
              f"running bootstrap gate ({BOOTSTRAP_GATE_GAMES} vs random_legal)", flush=True)
        res = _run_match(tournament, run_id, opp=None,
                         games=BOOTSTRAP_GATE_GAMES, seed=2001)
        rate = res["p1_wins"] / max(res["total"], 1)
        summary["bootstrap_rate"] = rate

        if rate < BOOTSTRAP_GATE_THRESHOLD:
            print(f"[bench] {label}: FAILED bootstrap ({rate:.1%}) — did_not_perform", flush=True)
            _set_elo_status(run_id, "did_not_perform")
            summary["final_status"] = "did_not_perform"
            summary["wall_s"] = time.perf_counter() - t0
            return summary

        print(f"[bench] {label}: passed bootstrap ({rate:.1%})", flush=True)
        summary["passed_bootstrap"] = True

        # Seed Elo from the bootstrap match itself and archive immediately
        _apply_elo(tournament, run_id, opp_run_id=None, result=res)
        _set_elo_status(run_id, "rated")
        summary["final_status"] = "rated"

        new_champ_id = _snapshot_run_into_archive(run_id, label, notes="bootstrap")
        _prune_archive_if_needed()
        summary["promoted"] = True
        summary["wall_s"] = time.perf_counter() - t0
        print(f"[bench] {label}: bootstrap done, archived as {new_champ_id[:8]}", flush=True)
        return summary

    # ------------------------------------------------------------------ #
    # Step 2: Archive sweep — 8 games vs every champion                  #
    # ------------------------------------------------------------------ #
    print(f"[bench] {label}: archive sweep — {len(arch)} champions × {SWEEP_GAMES} games",
          flush=True)
    bench_vector: dict[str, float] = {}
    win_rates: list[float] = []

    for i, champ in enumerate(arch, 1):
        champ_id   = str(champ["id"])
        champ_label = champ["label"]
        source_rid  = str(champ["source_run_id"])

        res = _run_match(tournament, run_id, opp=source_rid,
                         games=SWEEP_GAMES, seed=3000 + i)
        wr = res["p1_wins"] / max(res["total"], 1)
        bench_vector[champ_id] = round(wr, 4)
        win_rates.append(wr)

        new_elo = _apply_elo(tournament, run_id, opp_run_id=source_rid, result=res)
        print(f"[bench]   vs champ {champ_label[:40]}: {wr:.1%}  Elo→{new_elo:.0f}", flush=True)

    _write_bench_vector(run_id, bench_vector)
    summary["bench_vector"] = bench_vector

    # ------------------------------------------------------------------ #
    # Step 3: PFSP weight                                                 #
    # ------------------------------------------------------------------ #
    pfsp = _pfsp_weight(win_rates)
    _write_pfsp_weight(run_id, pfsp)
    summary["pfsp_weight"] = pfsp
    print(f"[bench] {label}: PFSP weight = {pfsp:.3f}", flush=True)

    # Mark rated regardless of win-rate (continuous — no gating)
    _set_elo_status(run_id, "rated")
    summary["final_status"] = "rated"

    # ------------------------------------------------------------------ #
    # Step 4: Champion promotion                                           #
    # ------------------------------------------------------------------ #
    current_champ = _most_recent_champion()
    if current_champ is not None:
        champ_source = str(current_champ["source_run_id"])
        print(f"[bench] {label}: promotion check vs {current_champ['label'][:40]} "
              f"({PROMO_GAMES} games)", flush=True)
        res_p = _run_match(tournament, run_id, opp=champ_source,
                           games=PROMO_GAMES, seed=4001)
        promo_wr = res_p["p1_wins"] / max(res_p["total"], 1)
        print(f"[bench] {label}: promotion win-rate = {promo_wr:.1%} "
              f"(need {PROMO_THRESHOLD:.0%})", flush=True)

        if promo_wr >= PROMO_THRESHOLD:
            _apply_elo(tournament, run_id, opp_run_id=champ_source, result=res_p)
            new_champ_id = _snapshot_run_into_archive(
                run_id, label,
                notes=f"promoted: {promo_wr:.1%} vs {current_champ['label']}"
            )
            _prune_archive_if_needed()
            summary["promoted"] = True
            print(f"[bench] {label}: PROMOTED — new champion {new_champ_id[:8]}", flush=True)
        else:
            print(f"[bench] {label}: not promoted ({promo_wr:.1%} < {PROMO_THRESHOLD:.0%})",
                  flush=True)
    else:
        # Archive just got populated via sweep but had no champion — auto-promote
        new_champ_id = _snapshot_run_into_archive(run_id, label, notes="first run")
        _prune_archive_if_needed()
        summary["promoted"] = True

    summary["wall_s"] = time.perf_counter() - t0
    print(f"[bench] {label}: DONE in {summary['wall_s']:.1f}s "
          f"(pfsp={pfsp:.3f}, promoted={summary['promoted']})", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI for one-off invocation
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Run archive bench-eval for a finished run.")
    ap.add_argument("run_id")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    label = args.label
    if label is None:
        with connect() as c:
            with c.cursor() as cur:
                cur.execute("SELECT label FROM runs WHERE id = %s", (args.run_id,))
                row = cur.fetchone()
        if not row:
            print(f"run {args.run_id} not found", file=sys.stderr)
            sys.exit(1)
        label = row[0]

    try:
        run_bench_eval(args.run_id, label)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli()
