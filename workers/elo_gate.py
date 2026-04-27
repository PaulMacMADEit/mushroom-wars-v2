"""4-step Elo gate run after every training-run finish.

Invoked by worker.py after a run completes. Replaces the old 3-match-vs-random
auto-rate path with a richer feedback signal:

  Step 1 — Gate (30 games vs random_legal)
      Filter out broken runs. >=70% wins -> proceed to ladder; else mark
      `elo_status='did_not_perform'`, leave elo_score=NULL/default.

  Step 2 — Seed vs anchor (8 games vs random_legal)
      Initial Elo against the 1000 baseline. Always run after gate passes.

  Step 3 — Adaptive ladder placement (8 games)
      Pick the existing rated run with Elo CLOSEST to this run's provisional
      Elo from step 2. Single match against that pivot. Cheap binary-search
      placement on a sparse ladder.

  Step 4 — Batch comparison (8 games)
      Match against the most-recent rated run from the SAME experiment-batch
      tag (parsed from the run's `label` prefix). If no peer in the batch,
      skip (this run becomes the anchor for the next batch member).

The cron-agent's existing 3h pulse continues to add top-3 matches over time
for runs that look strong. This module is the immediate-feedback path.
"""
from __future__ import annotations

import importlib
import re
import sys
import time
import traceback
from pathlib import Path

# Make scripts/ + cli/ importable when worker imports us.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cli.db import connect, PROJECT


# ---------------------------------------------------------------------------
# Tunables — kept at the top so we can sweep them without code-spelunking.
# ---------------------------------------------------------------------------

GATE_GAMES        = 30      # step 1: random_legal gate
GATE_THRESHOLD    = 0.70    # step 1: required win rate to pass

SEED_GAMES        = 8       # step 2: more matches vs random_legal for an anchor reading
LADDER_GAMES      = 8       # step 3: vs nearest-Elo peer
BATCH_GAMES       = 8       # step 4: vs last batch peer

LEVEL             = "random_8_16"
MAX_TICKS         = 200
ELO_K             = 32      # standard


# ---------------------------------------------------------------------------
# Batch-tag parsing — relies on label convention `cron-{date}-{HHMM}-{phase}-{size}-{NN}`.
# Strips the `-{size}-{NN}` suffix; whatever remains is the batch tag.
# Falls back to the full label if the suffix doesn't match.
# ---------------------------------------------------------------------------

_BATCH_SUFFIX = re.compile(r"-(?:short|med|long|tiny|huge|xs|s|m|l|xl)-\d+$")

def batch_tag(label: str | None) -> str | None:
    if not label:
        return None
    m = _BATCH_SUFFIX.search(label)
    if m:
        return label[: m.start()]
    # Generic numeric-suffix fallback: trailing "-NN".
    m2 = re.search(r"-\d+$", label)
    if m2:
        return label[: m2.start()]
    return label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_tournament():
    return importlib.import_module("scripts.tournament")


def _record_match(tournament, run_id: str, opp: str | None, games: int, seed: int) -> dict:
    """Run a match. opp=None means random_legal."""
    return tournament.run_match(
        p1=run_id,
        p2=opp if opp is not None else "random_legal",
        games=games, level=LEVEL, max_ticks=MAX_TICKS, seed=seed, verbose=False,
    )


def _apply_elo(tournament, run_id: str, opp_run_id: str | None, result: dict) -> float:
    """Write Elo delta to Supabase. Returns the run's new Elo."""
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


def _nearest_rated_run(provisional_elo: float, exclude_run_id: str) -> str | None:
    """Find the rated run (elo_status='rated') with Elo closest to provisional_elo.
    Excludes the current run itself."""
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT id, elo_score
                  FROM runs
                 WHERE project = %s
                   AND elo_status = 'rated'
                   AND elo_score IS NOT NULL
                   AND id <> %s
                 ORDER BY ABS(elo_score - %s) ASC
                 LIMIT 1
            """, (PROJECT, exclude_run_id, provisional_elo))
            row = cur.fetchone()
    return str(row[0]) if row else None


def _last_rated_in_batch(label: str, exclude_run_id: str) -> str | None:
    """Most-recent rated run sharing this run's batch tag (excluding self)."""
    tag = batch_tag(label)
    if not tag:
        return None
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT id
                  FROM runs
                 WHERE project = %s
                   AND elo_status = 'rated'
                   AND label LIKE %s
                   AND id <> %s
                 ORDER BY finished_at DESC NULLS LAST
                 LIMIT 1
            """, (PROJECT, f"{tag}%", exclude_run_id))
            row = cur.fetchone()
    return str(row[0]) if row else None


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

def run_elo_gate(run_id: str, label: str) -> dict:
    """Run the 4-step gate. Returns a summary dict (also printed to stdout).

    Failures are non-fatal: any exception is caught at the worker call site,
    so the training pipeline never blocks on a bad gate run.
    """
    t0 = time.perf_counter()
    tournament = _import_tournament()

    summary: dict = {
        "run_id": run_id, "label": label,
        "gate_rate": None, "passed_gate": False,
        "seed_elo": None, "ladder_elo": None, "batch_elo": None,
        "ladder_pivot": None, "batch_peer": None,
        "wall_s": None, "final_status": "unrated",
    }

    # ---- Step 1: gate vs random_legal ------------------------------------
    print(f"[gate] {label}: step 1 — gate ({GATE_GAMES} vs random_legal, "
          f"need >={int(GATE_THRESHOLD*100)}%)", flush=True)
    res = _record_match(tournament, run_id, opp=None, games=GATE_GAMES, seed=1001)
    rate = res["p1_wins"] / max(res["total"], 1)
    summary["gate_rate"] = rate

    if rate < GATE_THRESHOLD:
        print(f"[gate] {label}: FAILED gate ({rate:.1%} < {GATE_THRESHOLD:.0%}) — "
              f"marking did_not_perform", flush=True)
        _set_elo_status(run_id, "did_not_perform")
        summary["final_status"] = "did_not_perform"
        summary["wall_s"] = time.perf_counter() - t0
        return summary

    print(f"[gate] {label}: PASSED gate ({rate:.1%})", flush=True)
    summary["passed_gate"] = True

    # ---- Step 2: seed Elo vs random_legal --------------------------------
    print(f"[gate] {label}: step 2 — seed ({SEED_GAMES} vs random_legal)", flush=True)
    res2 = _record_match(tournament, run_id, opp=None, games=SEED_GAMES, seed=1002)
    seed_elo = _apply_elo(tournament, run_id, opp_run_id=None, result=res2)
    summary["seed_elo"] = seed_elo
    print(f"[gate] {label}: seed Elo -> {seed_elo:.0f}", flush=True)

    # Mark rated NOW so subsequent steps' lookups can see it (though we exclude
    # this run from its own pivot/peer queries).
    _set_elo_status(run_id, "rated")
    summary["final_status"] = "rated"

    # ---- Step 3: ladder placement (nearest-Elo peer) ---------------------
    pivot = _nearest_rated_run(seed_elo, exclude_run_id=run_id)
    if pivot is not None:
        print(f"[gate] {label}: step 3 — ladder vs {pivot[:8]} ({LADDER_GAMES} games)",
              flush=True)
        res3 = _record_match(tournament, run_id, opp=pivot, games=LADDER_GAMES, seed=1003)
        ladder_elo = _apply_elo(tournament, run_id, opp_run_id=pivot, result=res3)
        summary["ladder_elo"] = ladder_elo
        summary["ladder_pivot"] = pivot
        print(f"[gate] {label}: ladder Elo -> {ladder_elo:.0f}", flush=True)
    else:
        print(f"[gate] {label}: step 3 skipped (no rated peer to compare)", flush=True)

    # ---- Step 4: batch comparison ----------------------------------------
    peer = _last_rated_in_batch(label, exclude_run_id=run_id)
    if peer is not None:
        print(f"[gate] {label}: step 4 — batch vs {peer[:8]} ({BATCH_GAMES} games)",
              flush=True)
        res4 = _record_match(tournament, run_id, opp=peer, games=BATCH_GAMES, seed=1004)
        batch_elo = _apply_elo(tournament, run_id, opp_run_id=peer, result=res4)
        summary["batch_elo"] = batch_elo
        summary["batch_peer"] = peer
        print(f"[gate] {label}: batch Elo -> {batch_elo:.0f}", flush=True)
    else:
        print(f"[gate] {label}: step 4 skipped (no prior batch peer — "
              f"this run anchors the batch)", flush=True)

    summary["wall_s"] = time.perf_counter() - t0
    seed_s   = f"{summary['seed_elo']:.0f}"   if summary['seed_elo']   is not None else "—"
    ladder_s = f"{summary['ladder_elo']:.0f}" if summary['ladder_elo'] is not None else "—"
    batch_s  = f"{summary['batch_elo']:.0f}"  if summary['batch_elo']  is not None else "—"
    print(f"[gate] {label}: DONE in {summary['wall_s']:.1f}s "
          f"(seed={seed_s}, ladder={ladder_s}, batch={batch_s})", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI for one-off invocation / testing
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Run the Elo gate against a finished run.")
    ap.add_argument("run_id")
    ap.add_argument("--label", default=None,
                    help="Override label (defaults to runs.label from DB).")
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
        run_elo_gate(args.run_id, label)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli()
