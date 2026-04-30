"""Queue one Karpathy sweep — 3 cells of one axis, fixed-everything-else.

Reads configs/karpathy_loop.yaml for baseline cfg + axis definitions.
Picks the next axis after the most-recent karp- sweep (round-robin),
unless --axis is given.

Usage:
  python scripts/queue_karp_sweep.py                # auto-pick next axis
  python scripts/queue_karp_sweep.py --axis lr      # force this axis
  python scripts/queue_karp_sweep.py --dry-run      # show what it would do
  python scripts/queue_karp_sweep.py --override entropy_coef=0.005,n_envs=512
                                                    # bake into baseline first
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect, PROJECT
from cli.loop_config import load


_KARP_AXIS_RE = re.compile(r"^karpv2-\d{6}-\d{4}-([a-z_]+)-(?:lo|mid|hi)$")


def _last_karp_axis() -> str | None:
    """Most-recent karpv2- run's axis (from labels), or None if no karpv2- runs yet."""
    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT label
              FROM runs
             WHERE project = %s AND label LIKE 'karpv2-%%'
             ORDER BY queued_at DESC
             LIMIT 30
            """,
            (PROJECT,),
        )
        for (label,) in cur.fetchall():
            m = _KARP_AXIS_RE.match(label)
            if m:
                return m.group(1)
    return None


def _parse_overrides(spec: str | None) -> dict:
    """`a=1,b=2.0,c=true` → {a:1, b:2.0, c:True}. Best-effort type coercion."""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k.strip()] = v.lower() == "true"
        else:
            try:
                out[k.strip()] = int(v)
            except ValueError:
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    out[k.strip()] = v
    return out


def queue_sweep(axis: str | None, dry_run: bool, baseline_overrides: dict) -> None:
    cfg = load()
    last = _last_karp_axis()
    if axis is None:
        axis_obj = cfg.next_axis(last)
    else:
        axis_obj = cfg.get_axis(axis)

    base_hp = {**cfg.baseline_hyperparams, **baseline_overrides}
    cells = axis_obj.cells

    # Training opponent: read from configs/karpathy_loop.yaml `training_opponent`
    # block. Resolution per cell (each sweep cell gets its own opponent pick),
    # supporting:
    #   latest_champion  — most-recent archived champion
    #   random_champion  — uniform random from archive
    #   pfsp_champion    — PFSP-weighted random (favours recently-difficult)
    #   self_play        — enable trainer's self_play=true (disables fused_rollout)
    #   neural           — explicit run_id in kwargs (legacy)
    opp = dict(cfg.training_opponent or {})
    opp_mode = opp.get("name", "latest_champion")
    base_kwargs = dict(opp.get("kwargs", {}) or {})

    def _pick_opponent_for_cell(mode: str, cell_label: str):
        """Return (opponent_name, opponent_kwargs, self_play_override) for one cell."""
        if mode == "self_play":
            print(f"[karp]   {cell_label}: self_play")
            return "neural", {}, True
        if mode == "neural":
            return "neural", dict(base_kwargs), False
        if mode == "random_legal":
            # Bootstrap mode — used after a major encoder bump when the champion
            # archive is OBS_DIM-incompatible. No archive read; trainer picks
            # uniform-random legal moves for P2.
            print(f"[karp]   {cell_label}: random_legal")
            return "random_legal", {}, False
        if mode == "noop":
            print(f"[karp]   {cell_label}: noop")
            return "noop", {}, False
        # All champion-* modes pull from the archive
        with connect() as c, c.cursor() as cur:
            if mode == "latest_champion":
                cur.execute("SELECT source_run_id, label FROM champions ORDER BY archived_at DESC LIMIT 1")
                rows_db = cur.fetchall()
            elif mode == "random_champion":
                cur.execute("SELECT source_run_id, label FROM champions ORDER BY random() LIMIT 1")
                rows_db = cur.fetchall()
            elif mode == "pfsp_champion":
                # PFSP weight = 1 - run.elo_score normalised; use champion's source-run elo.
                # Higher-Elo opponents (recently-difficult) sampled more often.
                cur.execute("""
                    SELECT c.source_run_id, c.label, COALESCE(r.elo_score, 1000) AS elo
                      FROM champions c
                      LEFT JOIN runs r ON r.id = c.source_run_id
                     ORDER BY random() * (1.0 / GREATEST(1.0, COALESCE(r.elo_score, 1000) - 900)) DESC
                     LIMIT 1
                """)
                rows_db = cur.fetchall()
            else:
                raise RuntimeError(f"[karp] unknown training_opponent.name={mode!r}")
        if not rows_db:
            raise RuntimeError(
                f"[karp] training_opponent={mode!r} but no champions in archive. "
                "Seed the archive (e.g. via workers/bench_eval.py) before queueing."
            )
        run_id = str(rows_db[0][0])
        label  = rows_db[0][1]
        print(f"[karp]   {cell_label}: opponent={label} ({run_id[:8]}) via {mode}")
        return "neural", {"device": "cuda", "opponent_run_id": run_id, **base_kwargs}, False

    print(f"[karp] training_opponent.mode={opp_mode}")

    stamp = datetime.now().strftime("%y%m%d-%H%M")
    budget_ms = int(cfg.schedule["cell_budget_seconds"]) * 1000
    launch_at = int(time.time() * 1000)

    print(f"[karp] axis={axis_obj.axis} (last_used={last!r})")
    print(f"[karp] {len(cells)} cells × {budget_ms//60000} min each")

    rows = []
    for cell in cells:
        # Resolve opponent per-cell so each cell can train vs a different
        # archive member (under random_champion / pfsp_champion modes).
        opp_name, opp_kwargs, self_play_override = _pick_opponent_for_cell(
            opp_mode, cell["label"],
        )
        hp = {**base_hp, axis_obj.axis: cell["value"]}
        hp["opponent_name"] = opp_name
        if opp_kwargs:
            hp["opponent_kwargs"] = opp_kwargs
        if self_play_override:
            # self_play=true requires fused_rollout=false (trainer assertion).
            hp["self_play"] = True
            hp["fused_rollout"] = False
        label = f"karpv2-{stamp}-{axis_obj.axis}-{cell['label']}"
        desc  = f"Karpathy sweep: {axis_obj.axis}={cell['value']} (opp={opp_mode})"
        rows.append((cell, label, desc, hp))

    for cell, label, desc, hp in rows:
        print(f"  {label}  {axis_obj.axis}={cell['value']}")

    if dry_run:
        print("[karp] dry-run — no inserts")
        return

    inserted = []
    with connect() as c, c.cursor() as cur:
        for cell, label, desc, hp in rows:
            cur.execute(
                """
                INSERT INTO runs
                  (model_id, simulator_id, project, label, description,
                   status, budget_ms, seed, hyperparams, machine, launch_at)
                VALUES
                  (%s, %s, %s, %s, %s,
                   'queued', %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                (
                    cfg.model["model_id"], cfg.model["simulator_id"], PROJECT,
                    label, desc,
                    budget_ms, str(cell["label"]),
                    json.dumps(hp), "unassigned", launch_at,
                ),
            )
            inserted.append((label, str(cur.fetchone()[0])))
        c.commit()

    for label, rid in inserted:
        print(f"  queued {rid[:8]}  {label}")
    print(f"[karp] queued {len(inserted)} runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", help="force axis name (default: round-robin pick)")
    ap.add_argument("--override", help="comma-sep baseline overrides, eg 'lr=1e-4,n_envs=512'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue_sweep(
        axis=args.axis,
        dry_run=args.dry_run,
        baseline_overrides=_parse_overrides(args.override),
    )


if __name__ == "__main__":
    main()
