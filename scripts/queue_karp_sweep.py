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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect, PROJECT
from cli.loop_config import load


# Three label families this script must cope with:
#   legacy A: karpv2-260429-2104-lr-lo     (pre-2026-04-29)
#   legacy B: v10.1.5-lr-lo                (Paul's 2026-04-29 rename — exp num inline w/ model_id)
#   current: v10.2.02-LargeMap-lr-lo       (2026-04-30 onward — exp num two-digit, MajorChange tag)
#
# Sweep-cell pattern (the trailing portion that identifies axis + cell):
#   -<axis>-<lo|mid|hi>
_KARP_AXIS_RE = re.compile(
    r"^(?:"
    r"karpv2-\d{6}-\d{4}"                       # legacy A
    r"|v\d+\.\d+\.\d+-[A-Z][A-Za-z]+"           # current (v10.2.02-LargeMap)
    r"|v\d+(?:\.\d+)+"                           # legacy B (v10.1.5)
    r")-([a-z_]+)-(?:lo|mid|hi)$"
)


def _last_karp_axis() -> str | None:
    """Most-recent karp-style run's axis (from labels), or None if none yet.

    Scans recent runs for any of the three sweep label families and returns
    the axis from the newest match.
    """
    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT label
              FROM runs
             WHERE project = %s
               AND (label LIKE 'karpv2-%%' OR label ~ '^v[0-9]+\\.[0-9]+\\.')
             ORDER BY queued_at DESC
             LIMIT 50
            """,
            (PROJECT,),
        )
        for (label,) in cur.fetchall():
            m = _KARP_AXIS_RE.match(label)
            if m:
                return m.group(1)
    return None


def _next_experiment_num(model_id: str) -> int:
    """Next experiment number for `model_id`. Parses the {NN} segment from
    existing `{model_id}.{NN}-...` labels and returns max+1.

    Replaces the old queued_at-distinct-count approach because under the new
    convention a single experiment can be a chain (6+ batches → 6 distinct
    queued_at values, but still experiment 01) — the count would be wrong.
    """
    pattern = rf"^{re.escape(model_id)}\.(\d+)-"
    with connect() as c, c.cursor() as cur:
        cur.execute(
            "SELECT label FROM runs WHERE project = %s AND label ~ %s",
            (PROJECT, pattern),
        )
        nums: list[int] = []
        for (label,) in cur.fetchall():
            m = re.match(pattern, label)
            if m:
                nums.append(int(m.group(1)))
    return (max(nums) if nums else 0) + 1


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


def queue_sweep(
    axis: str | None,
    dry_run: bool,
    baseline_overrides: dict,
    from_run_id: str | None = None,
) -> None:
    """Queue one sweep (3 cells of one axis).

    `from_run_id` (optional UUID): when set, each cell is queued as a
    continuation off that parent. The worker loads parent weights at run
    start; cells start warm instead of from random init. Used to test "same
    starting point, different lr/etc." sweeps from a known-good checkpoint.
    """
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

    model_id       = cfg.model["model_id"]                                  # FK
    version_prefix = cfg.model.get("version_prefix", model_id)               # label prefix
    major_change   = cfg.model.get("major_change", "Unspecified")
    exp_num = _next_experiment_num(version_prefix)
    budget_ms = int(cfg.schedule["cell_budget_seconds"]) * 1000
    launch_at = int(time.time() * 1000)

    print(f"[karp] axis={axis_obj.axis} (last_used={last!r})")
    print(f"[karp] model_id={model_id} version_prefix={version_prefix} major_change={major_change} "
          f"experiment={exp_num:02d} — {len(cells)} cells × {budget_ms//60000} min each")

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
        # Label format (2026-04-30 onward):
        #   {version_prefix}.{exp:02d}-{MajorChange}-{axis}-{cell}
        # e.g. v10.2.02-LargeMap-lr-lo, v10.2.02-LargeMap-lr-mid, v10.2.02-LargeMap-lr-hi
        # One karp fire = one exp num shared across the 3 cells.
        label = f"{version_prefix}.{exp_num:02d}-{major_change}-{axis_obj.axis}-{cell['label']}"
        # Step number derived from the .N suffix of version_prefix (e.g. "v10.2" -> "2")
        step_label = version_prefix.split('.')[-1] if '.' in version_prefix else '?'
        desc  = (f"Step {step_label} ({major_change}): "
                 f"{axis_obj.axis} sweep — {cell['label']} cell "
                 f"({axis_obj.axis}={cell['value']}, opp={opp_mode})")
        rows.append((cell, label, desc, hp))

    for cell, label, desc, hp in rows:
        print(f"  {label}  {axis_obj.axis}={cell['value']}")

    if dry_run:
        print("[karp] dry-run — no inserts")
        return

    # When --from-run-id is set, each cell becomes a warm-start continuation
    # off that parent run. Worker reads parent_run_id and loads weights at
    # job start (same path the chain script uses).
    if from_run_id:
        print(f"[karp] warm-starting all cells from run {from_run_id[:8]}")

    inserted = []
    with connect() as c, c.cursor() as cur:
        for cell, label, desc, hp in rows:
            if from_run_id:
                cur.execute(
                    """
                    INSERT INTO runs
                      (model_id, simulator_id, project, label, description,
                       status, budget_ms, seed, hyperparams, machine, launch_at,
                       parent_run_id, is_continuation)
                    VALUES
                      (%s, %s, %s, %s, %s,
                       'queued', %s, %s, %s::jsonb, %s, %s,
                       %s, true)
                    RETURNING id
                    """,
                    (
                        cfg.model["model_id"], cfg.model["simulator_id"], PROJECT,
                        label, desc,
                        budget_ms, str(cell["label"]),
                        json.dumps(hp), "unassigned", launch_at,
                        from_run_id,
                    ),
                )
            else:
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
    ap.add_argument("--from-run-id", default=None,
                    help="warm-start every cell from this run UUID (loads its weights)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue_sweep(
        axis=args.axis,
        dry_run=args.dry_run,
        baseline_overrides=_parse_overrides(args.override),
        from_run_id=args.from_run_id,
    )


if __name__ == "__main__":
    main()
