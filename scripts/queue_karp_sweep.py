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


_KARP_AXIS_RE = re.compile(r"^karp-\d{6}-\d{4}-([a-z_]+)-(?:lo|mid|hi)$")


def _latest_champion() -> tuple[str, str] | None:
    """Most-recent champion's (source_run_id, label). Used as training opponent
    since per-env neural opponents aren't supported on the JAX backend yet."""
    with connect() as c, c.cursor() as cur:
        cur.execute("""
            SELECT source_run_id, label
              FROM champions
             ORDER BY archived_at DESC
             LIMIT 1
        """)
        r = cur.fetchone()
    return (str(r[0]), r[1]) if r else None


def _last_karp_axis() -> str | None:
    """Most-recent karp- run's axis (from labels), or None if no karp- runs yet."""
    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT label
              FROM runs
             WHERE project = %s AND label LIKE 'karp-%%'
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

    # Inject training opponent = most-recent champion (jax backend doesn't
    # support per-env neural opponents yet; see trainer.py:210). Only inject
    # when self_play is False — otherwise the trainer would fight us.
    if not base_hp.get("self_play", False):
        champ = _latest_champion()
        if champ is None:
            print("[karp] WARN: no champion in archive yet — falling back to random_legal opponent")
        else:
            champ_id, champ_label = champ
            base_hp["opponent_name"] = "neural"
            base_hp["opponent_kwargs"] = {
                "device": "cuda",
                "opponent_run_id": champ_id,
            }
            print(f"[karp] training opponent: {champ_label[:50]} ({champ_id[:8]})")

    stamp = datetime.now().strftime("%y%m%d-%H%M")
    budget_ms = int(cfg.schedule["cell_budget_seconds"]) * 1000
    launch_at = int(time.time() * 1000)

    print(f"[karp] axis={axis_obj.axis} (last_used={last!r})")
    print(f"[karp] {len(cells)} cells × {budget_ms//60000} min each")

    rows = []
    for cell in cells:
        hp = {**base_hp, axis_obj.axis: cell["value"]}
        label = f"karp-{stamp}-{axis_obj.axis}-{cell['label']}"
        desc  = f"Karpathy sweep: {axis_obj.axis}={cell['value']}"
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
