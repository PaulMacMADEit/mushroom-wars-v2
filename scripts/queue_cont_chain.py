#!/usr/bin/env python
"""Queue the next batch in a continuation chain.

Naming convention (2026-04-30 onward):
    v{model}.{step}.{batch:02d}-{MajorChange}-{Kind}-{idx:02d}

Examples:
    v10.2.01-LargeMap-Base-01      # Step 2, batch 01, base chain index 01
    v10.2.01-LargeMap-Base-02      # ... index 02
    v10.1.02-SmallMap-FineTuneLR-lo  # not a chain — sweep cell

A "chain" is a sequence of cont- runs descending from one root. Each batch
inherits the previous batch's hyperparams + weights; an --override flag lets
you change specific hp (e.g. graduate level_mix between Step 1 and Step 2).

Idempotent — safe to call every loop fire:
  - if chain head is queued/running → no-op (head not ready)
  - if max-batches reached → no-op
  - else → queue next batch from chain head (or root, if chain empty)

Usage:
    python scripts/queue_cont_chain.py \\
        --root <run_uuid> \\
        --model 10 --step 2 --batch 1 --major-change LargeMap \\
        --max-batches 6 --budget-sec 1800 \\
        --override 'level_name=random_4_8,level_mix=[random_4_8,random_6_10,random_8_16,random_16_24]'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


def parse_overrides(s: str | None) -> dict:
    """Parse --override 'k1=v1,k2=v2,...' into a dict.

    Values may be JSON-ish: numbers, true/false, null, [list], "string".
    Heuristic decoder — keeps the CLI ergonomic without dragging YAML in.
    """
    if not s:
        return {}
    out = {}
    # Split on top-level commas (don't break commas inside [])
    parts, depth, buf = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1; buf += ch
        elif ch == "]":
            depth -= 1; buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    if buf:
        parts.append(buf)

    for part in parts:
        if "=" not in part:
            raise SystemExit(f"--override item missing '=': {part!r}")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        # Try JSON first, fall back to bracketed-list of bare words, else string
        try:
            out[k] = json.loads(v)
            continue
        except json.JSONDecodeError:
            pass
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                out[k] = []
            else:
                out[k] = [x.strip().strip('"').strip("'") for x in inner.split(",")]
            continue
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root run UUID — chain continues from this")
    ap.add_argument("--model", type=int, required=True, help="model major version (e.g. 10)")
    ap.add_argument("--step", type=int, required=True, help="curriculum step (1, 2, 3...)")
    ap.add_argument("--batch", type=int, required=True, help="experiment batch within step (1, 2, ...)")
    ap.add_argument("--major-change", required=True,
                    help="major change descriptor for this step, e.g. SmallMap, LargeMap, ChampOpp")
    ap.add_argument("--kind", default="Base", help="chain kind: Base, FineTune, etc. (default: Base)")
    ap.add_argument("--max-batches", type=int, default=6, help="stop after this many chain entries")
    ap.add_argument("--budget-sec", type=int, default=1800, help="per-batch budget (default 1800s)")
    ap.add_argument("--override", default=None,
                    help="override hyperparams: 'k1=v1,k2=v2'. Lists in [a,b,c] form.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prefix = f"v{args.model}.{args.step}.{args.batch:02d}-{args.major_change}-{args.kind}"
    overrides = parse_overrides(args.override)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, label, status FROM runs WHERE label LIKE %s ORDER BY queued_at ASC",
                (f"{prefix}-%",),
            )
            chain = cur.fetchall()

            if chain:
                # Numeric two-digit suffix only (skip non-chain children if any)
                numbered = [(id_, lbl, st) for id_, lbl, st in chain
                            if lbl.split("-")[-1].isdigit()]
                if numbered:
                    head_id, head_label, head_status = numbered[-1]
                    head_index = int(head_label.split("-")[-1])
                else:
                    head_id, head_label, head_status = args.root, None, "done"
                    head_index = 0
                print(f"[chain] {len(chain)} entries found, head: {head_label} ({head_status})")
            else:
                head_id, head_label, head_status = args.root, None, "done"
                head_index = 0
                print(f"[chain] empty; will start from root {str(args.root)[:8]}")

            if head_status != "done":
                print(f"[chain] head is {head_status} — wait for it to finish")
                return

            if head_index >= args.max_batches:
                print(f"[chain] reached cap of {args.max_batches} — chain complete")
                return

            cur.execute("""
                SELECT model_id, simulator_id, seed, hyperparams::text,
                       cumulative_budget_ms, budget_ms, status
                FROM runs WHERE id = %s
            """, (head_id,))
            row = cur.fetchone()
            if not row:
                raise SystemExit(f"parent {head_id} not found")
            model_id, sim_id, parent_seed, parent_hp_text, parent_cum_ms, parent_budget_ms, parent_status = row
            if parent_status != "done":
                raise SystemExit(f"parent {head_id} status is {parent_status}, not 'done'")

            hp = json.loads(parent_hp_text) if parent_hp_text else {}
            if overrides:
                print(f"[chain] applying {len(overrides)} override(s): {list(overrides.keys())}")
                hp.update(overrides)

            next_index = head_index + 1
            new_label = f"{prefix}-{next_index:02d}"
            seed = f"{parent_seed or 'a'}-c{next_index}"
            budget_ms = args.budget_sec * 1000
            cumulative = (parent_cum_ms or parent_budget_ms or 0) + budget_ms
            launch_at = int(time.time() * 1000)
            description = (f"Step {args.step} ({args.major_change}): {args.kind} chain "
                           f"— batch {next_index:02d}")

            print(f"[chain] queueing {new_label}")
            print(f"        budget={args.budget_sec}s, parent={str(head_id)[:8]}, seed={seed}")
            print(f"        lr={hp.get('lr')} n_envs={hp.get('n_envs')} "
                  f"level_name={hp.get('level_name')} level_mix={hp.get('level_mix')}")

            if args.dry_run:
                print("[chain] dry-run — no insert")
                return

            cur.execute("""
                INSERT INTO runs (
                    model_id, simulator_id, project, label, description,
                    status, budget_ms, seed, hyperparams, machine, launch_at,
                    parent_run_id, is_continuation, cumulative_budget_ms
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s::jsonb, 'unassigned', %s,
                        %s, true, %s)
                RETURNING id
            """, (
                model_id, sim_id, PROJECT, new_label, description,
                budget_ms, seed, json.dumps(hp), launch_at,
                head_id, cumulative,
            ))
            new_id = cur.fetchone()[0]
            conn.commit()
            print(f"[chain] queued {new_id}  {new_label}")


if __name__ == "__main__":
    main()
