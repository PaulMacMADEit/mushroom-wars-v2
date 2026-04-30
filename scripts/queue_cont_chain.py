#!/usr/bin/env python
"""Queue the next batch in a continuation chain.

A "chain" is a sequence of cont- runs descending from one root, named
`karpv2-cont-<root_short>-NN` where NN is the batch index. Each batch
inherits the previous batch's hyperparams and weights.

Idempotent — safe to call every loop fire:
  - if chain head is queued/running → no-op (head not ready)
  - if max-batches reached → no-op
  - else → queue next batch from chain head (or root, if chain empty)

Usage:
    python scripts/queue_cont_chain.py --root <run_uuid> --max-batches 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root run UUID — chain continues from this")
    ap.add_argument("--max-batches", type=int, default=6, help="stop after this many cont batches")
    ap.add_argument("--budget-sec", type=int, default=1200, help="per-batch budget (default 1200s = 20min)")
    ap.add_argument("--label-prefix", default=None, help="default: karpv2-cont-<root_short>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root_short = args.root[:8]
    prefix = args.label_prefix or f"karpv2-cont-{root_short}"

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, label, status FROM runs WHERE label LIKE %s ORDER BY queued_at ASC",
                (f"{prefix}-%",),
            )
            chain = cur.fetchall()

            if chain:
                head_id, head_label, head_status = chain[-1]
                head_index = int(head_label.split("-")[-1])
                print(f"[chain] {len(chain)} batches found, head: {head_label} ({head_status})")
            else:
                head_id, head_label, head_status = args.root, None, "done"
                head_index = 0
                print(f"[chain] empty; will start from root {root_short}")

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
            next_index = head_index + 1
            new_label = f"{prefix}-{next_index:02d}"
            seed = f"{parent_seed or 'a'}-c{next_index}"
            budget_ms = args.budget_sec * 1000
            cumulative = (parent_cum_ms or parent_budget_ms or 0) + budget_ms
            launch_at = int(time.time() * 1000)

            print(f"[chain] queueing {new_label}")
            print(f"        budget={args.budget_sec}s, parent={str(head_id)[:8]}, seed={seed}")
            print(f"        update_epochs={hp.get('update_epochs')} lr={hp.get('lr')} "
                  f"opp_pool_mode={hp.get('opponent_pool_mode')} reward_v={hp.get('reward_version')}")

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
                model_id, sim_id, PROJECT, new_label,
                f"Cont chain batch {next_index} from root {root_short}",
                budget_ms, seed, json.dumps(hp), launch_at,
                head_id, cumulative,
            ))
            new_id = cur.fetchone()[0]
            conn.commit()
            print(f"[chain] queued {new_id}  {new_label}")


if __name__ == "__main__":
    main()
