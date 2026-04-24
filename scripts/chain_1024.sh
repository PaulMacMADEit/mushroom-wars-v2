#!/usr/bin/env bash
# Chain-queue 7 continuations of kar-cap-1024 (60-min each).
# Waits for each parent to hit status='done' before queueing the next link.
# Run in the background; it's idempotent against already-queued labels.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"

# Start parent: kar-cap-1024 (the originally-queued 3h initial run)
PARENT_LABEL="kar-cap-1024"
LINKS=7
BUDGET=3600  # 60 min per link

current_label="$PARENT_LABEL"

for i in $(seq 1 "$LINKS"); do
  child_label="kar-cap-1024-c$i"

  # Short-circuit if the child already exists (re-run friendly).
  exists=$("$PY" -c "
from cli.db import connect
with connect() as c, c.cursor() as cur:
    cur.execute(\"SELECT id FROM runs WHERE label=%s LIMIT 1\", ('$child_label',))
    r = cur.fetchone()
    print(r[0] if r else '')
" 2>/dev/null || echo "")
  if [[ -n "$exists" ]]; then
    echo "$(date +%H:%M:%S) skip $child_label — already queued as $exists"
    current_label="$child_label"
    continue
  fi

  # Wait for current_label's run to be done. Returns "" while not done.
  echo "$(date +%H:%M:%S) waiting for $current_label to be done…"
  while true; do
    parent_id=$("$PY" -c "
from cli.db import connect
with connect() as c, c.cursor() as cur:
    cur.execute(\"SELECT id FROM runs WHERE label=%s AND status='done' LIMIT 1\", ('$current_label',))
    r = cur.fetchone()
    print(r[0] if r else '')
" 2>/dev/null || echo "")
    [[ -n "$parent_id" ]] && break
    sleep 60
  done

  echo "$(date +%H:%M:%S) $current_label done (id=$parent_id) — queueing $child_label"
  "$PY" cli/continue_training.py \
    --parent "$parent_id" \
    --budget "$BUDGET" \
    --label "$child_label" \
    --description "Capacity: 1024 chain link $i/$LINKS (+60m)"

  current_label="$child_label"
done

echo "$(date +%H:%M:%S) 1024 chain complete — all $LINKS continuations queued."
