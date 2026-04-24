#!/usr/bin/env bash
# Chain-queue 24 × 60-min continuations of kar-cap-256.
# Gated on kar-cap-1024-c7 (final 1024 link) being done — so this lands
# AFTER the full 1024 chain, not interleaved with it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"

GATE_LABEL="kar-cap-1024-c7"   # wait for this to be done before starting
PARENT_LABEL="kar-cap-256"     # first parent in the 256 chain
LINKS=24
BUDGET=3600

# Wait for the gate (final 1024 chain link) to finish.
echo "$(date +%H:%M:%S) chain_256: waiting for $GATE_LABEL to be done…"
while true; do
  gate_id=$("$PY" -c "
from cli.db import connect
with connect() as c, c.cursor() as cur:
    cur.execute(\"SELECT id FROM runs WHERE label=%s AND status='done' LIMIT 1\", ('$GATE_LABEL',))
    r = cur.fetchone()
    print(r[0] if r else '')
" 2>/dev/null || echo "")
  [[ -n "$gate_id" ]] && break
  sleep 120
done
echo "$(date +%H:%M:%S) chain_256: gate $GATE_LABEL done — starting 256 chain"

current_label="$PARENT_LABEL"
for i in $(seq 1 "$LINKS"); do
  child_label="kar-cap-256-c$i"

  # Re-run safety.
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
    --description "Capacity: 256 chain link $i/$LINKS (+60m)"

  current_label="$child_label"
done

echo "$(date +%H:%M:%S) 256 chain complete — all $LINKS continuations queued."
