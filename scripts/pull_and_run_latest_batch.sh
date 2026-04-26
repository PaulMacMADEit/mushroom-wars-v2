#!/bin/bash
# Pull main, find the highest scripts/queue_b*.py, and run it once if it
# hasn't been run yet. State tracked in ~/.local/state/mushroom-wars/last_run_batch
# (just the integer N of the last batch executed).
#
# Fired daily at 11:30 Pacific by mushroom-pull-batch.timer on PaulLinux. The
# cloud routine commits a new queue_b{N+1}.py at 11:00 Pacific; this picks
# it up.

set -euo pipefail

REPO=~/Projects/Personal/games/mushroom-wars-v2
STATE_DIR=~/.local/state/mushroom-wars
STATE_FILE="$STATE_DIR/last_run_batch"
LOG_DIR=~/.local/log/mushroom-wars
LOG_FILE="$LOG_DIR/pull_batch.log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

{
  echo "[$(stamp)] start"

  cd "$REPO"
  source .venv/bin/activate

  echo "[$(stamp)] git pull origin main"
  if ! git pull --ff-only origin main 2>&1; then
    echo "[$(stamp)] pull failed; aborting"
    exit 1
  fi

  # Find highest queue_bN.py.
  highest=$(ls scripts/queue_b*.py 2>/dev/null \
    | sed -E 's|scripts/queue_b([0-9]+)\.py|\1|' \
    | sort -n | tail -1)

  if [[ -z "$highest" ]]; then
    echo "[$(stamp)] no scripts/queue_b*.py found; nothing to run"
    exit 0
  fi

  last=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
  echo "[$(stamp)] highest=b${highest}  last_run=b${last}"

  if [[ "$highest" -le "$last" ]]; then
    echo "[$(stamp)] no new batch since last run; nothing to do"
    exit 0
  fi

  script="scripts/queue_b${highest}.py"
  echo "[$(stamp)] running $script"
  if python "$script" 2>&1; then
    echo "$highest" > "$STATE_FILE"
    echo "[$(stamp)] queued b${highest}; state file updated"
  else
    echo "[$(stamp)] $script exited non-zero; NOT updating state file (will retry tomorrow)"
    exit 2
  fi

  echo "[$(stamp)] done"
} >> "$LOG_FILE" 2>&1

# Tee the latest run to stdout for systemd journal too.
tail -n 30 "$LOG_FILE"
