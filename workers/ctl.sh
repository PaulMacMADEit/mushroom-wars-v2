#!/usr/bin/env bash
# Toggle the mushroom-wars worker on either host.
#
# Usage:
#   ./workers/ctl.sh mac on|off|status|logs
#   ./workers/ctl.sh pc  on|off|status|logs
#   ./workers/ctl.sh all status
#
# Mac worker runs via launchd (com.mushroomwars.worker). Install once with
# `./workers/install_service.sh install` before first use.
#
# PC worker runs via systemd --user on paul@192.168.1.137.

set -euo pipefail

PC_HOST="paul@192.168.1.137"
PC_UNIT="mushroom-worker.service"
# Mac manages the worker as a plain nohup process rather than a launchd
# agent — launchd runs outside TCC's granted-apps list, so it can't read
# files inside ~/Documents (where this repo lives).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAC_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
MAC_LOG_DIR="$HOME/Library/Logs/mushroom-worker"
MAC_PID_FILE="$MAC_LOG_DIR/worker.pid"

mac_pid() { [[ -f "$MAC_PID_FILE" ]] && cat "$MAC_PID_FILE" || echo ""; }
mac_running() {
  local pid; pid=$(mac_pid)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

mac() {
  mkdir -p "$MAC_LOG_DIR"
  case "$1" in
    on)
      if mac_running; then echo "mac: already running pid=$(mac_pid)"; return; fi
      (cd "$REPO_ROOT" && \
        PYTHONUNBUFFERED=1 nohup "$MAC_VENV_PYTHON" -u workers/worker.py \
          >>"$MAC_LOG_DIR/worker.out.log" 2>>"$MAC_LOG_DIR/worker.err.log" &
        echo $! > "$MAC_PID_FILE")
      sleep 1
      if mac_running; then
        echo "mac: started pid=$(mac_pid)"
      else
        echo "mac: failed to start — see $MAC_LOG_DIR/worker.err.log" >&2; exit 1
      fi
      ;;
    off)
      if ! mac_running; then echo "mac: not running"; rm -f "$MAC_PID_FILE"; return; fi
      kill "$(mac_pid)" 2>/dev/null || true
      sleep 1
      if mac_running; then kill -9 "$(mac_pid)" 2>/dev/null || true; fi
      rm -f "$MAC_PID_FILE"
      echo "mac: stopped"
      ;;
    status)
      if mac_running; then
        echo "mac: running pid=$(mac_pid)"
      else
        echo "mac: not running"
      fi
      ;;
    logs)
      tail -n 40 -f "$MAC_LOG_DIR/worker.out.log" "$MAC_LOG_DIR/worker.err.log"
      ;;
    *) echo "usage: ctl.sh mac on|off|status|logs" >&2; exit 1 ;;
  esac
}

pc() {
  case "$1" in
    on)
      ssh "$PC_HOST" "systemctl --user start $PC_UNIT" && echo "pc: started"
      ;;
    off)
      ssh "$PC_HOST" "systemctl --user stop $PC_UNIT" && echo "pc: stopped"
      ;;
    status)
      ssh "$PC_HOST" "systemctl --user is-active $PC_UNIT; systemctl --user show $PC_UNIT -p MainPID --value" \
        | paste -sd ' pid=' - | sed 's/^/pc: /'
      ;;
    logs)
      ssh "$PC_HOST" "journalctl --user -u $PC_UNIT -n 40 -f"
      ;;
    *) echo "usage: ctl.sh pc on|off|status|logs" >&2; exit 1 ;;
  esac
}

host="${1:-}"
cmd="${2:-status}"

case "$host" in
  mac) mac "$cmd" ;;
  pc)  pc  "$cmd" ;;
  all)
    if [[ "$cmd" != "status" ]]; then
      echo "'all' only supports status" >&2; exit 1
    fi
    mac status
    pc  status
    ;;
  *)
    cat <<EOF >&2
usage: $0 <host> <cmd>
  host: mac | pc | all
  cmd:  on | off | status | logs   (all only supports status)
EOF
    exit 1 ;;
esac
