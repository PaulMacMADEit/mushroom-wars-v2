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
MAC_LABEL="com.mushroomwars.worker"
MAC_DOMAIN="gui/$(id -u)"
MAC_PLIST="$HOME/Library/LaunchAgents/$MAC_LABEL.plist"
PC_UNIT="mushroom-worker.service"

mac() {
  case "$1" in
    on)
      if [[ ! -f "$MAC_PLIST" ]]; then
        echo "mac: plist missing — run './workers/install_service.sh install' first" >&2
        exit 1
      fi
      # bootstrap is idempotent-ish: errors if already loaded, so ignore.
      launchctl bootstrap "$MAC_DOMAIN" "$MAC_PLIST" 2>/dev/null || true
      launchctl kickstart -k "$MAC_DOMAIN/$MAC_LABEL"
      echo "mac: started"
      ;;
    off)
      # KeepAlive=true means `stop` auto-restarts; bootout is the only way
      # to keep it down. Service stays registered via the plist file.
      launchctl bootout "$MAC_DOMAIN/$MAC_LABEL" 2>/dev/null || true
      echo "mac: stopped"
      ;;
    status)
      if info=$(launchctl print "$MAC_DOMAIN/$MAC_LABEL" 2>/dev/null); then
        pid=$(echo "$info"   | awk -F= '/[[:space:]]pid[[:space:]]*=/ {gsub(/[[:space:]]/,"",$2); print $2; exit}')
        state=$(echo "$info" | awk -F= '/[[:space:]]state[[:space:]]*=/ {sub(/^[[:space:]]*/,"",$2); print $2; exit}')
        echo "mac: ${state:-unknown} pid=${pid:-none}"
      else
        echo "mac: not loaded (run './workers/ctl.sh mac on' to start, or install_service.sh to register)"
      fi
      ;;
    logs)
      tail -n 40 -f ~/Library/Logs/mushroom-worker/worker.out.log ~/Library/Logs/mushroom-worker/worker.err.log
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
