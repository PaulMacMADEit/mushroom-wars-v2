#!/usr/bin/env bash
# Install the worker as an unattended service on the current machine.
#
# Mac   → launchd user agent in ~/Library/LaunchAgents
# Linux → systemd user unit in ~/.config/systemd/user
#
# Usage (from repo root):
#   ./workers/install_service.sh                  # install + start the worker
#   ./workers/install_service.sh uninstall        # stop + remove the worker
#   ./workers/install_service.sh status           # show worker state
#   ./workers/install_service.sh logs             # tail worker logs
#   ./workers/install_service.sh install-cron     # install + start the 3h cron
#   ./workers/install_service.sh uninstall-cron   # stop + remove the cron
#   ./workers/install_service.sh status-cron      # show cron timer state
#   ./workers/install_service.sh logs-cron        # tail cron logs
#
# The .venv directory must already exist at repo root with the worker's deps
# installed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: $VENV_PYTHON not found or not executable" >&2
  echo "       create the venv first (python3 -m venv .venv && .venv/bin/pip install ...)" >&2
  exit 1
fi

cmd="${1:-install}"
os="$(uname -s)"

# ---------------------------------------------------------------------------
# macOS (launchd)
# ---------------------------------------------------------------------------
if [[ "$os" == "Darwin" ]]; then
  LABEL="com.mushroomwars.worker"
  PLIST_SRC="$REPO/workers/launchd/com.mushroomwars.worker.plist.template"
  PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOG_DIR="$HOME/Library/Logs/mushroom-worker"
  UID_NUM="$(id -u)"
  DOMAIN="gui/$UID_NUM"

  case "$cmd" in
    install)
      mkdir -p "$LOG_DIR" "$(dirname "$PLIST_DST")"
      sed -e "s|{{REPO}}|$REPO|g" \
          -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
          -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
          "$PLIST_SRC" > "$PLIST_DST"
      # bootout is a no-op if not loaded; suppress its error.
      launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
      launchctl bootstrap "$DOMAIN" "$PLIST_DST"
      launchctl enable "$DOMAIN/$LABEL"
      launchctl kickstart -k "$DOMAIN/$LABEL"
      echo "installed $LABEL (launchd), logs in $LOG_DIR"
      ;;
    uninstall)
      launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
      rm -f "$PLIST_DST"
      echo "uninstalled $LABEL"
      ;;
    status)
      launchctl print "$DOMAIN/$LABEL" 2>&1 | grep -E "state|pid|last exit|program" | head -10 || echo "not loaded"
      ;;
    logs)
      tail -n 40 -f "$LOG_DIR/worker.out.log" "$LOG_DIR/worker.err.log"
      ;;
    *)
      echo "unknown command: $cmd" >&2
      exit 1
      ;;
  esac

# ---------------------------------------------------------------------------
# Linux (systemd user)
# ---------------------------------------------------------------------------
elif [[ "$os" == "Linux" ]]; then
  UNIT="mushroom-worker.service"
  UNIT_SRC="$REPO/workers/systemd/mushroom-worker.service.template"
  UNIT_DIR="$HOME/.config/systemd/user"
  UNIT_DST="$UNIT_DIR/$UNIT"
  LOG_DIR="$HOME/.local/log/mushroom-worker"

  case "$cmd" in
    install)
      mkdir -p "$LOG_DIR" "$UNIT_DIR"
      sed -e "s|{{REPO}}|$REPO|g" \
          -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
          -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
          "$UNIT_SRC" > "$UNIT_DST"
      systemctl --user daemon-reload
      systemctl --user enable "$UNIT"
      # restart (not --now) so a re-install picks up template changes on a
      # service that was already running.
      systemctl --user restart "$UNIT"
      # Keep service running after logout. Ignore if already enabled.
      loginctl enable-linger "$(whoami)" 2>/dev/null || true
      echo "installed $UNIT (systemd --user), logs in $LOG_DIR"
      ;;
    uninstall)
      systemctl --user disable --now "$UNIT" 2>/dev/null || true
      rm -f "$UNIT_DST"
      systemctl --user daemon-reload
      echo "uninstalled $UNIT"
      ;;
    status)
      systemctl --user status "$UNIT" --no-pager || true
      ;;
    logs)
      journalctl --user -u "$UNIT" -n 40 -f
      ;;
    install-cron)
      CRON_SVC="mushroom-cron.service"
      CRON_TIMER="mushroom-cron.timer"
      CRON_SVC_SRC="$REPO/workers/systemd/mushroom-cron.service.template"
      CRON_TIMER_SRC="$REPO/workers/systemd/mushroom-cron.timer.template"
      mkdir -p "$LOG_DIR" "$UNIT_DIR"
      sed -e "s|{{REPO}}|$REPO|g" \
          -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
          -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
          "$CRON_SVC_SRC" > "$UNIT_DIR/$CRON_SVC"
      cp "$CRON_TIMER_SRC" "$UNIT_DIR/$CRON_TIMER"
      systemctl --user daemon-reload
      systemctl --user enable --now "$CRON_TIMER"
      loginctl enable-linger "$(whoami)" 2>/dev/null || true
      echo "installed $CRON_TIMER (systemd --user, fires every 3h at :07)"
      systemctl --user list-timers "$CRON_TIMER" --no-pager
      ;;
    uninstall-cron)
      systemctl --user disable --now "mushroom-cron.timer" 2>/dev/null || true
      rm -f "$UNIT_DIR/mushroom-cron.timer" "$UNIT_DIR/mushroom-cron.service"
      systemctl --user daemon-reload
      echo "uninstalled mushroom-cron.timer"
      ;;
    status-cron)
      systemctl --user status mushroom-cron.timer --no-pager || true
      systemctl --user list-timers mushroom-cron.timer --no-pager 2>/dev/null || true
      ;;
    logs-cron)
      journalctl --user -u mushroom-cron.service -n 40 -f
      ;;
    install-rerate)
      RERATE_SVC="mushroom-rerate.service"
      RERATE_TIMER="mushroom-rerate.timer"
      RERATE_SVC_SRC="$REPO/workers/systemd/mushroom-rerate.service.template"
      RERATE_TIMER_SRC="$REPO/workers/systemd/mushroom-rerate.timer.template"
      mkdir -p "$LOG_DIR" "$UNIT_DIR"
      sed -e "s|{{REPO}}|$REPO|g" \
          -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
          -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
          "$RERATE_SVC_SRC" > "$UNIT_DIR/$RERATE_SVC"
      cp "$RERATE_TIMER_SRC" "$UNIT_DIR/$RERATE_TIMER"
      systemctl --user daemon-reload
      systemctl --user enable --now "$RERATE_TIMER"
      loginctl enable-linger "$(whoami)" 2>/dev/null || true
      echo "installed $RERATE_TIMER (systemd --user, fires Sun 02:13 weekly)"
      systemctl --user list-timers "$RERATE_TIMER" --no-pager
      ;;
    uninstall-rerate)
      systemctl --user disable --now "mushroom-rerate.timer" 2>/dev/null || true
      rm -f "$UNIT_DIR/mushroom-rerate.timer" "$UNIT_DIR/mushroom-rerate.service"
      systemctl --user daemon-reload
      echo "uninstalled mushroom-rerate.timer"
      ;;
    run-rerate)
      systemctl --user start mushroom-rerate.service
      echo "kicked off mushroom-rerate.service"
      systemctl --user status mushroom-rerate.service --no-pager || true
      ;;
    status-rerate)
      systemctl --user status mushroom-rerate.timer --no-pager || true
      systemctl --user list-timers mushroom-rerate.timer --no-pager 2>/dev/null || true
      ;;
    logs-rerate)
      journalctl --user -u mushroom-rerate.service -n 40 -f
      ;;
    *)
      echo "unknown command: $cmd" >&2
      exit 1
      ;;
  esac

else
  echo "unsupported OS: $os" >&2
  exit 1
fi
