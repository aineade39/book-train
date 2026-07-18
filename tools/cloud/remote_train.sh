#!/usr/bin/env bash
# Runs on the VM. Primary shutdown authority — does NOT depend on your laptop.
#
# Usage (on VM):
#   remote_train.sh [--shutdown] [--max-hours N] -- <train args...>
#
# --shutdown   power off the instance when training exits (success OR failure)
# --max-hours  hard failsafe: power off after N hours even if training is hung
#              (default 14). Set 0 to disable.
set -uo pipefail

SHUTDOWN=0
MAX_HOURS=14
WATCHDOG_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shutdown) SHUTDOWN=1; shift ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

VENV="${REMOTE_VENV:-$HOME/venv-train}"
REPO="${REMOTE_REPO:-$HOME/book-train}"
DONE_MARKER="$REPO/TRAIN_DONE"
cd "$REPO"

# Hard failsafe: wall-clock kill switch independent of the trainer.
if [[ "$MAX_HOURS" -gt 0 ]]; then
  (
    sleep $((MAX_HOURS * 3600))
    echo "$(date -Is) WATCHDOG: max-hours=$MAX_HOURS reached — powering off" | tee -a watchdog.log
    sudo poweroff
  ) &
  WATCHDOG_PID=$!
  echo "$WATCHDOG_PID" > watchdog.pid
  echo "Watchdog armed: poweroff after ${MAX_HOURS}h (pid=$WATCHDOG_PID)"
fi

cleanup() {
  local ec=$?
  # Stop watchdog so a clean finish doesn't race a later poweroff from sleep.
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  date -Is > "$DONE_MARKER"
  echo "exit_code=$ec" >> "$DONE_MARKER"
  echo "$(date -Is) training exited with code $ec" | tee -a train_wrapper.log
  if [[ "$SHUTDOWN" -eq 1 ]]; then
    echo "$(date -Is) --shutdown set: powering off in 30s (cancel with: sudo pkill -f 'poweroff')" | tee -a train_wrapper.log
    sleep 30
    sudo poweroff
  fi
  exit "$ec"
}
trap cleanup EXIT

rm -f "$DONE_MARKER"
echo "$(date -Is) starting: $*" | tee -a train_wrapper.log
"$VENV/bin/python" tools/train_combined_obb.py "$@"
# exit trap handles shutdown
