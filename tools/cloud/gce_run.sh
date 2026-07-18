#!/usr/bin/env bash
# Usage:
#   gce_run.sh setup                # one-time: venv + pip installs on the VM
#   gce_run.sh smoke                # short sanity run (no auto-shutdown)
#   gce_run.sh full [extra args...] # full run; VM self-powers-off when done
#   gce_run.sh status               # trainer / DONE marker / VM state
#   gce_run.sh log [n]              # tail last n lines (default 60) of the active log
#   gce_run.sh stop-instance        # gcloud stop (keeps disk, halts compute+GPU billing)
#
# Shutdown model (most reliable):
#   The VM powers itself off via remote_train.sh --shutdown + a max-hours
#   watchdog. Your laptop is NOT in the critical path. Local gce_monitor.sh
#   is optional (pull results after TERMINATED).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./gce_config.sh

# Hard failsafe wall clock for a full run (hours). Override: MAX_HOURS=18 ./gce_run.sh full
MAX_HOURS="${MAX_HOURS:-14}"

cmd="${1:-}"
shift || true

case "$cmd" in
  setup)
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
      set -e
      sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv libgl1
      python3 -m venv $REMOTE_VENV
      $REMOTE_VENV/bin/pip install --upgrade pip
      $REMOTE_VENV/bin/pip install ultralytics opencv-python-headless pyyaml
      $REMOTE_VENV/bin/python -c 'import torch; print(\"cuda available:\", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
      chmod +x $REMOTE_REPO/tools/cloud/remote_train.sh 2>/dev/null || true
    "
    ;;

  smoke)
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
      set -e
      cd $REMOTE_REPO
      chmod +x tools/cloud/remote_train.sh
      rm -f smoke.log TRAIN_DONE
      # No --shutdown: smoke is interactive; don't kill the VM mid-debug.
      nohup tools/cloud/remote_train.sh --max-hours 2 -- \
        --smoke --skip-export \
        --device 0 --batch 16 \
        --data-yaml $REMOTE_DATA_ROOT/spines_train.yaml \
        --runs-out $REMOTE_RUNS_ROOT \
        > smoke.log 2>&1 < /dev/null &
      disown
      echo 'launched smoke (no auto-shutdown), pid='\$!
    "
    ;;

  full)
    extra_args="$*"
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
      set -e
      cd $REMOTE_REPO
      chmod +x tools/cloud/remote_train.sh
      rm -f full.log TRAIN_DONE train_wrapper.log watchdog.log
      # VM is shutdown authority: EXIT trap + ${MAX_HOURS}h watchdog.
      nohup tools/cloud/remote_train.sh --shutdown --max-hours $MAX_HOURS -- \
        --skip-export \
        --device 0 --batch 16 \
        --data-yaml $REMOTE_DATA_ROOT/spines_train.yaml \
        --runs-out $REMOTE_RUNS_ROOT \
        $extra_args \
        > full.log 2>&1 < /dev/null &
      disown
      echo 'launched full (auto-poweroff on exit + ${MAX_HOURS}h watchdog), pid='\$!
    "
    ;;

  status)
    echo "VM: $(gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --format='value(status)' 2>/dev/null || echo unreachable)"
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
      echo -n 'trainer: '
      pgrep -af '[p]ython.*train_combined_obb' || echo 'not running'
      echo -n 'DONE marker: '
      if [ -f $REMOTE_REPO/TRAIN_DONE ]; then cat $REMOTE_REPO/TRAIN_DONE; else echo '(none)'; fi
      echo -n 'watchdog: '
      if [ -f $REMOTE_REPO/watchdog.pid ] && kill -0 \$(cat $REMOTE_REPO/watchdog.pid) 2>/dev/null; then
        echo \"armed pid=\$(cat $REMOTE_REPO/watchdog.pid)\"
      else
        echo '(none)'
      fi
    " 2>/dev/null || echo "(VM not reachable — if status=TERMINATED, run finished and self-halted)"
    ;;

  log)
    n="${1:-60}"
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
      cd $REMOTE_REPO
      f=full.log; [ -f \$f ] || f=smoke.log
      tail -n $n \$f 2>/dev/null | tr '\r' '\n' | tail -n $n
    "
    ;;

  stop-instance)
    echo "Stopping $INSTANCE (disk retained, compute+GPU billing stops)..."
    gcloud compute instances stop "$INSTANCE" --project="$PROJECT" --zone="$ZONE"
    ;;

  *)
    echo "Usage: $0 {setup|smoke|full [args]|status|log [n]|stop-instance}" >&2
    exit 1
    ;;
esac
