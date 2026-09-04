#!/usr/bin/env bash
# Optional local watcher: when the VM reaches TERMINATED, pull results.
#
# Shutdown is NOT this script's job — the VM self-powers-off via
# remote_train.sh (--shutdown + max-hours watchdog). Your laptop can sleep.
#
# Pull strategy:
#   1) Try starting the training VM (needs GPU capacity).
#   2) If L4 is stocked out, detach the boot disk onto a cheap CPU-only VM,
#      pull, then reattach the disk to the training instance.
#
# Usage: gce_monitor.sh [poll_seconds]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./gce_config.sh

POLL="${1:-120}"
PULL_TMP="spine-pull-tmp"

wait_for_ssh() {
  local host="$1"
  for _ in $(seq 1 36); do
    if gcloud compute ssh "$host" --project="$PROJECT" --zone="$ZONE" \
         --command="echo ok" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

resolve_run_name() {
  local host="$1"
  local name
  name=$(gcloud compute ssh "$host" --project="$PROJECT" --zone="$ZONE" \
    --command="cd $REMOTE_REPO && (grep -m1 '^Run name:' full.log || grep -m1 '^Run name:' smoke.log || true) | sed 's/Run name: //'" || true)
  if [ -z "$name" ]; then
    name=$(gcloud compute ssh "$host" --project="$PROJECT" --zone="$ZONE" \
      --command="ls -1t $REMOTE_RUNS_ROOT 2>/dev/null | head -1" || true)
  fi
  echo "$name"
}

pull_from_host() {
  local host="$1"
  local run_name="$2"
  # Temporarily point sync at $host by overriding INSTANCE for one pull.
  INSTANCE="$host" ./gce_sync.sh pull-run "$run_name"
}

pull_via_gpu_start() {
  echo "Trying to start $INSTANCE for pull..."
  if ! gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$ZONE"; then
    return 1
  fi
  wait_for_ssh "$INSTANCE" || {
    echo "SSH did not come up after GPU start." >&2
    return 1
  }
  local run_name
  run_name=$(resolve_run_name "$INSTANCE")
  if [ -z "$run_name" ]; then
    echo "Could not determine run name on $INSTANCE" >&2
    return 1
  fi
  pull_from_host "$INSTANCE" "$run_name"
  echo "Pulled: $LOCAL_RUNS_ROOT/$run_name"
  echo "Re-stopping GPU instance..."
  ./gce_run.sh stop-instance
  return 0
}

pull_via_cpu_disk() {
  echo "GPU start unavailable — pulling via CPU-only temp VM on the same boot disk..."
  local disk
  disk=$(gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
    --format='get(disks[0].source.basename())')
  if [ -z "$disk" ]; then
    # Disk may already be detached from a prior attempt.
    disk="$INSTANCE"
  fi
  echo "boot disk=$disk"

  # Ensure temp VM is gone
  gcloud compute instances delete "$PULL_TMP" --project="$PROJECT" --zone="$ZONE" --quiet 2>/dev/null || true

  # Detach from terminated GPU VM (no-op if already detached)
  gcloud compute instances detach-disk "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --disk="$disk" 2>/dev/null || true

  gcloud compute instances create "$PULL_TMP" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type=e2-standard-2 \
    --disk="name=$disk,boot=yes,mode=rw"

  wait_for_ssh "$PULL_TMP" || {
    echo "SSH did not come up on $PULL_TMP" >&2
    return 1
  }

  local run_name
  run_name=$(resolve_run_name "$PULL_TMP")
  if [ -z "$run_name" ]; then
    echo "Could not determine run name on $PULL_TMP" >&2
    return 1
  fi
  pull_from_host "$PULL_TMP" "$run_name"
  echo "Pulled: $LOCAL_RUNS_ROOT/$run_name"

  echo "Deleting CPU pull VM and reattaching disk to $INSTANCE..."
  gcloud compute instances delete "$PULL_TMP" --project="$PROJECT" --zone="$ZONE" --quiet
  gcloud compute instances attach-disk "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" \
    --disk="$disk" --boot
  echo "Disk reattached. $INSTANCE remains TERMINATED (no GPU billing)."
  return 0
}

pull_and_restop() {
  if pull_via_gpu_start; then
    return 0
  fi
  echo "Falling back to CPU disk pull (common when L4 capacity is exhausted)..."
  pull_via_cpu_disk
}

echo "Watching $INSTANCE for TERMINATED (poll ${POLL}s)."
echo "Shutdown authority = VM (remote_train.sh). Ctrl-C only stops this watcher."

while true; do
  status=$(gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
    --format='value(status)' 2>/dev/null || echo "UNKNOWN")
  ts=$(date "+%H:%M:%S")

  case "$status" in
    TERMINATED|STOPPED)
      echo "[$ts] VM is $status — pulling results..."
      pull_and_restop
      exit 0
      ;;
    RUNNING)
      line=$(gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
        if [ -f $REMOTE_REPO/TRAIN_DONE ]; then echo \"DONE \$(cat $REMOTE_REPO/TRAIN_DONE)\"; exit 0; fi
        cd $REMOTE_REPO && tail -c 400 full.log 2>/dev/null | tr '\r' '\n' | tail -1
      " 2>/dev/null || echo "(ssh miss)")
      echo "[$ts] RUNNING: $line"
      ;;
    *)
      # After detach, instance may show no disks briefly; still treat as done if TERMINATED was seen.
      echo "[$ts] status=$status"
      ;;
  esac
  sleep "$POLL"
done
