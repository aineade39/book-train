#!/usr/bin/env bash
# Usage:
#   gce_sync.sh push-code       # tools/*.py to ~/book-train
#   gce_sync.sh push-dataset    # derived combined dataset (no .npy cache)
#   gce_sync.sh pull-run <run_name>   # pull a run dir into $LOCAL_RUNS_ROOT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./gce_config.sh

# Prevent macOS from writing AppleDouble (._*) sidecar files into the tar
# stream for xattrs like com.apple.provenance -- these get misread as real
# images/labels by tools that glob by extension on the Linux side.
export COPYFILE_DISABLE=1

cmd="${1:-}"

case "$cmd" in
  push-code)
    echo "Pushing tools/ to $INSTANCE:$REMOTE_REPO/tools ..."
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
      --command="mkdir -p $REMOTE_REPO"
    tar -C "$(dirname "$(dirname "$(pwd)")")" --exclude='__pycache__' -cf - tools \
      | gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
          --command="tar -xf - -C $REMOTE_REPO"
    echo "Done."
    ;;

  push-dataset)
    if [ ! -d "$LOCAL_DATA_ROOT" ]; then
      echo "Missing $LOCAL_DATA_ROOT" >&2; exit 1
    fi
    remote_parent="$(dirname "$REMOTE_DATA_ROOT")"
    echo "Pushing dataset (excluding .npy cache) to $INSTANCE:$REMOTE_DATA_ROOT ..."
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
      --command="mkdir -p $remote_parent"
    tar -C "$(dirname "$LOCAL_DATA_ROOT")" --exclude='*.npy' -cf - "$(basename "$LOCAL_DATA_ROOT")" \
      | gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
          --command="tar -xf - -C $remote_parent"
    echo "Done."
    ;;

  pull-run)
    run_name="${2:-}"
    if [ -z "$run_name" ]; then
      echo "Usage: gce_sync.sh pull-run <run_name>" >&2; exit 1
    fi
    mkdir -p "$LOCAL_RUNS_ROOT"
    echo "Pulling $REMOTE_RUNS_ROOT/$run_name -> $LOCAL_RUNS_ROOT/$run_name ..."
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
      --command="tar --exclude='*.npy' -C $REMOTE_RUNS_ROOT -cf - '$run_name'" \
      | tar -xf - -C "$LOCAL_RUNS_ROOT"
    echo "Done. -> $LOCAL_RUNS_ROOT/$run_name"
    ;;

  *)
    echo "Usage: $0 {push-code|push-dataset|pull-run <run_name>}" >&2
    exit 1
    ;;
esac
