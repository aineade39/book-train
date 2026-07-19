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
    # Ultralytics needs an absolute path on the VM (local spines.yaml has Mac paths).
    # Prefer val_train (stratified unrotated subset) for every-epoch training val;
    # full images/val stays for eval_rotation_sweep / spines.yaml.
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" \
      --command="
        if [ -d $REMOTE_DATA_ROOT/images/val_train ]; then
          VAL_REL=images/val_train
        else
          VAL_REL=images/val
        fi
        cat > $REMOTE_DATA_ROOT/spines_train.yaml <<EOF
path: $REMOTE_DATA_ROOT
train: images/train
val: \$VAL_REL
names:
  0: spine
EOF
        # Also refresh spines.yaml for sweeps (always full val).
        cat > $REMOTE_DATA_ROOT/spines.yaml <<EOF
path: $REMOTE_DATA_ROOT
train: images/train
val: images/val
names:
  0: spine
EOF
        echo wrote $REMOTE_DATA_ROOT/spines_train.yaml val=\$VAL_REL
        echo -n 'train '; find $REMOTE_DATA_ROOT/images/train -type f 2>/dev/null | wc -l
        echo -n 'val '; find $REMOTE_DATA_ROOT/images/val -type f 2>/dev/null | wc -l
        echo -n 'val_train '; find $REMOTE_DATA_ROOT/images/val_train -type f 2>/dev/null | wc -l
      "
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
