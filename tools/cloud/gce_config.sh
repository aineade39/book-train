#!/usr/bin/env bash
# Shared config for tools/cloud/*.sh — edit here, not in the individual scripts.
set -euo pipefail

export PROJECT="${PROJECT:-gen-lang-client-0891856003}"
export ZONE="${ZONE:-us-central1-b}"
export INSTANCE="${INSTANCE:-spine-train-l4}"
export MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
export ACCELERATOR="${ACCELERATOR:-type=nvidia-l4,count=1}"
export IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
export IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
export BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-250GB}"
export BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"

# Local <-> remote paths (mirrored layout so train_combined_obb.py's defaults
# work unmodified on both sides). Override BOOK_SPINES_DATA to relocate.
export BOOK_SPINES_DATA="${BOOK_SPINES_DATA:-$HOME/ml/book-spines}"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$BOOK_SPINES_DATA/derived/4tu-ieee_yolo-obb}"
export LOCAL_RUNS_ROOT="${LOCAL_RUNS_ROOT:-$BOOK_SPINES_DATA/runs}"
export REMOTE_REPO="${REMOTE_REPO:-~/book-train}"
export REMOTE_BOOK_SPINES_DATA="${REMOTE_BOOK_SPINES_DATA:-~/ml/book-spines}"
export REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-$REMOTE_BOOK_SPINES_DATA/derived/4tu-ieee_yolo-obb}"
export REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:-$REMOTE_BOOK_SPINES_DATA/runs}"
export REMOTE_VENV="${REMOTE_VENV:-~/venv-train}"

gce_ssh() {
  gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="$1"
}
