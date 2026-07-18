#!/usr/bin/env bash
# Create (idempotent) the on-demand L4 training VM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./gce_config.sh

if gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  echo "Instance $INSTANCE already exists in $ZONE. Status:"
  gcloud compute instances describe "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --format="value(status)"
  exit 0
fi

echo "Creating $INSTANCE ($MACHINE_TYPE, $ACCELERATOR, on-demand) in $ZONE..."
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="$BOOT_DISK_TYPE" \
  --metadata="install-nvidia-driver=True"

echo "Waiting for SSH to come up..."
for i in $(seq 1 30); do
  if gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="echo ssh-ok" 2>/dev/null | grep -q ssh-ok; then
    echo "SSH is up."
    exit 0
  fi
  sleep 10
done
echo "SSH did not come up in time; try 'gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE' manually." >&2
exit 1
