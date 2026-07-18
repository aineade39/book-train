#!/usr/bin/env bash
# Poll the GPUS_ALL_REGIONS quota until Google approves the increase, then
# automatically provision the VM, push code+dataset, run setup, and launch
# the smoke test. Meant to be left running in the background after you've
# filed the quota increase request in the console.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./gce_config.sh

POLL="${1:-300}"  # seconds between quota checks

check_quota() {
  gcloud compute project-info describe --project="$PROJECT" --format="json(quotas)" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('0'); sys.exit()
for q in d.get('quotas', []):
    if q['metric'] == 'GPUS_ALL_REGIONS':
        print(q['limit'])
        sys.exit()
print('0')
"
}

echo "Polling GPUS_ALL_REGIONS quota on $PROJECT every ${POLL}s until it's > 0..."
while true; do
  limit=$(check_quota)
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  if [ "$limit" != "0" ] && [ "$limit" != "0.0" ] && [ -n "$limit" ]; then
    echo "[$ts] Quota granted (limit=$limit). Proceeding with provisioning..."
    break
  fi
  echo "[$ts] still 0, waiting..."
  sleep "$POLL"
done

set -e
echo "=== provision ==="
./gce_provision.sh

echo "=== push code ==="
./gce_sync.sh push-code

echo "=== push dataset ==="
./gce_sync.sh push-dataset

echo "=== remote setup (venv + deps) ==="
./gce_run.sh setup

echo "=== launch smoke test ==="
./gce_run.sh smoke

echo "QUOTA_GRANTED_AND_SMOKE_LAUNCHED"
