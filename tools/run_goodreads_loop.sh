#!/usr/bin/env bash
# Repeatedly invokes scrape_goodreads_lists.py until every seeded list is
# done/deprecated (no more pending), then exits. Meant to be run under
# `caffeinate` for long unattended sessions -- see AGENTS.md / BOOK_CATALOG.md
# for the underlying orchestrator's session caps, retry backoff, and
# soft-block escalation, all of which still apply here.
set -euo pipefail
cd "$(dirname "$0")/.."

while true; do
  .venv/bin/python tools/scrape_goodreads_lists.py --max-attempts 20
  if .venv/bin/python tools/scrape_goodreads_lists.py --report-only 2>&1 | grep -q '"pending"'; then
    echo "[run_goodreads_loop] more pending — sleeping 60s before next session"
    sleep 60
  else
    echo "[run_goodreads_loop] all lists done — stopping loop"
    break
  fi
done
