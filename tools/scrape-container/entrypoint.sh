#!/usr/bin/env bash
set -euo pipefail

echo "[scrape-container] book-train.rev=${BOOK_TRAIN_REV:-unknown} scrape-harness.rev=${HARNESS_REV:-unknown}"

JSONL="/data/book-spines/derived/book-catalog/goodreads/book_show_api.jsonl"
MARKER="/data/book-spines/derived/book-catalog/goodreads/container_migration.json"

if [[ ! -s "$JSONL" ]]; then
  echo "[scrape-container] missing or empty $JSONL — refusing to scrape" >&2
  exit 1
fi
if [[ ! -f "$MARKER" ]]; then
  echo "[scrape-container] missing $MARKER — run prepare on the host first" >&2
  exit 1
fi

expected_bytes="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bytes"])' "$MARKER")"
expected_lines="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lines"])' "$MARKER")"
actual_bytes="$(wc -c < "$JSONL" | tr -d ' ')"
actual_lines="$(wc -l < "$JSONL" | tr -d ' ')"
if [[ "$actual_bytes" -lt "$expected_bytes" || "$actual_lines" -lt "$expected_lines" ]]; then
  echo "[scrape-container] JSONL shrank vs marker (bytes $actual_bytes<$expected_bytes lines $actual_lines<$expected_lines) — refusing" >&2
  exit 1
fi

export DISPLAY=:99
cd /app/book-train
.venv/bin/python tools/catalog/ensure_xvfb.py ensure || exit 1
exec tools/run_book_show_api_loop.sh
