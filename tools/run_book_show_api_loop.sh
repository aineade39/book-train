#!/usr/bin/env bash
# Repeatedly runs extract_remaining_ids.py -> scrape-harness doctor -> a
# session-chunked book_show_api scrape until no book IDs remain. Meant to run
# under `caffeinate -i` for long unattended sessions -- see docs/BOOK_CATALOG.md.
#
# Safe to interrupt and re-run at any point: scrape-harness appends to --out
# rather than truncating it on restart, and extract_remaining_ids.py only
# ever asks for the delta not already fetched (across every book_show_api*
# shard file, so an interrupted session under a different --out name is
# never stranded). ids_remaining.txt lists never-tried book IDs first,
# popularity-ordered within each bucket (highest ratings_count first; see
# docs/BOOK_CATALOG.md).
#
# DELAY_MS_MIN/DELAY_MS_MAX override the base profile's delay range (default:
# inherit book_show_api.yaml's 3000-8000ms). Only set these after
# tools/catalog/bench_book_show_api_pacing.py confirms the faster tier holds
# parity with baseline -- see docs/BOOK_CATALOG.md's "Pacing benchmark" section.
# Failure spikes: harness circuit-breaks 404 streaks in-session (8 consecutive)
# and mixed incomplete_record via a rolling rate / delay-cap (exit 3).
# session_health classify/recover (catalog repair, stale-build rediscover with
# 60s/180s retries, then 15m→45m→2h soft-block escalation; with MULLVAD_ROTATE=1
# rotate the exit instead of a hard-stop for a manual VPN change).
# Exit 3: escalate-cooldown on the same 15m→45m→2h ladder, no Mullvad rotate.
# Each exit: wipe book_show_api_chunked, warmup home → fantasy list → LOTR,
# discover the Next.js build id in that profile, then scrape with the same dir.
# Discover fail rotates immediately; a full city-pool walk sleeps 15m; a second
# walk hard-stops.
# Progress: each chunk mark-start/record via tools/catalog/book_show_api_progress.py
# (book_show_api_progress.jsonl) so last-N-hour ISBN scrape counts are exact.
set -euo pipefail
cd "$(dirname "$0")/.."

HARNESS_ROOT="${HARNESS_ROOT:-$(cd .. && pwd)/scrape-harness}"
MAX_REQUESTS="${MAX_REQUESTS:-120}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"
DOCTOR_RETRY_SECONDS="${DOCTOR_RETRY_SECONDS:-900}"
# After each chunk, if >= SPIKE_FAILURE_THRESHOLD of new JSONL rows are warnings
# (default 80%), or the chunk ends with >= SPIKE_TRAILING_ABORT consecutive
# warnings (default 8, matching the harness stale-build circuit), classify
# (catalog / stale_build / soft_block / hard_block) and recover.
# Stale-build: rediscover immediately, then 60s/180s retries, then the
# soft-block ladder 15m → 45m → 2h (SPIKE_MAX_TIERS, default 3)
# then hard-stop for a manual VPN change (or Mullvad rotate when MULLVAD_ROTATE=1). See docs/BOOK_CATALOG.md.
SPIKE_FAILURE_THRESHOLD="${SPIKE_FAILURE_THRESHOLD:-0.80}"
SPIKE_MIN_CHUNK_RECORDS="${SPIKE_MIN_CHUNK_RECORDS:-20}"
SPIKE_TRAILING_ABORT="${SPIKE_TRAILING_ABORT:-8}"
SPIKE_MAX_TIERS="${SPIKE_MAX_TIERS:-3}"
# Optional pacing override -- only set these after tools/catalog/bench_book_show_api_pacing.py
# confirms the faster tier holds parity with baseline on a benchmark sample.
DELAY_MS_MIN="${DELAY_MS_MIN:-}"
DELAY_MS_MAX="${DELAY_MS_MAX:-}"
delay_args=()
[[ -n "$DELAY_MS_MIN" ]] && delay_args+=(--delay-ms-min "$DELAY_MS_MIN")
[[ -n "$DELAY_MS_MAX" ]] && delay_args+=(--delay-ms-max "$DELAY_MS_MAX")
if [[ -n "$DELAY_MS_MIN" || -n "$DELAY_MS_MAX" ]]; then
  echo "[run_book_show_api_loop] delay override active: ${DELAY_MS_MIN:-<base>}-${DELAY_MS_MAX:-<base>}ms"
fi

BOOK_SHOW_API_PATH="$(.venv/bin/python -c 'from tools.paths import catalog_goodreads; print(catalog_goodreads("book_show_api.jsonl"))')"
IDS_REMAINING_PATH="$(.venv/bin/python -c 'from tools.paths import catalog_goodreads; print(catalog_goodreads("ids_remaining.txt"))')"

# New process: allow a full city-pool walk even if a previous run hard-stopped.
.venv/bin/python tools/catalog/book_show_api_exit_session.py reset-fail || true

while true; do
  .venv/bin/python tools/catalog/extract_remaining_ids.py
  remaining="$(wc -l < "$IDS_REMAINING_PATH" | tr -d ' ')"

  if [[ "$remaining" == "0" ]]; then
    echo "[run_book_show_api_loop] no book IDs remaining -- stopping loop"
    break
  fi

  .venv/bin/python tools/catalog/write_book_show_api_chunked_profile.py \
    --harness-root "$HARNESS_ROOT" --max-requests "$MAX_REQUESTS" "${delay_args[@]}"

  if ! (cd "$HARNESS_ROOT" && .venv/bin/scrape-harness doctor); then
    echo "[run_book_show_api_loop] doctor check failed -- sleeping ${DOCTOR_RETRY_SECONDS}s before retry"
    sleep "$DOCTOR_RETRY_SECONDS"
    continue
  fi

  if ! .venv/bin/python tools/catalog/ensure_xvfb.py ensure; then
    echo "[run_book_show_api_loop] display :99 down after ensure — stopping (not rotating)"
    exit 1
  fi

  prepare_rc=0
  .venv/bin/python tools/catalog/book_show_api_exit_session.py prepare-exit \
      --harness-root "$HARNESS_ROOT" || prepare_rc=$?

  if [[ "$prepare_rc" == "4" ]]; then
    echo "[run_book_show_api_loop] Chrome exited: no X display (not a discover/VPN miss)"
    if ! .venv/bin/python tools/catalog/ensure_xvfb.py ensure; then
      echo "[run_book_show_api_loop] display :99 down after ensure — stopping (not rotating)"
      exit 1
    fi
    prepare_rc=0
    .venv/bin/python tools/catalog/book_show_api_exit_session.py prepare-exit \
        --harness-root "$HARNESS_ROOT" || prepare_rc=$?
    if [[ "$prepare_rc" == "4" ]]; then
      echo "[run_book_show_api_loop] Chrome still has no X display after ensure — stopping (not rotating)"
      exit 1
    fi
  fi

  if [[ "$prepare_rc" != "0" ]]; then
    echo "[run_book_show_api_loop] prepare-exit (home → list → discover) failed"
    fail_rc=0
    .venv/bin/python tools/catalog/book_show_api_exit_session.py next-fail-action || fail_rc=$?
    if [[ "$fail_rc" == "2" ]]; then
      echo "[run_book_show_api_loop] discover failed across the city pool twice — stopping"
      exit 1
    fi
    if [[ "$fail_rc" == "3" ]]; then
      echo "[run_book_show_api_loop] city pool exhausted — sleeping ${DOCTOR_RETRY_SECONDS}s before rotating"
      sleep "$DOCTOR_RETRY_SECONDS"
    fi
    echo "[run_book_show_api_loop] rotating Mullvad after discover fail"
    .venv/bin/python tools/catalog/mullvad_rotate.py \
      || echo "[run_book_show_api_loop] mullvad rotate failed — will retry on current exit"
    continue
  fi
  .venv/bin/python tools/catalog/book_show_api_exit_session.py reset-fail || true

  lines_before=0
  if [[ -f "$BOOK_SHOW_API_PATH" ]]; then
    lines_before="$(wc -l < "$BOOK_SHOW_API_PATH" | tr -d ' ')"
  fi

  # Timestamped progress sidecar so "scrapes in last 24h" is a fast lookup
  # (book_show_api.jsonl has no per-row timestamps). Mid-chunk counts use the
  # open marker; closed chunks land in book_show_api_progress.jsonl.
  .venv/bin/python tools/catalog/book_show_api_progress.py mark-start \
    --jsonl "$BOOK_SHOW_API_PATH" --since-line "$lines_before"

  scrape_rc=0
  (cd "$HARNESS_ROOT" && .venv/bin/scrape-harness scrape goodreads \
    --profile book_show_api_chunked \
    --set-from-file "book_id=$IDS_REMAINING_PATH" \
    --out "$BOOK_SHOW_API_PATH") \
    || scrape_rc=$?

  .venv/bin/python tools/catalog/book_show_api_progress.py record \
    --jsonl "$BOOK_SHOW_API_PATH" --since-line "$lines_before"

  if [[ "$scrape_rc" == "3" ]]; then
    echo "[run_book_show_api_loop] controlled stop (exit 3) — cooling down without exit rotation"
    cooldown_rc=0
    SPIKE_MAX_TIERS="$SPIKE_MAX_TIERS" .venv/bin/python tools/catalog/book_show_api_session_health.py escalate-cooldown \
      --max-tiers "$SPIKE_MAX_TIERS" || cooldown_rc=$?
    if [[ "$cooldown_rc" == "2" ]]; then
      echo "[run_book_show_api_loop] hard stop after repeated controlled stops"
      exit 1
    fi
    delay_args=()
    continue
  fi
  if [[ "$scrape_rc" != "0" ]]; then
    echo "[run_book_show_api_loop] scrape session exited ${scrape_rc} -- will retry next iteration"
  fi

  if ! .venv/bin/python tools/catalog/book_show_api_session_health.py check \
      --jsonl "$BOOK_SHOW_API_PATH" \
      --since-line "$lines_before" \
      --min-chunk-records "$SPIKE_MIN_CHUNK_RECORDS" \
      --failure-threshold "$SPIKE_FAILURE_THRESHOLD" \
      --trailing-abort "$SPIKE_TRAILING_ABORT"; then
    echo "[run_book_show_api_loop] failure spike detected -- recovering"
    recover_rc=0
    SPIKE_MAX_TIERS="$SPIKE_MAX_TIERS" .venv/bin/python tools/catalog/book_show_api_session_health.py recover \
      --jsonl "$BOOK_SHOW_API_PATH" \
      --since-line "$lines_before" \
      --harness-root "$HARNESS_ROOT" \
      --max-tiers "$SPIKE_MAX_TIERS" || recover_rc=$?
    if [[ "$recover_rc" == "2" ]]; then
      echo "[run_book_show_api_loop] hard stop — change VPN exit, then re-run"
      exit 1
    fi
    # recover already slept for the tier; drop DELAY_MS_* so the next chunk
    # uses book_show_api.yaml baseline pacing (3000-8000ms).
    delay_args=()
    echo "[run_book_show_api_loop] recovered (rc=$recover_rc) — next chunk at baseline pacing"
    continue
  fi

  .venv/bin/python tools/catalog/book_show_api_session_health.py reset-state || true
  .venv/bin/python tools/catalog/book_show_api_exit_session.py reset-fail || true

  if [[ "${MULLVAD_ROTATE:-}" == "1" ]]; then
    echo "[run_book_show_api_loop] rotating Mullvad exit before cooldown"
    .venv/bin/python tools/catalog/mullvad_rotate.py --require-enabled \
      || echo "[run_book_show_api_loop] mullvad rotate failed — continuing on current exit"
  fi

  echo "[run_book_show_api_loop] session done -- sleeping ${COOLDOWN_SECONDS}s before next chunk"
  sleep "$COOLDOWN_SECONDS"
done
