---
name: book-show-api-scrape-counts
description: >-
  Answers how many Goodreads book_show_api / ISBN scrapes ran in a time window
  (last 24h, today, etc.). Use when the user asks about ISBN scrape volume,
  book_show_api progress, scrape rate over hours/days, or how many books were
  fetched recently. Runs tools/catalog/book_show_api_progress.py instead of
  estimating from jsonl mtimes or terminal scrollback.
---

# book_show_api scrape counts

## When to use

Any question like:

- “How many ISBN scrapes in the last 24 hours?”
- “How many book_show_api fetches today?”
- “Scrape progress / throughput over the last N hours?”

## Do this

From the book-train repo root:

```bash
.venv/bin/python tools/catalog/book_show_api_progress.py --hours 24
```

Optional JSON:

```bash
.venv/bin/python tools/catalog/book_show_api_progress.py summarize --hours 24 --json
```

Change `--hours` to match the window the user asked for.

## How to report

Prefer **`isbn_ok`** when they say “ISBN scrapes”; mention **`attempts`** and **`warnings`** if useful.

If the summary notes that the progress log is empty, say counts only cover chunks after the loop started recording (`mark-start` / `record` in `tools/run_book_show_api_loop.sh`). Do not invent estimates from jsonl size unless the user asks for an estimate.

## Background

- Output JSONL: `$BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl` (no per-row timestamps).
- Progress sidecar: `book_show_api_progress.jsonl` (+ `book_show_api_chunk_open.json` while a chunk is in flight).
- Orchestrator: `tools/run_book_show_api_loop.sh` (calls `mark-start` / `record` around each scrape).
- Detail: `docs/BOOK_CATALOG.md` (ISBN / book_show_api section).

If the running loop was started before these hooks existed, restart it (safe; JSONL is append-only) so new chunks are recorded. Mid-flight: `mark-start --since-line N` can seed the open marker; when that chunk ends without the new loop, run `record --since-line N` once.
