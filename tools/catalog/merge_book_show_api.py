#!/usr/bin/env python3
"""Dedupe-merge one or more book_show_api JSONL shards into a single
canonical file, keyed on `legacy_id`.

Bridges scrape-harness's append-only, restart-safe book_show_api output
(which can end up split across several files — e.g. an ad hoc `--out`
rename during a session, or a chunked orchestrator writing to a fresh path)
and `tools/catalog/match_goodreads.py`, which reads exactly one JSONL of
`{legacy_id, isbn13, ...}` records via `--book-show-api`.

Idempotent and safe to re-run: merging the same inputs twice produces the
same output, and passing the existing canonical file back in as one of the
inputs (the common case — see the default below) just re-confirms it.

Usage:
    python tools/catalog/merge_book_show_api.py --out book_show_api.jsonl \\
        book_show_api.jsonl book_show_api.part1.jsonl book_show_api.batch2.jsonl

    # Merge every catalog_goodreads('book_show_api*.jsonl') shard in place:
    python tools/catalog/merge_book_show_api.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.extract_remaining_ids import default_book_show_api_paths  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402


def _iter_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def merge_records(paths: list[Path]) -> list[dict]:
    """Merge records from `paths`, oldest-modified first, keyed on `legacy_id` —
    a later (more recently modified) shard's record for the same `legacy_id`
    overwrites an earlier one. Records with no usable `legacy_id` (e.g.
    `_scrape_warning` records) are kept too, deduped by `_url` so re-running
    the merge doesn't pile up duplicate warning lines — but the number of
    attempts is preserved rather than discarded: repeated warnings for the
    same `_url` accumulate into an `_attempt_count` field (the newest
    warning's content otherwise wins), which
    `tools/catalog/extract_remaining_ids.py`'s give-up tracking relies on to
    keep counting correctly across a manual merge. The accumulation is
    idempotent: re-merging an already-merged file preserves its existing
    `_attempt_count`.
    """
    ordered_paths = sorted((p for p in paths if p.exists()), key=lambda p: p.stat().st_mtime)

    by_legacy_id: dict[int, dict] = {}
    warning_records: dict[str, dict] = {}

    for path in ordered_paths:
        for record in _iter_records(path):
            legacy_id = record.get("legacy_id")
            if legacy_id is not None:
                try:
                    by_legacy_id[int(legacy_id)] = record
                    continue
                except (TypeError, ValueError):
                    pass
            key = record.get("_url") or json.dumps(record, sort_keys=True)
            prior = warning_records.get(key)
            prior_count = prior.get("_attempt_count", 1) if prior else 0
            merged_record = dict(record)
            merged_record["_attempt_count"] = prior_count + record.get("_attempt_count", 1)
            warning_records[key] = merged_record

    merged = list(by_legacy_id.values()) + list(warning_records.values())
    merged.sort(key=lambda r: (r.get("legacy_id") is None, r.get("legacy_id") or 0))
    return merged


def write_merged(records: list[dict], out_path: Path) -> None:
    """Atomic tmp-then-replace so a crash mid-write can't corrupt the
    canonical file that match_goodreads.py depends on."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Shard JSONL paths. Default: every catalog_goodreads('book_show_api*.jsonl') shard.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Default: catalog_goodreads('book_show_api.jsonl')")
    args = parser.parse_args(argv)

    inputs = args.inputs or default_book_show_api_paths(catalog_goodreads())
    out_path = args.out or catalog_goodreads("book_show_api.jsonl")

    records = merge_records(inputs)
    write_merged(records, out_path)

    usable = sum(1 for r in records if r.get("legacy_id") is not None)
    print(f"[merge_book_show_api] merged {len(inputs)} file(s) -> {len(records):,} record(s) ({usable:,} with legacy_id)")
    print(f"[merge_book_show_api] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
