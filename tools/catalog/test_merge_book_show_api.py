#!/usr/bin/env python3
"""Unit tests for tools/catalog/merge_book_show_api.py
(run: python tools/catalog/test_merge_book_show_api.py)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.merge_book_show_api import main, merge_records, write_merged  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestMergeRecords(unittest.TestCase):
    def test_dedupes_by_legacy_id_keeping_newest_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            older = tmp_path / "a.jsonl"
            newer = tmp_path / "b.jsonl"
            _write_jsonl(older, [{"legacy_id": 33, "isbn13": "old"}])
            _write_jsonl(newer, [{"legacy_id": 33, "isbn13": "new"}])
            now = time.time()
            os.utime(older, (now - 10, now - 10))
            os.utime(newer, (now, now))

            merged = merge_records([older, newer])

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["isbn13"], "new")

    def test_unions_distinct_legacy_ids_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "a.jsonl"
            b = tmp_path / "b.jsonl"
            _write_jsonl(a, [{"legacy_id": 33, "isbn13": "x"}])
            _write_jsonl(b, [{"legacy_id": 74, "isbn13": "y"}])

            merged = merge_records([a, b])

            self.assertEqual({r["legacy_id"] for r in merged}, {33, 74})

    def test_keeps_warning_records_deduped_by_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "a.jsonl"
            b = tmp_path / "b.jsonl"
            warning = {
                "_record_name": "book_page",
                "_scrape_warning": "blocked_suspected",
                "_url": "https://example.com/1",
            }
            _write_jsonl(a, [warning])
            _write_jsonl(b, [warning])  # same warning re-appended in a second shard

            merged = merge_records([a, b])

            # Deduped to one line, but the repeat count is preserved (not silently
            # dropped) via _attempt_count -- see test_aggregates_attempt_count_*.
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["_url"], warning["_url"])
            self.assertEqual(merged[0]["_attempt_count"], 2)

    def test_aggregates_attempt_count_across_repeated_warnings(self) -> None:
        """extract_remaining_ids.py's give-up tracking needs the number of
        distinct attempts a book made, not just its last warning -- dedup by
        _url must not silently discard that count."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "a.jsonl"
            b = tmp_path / "b.jsonl"
            c = tmp_path / "c.jsonl"
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(a, [{"_scrape_warning": "incomplete_record", "_url": url}])
            _write_jsonl(b, [{"_scrape_warning": "incomplete_record", "_url": url}])
            _write_jsonl(c, [{"_scrape_warning": "incomplete_record", "_url": url}])

            merged = merge_records([a, b, c])

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["_attempt_count"], 3)

    def test_attempt_count_is_idempotent_on_repeated_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "a.jsonl"
            b = tmp_path / "b.jsonl"
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(a, [{"_scrape_warning": "incomplete_record", "_url": url}])
            _write_jsonl(b, [{"_scrape_warning": "incomplete_record", "_url": url}])

            first = merge_records([a, b])
            self.assertEqual(first[0]["_attempt_count"], 2)

            # Re-merging the already-merged output (as its sole input) must not
            # keep accumulating -- the existing _attempt_count is preserved, not summed again.
            merged_path = tmp_path / "merged.jsonl"
            write_merged(first, merged_path)
            second = merge_records([merged_path])
            self.assertEqual(second[0]["_attempt_count"], 2)

    def test_missing_input_path_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            existing = tmp_path / "a.jsonl"
            _write_jsonl(existing, [{"legacy_id": 33, "isbn13": "x"}])

            merged = merge_records([existing, tmp_path / "missing.jsonl"])

            self.assertEqual(len(merged), 1)

    def test_idempotent_on_repeated_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "a.jsonl"
            b = tmp_path / "b.jsonl"
            _write_jsonl(a, [{"legacy_id": 33, "isbn13": "x"}])
            _write_jsonl(b, [{"legacy_id": 74, "isbn13": "y"}])

            first = merge_records([a, b])
            second = merge_records([a, b])

            self.assertEqual(first, second)


class TestWriteMerged(unittest.TestCase):
    def test_writes_one_json_line_per_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.jsonl"
            write_merged([{"legacy_id": 33}, {"legacy_id": 74}], out_path)
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["legacy_id"] for line in lines], [33, 74])

    def test_overwrites_existing_file_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.jsonl"
            _write_jsonl(out_path, [{"legacy_id": 1}])
            write_merged([{"legacy_id": 33}], out_path)
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["legacy_id"] for line in lines], [33])


class TestMainEndToEnd(unittest.TestCase):
    def test_merges_named_inputs_into_out_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a = tmp_path / "book_show_api.jsonl"
            b = tmp_path / "book_show_api.batch2.jsonl"
            out_path = tmp_path / "merged.jsonl"
            _write_jsonl(a, [{"legacy_id": 33, "isbn13": "x"}])
            _write_jsonl(b, [{"legacy_id": 74, "isbn13": "y"}])

            rc = main([str(a), str(b), "--out", str(out_path)])

            self.assertEqual(rc, 0)
            ids = {json.loads(line)["legacy_id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
            self.assertEqual(ids, {33, 74})

    def test_default_inputs_glob_book_show_api_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_jsonl(tmp_path / "book_show_api.jsonl", [{"legacy_id": 33, "isbn13": "x"}])
            _write_jsonl(tmp_path / "book_show_api.part1.jsonl", [{"legacy_id": 74, "isbn13": "y"}])
            out_path = tmp_path / "book_show_api.jsonl"

            import tools.catalog.merge_book_show_api as mod

            original = mod.catalog_goodreads
            mod.catalog_goodreads = lambda *parts: tmp_path.joinpath(*parts)  # type: ignore[assignment]
            try:
                rc = main([])
            finally:
                mod.catalog_goodreads = original

            self.assertEqual(rc, 0)
            ids = {json.loads(line)["legacy_id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
            self.assertEqual(ids, {33, 74})


if __name__ == "__main__":
    unittest.main()
