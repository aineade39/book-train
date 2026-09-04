#!/usr/bin/env python3
"""Unit tests for tools/catalog/consolidate_popularity_signals.py
(run: python tools/catalog/test_consolidate_popularity_signals.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.consolidate_popularity_signals import (  # noqa: E402
    ConsolidatedSignal,
    collect_book_show_api_signals,
    consolidate,
    load_consolidated_signals,
    write_consolidated_signals,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _list_show_record(book_id: int, rating: int = 100) -> dict:
    return {
        "book_urls": [f"/book/show/{book_id}"],
        "rating_texts": [f"4.20 avg rating — {rating:,} ratings"],
        "score_texts": ["score: 10"],
        "vote_texts": ["1 people voted"],
    }


def _success_record(book_id: int, isbn13: str = "9780000000000", ratings_count: int = 500, genres=("Fantasy",)) -> dict:
    return {
        "legacy_id": book_id,
        "isbn13": isbn13,
        "average_rating": 4.5,
        "ratings_count": ratings_count,
        "text_reviews_count": 42,
        "genres": list(genres),
    }


def _incomplete_record(book_id: int, ratings_count: int = 700, genres=("Fiction",)) -> dict:
    return {
        "_scrape_warning": "incomplete_record",
        "_missing_fields": ["legacy_id", "isbn13"],
        "_url": f"https://www.goodreads.com/_next/data/xyz/book/show/{book_id}.json",
        "isbn13": None,
        "legacy_id": None,
        "title": None,
        "average_rating": 4.1,
        "ratings_count": ratings_count,
        "text_reviews_count": 7,
        "genres": list(genres),
    }


class TestCollectBookShowApiSignals(unittest.TestCase):
    def test_success_record_is_captured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [_success_record(33)])
            fields = collect_book_show_api_signals([path])
            self.assertEqual(fields[33].isbn13, "9780000000000")
            self.assertTrue(fields[33].has_isbn)
            self.assertFalse(fields[33].has_partial)

    def test_incomplete_record_recovers_book_id_and_never_has_isbn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [_incomplete_record(1)])
            fields = collect_book_show_api_signals([path])
            self.assertIn(1, fields)
            self.assertIsNone(fields[1].isbn13)
            self.assertFalse(fields[1].has_isbn)
            self.assertTrue(fields[1].has_partial)
            self.assertEqual(fields[1].ratings_count, 700)
            self.assertEqual(fields[1].genres, ("Fiction",))

    def test_incomplete_record_with_no_usable_fields_is_not_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            record = _incomplete_record(1, ratings_count=700)
            record["ratings_count"] = None
            record["average_rating"] = None
            record["text_reviews_count"] = None
            record["genres"] = []
            _write_jsonl(path, [record])
            fields = collect_book_show_api_signals([path])
            self.assertFalse(fields[1].has_partial)

    def test_success_wins_over_incomplete_record_regardless_of_line_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path_a, [_incomplete_record(5), _success_record(5, isbn13="9781111111111")])
            fields_a = collect_book_show_api_signals([path_a])
            self.assertEqual(fields_a[5].isbn13, "9781111111111")
            self.assertTrue(fields_a[5].has_isbn)

            path_b = Path(tmp) / "book_show_api2.jsonl"
            _write_jsonl(path_b, [_success_record(5, isbn13="9781111111111"), _incomplete_record(5)])
            fields_b = collect_book_show_api_signals([path_b])
            self.assertEqual(fields_b[5].isbn13, "9781111111111")
            self.assertTrue(fields_b[5].has_isbn)

    def test_missing_paths_are_skipped(self) -> None:
        fields = collect_book_show_api_signals([Path("/tmp/does-not-exist-book-show-api.jsonl")])
        self.assertEqual(fields, {})

    def test_skips_malformed_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            path.write_text('{"legacy_id": 1, "isbn13": "9780000000000"}\nnot json\n', encoding="utf-8")
            fields = collect_book_show_api_signals([path])
            self.assertEqual(set(fields), {1})


class TestConsolidate(unittest.TestCase):
    def test_book_in_both_sources_keeps_both_field_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [_list_show_record(33, rating=745415)])
            api_path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(api_path, [_success_record(33, isbn13="9780000000000", ratings_count=800000)])

            signals = consolidate(raw_dir, [api_path])
            sig = signals[33]
            self.assertEqual(sig.list_ratings_count, 745415)
            self.assertEqual(sig.list_appearances, 1)
            self.assertEqual(sig.api_isbn13, "9780000000000")
            self.assertEqual(sig.api_ratings_count, 800000)
            self.assertTrue(sig.has_list_signal)
            self.assertTrue(sig.has_api_isbn)

    def test_list_only_book_has_default_api_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [_list_show_record(1)])
            signals = consolidate(raw_dir, [])
            sig = signals[1]
            self.assertTrue(sig.has_list_signal)
            self.assertFalse(sig.has_api_isbn)
            self.assertFalse(sig.has_api_partial)
            self.assertIsNone(sig.api_isbn13)

    def test_incomplete_record_only_book_has_default_list_fields_and_no_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            api_path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(api_path, [_incomplete_record(1, ratings_count=700, genres=("Fiction",))])
            signals = consolidate(raw_dir, [api_path])
            sig = signals[1]
            self.assertFalse(sig.has_list_signal)
            self.assertEqual(sig.list_ratings_count, 0)
            self.assertTrue(sig.has_api_partial)
            self.assertEqual(sig.api_ratings_count, 700)
            self.assertEqual(sig.genres, ("Fiction",))
            # ConsolidatedSignal has no `title` field at all: incomplete_record
            # rows never carry a usable title (confirmed empirically: 0/12,112
            # in the live scrape), so there is nothing to store or leak here.
            self.assertFalse(hasattr(sig, "title"))

    def test_missing_raw_dir_does_not_crash(self) -> None:
        signals = consolidate(Path("/tmp/does-not-exist-raw-dir"), [])
        self.assertEqual(signals, {})


class TestWriteAndLoadRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "consolidated_signals.jsonl.gz"
            signals = {
                33: ConsolidatedSignal(
                    book_id=33,
                    list_ratings_count=100,
                    list_appearances=2,
                    api_isbn13="9780000000000",
                    api_ratings_count=500,
                    genres=("Fantasy", "Fiction"),
                    has_list_signal=True,
                    has_api_isbn=True,
                )
            }
            n_written = write_consolidated_signals(out_path, signals)
            self.assertEqual(n_written, 1)
            loaded = load_consolidated_signals(out_path)
            self.assertEqual(loaded[33], signals[33])

    def test_missing_path_returns_empty(self) -> None:
        self.assertEqual(load_consolidated_signals(Path("/tmp/does-not-exist-consolidated.jsonl.gz")), {})


if __name__ == "__main__":
    unittest.main()
