#!/usr/bin/env python3
"""Unit tests for tools/catalog/list_show_popularity.py
(run: python tools/catalog/test_list_show_popularity.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.list_show_popularity import (  # noqa: E402
    BookPopularity,
    aggregate_popularity,
    iter_list_show_books,
    popularity_sort_key,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _list_show_record(
    book_ids: list[int],
    ratings: list[int] | None = None,
    scores: list[int] | None = None,
    votes: list[int] | None = None,
) -> dict:
    ratings = ratings or [0] * len(book_ids)
    scores = scores or [0] * len(book_ids)
    votes = votes or [0] * len(book_ids)
    return {
        "book_urls": [f"/book/show/{bid}" for bid in book_ids],
        "rating_texts": [f"4.20 avg rating — {r:,} ratings" for r in ratings],
        "score_texts": [f"score: {s:,}" for s in scores],
        "vote_texts": [f"{v:,} people voted" for v in votes],
    }


class TestIterListShowBooks(unittest.TestCase):
    def test_parses_ratings_score_and_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.jsonl"
            _write_jsonl(path, [_list_show_record([33], ratings=[745415], scores=[42463], votes=[430])])
            rows = list(iter_list_show_books(path))
            self.assertEqual(rows, [(33, 745415, 42463, 430)])

    def test_missing_rating_text_yields_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.jsonl"
            _write_jsonl(
                path,
                [{"book_urls": ["/book/show/33"], "rating_texts": [""], "score_texts": [""], "vote_texts": [""]}],
            )
            rows = list(iter_list_show_books(path))
            self.assertEqual(rows, [(33, 0, 0, 0)])

    def test_missing_path_yields_nothing(self) -> None:
        self.assertEqual(list(iter_list_show_books(Path("/tmp/does-not-exist-list-show.jsonl"))), [])


class TestAggregatePopularity(unittest.TestCase):
    def test_ratings_count_takes_max_within_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [_list_show_record([33], ratings=[100]), _list_show_record([33], ratings=[500])],
            )
            pop = aggregate_popularity(raw_dir)
            self.assertEqual(pop[33].ratings_count, 500)

    def test_ratings_count_takes_max_across_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [_list_show_record([33], ratings=[100])])
            _write_jsonl(raw_dir / "2.jsonl", [_list_show_record([33], ratings=[745415])])
            pop = aggregate_popularity(raw_dir)
            self.assertEqual(pop[33].ratings_count, 745415)

    def test_list_appearances_increments_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [_list_show_record([33])])
            _write_jsonl(raw_dir / "2.jsonl", [_list_show_record([33])])
            _write_jsonl(raw_dir / "3.jsonl", [_list_show_record([33])])
            pop = aggregate_popularity(raw_dir)
            self.assertEqual(pop[33].list_appearances, 3)

    def test_list_score_and_vote_sum_across_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [_list_show_record([33], scores=[100], votes=[10])])
            _write_jsonl(raw_dir / "2.jsonl", [_list_show_record([33], scores=[50], votes=[5])])
            pop = aggregate_popularity(raw_dir)
            self.assertEqual(pop[33].list_score_sum, 150)
            self.assertEqual(pop[33].list_vote_sum, 15)

    def test_missing_raw_dir_returns_empty(self) -> None:
        self.assertEqual(aggregate_popularity(Path("/tmp/does-not-exist-raw-dir")), {})


class TestPopularitySortKey(unittest.TestCase):
    def test_higher_ratings_count_sorts_first(self) -> None:
        low = BookPopularity(ratings_count=10)
        high = BookPopularity(ratings_count=500)
        keys = sorted([(1, low), (2, high)], key=lambda t: popularity_sort_key(t[0], t[1]))
        self.assertEqual([bid for bid, _ in keys], [2, 1])

    def test_ties_broken_by_list_appearances_then_score_then_book_id(self) -> None:
        a = BookPopularity(ratings_count=100, list_appearances=1, list_score_sum=50)
        b = BookPopularity(ratings_count=100, list_appearances=2, list_score_sum=10)
        c = BookPopularity(ratings_count=100, list_appearances=2, list_score_sum=99)
        keys = sorted([(1, a), (2, b), (3, c)], key=lambda t: popularity_sort_key(t[0], t[1]))
        self.assertEqual([bid for bid, _ in keys], [3, 2, 1])

    def test_full_tie_breaks_by_book_id_ascending(self) -> None:
        same = BookPopularity(ratings_count=100, list_appearances=1, list_score_sum=1)
        keys = sorted([(99, same), (5, same)], key=lambda t: popularity_sort_key(t[0], t[1]))
        self.assertEqual([bid for bid, _ in keys], [5, 99])


if __name__ == "__main__":
    unittest.main()
