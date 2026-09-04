#!/usr/bin/env python3
"""Unit tests for tools/catalog/eval_canaries.py
(run: python tools/catalog/test_eval_canaries.py)."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.eval_canaries import (  # noqa: E402
    count_gap_fill_candidates,
    distribution_delta,
    evaluate_canaries,
    load_matched_rows,
    match_method_distribution,
    suspicious_duplicate_targets,
)


def _row(book_id: int, title: str, author: str, *, work_key: str | None, method: str, ratings_count: int = 0) -> dict:
    return {
        "goodreads_book_id": book_id,
        "title": title,
        "author": author,
        "work_key": work_key,
        "match_method": method,
        "ratings_count": ratings_count,
    }


class TestSuspiciousDuplicateTargets(unittest.TestCase):
    def test_same_title_and_compatible_author_is_legitimate(self) -> None:
        rows = [
            _row(1, "Pride and Prejudice", "Jane Austen", work_key="/works/PP", method="title_author"),
            _row(2, "Pride and Prejudice", "Jane Austen", work_key="/works/PP", method="title_author"),
        ]
        report = suspicious_duplicate_targets(rows)
        self.assertEqual(report["suspicious_duplicate_targets"], 0)
        self.assertEqual(report["legitimate_duplicate_targets"], 1)

    def test_different_titles_sharing_a_work_key_is_suspicious(self) -> None:
        rows = [
            _row(1, "Twilight", "Stephenie Meyer", work_key="/works/AMBIGUOUS", method="title_author"),
            _row(2, "New Moon", "Stephenie Meyer", work_key="/works/AMBIGUOUS", method="title_author"),
        ]
        report = suspicious_duplicate_targets(rows)
        self.assertEqual(report["suspicious_duplicate_targets"], 1)
        self.assertEqual(report["suspicious_duplicate_examples"][0]["work_key"], "/works/AMBIGUOUS")

    def test_incompatible_authors_sharing_a_work_key_is_suspicious(self) -> None:
        rows = [
            _row(1, "Twilight", "Stephenie Meyer", work_key="/works/AMBIGUOUS", method="title_author"),
            _row(2, "Twilight", "Dean Koontz", work_key="/works/AMBIGUOUS", method="title_author"),
        ]
        report = suspicious_duplicate_targets(rows)
        self.assertEqual(report["suspicious_duplicate_targets"], 1)

    def test_single_book_per_work_key_is_not_counted_either_way(self) -> None:
        rows = [_row(1, "Dune", "Frank Herbert", work_key="/works/DUNE", method="title_author")]
        report = suspicious_duplicate_targets(rows)
        self.assertEqual(report["suspicious_duplicate_targets"], 0)
        self.assertEqual(report["legitimate_duplicate_targets"], 0)

    def test_non_title_author_matches_are_ignored(self) -> None:
        rows = [
            _row(1, "Twilight", "Stephenie Meyer", work_key="/works/X", method="isbn"),
            _row(2, "New Moon", "Stephenie Meyer", work_key="/works/X", method="isbn"),
        ]
        report = suspicious_duplicate_targets(rows)
        self.assertEqual(report["suspicious_duplicate_targets"], 0)
        self.assertEqual(report["legitimate_duplicate_targets"], 0)


class TestGapFillCandidates(unittest.TestCase):
    def test_counts_popular_unmatched_and_ambiguous_only(self) -> None:
        rows = [
            _row(1, "A", "X", work_key=None, method="unmatched", ratings_count=5000),
            _row(2, "B", "Y", work_key=None, method="ambiguous", ratings_count=2000),
            _row(3, "C", "Z", work_key=None, method="unmatched", ratings_count=10),  # below floor
            _row(4, "D", "W", work_key="/works/D", method="unmatched", ratings_count=5000),  # has work_key
            _row(5, "E", "V", work_key=None, method="title_author", ratings_count=5000),  # matched
        ]
        self.assertEqual(count_gap_fill_candidates(rows), 2)

    def test_missing_title_or_author_is_excluded(self) -> None:
        rows = [{"work_key": None, "match_method": "unmatched", "ratings_count": 5000, "title": None, "author": "X"}]
        self.assertEqual(count_gap_fill_candidates(rows), 0)


class TestMatchMethodDistribution(unittest.TestCase):
    def test_counts_per_method(self) -> None:
        rows = [
            _row(1, "A", "X", work_key="/w/1", method="isbn"),
            _row(2, "B", "Y", work_key="/w/2", method="title_author"),
            _row(3, "C", "Z", work_key=None, method="unmatched"),
        ]
        self.assertEqual(match_method_distribution(rows), {"isbn": 1, "title_author": 1, "unmatched": 1})

    def test_distribution_delta(self) -> None:
        baseline = {"unmatched": 100, "title_author": 50}
        current = {"unmatched": 80, "title_author": 65, "isbn": 5}
        delta = distribution_delta(baseline, current)
        self.assertEqual(delta, {"isbn": 5, "title_author": 15, "unmatched": -20})


class TestLoadMatchedRows(unittest.TestCase):
    def test_reads_gzipped_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matched.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json.dumps({"goodreads_book_id": 1}) + "\n")
            rows = load_matched_rows(path)
            self.assertEqual(rows, [{"goodreads_book_id": 1}])

    def test_missing_path_returns_empty(self) -> None:
        self.assertEqual(load_matched_rows(Path("/tmp/does-not-exist-eval-canaries.jsonl.gz")), [])


class TestEvaluateCanaries(unittest.TestCase):
    def test_includes_deltas_only_when_baseline_given(self) -> None:
        rows = [_row(1, "Dune", "Frank Herbert", work_key="/works/DUNE", method="title_author")]
        without_baseline = evaluate_canaries(rows)
        self.assertNotIn("gap_fill_candidates_delta", without_baseline)

        with_baseline = evaluate_canaries(
            rows, baseline={"gap_fill_candidates": 3, "match_method_distribution": {"title_author": 0}}
        )
        self.assertEqual(with_baseline["gap_fill_candidates_delta"], with_baseline["gap_fill_candidates"] - 3)
        self.assertIn("match_method_distribution_delta", with_baseline)


if __name__ == "__main__":
    unittest.main()
