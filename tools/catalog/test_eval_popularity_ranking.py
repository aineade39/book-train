#!/usr/bin/env python3
"""Unit tests for tools/catalog/eval_popularity_ranking.py
(run: python tools/catalog/test_eval_popularity_ranking.py)."""

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

from tools.catalog.bibliographic_join import fold_for  # noqa: E402
from tools.catalog.consolidate_popularity_signals import (  # noqa: E402
    ConsolidatedSignal,
    write_consolidated_signals,
)
from tools.catalog.eval_popularity_ranking import (  # noqa: E402
    evaluate_ranking,
    load_reference_ratings_counts,
    load_shelf_scores,
    spearman_correlation,
    top_k_overlap,
    worst_rank_displacements,
)


def _write_matched(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestLoadShelfScores(unittest.TestCase):
    def test_reads_book_id_and_shelf_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matched.jsonl.gz"
            _write_matched(path, [{"goodreads_book_id": 33, "shelf_score": 0.7}, {"goodreads_book_id": 74, "shelf_score": 0.2}])
            scores = load_shelf_scores(path)
            self.assertEqual(scores, {33: 0.7, 74: 0.2})

    def test_missing_path_returns_empty(self) -> None:
        self.assertEqual(load_shelf_scores(Path("/tmp/does-not-exist-matched.jsonl.gz")), {})


class TestLoadReferenceRatingsCounts(unittest.TestCase):
    def test_only_positive_ratings_counts_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consolidated_signals.jsonl.gz"
            write_consolidated_signals(
                path,
                {
                    1: ConsolidatedSignal(book_id=1, api_ratings_count=500),
                    2: ConsolidatedSignal(book_id=2, api_ratings_count=0),
                    3: ConsolidatedSignal(book_id=3),
                },
            )
            ref = load_reference_ratings_counts(path)
            self.assertEqual(ref, {1: 500})


class TestSpearmanCorrelation(unittest.TestCase):
    def test_perfect_agreement_is_one(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_disagreement_is_negative_one(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_constant_values_are_zero_not_nan(self) -> None:
        self.assertEqual(spearman_correlation([1, 1, 1], [1, 2, 3]), 0.0)

    def test_too_few_points_is_zero(self) -> None:
        self.assertEqual(spearman_correlation([1], [2]), 0.0)


class TestTopKOverlap(unittest.TestCase):
    def test_identical_top_k_is_one(self) -> None:
        self.assertEqual(top_k_overlap([1, 2, 3], [1, 2, 3], 3), 1.0)

    def test_disjoint_top_k_is_zero(self) -> None:
        self.assertEqual(top_k_overlap([1, 2], [3, 4], 2), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(top_k_overlap([1, 2, 5], [1, 2, 3], 3), 2 / 3)


class TestWorstRankDisplacements(unittest.TestCase):
    def test_orders_by_displacement_descending(self) -> None:
        shelf_rank = {1: 1, 2: 2, 3: 3}
        reference_rank = {1: 1, 2: 10, 3: 3}
        out = worst_rank_displacements([1, 2, 3], shelf_rank, reference_rank, limit=2)
        self.assertEqual(out[0]["book_id"], 2)
        self.assertEqual(out[0]["displacement"], 8)


class TestEvaluateRanking(unittest.TestCase):
    def test_only_books_in_both_sources_are_scored(self) -> None:
        shelf_scores = {1: 0.9, 2: 0.1, 3: 0.5}
        reference = {1: 900, 2: 100}  # book 3 has no reference signal
        report = evaluate_ranking(shelf_scores, reference)
        self.assertEqual(report["n"], 2)
        self.assertAlmostEqual(report["spearman"], 1.0)

    def test_fold_filters_to_the_requested_split(self) -> None:
        book_ids = list(range(1, 200))
        shelf_scores = {b: float(b) for b in book_ids}
        reference = {b: b * 10 for b in book_ids}
        tuning_report = evaluate_ranking(shelf_scores, reference, fold="tuning")
        validation_report = evaluate_ranking(shelf_scores, reference, fold="validation")
        expected_tuning = sum(1 for b in book_ids if fold_for(b) == "tuning")
        expected_validation = sum(1 for b in book_ids if fold_for(b) == "validation")
        self.assertEqual(tuning_report["n"], expected_tuning)
        self.assertEqual(validation_report["n"], expected_validation)
        self.assertEqual(tuning_report["n"] + validation_report["n"], len(book_ids))
        all_report = evaluate_ranking(shelf_scores, reference, fold="all")
        self.assertEqual(all_report["n"], len(book_ids))

    def test_too_few_points_returns_zeroed_report(self) -> None:
        report = evaluate_ranking({1: 0.5}, {1: 100})
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["spearman"], 0.0)
        self.assertEqual(report["worst_displacements"], [])


if __name__ == "__main__":
    unittest.main()
