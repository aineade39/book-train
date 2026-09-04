#!/usr/bin/env python3
"""Unit tests for tools/catalog/sample_matcher_residual_candidates.py."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bibliographic_join import TitleAuthorBlockIndex, load_residual_labels  # noqa: E402
from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.catalog.sample_matcher_residual_candidates import (  # noqa: E402
    build_residual_candidates,
    load_ol_rows_by_work_key,
    sample_low_margin_title_author,
    sample_unmatched_or_ambiguous,
    top_scored_candidate,
    write_residual_labels_yaml,
)


def _make_ol_db(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    """rows: (work_key, title, author, edition_count)."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            workKey TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            titleNormalized TEXT NOT NULL,
            authorNormalized TEXT NOT NULL,
            editionCount INTEGER
        )
        """
    )
    for work_key, title, author, edition_count in rows:
        conn.execute(
            "INSERT INTO books (workKey, title, author, titleNormalized, authorNormalized, editionCount) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (work_key, title, author, normalize_for_search(title), normalize_for_search(author), edition_count),
        )
    conn.commit()
    conn.close()


def _row(book_id: int, title: str, author: str, *, ratings_count: int, method: str, **extra) -> dict:
    return {
        "goodreads_book_id": book_id,
        "title": title,
        "author": author,
        "ratings_count": ratings_count,
        "match_method": method,
        "work_key": None,
        "match_score": None,
        "match_margin": None,
        **extra,
    }


class TestWriteResidualLabelsYaml(unittest.TestCase):
    def test_roundtrips_through_load_residual_labels(self) -> None:
        entries = [
            {
                "goodreads_book_id": 42,
                "sample_reason": "low_margin_title_author",
                "gr_title": 'A "Quoted" Title',
                "gr_author": "Some Author",
                "gr_ratings_count": 12345,
                "match_method": "title_author",
                "candidate_work_key": "/works/OL1W",
                "candidate_title": "A Quoted Title",
                "candidate_author": "Some Author",
                "candidate_edition_count": 3,
                "match_score": 91.5,
                "match_margin": 2.5,
                "verdict": "correct",
                "corrected_work_key": None,
                "notes": "looks fine",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "residual.yaml"
            write_residual_labels_yaml(entries, out_path)
            self.assertTrue(out_path.exists())
            labels = load_residual_labels(out_path)
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0].goodreads_book_id, 42)
            self.assertEqual(labels[0].verdict, "correct")
            self.assertEqual(labels[0].candidate_work_key, "/works/OL1W")
            self.assertEqual(labels[0].notes, "looks fine")

    def test_unreviewed_entries_write_verdict_null(self) -> None:
        entries = [
            {
                "goodreads_book_id": 1,
                "sample_reason": "unmatched_popular",
                "gr_title": "T",
                "gr_author": "A",
                "gr_ratings_count": 5000,
                "match_method": "unmatched",
                "candidate_work_key": None,
                "candidate_title": None,
                "candidate_author": None,
                "candidate_edition_count": None,
                "match_score": None,
                "match_margin": None,
                "verdict": None,
                "corrected_work_key": None,
                "notes": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "residual.yaml"
            write_residual_labels_yaml(entries, out_path)
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("verdict: null", text)
            # Unreviewed entries must not load as ground truth.
            self.assertEqual(load_residual_labels(out_path), [])


class TestSamplers(unittest.TestCase):
    def test_sample_unmatched_or_ambiguous_sorts_desc_and_respects_threshold_and_cap(self) -> None:
        rows = [
            _row(1, "A", "X", ratings_count=100, method="unmatched"),
            _row(2, "B", "Y", ratings_count=50_000, method="unmatched"),
            _row(3, "C", "Z", ratings_count=10_000, method="unmatched"),
            _row(4, "D", "W", ratings_count=20_000, method="ambiguous"),  # wrong method, excluded
        ]
        sampled = sample_unmatched_or_ambiguous(rows, method="unmatched", min_ratings_count=5_000, per_stratum=2)
        self.assertEqual([r["goodreads_book_id"] for r in sampled], [2, 3])  # book 1 below threshold, capped at 2

    def test_sample_low_margin_title_author_sorts_ascending_and_excludes_none_margin(self) -> None:
        rows = [
            _row(1, "A", "X", ratings_count=1, method="title_author", match_margin=10.0, work_key="/works/A"),
            _row(2, "B", "Y", ratings_count=1, method="title_author", match_margin=None, work_key="/works/B"),  # identity match, excluded
            _row(3, "C", "Z", ratings_count=1, method="title_author", match_margin=2.0, work_key="/works/C"),
        ]
        sampled = sample_low_margin_title_author(rows, per_stratum=10)
        self.assertEqual([r["goodreads_book_id"] for r in sampled], [3, 1])


class TestTopScoredCandidate(unittest.TestCase):
    def test_finds_best_name_compatible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "full.sqlite"
            _make_ol_db(
                db_path,
                [
                    ("/works/MEYER", "Twilight", "Stephenie Meyer", 131),
                    ("/works/KOONTZ", "Twilight", "Dean Koontz", 27),
                ],
            )
            conn = sqlite3.connect(db_path)
            from tools.catalog.match_goodreads import load_ol_candidates_for_titles

            index = TitleAuthorBlockIndex(load_ol_candidates_for_titles(conn, ["Twilight"]))
            conn.close()
            cand = top_scored_candidate("Twilight", "Stephenie Meyer", index)
            self.assertIsNotNone(cand)
            self.assertEqual(cand.work_key, "/works/MEYER")

    def test_returns_none_when_no_compatible_candidate(self) -> None:
        index = TitleAuthorBlockIndex([])
        self.assertIsNone(top_scored_candidate("Some Title", "Some Author", index))


class TestLoadOlRowsByWorkKey(unittest.TestCase):
    def test_batches_and_returns_title_author_edition_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 42)])
            conn = sqlite3.connect(db_path)
            out = load_ol_rows_by_work_key(conn, ["/works/OL1W", "/works/MISSING", None])
            conn.close()
            self.assertEqual(out, {"/works/OL1W": ("Dune", "Frank Herbert", 42)})


class TestBuildResidualCandidates(unittest.TestCase):
    def test_covers_all_four_strata_with_no_verdicts_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "full.sqlite"
            _make_ol_db(
                db_path,
                [
                    ("/works/HP5", "Harry Potter and the Order of the Phoenix", "J. K. Rowling", 300),
                    ("/works/DUPE", "Fahrenheit 451", "Ray Bradbury", 261),
                    ("/works/LOWM", "The Chronicles of Amber Volume I", "Roger Zelazny", 1),
                ],
            )
            rows = [
                # unmatched_popular: no OL row exists for this title at all.
                _row(1, "A Totally Obscure Title Nobody Has", "Nobody Author", ratings_count=50_000, method="unmatched"),
                # ambiguous_popular: a real OL candidate exists for this exact
                # title (production would have found it below accept for some
                # other reason -- this test only checks build_residual_candidates
                # wires the full-corpus probe through, not match_title_author's
                # own accept logic).
                _row(2, "Harry Potter and the Order of the Phoenix", "J.K. Rowling", ratings_count=40_000, method="ambiguous", match_score=63.0),
                # low_margin_title_author: matched, but a thin margin.
                _row(
                    3,
                    "The Chronicles of Amber Volume II",
                    "Roger Zelazny",
                    ratings_count=900,
                    method="title_author",
                    work_key="/works/LOWM",
                    match_margin=5.0,
                ),
                # suspicious_duplicate group: two GR books sharing one work_key
                # via title_author, with a title/author mismatch.
                _row(4, "Fahrenheit 451", "Ray Bradbury", ratings_count=2_900_000, method="title_author", work_key="/works/DUPE"),
                _row(5, "Farenheit 451", "Ray Bradbury", ratings_count=100, method="title_author", work_key="/works/DUPE"),
            ]
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                entries = build_residual_candidates(rows, conn, min_ratings_count=1_000, per_stratum=10)
            finally:
                conn.close()

        by_reason: dict[str, list[dict]] = {}
        for e in entries:
            by_reason.setdefault(e["sample_reason"], []).append(e)

        self.assertEqual(len(by_reason["unmatched_popular"]), 1)
        self.assertIsNone(by_reason["unmatched_popular"][0]["candidate_work_key"])

        self.assertEqual(len(by_reason["ambiguous_popular"]), 1)
        self.assertEqual(by_reason["ambiguous_popular"][0]["candidate_work_key"], "/works/HP5")

        self.assertEqual(len(by_reason["low_margin_title_author"]), 1)
        self.assertEqual(by_reason["low_margin_title_author"][0]["candidate_title"], "The Chronicles of Amber Volume I")

        # Both book_ids in the suspicious-duplicate group get their own entry.
        self.assertEqual({e["goodreads_book_id"] for e in by_reason["suspicious_duplicate"]}, {4, 5})
        for e in by_reason["suspicious_duplicate"]:
            self.assertEqual(e["candidate_work_key"], "/works/DUPE")

        # No entry is pre-verdicted -- that is the one part a human must do.
        for e in entries:
            self.assertIsNone(e["verdict"])

    def test_below_ratings_count_threshold_is_excluded_from_unmatched_and_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "full.sqlite"
            _make_ol_db(db_path, [])
            rows = [_row(1, "Obscure", "Nobody", ratings_count=10, method="unmatched")]
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                entries = build_residual_candidates(rows, conn, min_ratings_count=1_000, per_stratum=10)
            finally:
                conn.close()
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
