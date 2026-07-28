#!/usr/bin/env python3
"""Unit tests for tools/catalog/ol_common.py (run: python tools/catalog/test_ol_common.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import (  # noqa: E402
    build_work_rows,
    filter_work_rows,
    isbn10_to_13,
    normalize_for_search,
    normalize_isbn13,
    search_tokens,
)


class TestISBN(unittest.TestCase):
    def test_isbn13_normalize(self) -> None:
        self.assertEqual(normalize_isbn13("978-0-441-01359-3"), "9780441013593")
        self.assertEqual(normalize_isbn13("9780441013593"), "9780441013593")


class TestNormalizeForSearch(unittest.TestCase):
    """Mirrors SpineMatchingTests' coverage of normalizeForSearch (Normalization.swift)."""

    def test_lowercases_and_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_for_search("  Dune   Messiah  "), "dune messiah")

    def test_diacritic_folds(self) -> None:
        self.assertEqual(normalize_for_search("Émile Zola"), "emile zola")

    def test_strips_decorative_punctuation(self) -> None:
        self.assertEqual(normalize_for_search('The "Great" Gatsby (1925)!'), "the great gatsby 1925")

    def test_keeps_meaningful_marks(self) -> None:
        self.assertEqual(normalize_for_search("Jean-Paul O'Brien AT&T Vol. 2"), "jean-paul o'brien at&t vol. 2")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_for_search(""), "")

    def test_matches_across_case_and_accents(self) -> None:
        self.assertEqual(normalize_for_search("NAÏVE"), normalize_for_search("naive"))


class TestSearchTokens(unittest.TestCase):
    def test_splits_on_single_space(self) -> None:
        self.assertEqual(search_tokens("dune messiah"), ["dune", "messiah"])

    def test_empty_string_yields_no_tokens(self) -> None:
        self.assertEqual(search_tokens(""), [])


class TestOLMiniFixture(unittest.TestCase):
    def test_build_and_filter_eng(self) -> None:
        fixture = _REPO / "Tests" / "fixtures" / "ol-mini"
        rows = build_work_rows(
            editions_path=fixture / "editions.jsonl",
            works_path=fixture / "works.jsonl",
            authors_path=fixture / "authors.jsonl",
            min_editions=1,
        )
        self.assertEqual(len(rows), 5)
        eng = filter_work_rows(rows, languages=["eng"], max_works=None)
        self.assertEqual(len(eng), 4)
        self.assertEqual(eng[0].title, "Dune")
        self.assertEqual(eng[0].popularity_rank, 1)


if __name__ == "__main__":
    unittest.main()
