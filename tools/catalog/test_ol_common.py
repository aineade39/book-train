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
    normalize_isbn13,
)


class TestISBN(unittest.TestCase):
    def test_isbn13_normalize(self) -> None:
        self.assertEqual(normalize_isbn13("978-0-441-01359-3"), "9780441013593")
        self.assertEqual(normalize_isbn13("9780441013593"), "9780441013593")


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
