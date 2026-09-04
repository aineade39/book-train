#!/usr/bin/env python3
"""Unit tests for tools/catalog/ol_common.py (run: python tools/catalog/test_ol_common.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import (  # noqa: E402
    author_keys_from_work,
    build_work_rows,
    filter_work_rows,
    isbn10_to_13,
    join_author_names,
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
        """`-` and `'` survive unchanged -- unlike `&` (folds to "and", 2j-5c)
        and `.` (dropped, 2j-5b), neither has a same-shape ambiguity between
        GR and OL, so there's nothing to fold."""
        self.assertEqual(normalize_for_search("Jean-Paul O'Brien"), "jean-paul o'brien")

    def test_ampersand_folds_to_and(self) -> None:
        """2j-5c regression: GR/OL disagree on "&" vs "and" in the same
        title (Carissa Broadbent, "The Serpent & the Wings of Night" (OL) /
        "...and the Wings of Night" (GR))."""
        self.assertEqual(
            normalize_for_search("The Serpent & the Wings of Night"),
            normalize_for_search("The Serpent and the Wings of Night"),
        )
        self.assertEqual(normalize_for_search("AT&T"), "atandt")

    def test_period_is_dropped_like_slash(self) -> None:
        """2j-5b regression: GR/OL disagree on "." vs "/" as a date-title
        separator ("11.22.63" (GR) vs "11/22/63" (OL, 49 editions))."""
        self.assertEqual(normalize_for_search("11.22.63"), normalize_for_search("11/22/63"))
        self.assertEqual(normalize_for_search("J.R.R. Tolkien"), "jrr tolkien")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_for_search(""), "")

    def test_matches_across_case_and_accents(self) -> None:
        self.assertEqual(normalize_for_search("NAÏVE"), normalize_for_search("naive"))

    def test_curly_apostrophe_folds_to_straight(self) -> None:
        """Stage 2i regression: a curly right single quote (U+2019) is the
        standard apostrophe glyph in most scraped web text (Goodreads) and
        must normalize identically to a straight ASCII apostrophe (U+0027)
        -- OL's own catalog strings mix both for the same word. Previously
        both curly quote marks were stripped as "decorative" (grouped with
        the double-quote curlies), which silently diverged from the straight
        apostrophe (deliberately *not* stripped, see `test_keeps_meaningful_
        marks` above) and caused real OL rows to go unmatched purely because
        of which apostrophe glyph the two sides happened to use."""
        self.assertEqual(
            normalize_for_search("The Assassin\u2019s Blade"),
            normalize_for_search("The Assassin's Blade"),
        )
        self.assertEqual(normalize_for_search("The Assassin\u2019s Blade"), "the assassin's blade")

    def test_curly_left_single_quote_also_folds_to_straight(self) -> None:
        self.assertEqual(normalize_for_search("rock \u2018n\u2019 roll"), "rock 'n' roll")

    def test_zero_width_space_is_dropped(self) -> None:
        """2j-5a regression: a real Goodreads title, "The \u200bCrown of
        Gilded Bones", has a stray zero-width space (U+200B) right after the
        real space before "Crown" -- invisible on screen but a distinct
        character, so it silently broke the OL title match."""
        self.assertEqual(
            normalize_for_search("The \u200bCrown of Gilded Bones"),
            normalize_for_search("The Crown of Gilded Bones"),
        )

    def test_other_invisible_format_characters_are_dropped(self) -> None:
        # Zero-width non-joiner, zero-width joiner, byte-order mark.
        self.assertEqual(normalize_for_search("Dune\u200cMessiah"), "dunemessiah")
        self.assertEqual(normalize_for_search("Dune\u200dMessiah"), "dunemessiah")
        self.assertEqual(normalize_for_search("\ufeffDune Messiah"), "dune messiah")


class TestSearchTokens(unittest.TestCase):
    def test_splits_on_single_space(self) -> None:
        self.assertEqual(search_tokens("dune messiah"), ["dune", "messiah"])

    def test_empty_string_yields_no_tokens(self) -> None:
        self.assertEqual(search_tokens(""), [])


class TestAuthorKeysFromWork(unittest.TestCase):
    """Co-author fix: `author_key_from_work` used to keep only `authors[0]`."""

    def test_single_author(self) -> None:
        row = {"authors": [{"key": "/authors/OL1A"}]}
        self.assertEqual(author_keys_from_work(row), ["/authors/OL1A"])

    def test_multiple_authors_preserve_order(self) -> None:
        row = {"authors": [{"key": "/authors/OL1A"}, {"key": "/authors/OL2A"}, {"key": "/authors/OL3A"}]}
        self.assertEqual(author_keys_from_work(row), ["/authors/OL1A", "/authors/OL2A", "/authors/OL3A"])

    def test_nested_author_role_shape(self) -> None:
        row = {"authors": [{"author": {"key": "/authors/OL1A"}, "type": {"key": "/type/author_role"}}]}
        self.assertEqual(author_keys_from_work(row), ["/authors/OL1A"])

    def test_no_authors_returns_empty(self) -> None:
        self.assertEqual(author_keys_from_work({}), [])
        self.assertEqual(author_keys_from_work({"authors": []}), [])


class TestJoinAuthorNames(unittest.TestCase):
    def test_single_name(self) -> None:
        self.assertEqual(join_author_names(["Frank Herbert"]), "Frank Herbert")

    def test_two_names(self) -> None:
        self.assertEqual(join_author_names(["Terry Pratchett", "Neil Gaiman"]), "Terry Pratchett and Neil Gaiman")

    def test_three_or_more_names_preserve_order(self) -> None:
        self.assertEqual(
            join_author_names(["A", "B", "C"]),
            "A, B, and C",
        )

    def test_empty_list(self) -> None:
        self.assertEqual(join_author_names([]), "")

    def test_blank_names_are_dropped(self) -> None:
        self.assertEqual(join_author_names(["Frank Herbert", "  ", ""]), "Frank Herbert")

    def test_joined_output_splits_back_into_individual_names_via_comma_and_and(self) -> None:
        # Mirrors CustomWordsBuilder.individualWords (Swift): replace " and "
        # with "," then split on [,&] -- confirms the joiner picked here
        # round-trips through that existing splitting convention.
        joined = join_author_names(["Terry Pratchett", "Neil Gaiman", "Rob Wilkins"])
        normalized = joined.replace(" and ", ",")
        people = [p.strip() for p in normalized.replace("&", ",").split(",") if p.strip()]
        self.assertEqual(people, ["Terry Pratchett", "Neil Gaiman", "Rob Wilkins"])


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


class TestBuildWorkRowsMultiAuthor(unittest.TestCase):
    """`build_work_rows` (the CSV/`ol_to_csv.py` path) resolves the co-author
    fix the same way `process_ol.py`'s SQL-join path does -- separate temp
    fixture (not the shared ol-mini one) so multi-author rows don't shift
    ol-mini's existing rank/count assertions."""

    def test_multi_author_work_is_joined_and_partial_resolution_keeps_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "authors.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"key": "/authors/OL1A", "name": "Terry Pratchett"}),
                        json.dumps({"key": "/authors/OL2A", "name": "Neil Gaiman"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (tmp_path / "works.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "key": "/works/OL1W",
                                "title": "Good Omens",
                                "authors": [{"key": "/authors/OL1A"}, {"key": "/authors/OL2A"}],
                            }
                        ),
                        json.dumps(
                            {
                                "key": "/works/OL2W",
                                "title": "Partial Resolve",
                                "authors": [{"key": "/authors/OL2A"}, {"key": "/authors/OL9A"}],
                            }
                        ),
                        json.dumps({"key": "/works/OL3W", "title": "None Resolve", "authors": [{"key": "/authors/OL8A"}]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (tmp_path / "editions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/eng"}]}),
                        json.dumps({"works": [{"key": "/works/OL2W"}], "languages": [{"key": "/languages/eng"}]}),
                        json.dumps({"works": [{"key": "/works/OL3W"}], "languages": [{"key": "/languages/eng"}]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_work_rows(
                editions_path=tmp_path / "editions.jsonl",
                works_path=tmp_path / "works.jsonl",
                authors_path=tmp_path / "authors.jsonl",
                min_editions=1,
            )

        by_key = {row.work_key: row for row in rows}
        self.assertEqual(by_key["/works/OL1W"].author, "Terry Pratchett and Neil Gaiman")
        self.assertEqual(by_key["/works/OL2W"].author, "Neil Gaiman")
        self.assertNotIn("/works/OL3W", by_key, "a work with zero resolvable authors should be dropped, same as before Part A")


if __name__ == "__main__":
    unittest.main()
