#!/usr/bin/env python3
"""Unit tests for tools/catalog/match_goodreads.py (run: python tools/catalog/test_match_goodreads.py).

No network access and no real full.sqlite: OL candidates are built directly
from an in-memory-shaped SQLite fixture matching Sources/SpineCatalog/BookCatalog.swift's
schema.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.consolidate_popularity_signals import ConsolidatedSignal  # noqa: E402
from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.catalog.match_goodreads import (  # noqa: E402
    AuthorBlockIndex,
    GoodreadsBook,
    OLCandidate,
    SchemaError,
    SeedListMeta,
    author_block_key,
    build_genre_tag_mapping,
    DEFAULT_SHELF_SCORE_WEIGHTS,
    ShelfScoreWeights,
    compute_shelf_score,
    dedupe_books,
    git_head_sha,
    load_book_show_api_books,
    load_book_show_api_isbns,
    load_book_show_isbns,
    load_ol_candidates,
    load_ol_isbn_index,
    load_seed_metadata,
    main,
    match_book,
    matcher_eval_default_path,
    parse_book_id,
    parse_book_show_next_data,
    parse_list_show_file,
    parse_rating_text,
    parse_score_text,
    parse_vote_text,
    run,
    run_isbn_holdout_eval,
    strip_series_suffix,
    validate_list_show_schema,
    write_eval_report,
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
            isbn TEXT,
            titleNormalized TEXT NOT NULL,
            authorNormalized TEXT NOT NULL,
            popularityRank INTEGER,
            editionCount INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE book_isbns (isbn13 TEXT NOT NULL, workKey TEXT NOT NULL)")
    for work_key, title, author, edition_count in rows:
        conn.execute(
            "INSERT INTO books (workKey, title, author, titleNormalized, authorNormalized, editionCount) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (work_key, title, author, normalize_for_search(title), normalize_for_search(author), edition_count),
        )
    conn.commit()
    conn.close()


class TestFieldParsers(unittest.TestCase):
    def test_parse_book_id(self) -> None:
        self.assertEqual(parse_book_id("/book/show/33.The_Lord_of_the_Rings"), 33)
        self.assertIsNone(parse_book_id(None))
        self.assertIsNone(parse_book_id("/author/show/1.Someone"))

    def test_parse_rating_text(self) -> None:
        self.assertEqual(parse_rating_text("4.55 avg rating — 745,415 ratings"), (4.55, 745415))
        self.assertEqual(parse_rating_text("4.16 avg rating - 2,136,336 ratings"), (4.16, 2136336))
        self.assertEqual(parse_rating_text(None), (None, None))
        self.assertEqual(parse_rating_text("garbage"), (None, None))

    def test_parse_score_text(self) -> None:
        self.assertEqual(parse_score_text("score: 42,463"), 42463)
        self.assertIsNone(parse_score_text(None))

    def test_parse_vote_text(self) -> None:
        self.assertEqual(parse_vote_text("430 people voted"), 430)
        self.assertIsNone(parse_vote_text("garbage"))

    def test_strip_series_suffix_removes_hash_annotation(self) -> None:
        self.assertEqual(
            strip_series_suffix("The Lord of the Rings (The Lord of the Rings, #1-3)"),
            "The Lord of the Rings",
        )

    def test_strip_series_suffix_keeps_non_series_parens(self) -> None:
        self.assertEqual(
            strip_series_suffix("The Diamond Age: Or, a Young Lady's Illustrated Primer"),
            "The Diamond Age: Or, a Young Lady's Illustrated Primer",
        )
        self.assertEqual(strip_series_suffix("Good Omens (Illustrated Edition)"), "Good Omens (Illustrated Edition)")

    def test_author_block_key_ignores_period_spacing_differences(self) -> None:
        self.assertEqual(author_block_key("J.R.R. Tolkien"), author_block_key("J. R. R. Tolkien"))

    def test_author_block_key_distinguishes_different_authors(self) -> None:
        self.assertNotEqual(author_block_key("J.R.R. Tolkien"), author_block_key("George R.R. Martin"))


class TestParseListShowFile(unittest.TestCase):
    def test_parses_parallel_arrays_into_rows(self) -> None:
        record = {
            "book_urls": ["/book/show/33.The_Lord_of_the_Rings", "/book/show/2.Harry_Potter"],
            "titles": ["The Lord of the Rings (The Lord of the Rings, #1-3)", "Harry Potter and the Sorcerer's Stone"],
            "authors": ["J.R.R. Tolkien", "J.K. Rowling"],
            "rating_texts": ["4.55 avg rating — 745,415 ratings", "4.47 avg rating — 10,000 ratings"],
            "score_texts": ["score: 42,463", "score: 1,000"],
            "vote_texts": ["430 people voted", "10 people voted"],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "367.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            books = list(parse_list_show_file(path, "fantasy"))
        self.assertEqual(len(books), 2)
        self.assertEqual(books[0].book_id, 33)
        self.assertEqual(books[0].title, "The Lord of the Rings (The Lord of the Rings, #1-3)")
        self.assertEqual(books[0].avg_rating, 4.55)
        self.assertEqual(books[0].ratings_count, 745415)
        self.assertEqual(books[0].list_score_sum, 42463)
        self.assertEqual(books[0].vote_sum, 430)
        self.assertEqual(books[0].genres, {"fantasy"})

    def test_mismatched_array_lengths_truncates_to_shortest(self) -> None:
        record = {
            "book_urls": ["/book/show/1.A", "/book/show/2.B"],
            "titles": ["A", "B"],
            "authors": ["Author A"],  # short by one
            "rating_texts": ["4.0 avg rating — 1 ratings", "4.0 avg rating — 1 ratings"],
            "score_texts": ["score: 1", "score: 1"],
            "vote_texts": ["1 people voted", "1 people voted"],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "1.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            books = list(parse_list_show_file(path, ""))
        self.assertEqual(len(books), 1, "should truncate to the shortest array's length, not crash")

    def test_skips_rows_missing_required_fields(self) -> None:
        record = {
            "book_urls": ["/book/show/1.A", None],
            "titles": ["A", "B"],
            "authors": ["Author A", "Author B"],
            "rating_texts": [None, None],
            "score_texts": [None, None],
            "vote_texts": [None, None],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "1.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            books = list(parse_list_show_file(path, ""))
        self.assertEqual(len(books), 1, "row with no parseable book_id should be skipped, not crash")


class TestDedupeBooks(unittest.TestCase):
    def test_merges_appearances_across_lists(self) -> None:
        a = GoodreadsBook(book_id=33, title="LOTR", author="Tolkien", list_appearances=1, list_score_sum=100, vote_sum=10, genres={"fantasy"})
        b = GoodreadsBook(book_id=33, title="LOTR", author="Tolkien", list_appearances=1, list_score_sum=50, vote_sum=5, genres={"scifi_fantasy"})
        merged = dedupe_books([a, b])
        self.assertEqual(len(merged), 1)
        row = merged[33]
        self.assertEqual(row.list_appearances, 2)
        self.assertEqual(row.list_score_sum, 150)
        self.assertEqual(row.vote_sum, 15)
        self.assertEqual(row.genres, {"fantasy", "scifi_fantasy"})

    def test_keeps_distinct_books_separate(self) -> None:
        a = GoodreadsBook(book_id=1, title="A", author="X")
        b = GoodreadsBook(book_id=2, title="B", author="Y")
        merged = dedupe_books([a, b])
        self.assertEqual(set(merged.keys()), {1, 2})


class TestLoadSeedMetadata(unittest.TestCase):
    def test_reads_explicit_fields(self) -> None:
        yaml_text = (
            "lists:\n"
            "  - list_id: 367\n"
            "    slug: Best_Fantasy_Books\n"
            "    genre: fantasy\n"
            "    short_label: Fantasy\n"
            "    list_title: Best Fantasy Books\n"
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            meta = load_seed_metadata(path)
        self.assertEqual(
            meta[367],
            SeedListMeta(
                list_id=367,
                slug="Best_Fantasy_Books",
                genre="fantasy",
                short_label="Fantasy",
                list_title="Best Fantasy Books",
            ),
        )

    def test_falls_back_when_short_label_and_list_title_omitted(self) -> None:
        yaml_text = "lists:\n  - list_id: 8329\n    slug: Best_Romance_Books_Ever\n    genre: romance\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            meta = load_seed_metadata(path)
        self.assertEqual(meta[8329].short_label, "Romance")
        self.assertEqual(meta[8329].list_title, "Best Romance Books Ever")

    def test_falls_back_to_slug_when_genre_also_missing(self) -> None:
        yaml_text = "lists:\n  - list_id: 1\n    slug: Best_Books_Ever\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            meta = load_seed_metadata(path)
        self.assertEqual(meta[1].short_label, "Best_Books_Ever")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_seed_metadata(Path("/nonexistent/seed.yaml")), {})


class TestBuildGenreTagMapping(unittest.TestCase):
    def test_produces_expected_shape(self) -> None:
        meta = [
            SeedListMeta(367, "Best_Fantasy_Books", "fantasy", "Fantasy", "Best Fantasy Books"),
            SeedListMeta(8329, "Best_Romance_Books_Ever", "romance", "Romance", "Best Romance Books Ever"),
        ]
        mapping = build_genre_tag_mapping(meta)
        self.assertEqual(
            mapping,
            {
                "fantasy": {
                    "short_label": "Fantasy",
                    "list_title": "Best Fantasy Books",
                    "slug": "Best_Fantasy_Books",
                    "list_id": 367,
                },
                "romance": {
                    "short_label": "Romance",
                    "list_title": "Best Romance Books Ever",
                    "slug": "Best_Romance_Books_Ever",
                    "list_id": 8329,
                },
            },
        )

    def test_skips_entries_with_no_genre(self) -> None:
        meta = [SeedListMeta(1, "Best_Books_Ever", "", "General", "Best Books Ever")]
        self.assertEqual(build_genre_tag_mapping(meta), {})

    def test_duplicate_tag_keeps_first_and_warns(self) -> None:
        meta = [
            SeedListMeta(367, "Best_Fantasy_Books", "fantasy", "Fantasy", "Best Fantasy Books"),
            SeedListMeta(999, "Another_Fantasy_List", "fantasy", "Fantasy Redux", "Another Fantasy List"),
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            mapping = build_genre_tag_mapping(meta)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(mapping["fantasy"]["list_id"], 367, "first entry (seed-yaml order) must win")
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn("fantasy", stderr.getvalue())


class TestParseBookShowNextData(unittest.TestCase):
    def _next_data(self, book_key: str, book_obj: dict) -> str:
        return json.dumps({"props": {"pageProps": {"apolloState": {book_key: book_obj}}}})

    def test_extracts_isbn_and_title(self) -> None:
        next_data = self._next_data(
            "Book:kca://book/x",
            {
                "legacyId": 33,
                "title": "The Lord of the Rings",
                "details": {"isbn13": "9780618640157", "language": {"name": "English"}},
            },
        )
        parsed = parse_book_show_next_data(next_data)
        self.assertEqual(
            parsed,
            {
                "book_id": 33,
                "title": "The Lord of the Rings",
                "isbn13": "9780618640157",
                "language": "English",
                "avg_rating": None,
                "ratings_count": None,
            },
        )

    def test_malformed_json_returns_none(self) -> None:
        self.assertIsNone(parse_book_show_next_data("not json"))

    def test_missing_book_key_returns_none(self) -> None:
        self.assertIsNone(parse_book_show_next_data(json.dumps({"props": {"pageProps": {"apolloState": {}}}})))


class TestLoadBookShowIsbns(unittest.TestCase):
    def test_merges_isbn_by_book_id(self) -> None:
        record = {
            "next_data_json": json.dumps(
                {
                    "props": {
                        "pageProps": {
                            "apolloState": {
                                "Book:x": {"legacyId": 33, "title": "T", "details": {"isbn13": "9780618640157"}}
                            }
                        }
                    }
                }
            )
        }
        with tempfile.TemporaryDirectory() as d:
            book_show_dir = Path(d)
            (book_show_dir / "33.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            isbns = load_book_show_isbns(book_show_dir)
        self.assertEqual(isbns, {33: "9780618640157"})

    def test_missing_dir_returns_empty(self) -> None:
        self.assertEqual(load_book_show_isbns(None), {})
        self.assertEqual(load_book_show_isbns(Path("/nonexistent")), {})


class TestLoadBookShowApiIsbns(unittest.TestCase):
    def test_reads_direct_fields(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(
                json.dumps({"legacy_id": 33, "isbn13": "9780618640157", "title": "T"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_book_show_api_isbns(path), {33: "9780618640157"})

    def test_multiple_records(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"legacy_id": 33, "isbn13": "AAA"}) + "\n")
                f.write(json.dumps({"legacy_id": 74, "isbn13": "BBB"}) + "\n")
            self.assertEqual(load_book_show_api_isbns(path), {33: "AAA", 74: "BBB"})

    def test_missing_path_returns_empty(self) -> None:
        self.assertEqual(load_book_show_api_isbns(None), {})
        self.assertEqual(load_book_show_api_isbns(Path("/nonexistent.jsonl")), {})

    def test_skips_records_without_isbn13(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(
                json.dumps({"legacy_id": 33, "isbn13": None}) + "\n"
                + json.dumps({"_scrape_warning": "blocked_suspected"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_book_show_api_isbns(path), {})

    def test_skips_malformed_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text("not json\n" + json.dumps({"legacy_id": 33, "isbn13": "AAA"}) + "\n", encoding="utf-8")
            self.assertEqual(load_book_show_api_isbns(path), {33: "AAA"})


def _api_record(**overrides: object) -> dict:
    record = {
        "legacy_id": 2,
        "isbn13": "9780439686525",
        "title": "Harry Potter and the Order of the Phoenix",
        "author": "J.K. Rowling",
        "average_rating": 4.47,
        "ratings_count": 11742238,
        "genres": ["Fantasy", "Young Adult"],
    }
    record.update(overrides)
    return record


class TestLoadBookShowApiBooks(unittest.TestCase):
    def test_builds_a_full_goodreads_book(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(json.dumps(_api_record()) + "\n", encoding="utf-8")
            books = load_book_show_api_books(path)
            self.assertEqual(set(books), {2})
            book = books[2]
            self.assertEqual(book.title, "Harry Potter and the Order of the Phoenix")
            self.assertEqual(book.author, "J.K. Rowling")
            self.assertEqual(book.avg_rating, 4.47)
            self.assertEqual(book.ratings_count, 11742238)
            self.assertEqual(book.list_appearances, 0)
            self.assertIsNone(book.isbn13)  # left for _attach_isbns to set uniformly

    def test_genres_are_not_populated_from_the_api_record(self) -> None:
        # This module's `genres` field means list_show seed-list tags, not
        # Goodreads' own bookGenres -- must stay empty for a book with zero
        # list appearances, even though the raw record has a genres list.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(json.dumps(_api_record()) + "\n", encoding="utf-8")
            books = load_book_show_api_books(path)
            self.assertEqual(books[2].genres, set())

    def test_skips_records_missing_isbn13(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(json.dumps(_api_record(isbn13=None)) + "\n", encoding="utf-8")
            self.assertEqual(load_book_show_api_books(path), {})

    def test_skips_records_missing_title_or_author(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_api_record(title=None)) + "\n")
                f.write(json.dumps(_api_record(legacy_id=3, author=None)) + "\n")
            self.assertEqual(load_book_show_api_books(path), {})

    def test_skips_incomplete_record_warnings(self) -> None:
        # incomplete_record rows never carry a title -- confirmed empirically
        # in consolidate_popularity_signals.py; this is the deferred remainder.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(
                json.dumps({"_scrape_warning": "incomplete_record", "_url": "/book/show/5", "ratings_count": 500})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_book_show_api_books(path), {})

    def test_missing_path_returns_empty(self) -> None:
        self.assertEqual(load_book_show_api_books(None), {})
        self.assertEqual(load_book_show_api_books(Path("/nonexistent.jsonl")), {})

    def test_non_numeric_average_rating_and_ratings_count_become_none(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book_show_api.jsonl"
            path.write_text(
                json.dumps(_api_record(average_rating="n/a", ratings_count=None)) + "\n", encoding="utf-8"
            )
            book = load_book_show_api_books(path)[2]
            self.assertIsNone(book.avg_rating)
            self.assertIsNone(book.ratings_count)


class TestValidateListShowSchema(unittest.TestCase):
    _VALID_RECORD = {
        "book_urls": ["/book/show/33.T"],
        "titles": ["T"],
        "authors": ["A"],
        "rating_texts": ["4.0 avg rating — 1 ratings"],
        "score_texts": ["score: 1"],
        "vote_texts": ["1 people voted"],
    }

    def test_passes_when_all_required_fields_present(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            raw_dir = Path(d)
            (raw_dir / "367.jsonl").write_text(json.dumps(self._VALID_RECORD) + "\n", encoding="utf-8")
            validate_list_show_schema(raw_dir)  # should not raise

    def test_extra_fields_are_fine(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            raw_dir = Path(d)
            record = dict(self._VALID_RECORD, list_id="367", list_name="Best Fantasy Books")
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            validate_list_show_schema(raw_dir)  # should not raise

    def test_raises_when_a_required_field_is_missing(self) -> None:
        record = {k: v for k, v in self._VALID_RECORD.items() if k != "titles"}
        with tempfile.TemporaryDirectory() as d:
            raw_dir = Path(d)
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(SchemaError) as ctx:
                validate_list_show_schema(raw_dir)
            self.assertIn("titles", str(ctx.exception))

    def test_only_checks_first_record_per_file(self) -> None:
        # A malformed second line should not be reached/raise anything.
        with tempfile.TemporaryDirectory() as d:
            raw_dir = Path(d)
            (raw_dir / "367.jsonl").write_text(
                json.dumps(self._VALID_RECORD) + "\nnot valid json at all\n", encoding="utf-8"
            )
            validate_list_show_schema(raw_dir)  # should not raise

    def test_missing_dir_is_a_noop(self) -> None:
        validate_list_show_schema(Path("/nonexistent/raw/dir"))  # should not raise


class TestMatching(unittest.TestCase):
    def _candidates(self) -> list[OLCandidate]:
        rows = [
            ("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 500),
            ("/works/OL2W", "The Hobbit", "J. R. R. Tolkien", 400),
            ("/works/OL3W", "A Game of Thrones", "George R. R. Martin", 300),
        ]
        return [
            OLCandidate(
                work_key=wk,
                title=title,
                author=author,
                title_normalized=normalize_for_search(title),
                author_normalized=normalize_for_search(author),
                edition_count=ec,
            )
            for wk, title, author, ec in rows
        ]

    def test_isbn_match_takes_priority(self) -> None:
        book = GoodreadsBook(book_id=1, title="Some Totally Different Title", author="Some Other Author", isbn13="9780618640157")
        index = AuthorBlockIndex(self._candidates())
        isbn_index = {"9780618640157": ["/works/OL1W"]}
        result = match_book(book, index, isbn_index)
        self.assertEqual(result.method, "isbn")
        self.assertEqual(result.work_key, "/works/OL1W")
        self.assertEqual(result.score, 100.0)

    def test_title_author_match_accepts_close_title_author(self) -> None:
        book = GoodreadsBook(
            book_id=33, title="The Lord of the Rings (The Lord of the Rings, #1-3)", author="J.R.R. Tolkien"
        )
        index = AuthorBlockIndex(self._candidates())
        result = match_book(book, index, {})
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/OL1W")
        self.assertGreaterEqual(result.score, 90.0)

    def test_skip_isbn_uses_title_author_even_when_isbn_hits(self) -> None:
        book = GoodreadsBook(
            book_id=1, title="The Lord of the Rings", author="J.R.R. Tolkien", isbn13="9780618640157"
        )
        index = AuthorBlockIndex(self._candidates())
        isbn_index = {"9780618640157": ["/works/OL9W"]}
        result = match_book(book, index, isbn_index, use_isbn=False)
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/OL1W")

    def test_no_author_block_candidates_is_unmatched(self) -> None:
        book = GoodreadsBook(book_id=99, title="Some Book", author="Someone Nobody Wrote About")
        index = AuthorBlockIndex(self._candidates())
        result = match_book(book, index, {})
        self.assertEqual(result.method, "unmatched")

    def test_ambiguous_when_two_same_author_titles_score_similarly(self) -> None:
        # "The Lord of the Ring" (typo/near-dup of two different real
        # Tolkien titles) should not blow past the margin test.
        candidates = [
            OLCandidate("/works/A", "The Lord of the Rings", "Tolkien", normalize_for_search("The Lord of the Rings"), normalize_for_search("Tolkien"), 1),
            OLCandidate("/works/B", "The Lord of the Rings Companion", "Tolkien", normalize_for_search("The Lord of the Rings Companion"), normalize_for_search("Tolkien"), 1),
        ]
        book = GoodreadsBook(book_id=1, title="The Lord of the Rings Guide", author="Tolkien")
        index = AuthorBlockIndex(candidates)
        result = match_book(book, index, {})
        self.assertIn(result.method, ("ambiguous", "title_author"))
        if result.method == "ambiguous":
            self.assertIsNone(result.work_key)

    def test_dedupes_by_work_key_before_margin_test(self) -> None:
        # Two rows sharing a workKey (e.g. two editions) must not count as
        # two distinct runner-ups against each other.
        candidates = [
            OLCandidate("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", normalize_for_search("The Lord of the Rings"), normalize_for_search("J. R. R. Tolkien"), 100),
            OLCandidate("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", normalize_for_search("The Lord of the Rings"), normalize_for_search("J. R. R. Tolkien"), 50),
        ]
        book = GoodreadsBook(book_id=33, title="The Lord of the Rings", author="J.R.R. Tolkien")
        index = AuthorBlockIndex(candidates)
        result = match_book(book, index, {})
        self.assertEqual(result.method, "title_author")


class TestComputeShelfScore(unittest.TestCase):
    def test_higher_rating_and_more_ratings_scores_higher(self) -> None:
        low = GoodreadsBook(book_id=1, title="A", author="X", avg_rating=3.0, ratings_count=10, list_appearances=1)
        high = GoodreadsBook(book_id=2, title="B", author="Y", avg_rating=4.8, ratings_count=500000, list_appearances=5)
        self.assertGreater(compute_shelf_score(high, edition_count=50), compute_shelf_score(low, edition_count=50))

    def test_missing_rating_data_does_not_crash(self) -> None:
        book = GoodreadsBook(book_id=1, title="A", author="X")
        score = compute_shelf_score(book, edition_count=0)
        self.assertGreaterEqual(score, 0.0)

    def test_score_is_bounded_reasonably(self) -> None:
        book = GoodreadsBook(book_id=1, title="A", author="X", avg_rating=5.0, ratings_count=10_000_000, list_appearances=100)
        score = compute_shelf_score(book, edition_count=1_000_000)
        self.assertLessEqual(score, 1.5)  # each term capped at 1.0, weights sum to 1.0 -> should stay close to that

    def test_consolidated_signal_is_ignored_when_book_already_has_list_signal(self) -> None:
        # Additive-only guarantee: a book that already has real list_show
        # avg_rating/ratings_count must score identically whether or not a
        # consolidated signal is passed, even if that signal disagrees.
        book = GoodreadsBook(book_id=1, title="A", author="X", avg_rating=4.5, ratings_count=1000, list_appearances=2)
        signal = ConsolidatedSignal(book_id=1, api_avg_rating=1.0, api_ratings_count=1)
        without_signal = compute_shelf_score(book, edition_count=10)
        with_signal = compute_shelf_score(book, edition_count=10, signal=signal)
        self.assertEqual(without_signal, with_signal)

    def test_consolidated_signal_fills_gap_when_list_show_rating_is_missing(self) -> None:
        book = GoodreadsBook(book_id=1, title="A", author="X", list_appearances=1)
        signal = ConsolidatedSignal(book_id=1, api_avg_rating=4.5, api_ratings_count=500_000)
        without_signal = compute_shelf_score(book, edition_count=10)
        with_signal = compute_shelf_score(book, edition_count=10, signal=signal)
        self.assertGreater(with_signal, without_signal)

    def test_consolidated_signal_fills_only_the_missing_half(self) -> None:
        # ratings_count present from list_show, avg_rating missing -> the
        # gap-filled score should match a book that had avg_rating natively,
        # i.e. the real ratings_count is never overridden by the signal's
        # (deliberately different) api_ratings_count=1.
        book = GoodreadsBook(book_id=1, title="A", author="X", ratings_count=500_000, list_appearances=1)
        signal = ConsolidatedSignal(book_id=1, api_avg_rating=4.5, api_ratings_count=1)
        with_signal = compute_shelf_score(book, edition_count=10, signal=signal)
        as_if_native = compute_shelf_score(
            GoodreadsBook(book_id=1, title="A", author="X", avg_rating=4.5, ratings_count=500_000, list_appearances=1),
            edition_count=10,
        )
        self.assertAlmostEqual(with_signal, as_if_native)

    def test_default_weights_sum_to_one(self) -> None:
        w = DEFAULT_SHELF_SCORE_WEIGHTS
        self.assertAlmostEqual(w.rating + w.ratings_count + w.list_ + w.edition, 1.0)

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ShelfScoreWeights(rating=0.5, ratings_count=0.5, list_=0.5, edition=0.5)

    def test_high_ratings_count_beats_high_avg_rating_alone_2b1(self) -> None:
        # 2b-0/2b-1's whole point: ratings_count (true engagement volume),
        # not avg_rating (weakly *inversely* correlated with it -- see
        # DEFAULT_SHELF_SCORE_WEIGHTS's comment), should drive the ranking.
        # A perfectly-rated niche book must not outrank a merely-good
        # mega-bestseller.
        niche_perfect = GoodreadsBook(book_id=1, title="A", author="X", avg_rating=5.0, ratings_count=200)
        mega_bestseller = GoodreadsBook(book_id=2, title="B", author="Y", avg_rating=3.9, ratings_count=5_000_000)
        self.assertGreater(
            compute_shelf_score(mega_bestseller, edition_count=100),
            compute_shelf_score(niche_perfect, edition_count=100),
        )

    def test_ratings_count_term_is_uncapped_past_one_million(self) -> None:
        # The old min(ratings_count_term, 1.0) ceiling made every book past
        # ~1M ratings tie on this term -- exactly the range a 50k-capped
        # catalog cares about getting right. 10M ratings must now score
        # strictly higher than 1M, all else equal.
        one_million = GoodreadsBook(book_id=1, title="A", author="X", ratings_count=1_000_000)
        ten_million = GoodreadsBook(book_id=2, title="B", author="Y", ratings_count=10_000_000)
        self.assertGreater(
            compute_shelf_score(ten_million, edition_count=0),
            compute_shelf_score(one_million, edition_count=0),
        )

    def test_list_term_normalizes_against_the_live_seed_list_count(self) -> None:
        # Same list_appearances, different total_seed_lists -> the book
        # should score a *smaller* list contribution when there are more
        # real lists it could have been on but wasn't (this is the "stale
        # 11 vs live 21" bug's fix).
        book = GoodreadsBook(book_id=1, title="A", author="X", list_appearances=10)
        score_against_11 = compute_shelf_score(book, edition_count=0, total_seed_lists=11)
        score_against_21 = compute_shelf_score(book, edition_count=0, total_seed_lists=21)
        self.assertGreater(score_against_11, score_against_21)

    def test_list_term_reaches_full_weight_only_at_the_live_seed_list_count(self) -> None:
        weights = DEFAULT_SHELF_SCORE_WEIGHTS
        appeared_on_all = GoodreadsBook(book_id=1, title="A", author="X", list_appearances=21)
        score = compute_shelf_score(appeared_on_all, edition_count=0, total_seed_lists=21)
        # Every other term is 0 (no rating/ratings_count/editions) -> the
        # full score should be exactly the list weight.
        self.assertAlmostEqual(score, weights.list_)

    def test_zero_total_seed_lists_does_not_crash(self) -> None:
        book = GoodreadsBook(book_id=1, title="A", author="X")
        score = compute_shelf_score(book, edition_count=0, total_seed_lists=0)
        self.assertGreaterEqual(score, 0.0)

    def test_custom_weights_override_the_default(self) -> None:
        book = GoodreadsBook(book_id=1, title="A", author="X", avg_rating=5.0)
        all_weight_on_rating = ShelfScoreWeights(rating=1.0, ratings_count=0.0, list_=0.0, edition=0.0)
        score = compute_shelf_score(book, edition_count=0, weights=all_weight_on_rating)
        self.assertAlmostEqual(score, 1.0)  # avg_rating=5.0 -> rating_term=1.0, sole weight=1.0


class TestOLLoaders(unittest.TestCase):
    def test_load_ol_candidates_and_isbn_index(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 42)])
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO book_isbns (isbn13, workKey) VALUES (?, ?)", ("9780441013593", "/works/OL1W"))
            conn.commit()
            candidates = load_ol_candidates(conn)
            isbn_index = load_ol_isbn_index(conn)
            conn.close()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edition_count, 42)
        self.assertEqual(isbn_index, {"9780441013593": ["/works/OL1W"]})


class TestRunEndToEnd(unittest.TestCase):
    def test_run_writes_matched_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            record = {
                "book_urls": ["/book/show/33.The_Lord_of_the_Rings"],
                "titles": ["The Lord of the Rings (The Lord of the Rings, #1-3)"],
                "authors": ["J.R.R. Tolkien"],
                "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                "score_texts": ["score: 42,463"],
                "vote_texts": ["430 people voted"],
            }
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 500)])

            seed_lists_path = tmp / "seed.yaml"
            seed_lists_path.write_text(
                "lists:\n  - list_id: 367\n    slug: Best_Fantasy_Books\n    genre: fantasy\n", encoding="utf-8"
            )

            out_path = tmp / "matched.jsonl.gz"
            counts = run(raw_dir, db_path, out_path, seed_lists_path=seed_lists_path)

            self.assertEqual(counts, {"title_author": 1, "book_show_api_only": 0})
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

            genre_tags_path = tmp / "genre_tags.json"
            self.assertTrue(genre_tags_path.exists(), "genre_tags.json should be written next to matched output")
            genre_tags = json.loads(genre_tags_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["goodreads_book_id"], 33)
        self.assertEqual(rows[0]["work_key"], "/works/OL1W")
        self.assertEqual(rows[0]["genres"], ["fantasy"])
        self.assertIsNotNone(rows[0]["shelf_score"])
        self.assertEqual(rows[0]["edition_count"], 500)  # persisted for a reweight sweep -- see 2b-1
        self.assertEqual(
            genre_tags,
            {"fantasy": {"short_label": "Fantasy", "list_title": "Best Fantasy Books", "slug": "Best_Fantasy_Books", "list_id": 367}},
        )

    def test_run_respects_explicit_genre_tags_out_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            record = {
                "book_urls": ["/book/show/33.The_Lord_of_the_Rings"],
                "titles": ["The Lord of the Rings"],
                "authors": ["J.R.R. Tolkien"],
                "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                "score_texts": ["score: 42,463"],
                "vote_texts": ["430 people voted"],
            }
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 500)])

            seed_lists_path = tmp / "seed.yaml"
            seed_lists_path.write_text("lists:\n  - list_id: 367\n    slug: X\n    genre: fantasy\n", encoding="utf-8")

            out_path = tmp / "nested" / "matched.jsonl.gz"
            genre_tags_out_path = tmp / "elsewhere" / "tags.json"
            run(
                raw_dir,
                db_path,
                out_path,
                seed_lists_path=seed_lists_path,
                genre_tags_out_path=genre_tags_out_path,
            )
            self.assertTrue(genre_tags_out_path.exists())

    def test_run_raises_schema_error_on_broken_list_show_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            # Missing titles/authors/rating_texts — the exact breakage this
            # validation exists to catch (see module's SchemaError docstring).
            record = {"book_urls": ["/book/show/33.T"], "score_texts": ["score: 1"], "vote_texts": ["1 people voted"]}
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [])

            with self.assertRaises(SchemaError):
                run(raw_dir, db_path, tmp / "matched.jsonl.gz")

    def test_run_merges_book_show_api_isbns_with_precedence_over_book_show_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            record = {
                "book_urls": ["/book/show/33.The_Lord_of_the_Rings"],
                "titles": ["The Lord of the Rings"],
                "authors": ["J.R.R. Tolkien"],
                "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                "score_texts": ["score: 42,463"],
                "vote_texts": ["430 people voted"],
            }
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 500)])
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO book_isbns (isbn13, workKey) VALUES ('9780000000000', '/works/OL1W')")
            conn.commit()
            conn.close()

            seed_lists_path = tmp / "seed.yaml"
            seed_lists_path.write_text(
                "lists:\n  - list_id: 367\n    slug: Best_Fantasy_Books\n    genre: fantasy\n", encoding="utf-8"
            )

            # book_show_dir (blob) gives a stale ISBN; book_show_api (direct
            # fields) gives the correct one and should win.
            book_show_dir = tmp / "book_show"
            book_show_dir.mkdir()
            (book_show_dir / "33.jsonl").write_text(
                json.dumps(
                    {
                        "next_data_json": json.dumps(
                            {
                                "props": {
                                    "pageProps": {
                                        "apolloState": {
                                            "Book:x": {
                                                "legacyId": 33,
                                                "title": "T",
                                                "details": {"isbn13": "9999999999999"},
                                            }
                                        }
                                    }
                                }
                            }
                        )
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            book_show_api_path = tmp / "book_show_api.jsonl"
            book_show_api_path.write_text(
                json.dumps({"legacy_id": 33, "isbn13": "9780000000000"}) + "\n", encoding="utf-8"
            )

            out_path = tmp / "matched.jsonl.gz"
            counts = run(
                raw_dir,
                db_path,
                out_path,
                book_show_dir=book_show_dir,
                book_show_api_path=book_show_api_path,
                seed_lists_path=seed_lists_path,
            )

            self.assertEqual(counts, {"isbn": 1, "book_show_api_only": 0})
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(rows[0]["work_key"], "/works/OL1W")
            self.assertEqual(rows[0]["match_method"], "isbn")

    def test_run_passes_the_live_seed_list_count_to_shelf_score(self) -> None:
        # Two seed lists defined, but the book only appears on one --
        # run() must normalize compute_shelf_score's list_term against the
        # live count (2), not a stale hardcoded constant.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            record = {
                "book_urls": ["/book/show/33.The_Lord_of_the_Rings"],
                "titles": ["The Lord of the Rings"],
                "authors": ["J.R.R. Tolkien"],
                "rating_texts": [None],
                "score_texts": [None],
                "vote_texts": [None],
            }
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 0)])

            seed_lists_path = tmp / "seed.yaml"
            seed_lists_path.write_text(
                "lists:\n"
                "  - list_id: 367\n    slug: Best_Fantasy_Books\n    genre: fantasy\n"
                "  - list_id: 999\n    slug: Best_Scifi_Books\n    genre: scifi\n",
                encoding="utf-8",
            )

            out_path = tmp / "matched.jsonl.gz"
            run(raw_dir, db_path, out_path, seed_lists_path=seed_lists_path)
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                row = json.loads(next(iter(f)))

            expected = compute_shelf_score(
                GoodreadsBook(book_id=33, title="x", author="y", list_appearances=1), edition_count=0,
                total_seed_lists=2,
            )
            self.assertAlmostEqual(row["shelf_score"], expected)

    def test_book_show_api_only_book_is_added_and_resolved_via_isbn(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            # list_show only knows about book 33; book 2 never appeared on
            # any seeded list, but book_show_api has full signal for it.
            record = {
                "book_urls": ["/book/show/33.The_Lord_of_the_Rings"],
                "titles": ["The Lord of the Rings"],
                "authors": ["J.R.R. Tolkien"],
                "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                "score_texts": ["score: 42,463"],
                "vote_texts": ["430 people voted"],
            }
            (raw_dir / "367.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(
                db_path,
                [
                    ("/works/OL1W", "The Lord of the Rings", "J. R. R. Tolkien", 500),
                    ("/works/HP5", "Harry Potter and the Order of the Phoenix", "J. K. Rowling", 300),
                ],
            )
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO book_isbns (isbn13, workKey) VALUES ('9780439686525', '/works/HP5')")
            conn.commit()
            conn.close()

            book_show_api_path = tmp / "book_show_api.jsonl"
            book_show_api_path.write_text(json.dumps(_api_record()) + "\n", encoding="utf-8")

            out_path = tmp / "matched.jsonl.gz"
            counts = run(raw_dir, db_path, out_path, book_show_api_path=book_show_api_path)

            self.assertEqual(counts["book_show_api_only"], 1)
            self.assertEqual(counts["isbn"], 1)  # book 2, via the newly-added isbn13
            self.assertEqual(counts["title_author"], 1)  # book 33, via list_show + bibliographic join

            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                parsed_rows = [json.loads(line) for line in f]
            rows = {r["goodreads_book_id"]: r for r in parsed_rows}
            row = rows[2]
            self.assertEqual(row["match_method"], "isbn")
            self.assertEqual(row["work_key"], "/works/HP5")
            self.assertEqual(row["list_appearances"], 0)
            self.assertEqual(row["genres"], [])  # not populated from the api record's own genres
            self.assertIsNotNone(row["shelf_score"])

    def test_book_show_api_record_does_not_duplicate_an_existing_list_show_book(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            record = {
                "book_urls": ["/book/show/2.Harry_Potter"],
                "titles": ["Harry Potter and the Order of the Phoenix"],
                "authors": ["J.K. Rowling"],
                "rating_texts": ["4.47 avg rating — 11,742,238 ratings"],
                "score_texts": ["score: 1"],
                "vote_texts": ["1 people voted"],
            }
            (raw_dir / "1.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/HP5", "Harry Potter and the Order of the Phoenix", "J. K. Rowling", 300)])

            book_show_api_path = tmp / "book_show_api.jsonl"
            book_show_api_path.write_text(json.dumps(_api_record()) + "\n", encoding="utf-8")

            out_path = tmp / "matched.jsonl.gz"
            counts = run(raw_dir, db_path, out_path, book_show_api_path=book_show_api_path)

            self.assertEqual(counts["book_show_api_only"], 0)  # book 2 already existed via list_show
            with gzip.open(out_path, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["list_appearances"], 1)  # the real list_show appearance, not 0

    def test_book_show_api_only_book_still_added_when_use_isbn_false(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            (raw_dir / "1.jsonl").write_text("", encoding="utf-8")

            db_path = tmp / "full.sqlite"
            _make_ol_db(db_path, [("/works/HP5", "Harry Potter and the Order of the Phoenix", "J. K. Rowling", 300)])

            book_show_api_path = tmp / "book_show_api.jsonl"
            book_show_api_path.write_text(json.dumps(_api_record()) + "\n", encoding="utf-8")

            out_path = tmp / "matched.jsonl.gz"
            counts = run(raw_dir, db_path, out_path, book_show_api_path=book_show_api_path, use_isbn=False)

            self.assertEqual(counts["book_show_api_only"], 1)
            self.assertEqual(counts.get("isbn", 0), 0)  # isbn overlay disabled -- resolves via bibliographic join
            self.assertEqual(counts["title_author"], 1)


def _isbn_holdout_fixture(tmp: Path, n_books: int) -> tuple[Path, Path]:
    """`n_books` distinct GR books, each with a real ISBN that hits a
    distinct OL work with matching title+author -- every one should be a
    clean identity-recall hit, so fold filtering is the only thing under
    test (not the matcher itself)."""
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    records = []
    ol_rows: list[tuple[str, str, str, int]] = []
    isbn_rows: list[tuple[str, str]] = []
    for i in range(n_books):
        book_id = i + 1
        title = f"Book Number {book_id}"
        author = f"Author {book_id}"
        isbn13 = f"97800000{book_id:05d}"
        work_key = f"/works/OL{book_id}"
        records.append(
            {
                "book_urls": [f"/book/show/{book_id}"],
                "titles": [title],
                "authors": [author],
                "rating_texts": ["4.00 avg rating — 100 ratings"],
                "score_texts": ["score: 1"],
                "vote_texts": ["1 people voted"],
            }
        )
        ol_rows.append((work_key, title, author, 10))
        isbn_rows.append((isbn13, work_key))
    with (raw_dir / "1.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    db_path = tmp / "full.sqlite"
    _make_ol_db(db_path, ol_rows)
    conn = sqlite3.connect(db_path)
    conn.executemany("INSERT INTO book_isbns (isbn13, workKey) VALUES (?, ?)", isbn_rows)
    conn.commit()
    conn.close()

    book_show_api_path = tmp / "book_show_api.jsonl"
    with book_show_api_path.open("w", encoding="utf-8") as f:
        for i in range(n_books):
            book_id = i + 1
            f.write(json.dumps({"legacy_id": book_id, "isbn13": f"97800000{book_id:05d}"}) + "\n")

    return raw_dir, db_path, book_show_api_path


class TestRunIsbnHoldoutEvalFold(unittest.TestCase):
    def test_fold_all_covers_every_gold_pair(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(Path(d), 40)
            report = run_isbn_holdout_eval(raw_dir, db_path, book_show_api_path=book_show_api_path)
            self.assertEqual(report["gold_pairs"], 40)
            self.assertEqual(report["fold"], "all")

    def test_tuning_and_validation_folds_partition_the_gold_set(self) -> None:
        from tools.catalog.bibliographic_join import fold_for

        with tempfile.TemporaryDirectory() as d:
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(Path(d), 40)
            tuning = run_isbn_holdout_eval(raw_dir, db_path, book_show_api_path=book_show_api_path, fold="tuning")
            validation = run_isbn_holdout_eval(
                raw_dir, db_path, book_show_api_path=book_show_api_path, fold="validation"
            )
            expected_tuning = sum(1 for i in range(1, 41) if fold_for(i) == "tuning")
            expected_validation = sum(1 for i in range(1, 41) if fold_for(i) == "validation")
            self.assertEqual(tuning["gold_pairs"], expected_tuning)
            self.assertEqual(validation["gold_pairs"], expected_validation)
            self.assertEqual(tuning["gold_pairs"] + validation["gold_pairs"], 40)

    def test_adversarial_pairs_are_folded_into_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(tmp, 2)
            # run_isbn_holdout_eval only probes OL for titles it saw among
            # the *raw* GR books (load_ol_candidates_for_titles) -- add a
            # "Twilight" list_show row (no ISBN needed) purely so that
            # probe fires and pulls in the rigged /works/BAD candidate
            # below into the same index the adversarial check runs against.
            with (raw_dir / "1.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "book_urls": ["/book/show/9999"],
                            "titles": ["Twilight"],
                            "authors": ["Nobody In Particular"],
                            "rating_texts": ["4.00 avg rating — 1 ratings"],
                            "score_texts": ["score: 1"],
                            "vote_texts": ["1 people voted"],
                        }
                    )
                    + "\n"
                )
            # A bad candidate that wrongly ties two unrelated authors
            # together under one work, to exercise real false-merge
            # detection through the actual OL-index-building code path.
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO books (workKey, title, author, titleNormalized, authorNormalized, editionCount) "
                "VALUES ('/works/BAD', 'Twilight', 'Stephenie Meyer and Dean Koontz', "
                "'twilight', 'stephenie meyer and dean koontz', 5)"
            )
            conn.commit()
            conn.close()

            pairs_path = tmp / "adversarial.yaml"
            pairs_path.write_text(
                "pairs:\n"
                '  - title_a: "Twilight"\n'
                '    author_a: "Stephenie Meyer"\n'
                '    title_b: "Twilight"\n'
                '    author_b: "Dean Koontz"\n',
                encoding="utf-8",
            )
            report = run_isbn_holdout_eval(
                raw_dir, db_path, book_show_api_path=book_show_api_path, adversarial_pairs_path=pairs_path
            )
            self.assertEqual(report["adversarial_total"], 1)
            self.assertEqual(report["false_merges"], 1)

    def test_no_adversarial_pairs_path_omits_false_merges_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(Path(d), 2)
            report = run_isbn_holdout_eval(raw_dir, db_path, book_show_api_path=book_show_api_path)
            self.assertNotIn("false_merges", report)


class TestEvalPersistence(unittest.TestCase):
    def test_git_head_sha_returns_unknown_outside_a_repo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(git_head_sha(Path(d)), "unknown")

    def test_git_head_sha_is_nonempty_inside_this_repo(self) -> None:
        sha = git_head_sha(Path(__file__).resolve().parents[2])
        self.assertNotEqual(sha, "unknown")
        self.assertEqual(len(sha), 8)

    def test_matcher_eval_default_path_naming(self) -> None:
        base = Path("/tmp/base")
        ts = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
        path = matcher_eval_default_path(base, "abcd1234", ts)
        self.assertEqual(path, base / "matcher_eval" / "20260903T120000Z_abcd1234.json")

    def test_write_eval_report_includes_sha_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            ts = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
            write_eval_report(path, {"identity_recall": 0.9}, sha="abcd1234", timestamp=ts)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["git_sha"], "abcd1234")
            self.assertEqual(payload["timestamp_utc"], "2026-09-03T12:00:00Z")
            self.assertEqual(payload["identity_recall"], 0.9)


class TestMainEvalOutCli(unittest.TestCase):
    def test_eval_out_auto_writes_under_matcher_eval(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(tmp, 3)
            # catalog_goodreads() base dir is monkeypatched via --eval-out
            # explicit path instead of 'auto', so this test doesn't depend
            # on the real $BOOK_SPINES_DATA env var being set.
            out_path = tmp / "report.json"
            rc = main(
                [
                    "--ol-db",
                    str(db_path),
                    "--raw-dir",
                    str(raw_dir),
                    "--book-show-api",
                    str(book_show_api_path),
                    "--eval-isbn-holdout",
                    "--eval-out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["gold_pairs"], 3)
            self.assertIn("git_sha", payload)
            self.assertIn("timestamp_utc", payload)

    def test_no_eval_out_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir, db_path, book_show_api_path = _isbn_holdout_fixture(tmp, 1)
            rc = main(
                [
                    "--ol-db",
                    str(db_path),
                    "--raw-dir",
                    str(raw_dir),
                    "--book-show-api",
                    str(book_show_api_path),
                    "--eval-isbn-holdout",
                ]
            )
            self.assertEqual(rc, 0)
            # Nothing under tmp besides the fixture's own raw/book_show_api files.
            self.assertFalse((tmp / "matcher_eval").exists())


if __name__ == "__main__":
    unittest.main()
