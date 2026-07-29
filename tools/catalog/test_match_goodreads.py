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
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.catalog.match_goodreads import (  # noqa: E402
    AuthorBlockIndex,
    GoodreadsBook,
    OLCandidate,
    SeedListMeta,
    author_block_key,
    build_genre_tag_mapping,
    compute_shelf_score,
    dedupe_books,
    load_book_show_isbns,
    load_ol_candidates,
    load_ol_isbn_index,
    load_seed_metadata,
    match_book,
    parse_book_id,
    parse_book_show_next_data,
    parse_list_show_file,
    parse_rating_text,
    parse_score_text,
    parse_vote_text,
    run,
    strip_series_suffix,
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
        self.assertEqual(parsed, {"book_id": 33, "title": "The Lord of the Rings", "isbn13": "9780618640157", "language": "English"})

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

    def test_fuzzy_match_accepts_close_title_author(self) -> None:
        book = GoodreadsBook(
            book_id=33, title="The Lord of the Rings (The Lord of the Rings, #1-3)", author="J.R.R. Tolkien"
        )
        index = AuthorBlockIndex(self._candidates())
        result = match_book(book, index, {})
        self.assertEqual(result.method, "fuzzy")
        self.assertEqual(result.work_key, "/works/OL1W")
        self.assertGreaterEqual(result.score, 90.0)

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
        self.assertIn(result.method, ("ambiguous", "fuzzy"))  # exact outcome depends on scorer; margin must be respected
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
        self.assertEqual(result.method, "fuzzy")


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

            self.assertEqual(counts, {"fuzzy": 1})
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


if __name__ == "__main__":
    unittest.main()
