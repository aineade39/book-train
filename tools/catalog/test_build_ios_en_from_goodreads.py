#!/usr/bin/env python3
"""Unit tests for tools/catalog/build_ios_en_from_goodreads.py
(run: python tools/catalog/test_build_ios_en_from_goodreads.py).

No Swift toolchain required: `swift run catalog-build` is mocked throughout.
No real full.sqlite/works.jsonl.gz: both are built as small fixtures.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import normalize_for_search, write_jsonl_gz  # noqa: E402
from tools.catalog.build_ios_en_from_goodreads import (  # noqa: E402
    GAP_FILL_WORK_KEY_PREFIX,
    IntermediateMismatchError,
    add_languages_column,
    copy_to_scratch,
    create_goodreads_signals_table,
    default_work_key,
    gap_fill_unmatched,
    load_matched_goodreads,
    load_work_languages,
    popularity_signal,
    populate_goodreads_signals,
    populate_languages,
    prune_to_languages,
    rerank_popularity,
    run,
)


def _make_full_db(path: Path, rows: list[tuple[str, str, str, int, int]]) -> None:
    """rows: (workKey, title, author, popularityRank, editionCount) — mirrors
    Sources/SpineCatalog/BookCatalog.swift's `books` table (no languages column)."""
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
    for work_key, title, author, rank, edition_count in rows:
        conn.execute(
            "INSERT INTO books (workKey, title, author, titleNormalized, authorNormalized, popularityRank, editionCount) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (work_key, title, author, normalize_for_search(title), normalize_for_search(author), rank, edition_count),
        )
    conn.commit()
    conn.close()


def _make_intermediate(dir_path: Path, rows: list[tuple[str, list[str]]]) -> None:
    """rows: (workKey, languages)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(
        dir_path / "works.jsonl.gz",
        (
            {"workKey": wk, "title": "t", "author": "a", "isbn13": None, "editionCount": 1, "popularityRank": i, "languages": langs}
            for i, (wk, langs) in enumerate(rows, start=1)
        ),
    )


def _write_matched_goodreads(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestCopyToScratch(unittest.TestCase):
    def test_copies_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "full.sqlite"
            _make_full_db(src, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            dst = Path(d) / "scratch.sqlite"
            copy_to_scratch(src, dst)
            self.assertTrue(dst.exists())
            conn = sqlite3.connect(dst)
            count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            conn.close()
        self.assertEqual(count, 1)

    def test_overwrites_existing_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "full.sqlite"
            _make_full_db(src, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            dst = Path(d) / "scratch.sqlite"
            dst.write_text("garbage", encoding="utf-8")
            copy_to_scratch(src, dst)
            conn = sqlite3.connect(dst)
            count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            conn.close()
        self.assertEqual(count, 1)


class TestLanguages(unittest.TestCase):
    def test_add_languages_column_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            add_languages_column(conn)  # must not raise "duplicate column"
            columns = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
            conn.close()
        self.assertIn("languages", columns)

    def test_populate_languages_matches_by_work_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            total, matched = populate_languages(conn, {"/works/OL1W": {"eng"}})
            langs = conn.execute("SELECT languages FROM books WHERE workKey = ?", ("/works/OL1W",)).fetchone()[0]
            conn.close()
        self.assertEqual((total, matched), (1, 1))
        self.assertEqual(langs, "eng")

    def test_populate_languages_unmatched_work_key_stays_null(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            total, matched = populate_languages(conn, {})
            conn.close()
        self.assertEqual((total, matched), (1, 0))

    def test_prune_to_languages_removes_non_matching_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(
                db_path,
                [
                    ("/works/OL1W", "Dune", "Frank Herbert", 1, 10),
                    ("/works/OL2W", "Le Petit Prince", "Antoine de Saint-Exupery", 2, 5),
                    ("/works/OL3W", "Unknown Lang Book", "Someone", 3, 1),
                ],
            )
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            populate_languages(conn, {"/works/OL1W": {"eng"}, "/works/OL2W": {"fre"}})
            removed = prune_to_languages(conn, frozenset({"eng"}))
            remaining = [row[0] for row in conn.execute("SELECT workKey FROM books")]
            conn.close()
        self.assertEqual(removed, 2)
        self.assertEqual(remaining, ["/works/OL1W"])

    def test_prune_also_cleans_up_orphaned_isbns(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10), ("/works/OL2W", "French Book", "X", 2, 1)])
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO book_isbns (isbn13, workKey) VALUES ('9780441013593', '/works/OL1W')")
            conn.execute("INSERT INTO book_isbns (isbn13, workKey) VALUES ('9999999999999', '/works/OL2W')")
            conn.commit()
            add_languages_column(conn)
            populate_languages(conn, {"/works/OL1W": {"eng"}, "/works/OL2W": {"fre"}})
            prune_to_languages(conn, frozenset({"eng"}))
            remaining_isbns = [row[0] for row in conn.execute("SELECT isbn13 FROM book_isbns")]
            conn.close()
        self.assertEqual(remaining_isbns, ["9780441013593"])

    def test_load_work_languages_from_intermediate(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            intermediate_dir = Path(d) / "intermediate"
            _make_intermediate(intermediate_dir, [("/works/OL1W", ["eng"]), ("/works/OL2W", ["fre", "ger"])])
            langs = load_work_languages(intermediate_dir)
        self.assertEqual(langs, {"/works/OL1W": {"eng"}, "/works/OL2W": {"fre", "ger"}})


class TestGoodreadsSignals(unittest.TestCase):
    def test_populate_goodreads_signals_skips_rows_without_work_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            n = populate_goodreads_signals(
                conn,
                [
                    {"work_key": "/works/OL1W", "goodreads_book_id": 1, "shelf_score": 0.8},
                    {"work_key": None, "goodreads_book_id": 2, "shelf_score": None},
                ],
            )
            rows = conn.execute("SELECT workKey, shelfScore FROM goodreads_signals").fetchall()
            conn.close()
        self.assertEqual(n, 1)
        self.assertEqual(rows, [("/works/OL1W", 0.8)])

    def test_load_matched_goodreads_reads_gzipped_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "matched.jsonl.gz"
            _write_matched_goodreads(path, [{"work_key": "/works/OL1W"}, {"work_key": None}])
            rows = load_matched_goodreads(path)
        self.assertEqual(len(rows), 2)


class TestPopularitySignal(unittest.TestCase):
    def test_rank_one_is_near_one(self) -> None:
        self.assertAlmostEqual(popularity_signal(1, 1000), 1.0, places=6)

    def test_last_rank_is_near_zero(self) -> None:
        self.assertAlmostEqual(popularity_signal(1000, 1000), 0.0, places=6)

    def test_none_rank_is_zero(self) -> None:
        self.assertEqual(popularity_signal(None, 1000), 0.0)

    def test_monotonically_decreasing_with_rank(self) -> None:
        self.assertGreater(popularity_signal(10, 1000), popularity_signal(500, 1000))


class TestRerankPopularity(unittest.TestCase):
    def test_high_shelf_score_can_outrank_higher_ol_popularity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            # OL1W has better (lower) OL popularityRank than OL2W, but OL2W
            # has a much higher Goodreads shelf_score.
            _make_full_db(
                db_path,
                [("/works/OL1W", "Obscure But Many Editions", "X", 1, 100), ("/works/OL2W", "Beloved Classic", "Y", 500, 5)],
            )
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            populate_goodreads_signals(conn, [{"work_key": "/works/OL2W", "shelf_score": 0.95}])
            rerank_popularity(conn, shelf_weight=0.9)
            ranks = dict(conn.execute("SELECT workKey, popularityRank FROM books"))
            conn.close()
        self.assertLess(ranks["/works/OL2W"], ranks["/works/OL1W"], "high shelf_score should win rank 1 at shelf_weight=0.9")

    def test_shelf_weight_zero_preserves_original_ol_order(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "A", "X", 1, 100), ("/works/OL2W", "B", "Y", 2, 5)])
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            populate_goodreads_signals(conn, [{"work_key": "/works/OL2W", "shelf_score": 0.99}])
            rerank_popularity(conn, shelf_weight=0.0)
            ranks = dict(conn.execute("SELECT workKey, popularityRank FROM books"))
            conn.close()
        self.assertLess(ranks["/works/OL1W"], ranks["/works/OL2W"])

    def test_ranks_are_dense_1_to_n(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "A", "X", 1, 1), ("/works/OL2W", "B", "Y", 2, 1), ("/works/OL3W", "C", "Z", 3, 1)])
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            n = rerank_popularity(conn)
            ranks = sorted(row[0] for row in conn.execute("SELECT popularityRank FROM books"))
            conn.close()
        self.assertEqual(n, 3)
        self.assertEqual(ranks, [1, 2, 3])


class TestGapFill(unittest.TestCase):
    def test_default_work_key_matches_swift_format(self) -> None:
        self.assertEqual(default_work_key("Dune", "Frank Herbert"), "dune|frank herbert")

    def test_inserts_popular_unmatched_book(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            create_goodreads_signals_table(conn)
            matched_rows = [
                {
                    "work_key": None,
                    "match_method": "unmatched",
                    "title": "Some Popular Missing Book",
                    "author": "Some Author",
                    "goodreads_book_id": 42,
                    "ratings_count": 50_000,
                    "shelf_score": 0.6,
                }
            ]
            inserted = gap_fill_unmatched(conn, matched_rows, min_ratings_count=1000, placeholder_rank=1)
            row = conn.execute(
                f"SELECT title, author, editionCount, languages FROM books WHERE workKey LIKE '{GAP_FILL_WORK_KEY_PREFIX}%'"
            ).fetchone()
            signal = conn.execute(
                f"SELECT shelfScore FROM goodreads_signals WHERE workKey LIKE '{GAP_FILL_WORK_KEY_PREFIX}%'"
            ).fetchone()
            conn.close()
        self.assertEqual(inserted, 1)
        self.assertEqual(row, ("Some Popular Missing Book", "Some Author", 0, "eng"))
        self.assertEqual(signal, (0.6,))

    def test_skips_book_below_ratings_floor(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            matched_rows = [
                {"work_key": None, "match_method": "unmatched", "title": "Obscure", "author": "X", "ratings_count": 5}
            ]
            inserted = gap_fill_unmatched(conn, matched_rows, min_ratings_count=1000, placeholder_rank=1)
            conn.close()
        self.assertEqual(inserted, 0)

    def test_skips_already_matched_books(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            create_goodreads_signals_table(conn)
            matched_rows = [
                {"work_key": "/works/OL1W", "match_method": "fuzzy", "title": "Dune", "author": "Frank Herbert", "ratings_count": 50_000}
            ]
            inserted = gap_fill_unmatched(conn, matched_rows, min_ratings_count=1000, placeholder_rank=1)
            conn.close()
        self.assertEqual(inserted, 0)

    def test_ambiguous_method_is_eligible_for_gap_fill(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            create_goodreads_signals_table(conn)
            matched_rows = [
                {"work_key": None, "match_method": "ambiguous", "title": "Ambiguous Popular Book", "author": "X", "ratings_count": 50_000, "shelf_score": 0.5}
            ]
            inserted = gap_fill_unmatched(conn, matched_rows, min_ratings_count=1000, placeholder_rank=1)
            conn.close()
        self.assertEqual(inserted, 1)


class TestNoDataSourceLeakInShipped(unittest.TestCase):
    """The shipped on-device catalog must never reveal Goodreads as a data
    source. Table/column names (`goodreads_signals`, `goodreadsBookId`)
    never ship -- `CatalogOLBuild.buildFromSubset`
    (`Sources/catalog-build/CatalogOLBuild.swift`) only reads the fixed
    `BookRecord` field set (workKey/title/author/isbn/titleNormalized/
    authorNormalized/popularityRank/editionCount) plus `book_isbns`, and
    copies `record.workKey`/`.title`/`.author` **verbatim** with no
    transformation -- so asserting on the scratch db's `books` table here is
    equivalent to asserting on the real shipped output for these columns,
    without needing the Swift toolchain (this test suite's existing
    no-Swift-required convention; see module docstring). If that Swift copy
    ever stops being a verbatim pass-through, this equivalence breaks and
    this test would need a real `swift run catalog-build` invocation
    instead.
    """

    def test_gap_fill_prefix_constant_has_no_data_source_name(self) -> None:
        self.assertNotIn("goodreads", GAP_FILL_WORK_KEY_PREFIX.lower())
        self.assertNotIn("gr", GAP_FILL_WORK_KEY_PREFIX.lower())

    def test_gap_filled_workkey_contains_no_goodreads_string(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "s.sqlite"
            _make_full_db(db_path, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            conn = sqlite3.connect(db_path)
            add_languages_column(conn)
            create_goodreads_signals_table(conn)
            matched_rows = [
                {
                    "work_key": None,
                    "match_method": "unmatched",
                    "title": "Some Popular Missing Book",
                    "author": "Some Author",
                    "goodreads_book_id": 42,
                    "ratings_count": 50_000,
                    "shelf_score": 0.6,
                }
            ]
            gap_fill_unmatched(conn, matched_rows, min_ratings_count=1000, placeholder_rank=1)
            rows = conn.execute("SELECT workKey, title, author FROM books").fetchall()
            conn.close()
        self.assertTrue(rows)
        for work_key, title, author in rows:
            for value in (work_key, title, author):
                self.assertNotIn("goodreads", (value or "").lower())


class TestRunEndToEnd(unittest.TestCase):
    def test_run_invokes_catalog_build_with_expected_args_and_cleans_up_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            full_db = tmp / "full.sqlite"
            _make_full_db(
                full_db,
                [
                    ("/works/OL1W", "Dune", "Frank Herbert", 1, 10),
                    ("/works/OL2W", "Le Petit Prince", "Antoine de Saint-Exupery", 2, 20),
                ],
            )
            intermediate_dir = tmp / "intermediate"
            _make_intermediate(intermediate_dir, [("/works/OL1W", ["eng"]), ("/works/OL2W", ["fre"])])

            matched_path = tmp / "matched.jsonl.gz"
            _write_matched_goodreads(
                matched_path,
                [{"work_key": "/works/OL1W", "goodreads_book_id": 1, "shelf_score": 0.7, "match_method": "fuzzy"}],
            )

            output_path = tmp / "ios_en_shelf.sqlite"
            captured_cmds = []

            def fake_run(cmd, cwd, check):
                captured_cmds.append(cmd)
                # Simulate catalog-build actually producing the output file.
                Path(cmd[cmd.index("--output") + 1]).write_text("fake db", encoding="utf-8")
                return None

            with patch("tools.catalog.build_ios_en_from_goodreads.subprocess.run", side_effect=fake_run):
                result = run(full_db, intermediate_dir, matched_path, output_path, min_intermediate_match_ratio=0.5)

        self.assertEqual(len(captured_cmds), 1)
        cmd = captured_cmds[0]
        self.assertIn("--subset-from", cmd)
        self.assertIn("--min-editions", cmd)
        self.assertIn("--max-works", cmd)
        self.assertEqual(result["total_books"], 2)
        self.assertEqual(result["pruned"], 1, "the French book should have been pruned by the default eng filter")
        self.assertFalse((tmp / "ios_en_shelf.scratch.sqlite").exists(), "scratch db should be cleaned up by default")

    def test_keep_scratch_leaves_scratch_db_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            full_db = tmp / "full.sqlite"
            _make_full_db(full_db, [("/works/OL1W", "Dune", "Frank Herbert", 1, 10)])
            intermediate_dir = tmp / "intermediate"
            _make_intermediate(intermediate_dir, [("/works/OL1W", ["eng"])])
            matched_path = tmp / "matched.jsonl.gz"
            _write_matched_goodreads(matched_path, [])
            output_path = tmp / "out.sqlite"

            def fake_run(cmd, cwd, check):
                Path(cmd[cmd.index("--output") + 1]).write_text("fake db", encoding="utf-8")
                return None

            with patch("tools.catalog.build_ios_en_from_goodreads.subprocess.run", side_effect=fake_run):
                run(full_db, intermediate_dir, matched_path, output_path, keep_scratch=True, min_intermediate_match_ratio=0.5)

            self.assertTrue((tmp / "out.scratch.sqlite").exists())

    def test_raises_on_intermediate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            full_db = tmp / "full.sqlite"
            _make_full_db(
                full_db,
                [(f"/works/OL{i}W", f"Title {i}", f"Author {i}", i, 1) for i in range(1, 11)],
            )
            intermediate_dir = tmp / "intermediate"
            # Only one of ten workKeys resolves -> well below the 0.5 default ratio.
            _make_intermediate(intermediate_dir, [("/works/OL1W", ["eng"])])
            matched_path = tmp / "matched.jsonl.gz"
            _write_matched_goodreads(matched_path, [])
            output_path = tmp / "out.sqlite"

            with self.assertRaises(IntermediateMismatchError):
                run(full_db, intermediate_dir, matched_path, output_path)


if __name__ == "__main__":
    unittest.main()
