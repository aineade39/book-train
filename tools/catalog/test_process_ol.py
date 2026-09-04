#!/usr/bin/env python3
"""Unit tests for tools/catalog/process_ol.py (run: python tools/catalog/test_process_ol.py).

Covers the D1 SQL-join rewrite (`ingest_works_raw` + `build_work_author_names`
+ `build_works_out`, replacing the old per-row 3-`SELECT` Python loop), D2's
`executemany` batching (`ingest_editions`), and the co-author fix's effect on
the staging-DB join (Part A) -- all against a real (temp-file) SQLite
connection, not mocks, since the whole point of D1/D2 is the SQL these
functions run.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.process_ol import (  # noqa: E402
    build_work_author_names,
    build_works_out,
    init_db,
    ingest_authors,
    ingest_editions,
    ingest_works_raw,
)


class ProcessOLTestCase(unittest.TestCase):
    """Each test gets its own temp-file SQLite staging DB (matching
    `init_db`'s real on-disk usage -- `executemany`/transaction behavior
    isn't guaranteed identical on `:memory:`)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.conn: sqlite3.Connection = init_db(self.tmp_path / "staging.sqlite")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def _write_editions_jsonl(self, rows: list[dict]) -> Path:
        path = self.tmp_path / "editions.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path


class TestIngestEditionsBatching(ProcessOLTestCase):
    """D2: executemany-batched ingest_editions produces the same
    edition_stats/edition_langs/work_isbns row shape as the pre-change
    per-row `conn.execute()` version."""

    def test_edition_stats_counts_editions_per_work(self) -> None:
        path = self._write_editions_jsonl(
            [
                {"works": [{"key": "/works/OL1W"}], "isbn_13": ["9780441013593"], "languages": [{"key": "/languages/eng"}]},
                {"works": [{"key": "/works/OL1W"}], "isbn_13": ["9780441013609"], "languages": [{"key": "/languages/eng"}]},
                {"works": [{"key": "/works/OL2W"}], "isbn_13": ["9780593135204"], "languages": [{"key": "/languages/eng"}]},
            ]
        )
        n = ingest_editions(self.conn, path)
        self.assertEqual(n, 3)
        self.assertEqual(
            dict(self.conn.execute("SELECT work_key, edition_count FROM edition_stats").fetchall()),
            {"/works/OL1W": 2, "/works/OL2W": 1},
        )

    def test_edition_langs_dedupes_repeated_language_per_work(self) -> None:
        path = self._write_editions_jsonl(
            [
                {"works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/eng"}]},
                {"works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/eng"}]},
                {"works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/fre"}]},
            ]
        )
        ingest_editions(self.conn, path)
        langs = {r[0] for r in self.conn.execute("SELECT lang FROM edition_langs WHERE work_key = '/works/OL1W'")}
        self.assertEqual(langs, {"eng", "fre"})

    def test_work_isbns_collects_all_isbn13_and_converted_isbn10(self) -> None:
        path = self._write_editions_jsonl(
            [
                {"works": [{"key": "/works/OL1W"}], "isbn_13": ["9780441013593"]},
                # A different edition's ISBN-10 -> converts to a distinct
                # valid ISBN-13, both should land in work_isbns.
                {"works": [{"key": "/works/OL1W"}], "isbn_10": ["0553418025"]},
            ]
        )
        ingest_editions(self.conn, path)
        isbns = {r[0] for r in self.conn.execute("SELECT isbn13 FROM work_isbns WHERE work_key = '/works/OL1W'")}
        self.assertEqual(isbns, {"9780441013593", "9780553418026"})

    def test_editions_with_no_resolvable_work_key_are_skipped(self) -> None:
        path = self._write_editions_jsonl([{"works": [], "isbn_13": ["9780441013593"]}, {"isbn_13": ["9780441013593"]}])
        n = ingest_editions(self.conn, path)
        self.assertEqual(n, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM edition_stats").fetchone()[0], 0)

    def test_batching_across_multiple_flushes_matches_unbatched_totals(self) -> None:
        # Forces >1 `executemany` flush (BATCH=50_000) to exercise D2's
        # batching boundary itself, not just the single-flush case above.
        import tools.catalog.process_ol as process_ol

        original_batch = process_ol.BATCH
        process_ol.BATCH = 3
        try:
            path = self._write_editions_jsonl(
                [{"works": [{"key": f"/works/OL{i}W"}], "isbn_13": [], "languages": []} for i in range(10)]
            )
            n = ingest_editions(self.conn, path)
        finally:
            process_ol.BATCH = original_batch
        self.assertEqual(n, 10)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM edition_stats").fetchone()[0], 10)


class TestBuildWorksOutJoin(ProcessOLTestCase):
    """D1: the set-based `works_raw`/`work_author_names`/`edition_stats`/
    `work_isbns` join in `build_works_out` must produce the same
    `works_out` rows the old per-row 3-`SELECT` loop did, for every one of
    the filter/tie-break cases it had to handle: no author match, below
    min_editions, no ISBN, multiple ISBNs (lowest wins), and (Part A) a
    multi-author work."""

    def _seed_authors(self, pairs: dict[str, str]) -> None:
        self.conn.executemany(
            "INSERT INTO authors(author_key, name) VALUES (?, ?)", list(pairs.items())
        )
        self.conn.commit()

    def _seed_work(self, work_key: str, title: str, author_keys: list[str]) -> None:
        self.conn.execute("INSERT INTO works_raw(work_key, title) VALUES (?, ?)", (work_key, title))
        self.conn.executemany(
            "INSERT INTO work_authors(work_key, author_key, author_ord) VALUES (?, ?, ?)",
            [(work_key, key, i) for i, key in enumerate(author_keys)],
        )
        self.conn.commit()

    def _seed_edition_stats(self, work_key: str, edition_count: int) -> None:
        self.conn.execute("INSERT INTO edition_stats(work_key, edition_count) VALUES (?, ?)", (work_key, edition_count))
        self.conn.commit()

    def _seed_isbns(self, work_key: str, isbns: list[str]) -> None:
        self.conn.executemany(
            "INSERT INTO work_isbns(work_key, isbn13) VALUES (?, ?)", [(work_key, isbn) for isbn in isbns]
        )
        self.conn.commit()

    def _works_out(self) -> dict[str, tuple[str, str, int, str | None]]:
        rows = self.conn.execute("SELECT work_key, title, author, edition_count, isbn13 FROM works_out").fetchall()
        return {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}

    def test_work_with_no_author_match_is_dropped(self) -> None:
        self._seed_authors({})
        self._seed_work("/works/OL1W", "Dune", ["/authors/MISSING"])
        self._seed_edition_stats("/works/OL1W", 5)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)
        self.assertEqual(self._works_out(), {})

    def test_work_below_min_editions_is_dropped(self) -> None:
        self._seed_authors({"/authors/OL1A": "Frank Herbert"})
        self._seed_work("/works/OL1W", "Dune", ["/authors/OL1A"])
        self._seed_edition_stats("/works/OL1W", 1)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=2)
        self.assertEqual(self._works_out(), {})

    def test_work_with_no_isbn_gets_null_isbn13(self) -> None:
        self._seed_authors({"/authors/OL1A": "Frank Herbert"})
        self._seed_work("/works/OL1W", "Dune", ["/authors/OL1A"])
        self._seed_edition_stats("/works/OL1W", 1)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)
        self.assertEqual(self._works_out()["/works/OL1W"], ("Dune", "Frank Herbert", 1, None))

    def test_work_with_multiple_isbns_keeps_the_lowest(self) -> None:
        self._seed_authors({"/authors/OL1A": "Frank Herbert"})
        self._seed_work("/works/OL1W", "Dune", ["/authors/OL1A"])
        self._seed_edition_stats("/works/OL1W", 2)
        self._seed_isbns("/works/OL1W", ["9780441172696", "9780441013593"])
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)
        self.assertEqual(self._works_out()["/works/OL1W"][3], "9780441013593")

    def test_multi_author_work_joins_names_in_order(self) -> None:
        self._seed_authors({"/authors/OL1A": "Terry Pratchett", "/authors/OL2A": "Neil Gaiman"})
        self._seed_work("/works/OL1W", "Good Omens", ["/authors/OL1A", "/authors/OL2A"])
        self._seed_edition_stats("/works/OL1W", 3)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)
        self.assertEqual(self._works_out()["/works/OL1W"][1], "Terry Pratchett and Neil Gaiman")

    def test_multi_author_work_with_one_unresolvable_coauthor_keeps_the_resolved_ones(self) -> None:
        self._seed_authors({"/authors/OL2A": "Neil Gaiman"})
        self._seed_work("/works/OL1W", "Partial Resolve", ["/authors/OL1A", "/authors/OL2A"])
        self._seed_edition_stats("/works/OL1W", 1)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)
        self.assertEqual(self._works_out()["/works/OL1W"][1], "Neil Gaiman")


class TestFullStagingPipelineRoundTrip(ProcessOLTestCase):
    """A multi-author work round-trips through the real staging functions
    end to end (`ingest_authors` + `ingest_works_raw` -> `stream_jsonl`'d
    from real fixture files, not directly-seeded tables like
    `TestBuildWorksOutJoin` above) -- exercises the actual works-dump
    parsing path, not just the join."""

    def test_multi_author_work_round_trips_through_staging_db(self) -> None:
        (self.tmp_path / "authors.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"key": "/authors/OL1A", "name": "Terry Pratchett"}),
                    json.dumps({"key": "/authors/OL2A", "name": "Neil Gaiman"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        works_path = self.tmp_path / "works.jsonl"
        works_path.write_text(
            json.dumps(
                {
                    "key": "/works/OL1W",
                    "title": "Good Omens",
                    "authors": [{"key": "/authors/OL1A"}, {"key": "/authors/OL2A"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        editions_path = self._write_editions_jsonl(
            [
                {"works": [{"key": "/works/OL1W"}], "isbn_13": ["9780575049222"], "languages": [{"key": "/languages/eng"}]},
            ]
        )

        ingest_editions(self.conn, editions_path)
        ingest_authors(self.conn, self.tmp_path / "authors.jsonl")
        kept = ingest_works_raw(self.conn, works_path)
        self.assertEqual(kept, 1)
        build_work_author_names(self.conn)
        build_works_out(self.conn, min_editions=1)

        row = self.conn.execute(
            "SELECT title, author, edition_count, isbn13 FROM works_out WHERE work_key = '/works/OL1W'"
        ).fetchone()
        self.assertEqual(row, ("Good Omens", "Terry Pratchett and Neil Gaiman", 1, "9780575049222"))


if __name__ == "__main__":
    unittest.main()
