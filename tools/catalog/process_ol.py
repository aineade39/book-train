#!/usr/bin/env python3
"""Stream OL dumps → gzipped JSONL intermediate (works + isbns + manifest).

Uses SQLite staging for memory-safe processing of full OL dumps.

Works are staged (`works_raw`/`work_authors`) then joined against
`authors`/`edition_stats`/`work_isbns` with a small number of set-based SQL
queries rather than one-row-at-a-time Python lookups -- see
`build_work_author_names`/`build_works_out` (the "Catalog match quality"
plan's D1: ~36-39M individual point-lookup round trips replaced with a
handful of sequential index scans). `ingest_editions`/`export_intermediate`
batch their writes with `executemany` for the same reason (D2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import (  # noqa: E402
    author_keys_from_work,
    join_author_names,
    normalize_isbn13,
    normalize_language,
    stream_jsonl,
    write_jsonl_gz,
)
from tools.derived_meta import git_commit_short, write_derived_source  # noqa: E402
from tools.paths import catalog_intermediate, catalog_raw_ol  # noqa: E402

DEFAULT_DUMPS = {
    "editions": "ol_dump_editions_latest.txt.gz",
    "works": "ol_dump_works_latest.txt.gz",
    "authors": "ol_dump_authors_latest.txt.gz",
}

BATCH = 50_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE edition_stats (
            work_key TEXT PRIMARY KEY,
            edition_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE edition_langs (
            work_key TEXT NOT NULL,
            lang TEXT NOT NULL,
            PRIMARY KEY (work_key, lang)
        );
        CREATE TABLE work_isbns (
            work_key TEXT NOT NULL,
            isbn13 TEXT NOT NULL,
            PRIMARY KEY (work_key, isbn13)
        );
        CREATE TABLE authors (
            author_key TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE works_raw (
            work_key TEXT PRIMARY KEY,
            title TEXT NOT NULL
        );
        CREATE TABLE work_authors (
            work_key TEXT NOT NULL,
            author_key TEXT NOT NULL,
            author_ord INTEGER NOT NULL,
            PRIMARY KEY (work_key, author_ord)
        );
        CREATE TABLE work_author_names (
            work_key TEXT PRIMARY KEY,
            author_names TEXT NOT NULL
        );
        CREATE TABLE works_out (
            work_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            edition_count INTEGER NOT NULL,
            isbn13 TEXT
        );
        """
    )
    return conn


def edition_work_key(row: dict) -> str | None:
    works = row.get("works") or []
    if works and isinstance(works[0], dict):
        key = works[0].get("key")
        if isinstance(key, str) and key.startswith("/works/"):
            return key
    return None


def ingest_editions(conn: sqlite3.Connection, editions_path: Path) -> int:
    """D2: buffers rows per `BATCH` and writes with `executemany` instead of
    one `conn.execute()` per edition/language/isbn -- same statement count,
    far less per-call Python/interpreter overhead across ~40M+ editions."""
    n = 0
    stats_batch: list[tuple[str]] = []
    langs_batch: list[tuple[str, str]] = []
    isbns_batch: list[tuple[str, str]] = []

    def flush() -> None:
        if stats_batch:
            conn.executemany(
                """
                INSERT INTO edition_stats(work_key, edition_count) VALUES (?, 1)
                ON CONFLICT(work_key) DO UPDATE SET edition_count = edition_count + 1
                """,
                stats_batch,
            )
            stats_batch.clear()
        if langs_batch:
            conn.executemany("INSERT OR IGNORE INTO edition_langs(work_key, lang) VALUES (?, ?)", langs_batch)
            langs_batch.clear()
        if isbns_batch:
            conn.executemany("INSERT OR IGNORE INTO work_isbns(work_key, isbn13) VALUES (?, ?)", isbns_batch)
            isbns_batch.clear()

    for row in stream_jsonl(editions_path):
        work_key = edition_work_key(row)
        if not work_key:
            continue
        stats_batch.append((work_key,))
        for lang in row.get("languages") or []:
            if isinstance(lang, dict):
                code = normalize_language(lang.get("key"))
            else:
                code = normalize_language(str(lang))
            if code:
                langs_batch.append((work_key, code))
        for raw in (row.get("isbn_13") or []) + (row.get("isbn_10") or []):
            if not isinstance(raw, str):
                continue
            isbn = normalize_isbn13(raw)
            if isbn:
                isbns_batch.append((work_key, isbn))
        n += 1
        if n % BATCH == 0:
            flush()
            conn.commit()
            print(f"  editions: {n:,}", file=sys.stderr)
    flush()
    conn.commit()
    return n


def ingest_authors(conn: sqlite3.Connection, authors_path: Path) -> int:
    n = 0
    batch: list[tuple[str, str]] = []
    for row in stream_jsonl(authors_path):
        key = row.get("key")
        name = row.get("name")
        if isinstance(key, str) and isinstance(name, str) and name.strip():
            batch.append((key, name.strip()))
            n += 1
            if len(batch) >= BATCH:
                conn.executemany("INSERT OR REPLACE INTO authors(author_key, name) VALUES (?, ?)", batch)
                batch.clear()
                conn.commit()
                print(f"  authors: {n:,}", file=sys.stderr)
    if batch:
        conn.executemany("INSERT OR REPLACE INTO authors(author_key, name) VALUES (?, ?)", batch)
    conn.commit()
    return n


def ingest_works_raw(conn: sqlite3.Connection, works_path: Path) -> int:
    """Stages the works dump into `works_raw`/`work_authors` (D2: batched
    `executemany` writes) -- the join against authors/editions/isbns that
    used to happen inline, one `SELECT` per work, now happens once in
    `build_work_author_names`/`build_works_out` as a handful of set-based
    SQL queries (D1)."""
    n = 0
    kept = 0
    work_batch: list[tuple[str, str]] = []
    author_batch: list[tuple[str, str, int]] = []

    def flush() -> None:
        if work_batch:
            conn.executemany("INSERT OR REPLACE INTO works_raw(work_key, title) VALUES (?, ?)", work_batch)
            work_batch.clear()
        if author_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO work_authors(work_key, author_key, author_ord) VALUES (?, ?, ?)",
                author_batch,
            )
            author_batch.clear()

    for row in stream_jsonl(works_path):
        n += 1
        work_key = row.get("key")
        if not isinstance(work_key, str) or not work_key.startswith("/works/"):
            continue
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        author_keys = author_keys_from_work(row)
        if not author_keys:
            continue
        work_batch.append((work_key, title.strip()))
        for author_ord, author_key in enumerate(author_keys):
            author_batch.append((work_key, author_key, author_ord))
        kept += 1
        if n % BATCH == 0:
            flush()
            conn.commit()
            print(f"  works scanned: {n:,}, kept: {kept:,}", file=sys.stderr)
    flush()
    conn.commit()
    return kept


def build_work_author_names(conn: sqlite3.Connection) -> int:
    """One bulk ordered join (`work_authors` JOIN `authors`, `ORDER BY
    work_key, author_ord`) streamed through a single linear groupby pass --
    replaces the old per-work `SELECT name FROM authors WHERE author_key = ?`
    single-author lookup with one sequential scan instead of ~36-39M
    individual point queries (D1), while preserving OL's author order and
    supporting multiple authors per work (co-author fix, Part A).

    A co-author key that doesn't resolve to a name (missing from the
    authors dump) is simply absent from the joined name for that work,
    rather than dropping the whole work -- matches the old single-author
    behavior exactly when there's only one author (drop iff it doesn't
    resolve) while degrading gracefully for multi-author works (keep
    whichever co-authors did resolve). A work with *zero* resolved authors
    has no row here at all, and is dropped by `build_works_out`'s join.
    """
    cur = conn.execute(
        """
        SELECT wa.work_key, a.name
        FROM work_authors wa
        JOIN authors a ON a.author_key = wa.author_key
        ORDER BY wa.work_key, wa.author_ord
        """
    )
    batch: list[tuple[str, str]] = []
    current_key: str | None = None
    current_names: list[str] = []
    count = 0

    def flush_group() -> None:
        nonlocal count
        if current_key is not None and current_names:
            batch.append((current_key, join_author_names(current_names)))
            count += 1

    def flush_batch() -> None:
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO work_author_names(work_key, author_names) VALUES (?, ?)", batch
            )
            batch.clear()

    for work_key, name in cur:
        if work_key != current_key:
            flush_group()
            current_key = work_key
            current_names = []
            if len(batch) >= BATCH:
                flush_batch()
                conn.commit()
        current_names.append(name)
    flush_group()
    flush_batch()
    conn.commit()
    return count


def build_works_out(conn: sqlite3.Connection, min_editions: int) -> int:
    """D1: single set-based join replacing the old per-work 3-`SELECT`
    Python loop (author name, edition_stats, min isbn13) -- SQLite's own
    query engine does this join in one pass over indexed tables instead of
    ~36-39M individual point-lookup round trips."""
    conn.execute(
        """
        INSERT OR REPLACE INTO works_out (work_key, title, author, edition_count, isbn13)
        SELECT w.work_key, w.title, an.author_names, e.edition_count, m.isbn13
        FROM works_raw w
        JOIN work_author_names an ON an.work_key = w.work_key
        JOIN edition_stats e ON e.work_key = w.work_key AND e.edition_count >= ?
        LEFT JOIN (
            SELECT work_key, MIN(isbn13) AS isbn13 FROM work_isbns GROUP BY work_key
        ) m ON m.work_key = w.work_key
        """,
        (min_editions,),
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM works_out").fetchone()[0]


def export_intermediate(conn: sqlite3.Connection, out_dir: Path) -> tuple[int, int]:
    works_path = out_dir / "works.jsonl.gz"
    isbns_path = out_dir / "isbns.jsonl.gz"

    ranked = conn.execute(
        """
        SELECT work_key, title, author, edition_count, isbn13
        FROM works_out
        ORDER BY edition_count DESC, work_key
        """
    ).fetchall()

    # D2: one bulk GROUP_CONCAT pass per side-table instead of a
    # per-work_key subquery inside the row-emitting loop below. Same
    # O(unique-work-key)-scale dict shape `ranked` above already accepts at
    # this point in the pipeline (works_out is already filtered down to
    # shipped works, same cardinality as `ranked`) -- just two small
    # comma-joined strings per work instead of full rows.
    langs_by_work: dict[str, str] = dict(
        conn.execute(
            """
            SELECT work_key, GROUP_CONCAT(lang, ',')
            FROM (SELECT DISTINCT work_key, lang FROM edition_langs ORDER BY work_key, lang)
            GROUP BY work_key
            """
        ).fetchall()
    )
    extra_isbns_by_work: dict[str, str] = dict(
        conn.execute(
            """
            SELECT work_key, GROUP_CONCAT(isbn13, ',')
            FROM (SELECT DISTINCT work_key, isbn13 FROM work_isbns ORDER BY work_key, isbn13)
            GROUP BY work_key
            """
        ).fetchall()
    )

    def work_rows():
        for rank, (work_key, title, author, edition_count, isbn13) in enumerate(ranked, start=1):
            langs_csv = langs_by_work.get(work_key)
            yield {
                "workKey": work_key,
                "title": title,
                "author": author,
                "isbn13": isbn13,
                "editionCount": edition_count,
                "popularityRank": rank,
                "languages": langs_csv.split(",") if langs_csv else [],
            }

    def isbn_rows():
        for work_key, _title, _author, _edition_count, isbn13 in ranked:
            if isbn13:
                yield {"isbn13": isbn13, "workKey": work_key}
            extra_csv = extra_isbns_by_work.get(work_key)
            if not extra_csv:
                continue
            for extra_isbn in extra_csv.split(","):
                if extra_isbn and extra_isbn != (isbn13 or ""):
                    yield {"isbn13": extra_isbn, "workKey": work_key}

    work_count = write_jsonl_gz(works_path, work_rows())
    isbn_count = write_jsonl_gz(isbns_path, isbn_rows())
    return work_count, isbn_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--editions", type=Path, default=None)
    parser.add_argument("--works", type=Path, default=None)
    parser.add_argument("--authors", type=Path, default=None)
    parser.add_argument("--min-editions", type=int, default=1)
    args = parser.parse_args()

    raw = args.raw_dir or catalog_raw_ol()
    out_dir = args.out_dir or catalog_intermediate()
    editions = args.editions or raw / DEFAULT_DUMPS["editions"]
    works = args.works or raw / DEFAULT_DUMPS["works"]
    authors = args.authors or raw / DEFAULT_DUMPS["authors"]
    for path in (editions, works, authors):
        if not path.exists():
            print(f"Missing dump: {path}", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "staging.sqlite"
    print("Ingesting editions...", file=sys.stderr)
    conn = init_db(staging)
    ingest_editions(conn, editions)
    print("Ingesting authors...", file=sys.stderr)
    ingest_authors(conn, authors)
    print("Staging works...", file=sys.stderr)
    ingest_works_raw(conn, works)
    print("Joining author names...", file=sys.stderr)
    build_work_author_names(conn)
    print("Joining works × authors × editions × isbns...", file=sys.stderr)
    kept = build_works_out(conn, args.min_editions)
    print(f"  works kept: {kept:,}", file=sys.stderr)
    print("Exporting intermediate...", file=sys.stderr)
    work_count, isbn_count = export_intermediate(conn, out_dir)
    conn.close()
    staging.unlink(missing_ok=True)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit_short(),
        "source_hashes": {
            "editions": sha256_file(editions),
            "works": sha256_file(works),
            "authors": sha256_file(authors),
        },
        "work_count": work_count,
        "isbn_count": isbn_count,
        "min_editions": args.min_editions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_derived_source(
        out_dir.parent,
        derived_id="book-catalog",
        title="Open Library book catalog intermediate",
        sources=[str(editions), str(works), str(authors)],
        script="tools/catalog/process_ol.py",
        flags={"work_count": work_count, "isbn_count": isbn_count},
    )
    print(f"Wrote {work_count:,} works, {isbn_count:,} isbns → {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
