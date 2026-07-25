#!/usr/bin/env python3
"""Stream OL dumps → gzipped JSONL intermediate (works + isbns + manifest).

Uses SQLite staging for memory-safe processing of full OL dumps.
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
    author_key_from_work,
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
    n = 0
    for row in stream_jsonl(editions_path):
        work_key = edition_work_key(row)
        if not work_key:
            continue
        conn.execute(
            """
            INSERT INTO edition_stats(work_key, edition_count) VALUES (?, 1)
            ON CONFLICT(work_key) DO UPDATE SET edition_count = edition_count + 1
            """,
            (work_key,),
        )
        for lang in row.get("languages") or []:
            if isinstance(lang, dict):
                code = normalize_language(lang.get("key"))
            else:
                code = normalize_language(str(lang))
            if code:
                conn.execute(
                    "INSERT OR IGNORE INTO edition_langs(work_key, lang) VALUES (?, ?)",
                    (work_key, code),
                )
        for raw in (row.get("isbn_13") or []) + (row.get("isbn_10") or []):
            if not isinstance(raw, str):
                continue
            isbn = normalize_isbn13(raw)
            if isbn:
                conn.execute(
                    "INSERT OR IGNORE INTO work_isbns(work_key, isbn13) VALUES (?, ?)",
                    (work_key, isbn),
                )
        n += 1
        if n % BATCH == 0:
            conn.commit()
            print(f"  editions: {n:,}", file=sys.stderr)
    conn.commit()
    return n


def ingest_authors(conn: sqlite3.Connection, authors_path: Path) -> int:
    n = 0
    for row in stream_jsonl(authors_path):
        key = row.get("key")
        name = row.get("name")
        if isinstance(key, str) and isinstance(name, str) and name.strip():
            conn.execute(
                "INSERT OR REPLACE INTO authors(author_key, name) VALUES (?, ?)",
                (key, name.strip()),
            )
            n += 1
            if n % BATCH == 0:
                conn.commit()
                print(f"  authors: {n:,}", file=sys.stderr)
    conn.commit()
    return n


def ingest_works(conn: sqlite3.Connection, works_path: Path, min_editions: int) -> int:
    n = 0
    kept = 0
    for row in stream_jsonl(works_path):
        n += 1
        work_key = row.get("key")
        if not isinstance(work_key, str) or not work_key.startswith("/works/"):
            continue
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        author_key = author_key_from_work(row)
        if not author_key:
            continue
        author_row = conn.execute(
            "SELECT name FROM authors WHERE author_key = ?", (author_key,)
        ).fetchone()
        if not author_row:
            continue
        stats = conn.execute(
            "SELECT edition_count FROM edition_stats WHERE work_key = ?", (work_key,)
        ).fetchone()
        if not stats or stats[0] < min_editions:
            continue
        isbn_row = conn.execute(
            "SELECT isbn13 FROM work_isbns WHERE work_key = ? ORDER BY isbn13 LIMIT 1",
            (work_key,),
        ).fetchone()
        isbn13 = isbn_row[0] if isbn_row else None
        conn.execute(
            """
            INSERT OR REPLACE INTO works_out(work_key, title, author, edition_count, isbn13)
            VALUES (?, ?, ?, ?, ?)
            """,
            (work_key, title.strip(), author_row[0], stats[0], isbn13),
        )
        kept += 1
        if n % BATCH == 0:
            conn.commit()
            print(f"  works scanned: {n:,}, kept: {kept:,}", file=sys.stderr)
    conn.commit()
    return kept


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

    def work_rows():
        for rank, (work_key, title, author, edition_count, isbn13) in enumerate(ranked, start=1):
            langs = [
                r[0]
                for r in conn.execute(
                    "SELECT lang FROM edition_langs WHERE work_key = ? ORDER BY lang", (work_key,)
                )
            ]
            yield {
                "workKey": work_key,
                "title": title,
                "author": author,
                "isbn13": isbn13,
                "editionCount": edition_count,
                "popularityRank": rank,
                "languages": langs,
            }

    def isbn_rows():
        for work_key, _title, _author, _edition_count, isbn13 in ranked:
            if isbn13:
                yield {"isbn13": isbn13, "workKey": work_key}
            for (extra_isbn,) in conn.execute(
                "SELECT isbn13 FROM work_isbns WHERE work_key = ? AND isbn13 != ?",
                (work_key, isbn13 or ""),
            ):
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
    print("Ingesting works...", file=sys.stderr)
    ingest_works(conn, works, args.min_editions)
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
