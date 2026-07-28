#!/usr/bin/env python3
"""Rebuild `ios_en_shelf`: re-rank + gap-fill `ios_en` using Goodreads shelf signals.

The shipped ~250k catalog on iOS only needs what
`Sources/SpineCatalog/BookCatalog.swift` already stores (workKey, title,
author, isbn, titleNormalized, authorNormalized, popularityRank,
editionCount) — nothing about Goodreads. Per an explicit project
constraint, this script must NOT add columns/tables to what actually
ships; `swift run catalog-build --subset-from ... --output
ios_en_shelf.sqlite` (unmodified — the exact same code path the existing
`ios_en` profile uses) is what produces the shipped file and schema.

Everything Goodreads-specific — a `languages` column (full.sqlite's `books`
table carries none; see `CatalogOLBuild.buildFromSubset`, which has no
language filter of its own) and a `goodreads_signals` table — lives ONLY on
a *scratch copy* of full.sqlite that this script creates and mutates. The
scratch copy is deleted afterward unless `--keep-scratch` is passed; it
never ships anywhere.

Pipeline on the scratch copy:
  1. Copy full.sqlite -> scratch path.
  2. ALTER TABLE books ADD COLUMN languages; populate from the
     `works.jsonl.gz` intermediate that was used to build full.sqlite
     (that per-work language data doesn't otherwise exist in full.sqlite).
     Aborts if too few `books` rows resolve against the given
     `--intermediate-dir` (almost certainly the wrong intermediate for this
     full.sqlite) rather than silently pruning on an unreliable signal.
  3. DELETE rows whose languages don't intersect `--languages` (the only
     place a subset build can apply a language filter at all).
  4. CREATE TABLE goodreads_signals from `match_goodreads.py`'s output.
  5. Gap-fill: Goodreads books with `ratings_count` above a floor that
     *didn't* match any OL work get inserted as new synthetic rows
     (assumed English; isbn/editionCount left unset — best-effort
     placeholders, not OL-verified facts).
  6. Re-rank every row's `popularityRank`: blend the existing OL-derived
     rank (edition-count based) with `goodreads_signals.shelfScore`, so a
     Goodreads-validated "really on shelves" book can out-rank a
     higher-edition-count-but-obscure OL work, while books absent from the
     Goodreads scrape still rank by their original OL popularity.
  7. Invoke `swift run catalog-build --subset-from <scratch> --output
     <output> --min-editions N --max-works M` (same shape as the existing
     `ios_en` profile in tools/catalog/profiles.yaml).

Usage:
    python tools/catalog/build_ios_en_from_goodreads.py \\
        --full-db /path/to/full.sqlite \\
        --intermediate-dir /path/to/intermediate \\
        --matched-goodreads /path/to/matched_goodreads.jsonl.gz \\
        --output /path/to/ios_en_shelf.sqlite
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import normalize_for_search, stream_jsonl  # noqa: E402

DEFAULT_MIN_EDITIONS = 2
DEFAULT_MAX_WORKS = 250_000
DEFAULT_LANGUAGES = frozenset({"eng"})
# How much a book's rerank position depends on Goodreads shelf_score vs. its
# original OL edition-count-based popularity. Not tuned against real scrape
# output (see compute_shelf_score's docstring in match_goodreads.py) —
# revisit once a real run exists to look at.
DEFAULT_SHELF_WEIGHT = 0.5
DEFAULT_GAP_FILL_MIN_RATINGS_COUNT = 1_000
DEFAULT_CATALOG_BUILD_CMD = ("swift", "run", "catalog-build")


class IntermediateMismatchError(RuntimeError):
    pass


def copy_to_scratch(full_db: Path, scratch_db: Path) -> None:
    if scratch_db.exists():
        scratch_db.unlink()
    scratch_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(full_db, scratch_db)


def add_languages_column(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
    if "languages" not in columns:
        conn.execute("ALTER TABLE books ADD COLUMN languages TEXT")
        conn.commit()


def load_work_languages(intermediate_dir: Path) -> dict[str, set[str]]:
    """`{workKey: {language codes}}` from the `works.jsonl.gz` intermediate
    (see tools/catalog/process_ol.py's export_intermediate)."""
    out: dict[str, set[str]] = {}
    for row in stream_jsonl(intermediate_dir / "works.jsonl.gz"):
        work_key = row.get("workKey")
        if isinstance(work_key, str):
            out[work_key] = set(row.get("languages") or [])
    return out


def populate_languages(conn: sqlite3.Connection, work_languages: dict[str, set[str]]) -> tuple[int, int]:
    """Returns (total_books, books_found_in_the_intermediate) for the caller's
    mismatch sanity check."""
    rows = conn.execute("SELECT workKey FROM books").fetchall()
    total = len(rows)
    matched = 0
    updates: list[tuple[str, str]] = []
    for (work_key,) in rows:
        langs = work_languages.get(work_key)
        if langs is not None:
            matched += 1
            updates.append((",".join(sorted(langs)), work_key))
    conn.executemany("UPDATE books SET languages = ? WHERE workKey = ?", updates)
    conn.commit()
    return total, matched


def prune_to_languages(conn: sqlite3.Connection, languages: frozenset[str]) -> int:
    """Deletes rows whose `languages` column has no overlap with `languages`
    — including rows with no language data at all (treated as unknown,
    hence excluded; see module docstring's intermediate-mismatch caveat for
    why that's checked separately rather than silently trusted here)."""
    rows = conn.execute("SELECT id, languages FROM books").fetchall()
    to_delete = [
        row_id for row_id, langs_str in rows if not (set((langs_str or "").split(",")) & languages)
    ]
    conn.executemany("DELETE FROM books WHERE id = ?", [(i,) for i in to_delete])
    conn.execute("DELETE FROM book_isbns WHERE workKey NOT IN (SELECT workKey FROM books)")
    conn.commit()
    return len(to_delete)


def create_goodreads_signals_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS goodreads_signals (
            workKey TEXT PRIMARY KEY,
            goodreadsBookId INTEGER,
            shelfScore REAL,
            avgRating REAL,
            ratingsCount INTEGER,
            listAppearances INTEGER,
            matchMethod TEXT
        )
        """
    )
    conn.commit()


def load_matched_goodreads(matched_path: Path) -> list[dict]:
    """Reads match_goodreads.py's gzipped-JSONL output."""
    rows: list[dict] = []
    with gzip.open(matched_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def populate_goodreads_signals(conn: sqlite3.Connection, matched_rows: list[dict]) -> int:
    n = 0
    for row in matched_rows:
        work_key = row.get("work_key")
        if not work_key:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO goodreads_signals
                (workKey, goodreadsBookId, shelfScore, avgRating, ratingsCount, listAppearances, matchMethod)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_key,
                row.get("goodreads_book_id"),
                row.get("shelf_score"),
                row.get("avg_rating"),
                row.get("ratings_count"),
                row.get("list_appearances"),
                row.get("match_method"),
            ),
        )
        n += 1
    conn.commit()
    return n


def default_work_key(title: str, author: str) -> str:
    """Mirrors Sources/SpineCatalog/BookCatalog.swift's `defaultWorkKey` —
    used only for gap-filled rows that have no real OL workKey."""
    return f"{normalize_for_search(title)}|{normalize_for_search(author)}"


def gap_fill_unmatched(
    conn: sqlite3.Connection,
    matched_rows: list[dict],
    *,
    min_ratings_count: int,
    placeholder_rank: int,
) -> int:
    """Books Goodreads clearly shows are popular (`ratings_count` above
    `min_ratings_count`) but that didn't fuzzy-match any OL work
    (`match_method` in {"unmatched", "ambiguous"}) get inserted as new
    synthetic rows — the "gap-fill" half of "re-rank and gap-fill". Their
    `popularityRank` is set to `placeholder_rank` (meaning ~zero OL
    popularity signal) and corrected by the caller's subsequent
    `rerank_popularity()` pass, which is what actually gives them a
    shelf_score-driven rank.
    """
    inserted = 0
    for row in matched_rows:
        if row.get("work_key") or row.get("match_method") not in ("unmatched", "ambiguous"):
            continue
        if (row.get("ratings_count") or 0) < min_ratings_count:
            continue
        title, author = row.get("title"), row.get("author")
        if not title or not author:
            continue
        work_key = "goodreads-gapfill:" + default_work_key(title, author)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO books
                (workKey, title, author, isbn, titleNormalized, authorNormalized, popularityRank, editionCount, languages)
            VALUES (?, ?, ?, NULL, ?, ?, ?, 0, 'eng')
            """,
            (work_key, title, author, normalize_for_search(title), normalize_for_search(author), placeholder_rank),
        )
        if cur.rowcount:
            inserted += 1
            # `populate_goodreads_signals` never inserted a row for this
            # book (it only stores rows that already had a real OL
            # work_key) — insert one now under the synthetic key so the
            # immediately-following rerank pass's shelf_score lookup finds it.
            conn.execute(
                """
                INSERT OR REPLACE INTO goodreads_signals
                    (workKey, goodreadsBookId, shelfScore, avgRating, ratingsCount, listAppearances, matchMethod)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_key,
                    row.get("goodreads_book_id"),
                    row.get("shelf_score"),
                    row.get("avg_rating"),
                    row.get("ratings_count"),
                    row.get("list_appearances"),
                    row.get("match_method"),
                ),
            )
    conn.commit()
    return inserted


def popularity_signal(rank: int | None, catalog_size: int) -> float:
    """0..1 popularity signal from an OL `popularityRank` — mirrors
    Sources/SpineMatching/FieldAwareScore.swift's `popularityScore`
    (there 0..100; normalized to 0..1 here), just without its 0..100 scale."""
    if rank is None or rank < 1 or catalog_size <= 1:
        return 0.0
    value = 1 - math.log(rank) / math.log(catalog_size)
    return min(max(value, 0.0), 1.0)


def rerank_popularity(conn: sqlite3.Connection, *, shelf_weight: float = DEFAULT_SHELF_WEIGHT) -> int:
    rows = conn.execute("SELECT id, workKey, popularityRank FROM books").fetchall()
    catalog_size = max(len(rows), 2)
    shelf_by_work = dict(conn.execute("SELECT workKey, shelfScore FROM goodreads_signals").fetchall())

    scored = []
    for row_id, work_key, old_rank in rows:
        ol_signal = popularity_signal(old_rank, catalog_size)
        shelf = shelf_by_work.get(work_key) or 0.0
        combined = (1 - shelf_weight) * ol_signal + shelf_weight * shelf
        scored.append((combined, old_rank if old_rank is not None else catalog_size, row_id))

    # Sort by combined score descending (more popular -> rank 1); ties
    # broken by the original OL rank ascending for determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))

    conn.executemany(
        "UPDATE books SET popularityRank = ? WHERE id = ?",
        [(new_rank, row_id) for new_rank, (_combined, _old, row_id) in enumerate(scored, start=1)],
    )
    conn.commit()
    return len(scored)


def invoke_catalog_build(
    scratch_db: Path,
    output_db: Path,
    *,
    min_editions: int,
    max_works: int,
    catalog_build_cmd: Sequence[str] = DEFAULT_CATALOG_BUILD_CMD,
) -> subprocess.CompletedProcess:
    cmd = [
        *catalog_build_cmd,
        "--subset-from",
        str(scratch_db),
        "--output",
        str(output_db),
        "--min-editions",
        str(min_editions),
        "--max-works",
        str(max_works),
    ]
    print(f"[build_ios_en_from_goodreads] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=_REPO, check=True)


def run(
    full_db: Path,
    intermediate_dir: Path,
    matched_goodreads_path: Path,
    output_path: Path,
    *,
    scratch_db: Path | None = None,
    keep_scratch: bool = False,
    languages: frozenset[str] = DEFAULT_LANGUAGES,
    min_editions: int = DEFAULT_MIN_EDITIONS,
    max_works: int = DEFAULT_MAX_WORKS,
    shelf_weight: float = DEFAULT_SHELF_WEIGHT,
    gap_fill: bool = True,
    gap_fill_min_ratings_count: int = DEFAULT_GAP_FILL_MIN_RATINGS_COUNT,
    catalog_build_cmd: Sequence[str] = DEFAULT_CATALOG_BUILD_CMD,
    min_intermediate_match_ratio: float = 0.5,
) -> dict[str, int]:
    scratch_path = scratch_db or output_path.with_name(output_path.stem + ".scratch.sqlite")
    copy_to_scratch(full_db, scratch_path)

    conn = sqlite3.connect(scratch_path)
    try:
        add_languages_column(conn)
        work_languages = load_work_languages(intermediate_dir)
        total, lang_matched = populate_languages(conn, work_languages)
        if total > 0 and (lang_matched / total) < min_intermediate_match_ratio:
            raise IntermediateMismatchError(
                f"Only {lang_matched}/{total} books resolved against --intermediate-dir "
                f"{intermediate_dir}'s works.jsonl.gz workKeys (< {min_intermediate_match_ratio:.0%}) "
                "— this is almost certainly the wrong intermediate for this full.sqlite. Aborting "
                "rather than silently pruning rows on an unreliable language signal."
            )

        create_goodreads_signals_table(conn)
        matched_rows = load_matched_goodreads(matched_goodreads_path)
        signal_count = populate_goodreads_signals(conn, matched_rows)

        pruned = prune_to_languages(conn, languages) if languages else 0

        placeholder_rank = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] or 1
        gap_filled = (
            gap_fill_unmatched(
                conn, matched_rows, min_ratings_count=gap_fill_min_ratings_count, placeholder_rank=placeholder_rank
            )
            if gap_fill
            else 0
        )

        reranked = rerank_popularity(conn, shelf_weight=shelf_weight)
    finally:
        conn.close()

    invoke_catalog_build(
        scratch_path, output_path, min_editions=min_editions, max_works=max_works, catalog_build_cmd=catalog_build_cmd
    )

    if not keep_scratch:
        scratch_path.unlink(missing_ok=True)

    return {
        "total_books": total,
        "language_matched": lang_matched,
        "pruned": pruned,
        "goodreads_signals": signal_count,
        "gap_filled": gap_filled,
        "reranked": reranked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-db", type=Path, required=True)
    parser.add_argument("--intermediate-dir", type=Path, required=True)
    parser.add_argument("--matched-goodreads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-db", type=Path, default=None)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--languages", nargs="*", default=sorted(DEFAULT_LANGUAGES))
    parser.add_argument("--min-editions", type=int, default=DEFAULT_MIN_EDITIONS)
    parser.add_argument("--max-works", type=int, default=DEFAULT_MAX_WORKS)
    parser.add_argument("--shelf-weight", type=float, default=DEFAULT_SHELF_WEIGHT)
    parser.add_argument("--no-gap-fill", action="store_true")
    parser.add_argument("--gap-fill-min-ratings-count", type=int, default=DEFAULT_GAP_FILL_MIN_RATINGS_COUNT)
    args = parser.parse_args(argv)

    result = run(
        args.full_db,
        args.intermediate_dir,
        args.matched_goodreads,
        args.output,
        scratch_db=args.scratch_db,
        keep_scratch=args.keep_scratch,
        languages=frozenset(args.languages),
        min_editions=args.min_editions,
        max_works=args.max_works,
        shelf_weight=args.shelf_weight,
        gap_fill=not args.no_gap_fill,
        gap_fill_min_ratings_count=args.gap_fill_min_ratings_count,
    )
    print("[build_ios_en_from_goodreads] " + json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
