#!/usr/bin/env python3
"""Build a small catalog SQLite containing every oracle fixture book title.

Matches books from optimize-gemini's oracle JSON fixtures against
`full.sqlite` (FTS shortlist + fuzzy title/author scoring), then
materializes a proper `SpineCatalog` database via `catalog-build`.
Oracle books that don't match any OL work are gap-filled as synthetic rows
so the test DB still contains every oracle title.

Usage:
    python tools/catalog/build_oracle_test_catalog.py
    python tools/catalog/build_oracle_test_catalog.py \\
        --full-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \\
        --oracles-dir /path/to/optimize-gemini/fixtures/oracles \\
        --output $BOOK_SPINES_DATA/derived/book-catalog/oracle_test.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.match_goodreads import (  # noqa: E402
    ACCEPT_THRESHOLD,
    OLCandidate,
    _score_candidate,
    strip_series_suffix,
)
from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.derived_meta import git_commit_short  # noqa: E402
from tools.paths import catalog_dir  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install rapidfuzz: pip install rapidfuzz") from exc

DEFAULT_ORACLES_DIR = _REPO.parent / "optimize-gemini" / "fixtures" / "oracles"
MIN_FTS_TOKEN_LEN = 3
DEFAULT_CATALOG_BUILD_CMD = ("swift", "run", "-c", "release", "catalog-build")
_FTS_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "you",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "into",
        "about",
        "book",
        "books",
        "guide",
        "history",
        "edition",
    }
)
# Below AcceptPolicy's 90/8 bar, but still rejects obvious junk when gap-filling
# isn't needed. Ambiguous matches fall through to gap-fill so every oracle title
# is present in the output DB.
RELAXED_MATCH_THRESHOLD = 75.0


@dataclass(frozen=True)
class OracleBook:
    scene: str
    title: str
    author: str
    book_id: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.title.strip(), self.author.strip())


@dataclass
class OracleMatch:
    book: OracleBook
    method: str  # "fuzzy" | "gapfill"
    work_key: str | None = None
    ol_title: str | None = None
    ol_author: str | None = None
    score: float | None = None
    margin: float | None = None


def load_oracle_books(oracles_dir: Path) -> list[OracleBook]:
    books: list[OracleBook] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(oracles_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("books", []):
            title = (row.get("title") or "").strip()
            author = (row.get("author") or "").strip()
            if not title:
                continue
            key = (title, author)
            if key in seen:
                continue
            seen.add(key)
            book_id = (row.get("bookId") or "").strip()
            books.append(OracleBook(scene=path.stem, title=title, author=author, book_id=book_id))
    return books


def load_scene_books(oracles_dir: Path) -> list[OracleBook]:
    """Every oracle `bookId` in scene fixtures (not deduped across scenes)."""
    books: list[OracleBook] = []
    for path in sorted(oracles_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("books", []):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            author = (row.get("author") or "").strip()
            book_id = (row.get("bookId") or "").strip()
            books.append(OracleBook(scene=path.stem, title=title, author=author, book_id=book_id))
    return books


def fts_tokens(text: str) -> list[str]:
    tokens = [t for t in normalize_for_search(text).split() if len(t) >= MIN_FTS_TOKEN_LEN]
    if not tokens:
        return []
    filtered = [t for t in tokens if t not in _FTS_STOPWORDS]
    return filtered or tokens


def fts_shortlist(conn: sqlite3.Connection, title: str, author: str, cap: int) -> list[OLCandidate]:
    tokens = fts_tokens(f"{title} {author}".strip()) or fts_tokens(title)
    if not tokens:
        return []
    selected = tokens[:3]
    if len(selected) == 1:
        query = f'"{selected[0]}"'
    else:
        query = " AND ".join(f'"{t}"' for t in selected)
    try:
        rows = conn.execute(
            """
            SELECT b.workKey, b.title, b.author, b.titleNormalized, b.authorNormalized, b.editionCount
            FROM books_fts f
            JOIN books b ON b.id = f.rowid
            WHERE books_fts MATCH ?
            LIMIT ?
            """,
            (query, cap),
        ).fetchall()
    except sqlite3.OperationalError:
        # Rare tokenization edge cases — fall back to a looser OR query.
        fallback = " OR ".join(f'"{t}"' for t in selected)
        try:
            rows = conn.execute(
                """
                SELECT b.workKey, b.title, b.author, b.titleNormalized, b.authorNormalized, b.editionCount
                FROM books_fts f
                JOIN books b ON b.id = f.rowid
                WHERE books_fts MATCH ?
                LIMIT ?
                """,
                (fallback, cap),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        OLCandidate(
            work_key=row[0],
            title=row[1],
            author=row[2],
            title_normalized=row[3],
            author_normalized=row[4],
            edition_count=row[5] or 0,
        )
        for row in rows
    ]



def score_oracle_book(book: OracleBook, candidates: list[OLCandidate]) -> tuple[OLCandidate | None, float | None, float | None]:
    title_norm = normalize_for_search(strip_series_suffix(book.title))
    author_norm = normalize_for_search(book.author)
    if not candidates:
        return None, None, None

    scored = [(c, _score_candidate(title_norm, author_norm, c)) for c in candidates]
    best_per_work: dict[str, tuple[OLCandidate, float]] = {}
    for candidate, score in scored:
        existing = best_per_work.get(candidate.work_key)
        if existing is None or existing[1] < score:
            best_per_work[candidate.work_key] = (candidate, score)
    ranked = sorted(best_per_work.values(), key=lambda item: item[1], reverse=True)
    top_candidate, top_score = ranked[0]
    margin = (top_score - ranked[1][1]) if len(ranked) > 1 else None
    return top_candidate, top_score, margin


def accept_match(score: float | None, margin: float | None) -> bool:
    if score is None:
        return False
    if score >= ACCEPT_THRESHOLD:
        return True
    if score >= RELAXED_MATCH_THRESHOLD:
        return True
    return False


def default_work_key(title: str, author: str) -> str:
    return f"{normalize_for_search(title)}|{normalize_for_search(author)}"


def match_oracle_books(conn: sqlite3.Connection, books: list[OracleBook], *, shortlist_cap: int) -> list[OracleMatch]:
    matches: list[OracleMatch] = []
    for index, book in enumerate(books, start=1):
        if index % 50 == 0 or index == len(books):
            print(f"Matching oracle books: {index}/{len(books)}...", file=sys.stderr)
        candidates = fts_shortlist(conn, book.title, book.author, shortlist_cap)
        top, score, margin = score_oracle_book(book, candidates)
        if top is not None and accept_match(score, margin):
            matches.append(
                OracleMatch(
                    book=book,
                    method="fuzzy",
                    work_key=top.work_key,
                    ol_title=top.title,
                    ol_author=top.author,
                    score=score,
                    margin=margin,
                )
            )
            continue
        work_key = "oracle-gapfill:" + default_work_key(book.title, book.author)
        matches.append(
            OracleMatch(
                book=book,
                method="gapfill",
                work_key=work_key,
                score=score,
                margin=margin,
            )
        )
    return matches


@dataclass(frozen=True)
class OLBookMeta:
    isbn: str | None
    popularity_rank: int | None
    edition_count: int | None


def fetch_ol_rows(full_db: Path, work_keys: list[str]) -> dict[str, OLBookMeta]:
    if not work_keys:
        return {}
    conn = sqlite3.connect(f"file:{full_db}?mode=ro", uri=True)
    out: dict[str, OLBookMeta] = {}
    batch_size = 200
    for start in range(0, len(work_keys), batch_size):
        chunk = work_keys[start : start + batch_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT workKey, isbn, popularityRank, editionCount
            FROM books
            WHERE workKey IN ({placeholders})
            ORDER BY popularityRank IS NULL, popularityRank ASC
            """,
            chunk,
        ).fetchall()
        for work_key, isbn, popularity_rank, edition_count in rows:
            if work_key in out:
                continue
            out[work_key] = OLBookMeta(
                isbn=isbn,
                popularity_rank=popularity_rank,
                edition_count=edition_count,
            )
    conn.close()
    return out


def enrich_popularity_from_full(output_db: Path, full_db: Path) -> int:
    """Copy `popularityRank` / `editionCount` from `full.sqlite` into the test DB."""
    conn = sqlite3.connect(output_db)
    conn.execute("ATTACH DATABASE ? AS full", (str(full_db),))
    updated = conn.execute(
        """
        UPDATE books AS b
        SET popularityRank = (
            SELECT MIN(f.popularityRank)
            FROM full.books AS f
            WHERE f.workKey = b.workKey AND f.popularityRank IS NOT NULL
        ),
        editionCount = (
            SELECT MAX(f.editionCount)
            FROM full.books AS f
            WHERE f.workKey = b.workKey AND f.editionCount IS NOT NULL
        )
        WHERE b.workKey NOT LIKE 'oracle-gapfill:%'
        """
    ).rowcount
    conn.commit()
    conn.close()
    return updated


def _csv_field(value: str) -> str:
    if any(ch in value for ch in (",", '"', "\n", "\r")):
        return '"' + value.replace('"', '""') + '"'
    return value


def write_catalog_csv(matches: list[OracleMatch], full_db: Path, csv_path: Path) -> None:
    """Write catalog rows using oracle titles so every fixture label is searchable."""
    matched_keys = [m.work_key for m in matches if m.method == "fuzzy" and m.work_key]
    ol_by_work = fetch_ol_rows(full_db, matched_keys)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["title,author,isbn,workKey"]
    for match in matches:
        meta = ol_by_work.get(match.work_key or "") if match.method == "fuzzy" else None
        isbn = meta.isbn if meta else ""
        lines.append(
            ",".join(
                [
                    _csv_field(match.book.title),
                    _csv_field(match.book.author or "Unknown"),
                    _csv_field(isbn or ""),
                    _csv_field(match.work_key or ""),
                ]
            )
        )
    # catalog-build's hand-rolled CSV parser only handles LF line endings.
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_catalog_build(csv_path: Path, output_db: Path) -> None:
    for path in (output_db, output_db.with_suffix(".json")):
        if path.exists():
            path.unlink()
    cmd = [
        *DEFAULT_CATALOG_BUILD_CMD,
        str(csv_path),
        "--db",
        str(output_db),
    ]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=_REPO)


def write_sidecar(
    output_db: Path,
    *,
    work_count: int,
    matched_fuzzy: int,
    gapfilled: int,
    oracles_dir: Path,
    full_db: Path,
    scene_book_count: int,
    with_rank: int,
    catalog_rows_with_rank: int,
) -> None:
    sidecar = {
        "profile": "oracle_test",
        "works": work_count,
        "bytes": output_db.stat().st_size if output_db.exists() else 0,
        "git_commit": git_commit_short(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "full_db": str(full_db),
            "oracles_dir": str(oracles_dir),
        },
        "stats": {
            "oracle_books": matched_fuzzy + gapfilled,
            "scene_book_ids": scene_book_count,
            "matched_fuzzy": matched_fuzzy,
            "gapfilled": gapfilled,
            "with_popularity_rank": with_rank,
            "catalog_rows_with_rank": catalog_rows_with_rank,
        },
    }
    output_db.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def write_match_report(
    path: Path,
    matches: list[OracleMatch],
    *,
    ol_by_work: dict[str, OLBookMeta],
) -> None:
    rows = []
    for match in matches:
        meta = ol_by_work.get(match.work_key or "") if match.method == "fuzzy" else None
        rows.append(
            {
                "scene": match.book.scene,
                "oracle_title": match.book.title,
                "oracle_author": match.book.author,
                "method": match.method,
                "work_key": match.work_key,
                "ol_title": match.ol_title,
                "ol_author": match.ol_author,
                "score": match.score,
                "margin": match.margin,
                "popularityRank": meta.popularity_rank if meta else None,
                "editionCount": meta.edition_count if meta else None,
            }
        )
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def write_scene_book_popularity_report(
    path: Path,
    scene_books: list[OracleBook],
    matches: list[OracleMatch],
    ol_by_work: dict[str, OLBookMeta],
) -> None:
    """One row per oracle `bookId` in scene images with OL popularity metadata."""
    key_to_match: dict[tuple[str, str], OracleMatch] = {m.book.key: m for m in matches}
    rows = []
    for book in scene_books:
        match = key_to_match.get(book.key)
        work_key = match.work_key if match else None
        meta = ol_by_work.get(work_key or "") if match and match.method == "fuzzy" else None
        rows.append(
            {
                "scene": book.scene,
                "bookId": book.book_id,
                "title": book.title,
                "author": book.author,
                "method": match.method if match else None,
                "work_key": work_key,
                "popularityRank": meta.popularity_rank if meta else None,
                "editionCount": meta.edition_count if meta else None,
            }
        )
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-db", type=Path, default=catalog_dir("full.sqlite"))
    parser.add_argument("--oracles-dir", type=Path, default=DEFAULT_ORACLES_DIR)
    parser.add_argument("--output", type=Path, default=catalog_dir("oracle_test.sqlite"))
    parser.add_argument("--shortlist-cap", type=int, default=80)
    parser.add_argument(
        "--match-report",
        type=Path,
        default=None,
        help="Optional JSON report of oracle -> OL matches (default: next to output)",
    )
    parser.add_argument(
        "--scene-popularity-report",
        type=Path,
        default=None,
        help="Per-scene bookId popularity report (default: <output>.scene_popularity.json)",
    )
    args = parser.parse_args()

    if not args.full_db.exists():
        print(f"Missing full catalog: {args.full_db}", file=sys.stderr)
        return 1
    if not args.oracles_dir.is_dir():
        print(f"Missing oracles dir: {args.oracles_dir}", file=sys.stderr)
        return 1

    books = load_oracle_books(args.oracles_dir)
    scene_books = load_scene_books(args.oracles_dir)
    if not books:
        print(f"No oracle books found in {args.oracles_dir}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{args.full_db}?mode=ro", uri=True)
    matches = match_oracle_books(conn, books, shortlist_cap=args.shortlist_cap)
    conn.close()

    matched_fuzzy = sum(1 for m in matches if m.method == "fuzzy")
    gapfilled = sum(1 for m in matches if m.method == "gapfill")
    ol_by_work = fetch_ol_rows(
        args.full_db,
        [m.work_key for m in matches if m.method == "fuzzy" and m.work_key],
    )
    with_rank = sum(1 for m in matches if m.method == "fuzzy" and ol_by_work.get(m.work_key or "").popularity_rank is not None)
    print(
        f"Oracle books: {len(matches)} unique title/author pairs "
        f"({matched_fuzzy} matched in full.sqlite, {gapfilled} gap-filled)",
        file=sys.stderr,
    )

    scratch_dir = args.output.parent
    csv_path = scratch_dir / ".oracle_test_catalog.csv"
    try:
        write_catalog_csv(matches, args.full_db, csv_path)
        run_catalog_build(csv_path, args.output)
        enriched = enrich_popularity_from_full(args.output, args.full_db)
        print(f"Enriched popularityRank from full.sqlite: {enriched} rows", file=sys.stderr)
    finally:
        if csv_path.exists():
            csv_path.unlink()

    out_conn = sqlite3.connect(args.output)
    work_count = out_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    rank_count = out_conn.execute("SELECT COUNT(*) FROM books WHERE popularityRank IS NOT NULL").fetchone()[0]
    out_conn.close()

    write_sidecar(
        args.output,
        work_count=work_count,
        matched_fuzzy=matched_fuzzy,
        gapfilled=gapfilled,
        oracles_dir=args.oracles_dir,
        full_db=args.full_db,
        scene_book_count=len(scene_books),
        with_rank=with_rank,
        catalog_rows_with_rank=rank_count,
    )

    report_path = args.match_report or args.output.with_suffix(".matches.json")
    write_match_report(report_path, matches, ol_by_work=ol_by_work)
    scene_report_path = args.scene_popularity_report or args.output.with_suffix(".scene_popularity.json")
    write_scene_book_popularity_report(scene_report_path, scene_books, matches, ol_by_work)
    print(f"Built {args.output} ({work_count} works, {rank_count} with popularityRank)", file=sys.stderr)
    print(f"Match report: {report_path}", file=sys.stderr)
    print(f"Scene bookId popularity: {scene_report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
