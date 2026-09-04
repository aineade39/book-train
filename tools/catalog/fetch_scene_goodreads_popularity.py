#!/usr/bin/env python3
"""Look up Goodreads popularity signals for every oracle `bookId` in scene fixtures.

For each book in optimize-gemini's oracle JSON fixtures, this script:
  1. Tries a fuzzy match against scraped `matched_goodreads.jsonl.gz` (fast).
  2. Otherwise searches goodreads.com, scores the top search hits, and reads
     `ratingsCount` / `avgRating` from the search row (or book_show fallback).
  3. Writes a per-book report with `popularityRank` (1 = highest `ratingsCount`
     among resolved scene books).

Checkpointed in SQLite so an interrupted run resumes without re-fetching.
Use `--engine playwright` when Goodreads blocks plain HTTP (HTTP 202 / empty body).

Usage:
    python tools/catalog/fetch_scene_goodreads_popularity.py
    python tools/catalog/fetch_scene_goodreads_popularity.py --engine playwright
    python tools/catalog/fetch_scene_goodreads_popularity.py --retry-failed --limit 20
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.build_oracle_test_catalog import (  # noqa: E402
    DEFAULT_ORACLES_DIR,
    load_scene_books,
)
from tools.catalog.match_goodreads import (  # noqa: E402
    ACCEPT_THRESHOLD,
    GoodreadsBook,
    OLCandidate,
    _score_candidate,
    author_block_key,
    parse_book_show_html,
    strip_series_suffix,
)
from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.derived_meta import git_commit_short  # noqa: E402
from tools.paths import catalog_dir, catalog_goodreads  # noqa: E402

if TYPE_CHECKING:
    from playwright.sync_api import Page

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SEARCH_URL = "https://www.goodreads.com/search?q={query}"
BOOK_SHOW_RE = re.compile(r"/book/show/(\d+)[.\-]([^\"?&]+)")
SEARCH_RATINGS_RE = re.compile(
    r"([\d.]+)\s*avg rating\s*(?:—|-)\s*([\d,]+)\s*ratings",
    re.IGNORECASE,
)
DEFAULT_PAUSE_RANGE = (1.5, 3.0)
STATUS_DONE = "done"
STATUS_FAILED = "failed"
RELAXED_MATCH_THRESHOLD = 75.0


@dataclass(frozen=True)
class SceneBookRow:
    scene: str
    book_id: str
    title: str
    author: str


@dataclass
class GoodreadsLookup:
    goodreads_book_id: int | None = None
    goodreads_title: str | None = None
    avg_rating: float | None = None
    ratings_count: int | None = None
    source: str = "unresolved"
    match_score: float | None = None
    error: str | None = None


class MatchedGoodreadsIndex:
    """Block scraped Goodreads rows by author for fast fuzzy shortlists."""

    def __init__(self, rows: list[GoodreadsBook]) -> None:
        from collections import defaultdict

        self._by_block: dict[str, list[GoodreadsBook]] = defaultdict(list)
        self._all: list[GoodreadsBook] = rows
        for row in rows:
            self._by_block[author_block_key(row.author)].append(row)

    def shortlist(self, title: str, author: str, *, cap: int = 40) -> list[GoodreadsBook]:
        block = self._by_block.get(author_block_key(author), [])
        if block:
            return block[:cap]
        if not author:
            title_norm = normalize_for_search(strip_series_suffix(title))
            scored = []
            for row in self._all:
                candidate = OLCandidate(
                    "",
                    row.title,
                    row.author,
                    normalize_for_search(strip_series_suffix(row.title)),
                    normalize_for_search(row.author),
                    0,
                )
                title_score = _score_candidate(title_norm, "", candidate)
                if title_score >= 50:
                    scored.append((title_score, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [row for _, row in scored[:cap]]
        return []


def load_matched_goodreads_index(path: Path) -> MatchedGoodreadsIndex:
    rows: list[GoodreadsBook] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                GoodreadsBook(
                    book_id=int(record["goodreads_book_id"]),
                    title=str(record["title"]),
                    author=str(record["author"]),
                    avg_rating=record.get("avg_rating"),
                    ratings_count=record.get("ratings_count"),
                )
            )
    return MatchedGoodreadsIndex(rows)


def score_oracle_to_goodreads(
    title: str,
    author: str,
    candidates: list[GoodreadsBook],
) -> tuple[GoodreadsBook | None, float | None]:
    title_norm = normalize_for_search(strip_series_suffix(title))
    author_norm = normalize_for_search(author)
    if not candidates:
        return None, None
    scored = [
        (
            row,
            _score_candidate(
                title_norm,
                author_norm,
                OLCandidate(
                    "",
                    row.title,
                    row.author,
                    normalize_for_search(strip_series_suffix(row.title)),
                    normalize_for_search(row.author),
                    0,
                ),
            ),
        )
        for row in candidates
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_row, best_score = scored[0]
    return best_row, best_score


def lookup_to_result(
    book: GoodreadsBook,
    *,
    score: float,
    source: str,
) -> GoodreadsLookup:
    return GoodreadsLookup(
        goodreads_book_id=book.book_id,
        goodreads_title=book.title,
        avg_rating=book.avg_rating,
        ratings_count=book.ratings_count,
        source=source,
        match_score=score,
    )


def lookup_from_matched_goodreads(
    title: str,
    author: str,
    index: MatchedGoodreadsIndex,
) -> GoodreadsLookup:
    candidates = index.shortlist(title, author)
    best, score = score_oracle_to_goodreads(title, author, candidates)
    if best is None or score is None:
        return GoodreadsLookup(source="matched_unresolved")
    if score >= ACCEPT_THRESHOLD and best.ratings_count is not None:
        return lookup_to_result(best, score=score, source="matched_goodreads")
    if score >= RELAXED_MATCH_THRESHOLD and best.ratings_count is not None:
        return lookup_to_result(best, score=score, source="matched_goodreads_relaxed")
    return GoodreadsLookup(source="matched_unresolved", match_score=score)


def parse_search_candidates(html: str) -> list[GoodreadsBook]:
    """Parse Goodreads search HTML into candidate rows (best-effort)."""
    candidates: list[GoodreadsBook] = []
    seen: set[int] = set()
    for row_html in re.findall(r"<tr[^>]*>.*?</tr>", html, flags=re.DOTALL | re.IGNORECASE):
        if "/book/show/" not in row_html:
            continue
        link_match = BOOK_SHOW_RE.search(row_html)
        if not link_match:
            continue
        book_id = int(link_match.group(1))
        if book_id in seen:
            continue
        seen.add(book_id)
        title_match = re.search(
            r'class="bookTitle"[^>]*>\s*<span[^>]*>([^<]+)</span>',
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        author_match = re.search(
            r'class="authorName[^"]*"[^>]*>([^<]+)</a>',
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        ratings_match = SEARCH_RATINGS_RE.search(row_html)
        avg_rating = float(ratings_match.group(1)) if ratings_match else None
        ratings_count = (
            int(ratings_match.group(2).replace(",", "")) if ratings_match else None
        )
        candidates.append(
            GoodreadsBook(
                book_id=book_id,
                title=(title_match.group(1).strip() if title_match else ""),
                author=(author_match.group(1).strip() if author_match else ""),
                avg_rating=avg_rating,
                ratings_count=ratings_count,
            )
        )
        if len(candidates) >= 8:
            break
    return candidates


def pick_search_match(title: str, author: str, candidates: list[GoodreadsBook]) -> GoodreadsLookup:
    best, score = score_oracle_to_goodreads(title, author, candidates)
    if best is None or score is None or score < RELAXED_MATCH_THRESHOLD:
        return GoodreadsLookup(source="search_unmatched", match_score=score)
    if best.ratings_count is None:
        return GoodreadsLookup(
            goodreads_book_id=best.book_id,
            goodreads_title=best.title,
            source="search_no_ratings",
            match_score=score,
        )
    return lookup_to_result(best, score=score, source="goodreads_search")


def fetch_url(url: str, *, timeout: float = 45.0, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            if body:
                return body
            last_error = ValueError("empty response body")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
        time.sleep(1.5 + attempt * 2.0 + random.uniform(0, 1.0))
    raise last_error or RuntimeError("fetch failed")


def lookup_from_goodreads_web(title: str, author: str) -> GoodreadsLookup:
    query = urllib.parse.quote(f"{title} {author}".strip())
    search_html = fetch_url(SEARCH_URL.format(query=query))
    candidates = parse_search_candidates(search_html)
    if candidates:
        picked = pick_search_match(title, author, candidates)
        if picked.ratings_count is not None:
            return picked

    parsed = BOOK_SHOW_RE.search(search_html)
    if parsed is None:
        return GoodreadsLookup(source="search_failed", error="no search result")
    book_id, slug = int(parsed.group(1)), parsed.group(2)
    book_html = fetch_url(f"https://www.goodreads.com/book/show/{book_id}.{slug}")
    page = parse_book_show_html(book_html)
    if page is None or page.get("book_id") is None:
        return GoodreadsLookup(
            goodreads_book_id=book_id,
            source="book_show_failed",
            error="could not parse book_show page",
        )
    return GoodreadsLookup(
        goodreads_book_id=int(page["book_id"]),
        goodreads_title=page.get("title"),
        avg_rating=page.get("avg_rating"),
        ratings_count=page.get("ratings_count"),
        source="goodreads_book_show",
    )


class PlaywrightGoodreadsClient:
    """Headless-browser Goodreads search for when plain HTTP is blocked."""

    def __init__(self, *, pause_min: float, pause_max: float) -> None:
        from playwright.sync_api import sync_playwright

        self._pause_min = pause_min
        self._pause_max = pause_max
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(user_agent=BROWSER_USER_AGENT)

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()

    def lookup(self, title: str, author: str) -> GoodreadsLookup:
        query = f"{title} {author}".strip()
        url = SEARCH_URL.format(query=urllib.parse.quote(query))
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as exc:
            return GoodreadsLookup(source="error", error=f"search navigation: {exc}")
        candidates = self._extract_candidates()
        if candidates:
            picked = pick_search_match(title, author, candidates)
            if picked.ratings_count is not None:
                time.sleep(random.uniform(self._pause_min, self._pause_max))
                return picked
        time.sleep(random.uniform(self._pause_min, self._pause_max))
        return GoodreadsLookup(source="search_failed", error="no scored search result")

    def _extract_candidates(self) -> list[GoodreadsBook]:
        raw_rows: list[dict] = self._page.evaluate(
            """
            () => {
              const rows = document.querySelectorAll('table.tableList tr');
              const out = [];
              for (const row of rows) {
                const link = row.querySelector('a[href*="/book/show/"]');
                if (!link) continue;
                const href = link.getAttribute('href') || '';
                const m = href.match(/\\/book\\/show\\/(\\d+)/);
                if (!m) continue;
                const titleEl = row.querySelector('.bookTitle span') || row.querySelector('[itemprop="name"]');
                const authorEl = row.querySelector('a.authorName');
                const text = row.textContent || '';
                const ratingsM = text.match(/([\\d.]+)\\s*avg rating\\s*(?:—|-)\\s*([\\d,]+)\\s*ratings/i);
                out.push({
                  book_id: parseInt(m[1], 10),
                  title: (titleEl && titleEl.textContent || '').trim(),
                  author: (authorEl && authorEl.textContent || '').trim(),
                  avg_rating: ratingsM ? parseFloat(ratingsM[1]) : null,
                  ratings_count: ratingsM ? parseInt(ratingsM[2].replace(/,/g, ''), 10) : null,
                });
                if (out.length >= 8) break;
              }
              return out;
            }
            """
        )
        return [
            GoodreadsBook(
                book_id=int(row["book_id"]),
                title=str(row.get("title") or ""),
                author=str(row.get("author") or ""),
                avg_rating=row.get("avg_rating"),
                ratings_count=row.get("ratings_count"),
            )
            for row in raw_rows
        ]


def init_checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_goodreads (
            bookId TEXT PRIMARY KEY,
            scene TEXT,
            title TEXT,
            author TEXT,
            status TEXT,
            goodreadsBookId INTEGER,
            goodreadsTitle TEXT,
            avgRating REAL,
            ratingsCount INTEGER,
            source TEXT,
            matchScore REAL,
            error TEXT,
            updatedAt TEXT
        )
        """
    )
    conn.commit()
    return conn


def clear_failed_checkpoint_rows(conn: sqlite3.Connection) -> int:
    deleted = conn.execute(
        "DELETE FROM scene_goodreads WHERE status != ? OR ratingsCount IS NULL",
        (STATUS_DONE,),
    ).rowcount
    conn.commit()
    return deleted


def load_all_checkpoint_rows(conn: sqlite3.Connection) -> dict[str, GoodreadsLookup]:
    rows = conn.execute(
        """
        SELECT bookId, goodreadsBookId, goodreadsTitle, avgRating, ratingsCount, source, matchScore, error
        FROM scene_goodreads
        WHERE status = ? AND ratingsCount IS NOT NULL
        """,
        (STATUS_DONE,),
    ).fetchall()
    out: dict[str, GoodreadsLookup] = {}
    for book_id, gr_id, gr_title, avg_rating, ratings_count, source, match_score, error in rows:
        out[book_id] = GoodreadsLookup(
            goodreads_book_id=gr_id,
            goodreads_title=gr_title,
            avg_rating=avg_rating,
            ratings_count=ratings_count,
            source=source or "cache",
            match_score=match_score,
            error=error,
        )
    return out


def load_checkpoint_row(conn: sqlite3.Connection, book_id: str) -> GoodreadsLookup | None:
    return load_all_checkpoint_rows(conn).get(book_id)


def save_checkpoint_row(
    conn: sqlite3.Connection,
    book: SceneBookRow,
    lookup: GoodreadsLookup,
    *,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO scene_goodreads (
            bookId, scene, title, author, status, goodreadsBookId, goodreadsTitle,
            avgRating, ratingsCount, source, matchScore, error, updatedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            book.book_id,
            book.scene,
            book.title,
            book.author,
            status,
            lookup.goodreads_book_id,
            lookup.goodreads_title,
            lookup.avg_rating,
            lookup.ratings_count,
            lookup.source,
            lookup.match_score,
            lookup.error,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def assign_popularity_ranks(rows: list[dict]) -> None:
    """Rank resolved scene books by Goodreads ratingsCount (1 = most ratings)."""
    ranked = sorted(
        [row for row in rows if row.get("ratingsCount") is not None],
        key=lambda row: (-int(row["ratingsCount"]), row.get("bookId", "")),
    )
    for rank, row in enumerate(ranked, start=1):
        row["popularityRank"] = rank
    for row in rows:
        if row.get("ratingsCount") is None:
            row["popularityRank"] = None


def write_report(path: Path, rows: list[dict], *, oracles_dir: Path, checkpoint_db: Path) -> None:
    sidecar = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit_short(),
        "oracles_dir": str(oracles_dir),
        "checkpoint_db": str(checkpoint_db),
        "book_count": len(rows),
        "with_ratings_count": sum(1 for row in rows if row.get("ratingsCount") is not None),
    }
    path.write_text(json.dumps({"meta": sidecar, "books": rows}, indent=2) + "\n", encoding="utf-8")


def resolve_book(
    book: SceneBookRow,
    gr_index: MatchedGoodreadsIndex,
    *,
    engine: str,
    playwright_client: PlaywrightGoodreadsClient | None,
) -> GoodreadsLookup:
    lookup = lookup_from_matched_goodreads(book.title, book.author, gr_index)
    if lookup.ratings_count is not None:
        return lookup
    try:
        if engine == "playwright" and playwright_client is not None:
            return playwright_client.lookup(book.title, book.author)
        return lookup_from_goodreads_web(book.title, book.author)
    except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
        return GoodreadsLookup(source="error", error=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracles-dir", type=Path, default=DEFAULT_ORACLES_DIR)
    parser.add_argument(
        "--matched-goodreads",
        type=Path,
        default=catalog_goodreads("matched_goodreads.jsonl.gz"),
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=catalog_goodreads("scene_goodreads_checkpoint.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=catalog_dir("oracle_scene_goodreads_popularity.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only N unresolved books")
    parser.add_argument("--pause-min", type=float, default=DEFAULT_PAUSE_RANGE[0])
    parser.add_argument("--pause-max", type=float, default=DEFAULT_PAUSE_RANGE[1])
    parser.add_argument(
        "--engine",
        choices=("urllib", "playwright"),
        default="urllib",
        help="HTTP engine for Goodreads search (use playwright when urllib is blocked)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Drop failed checkpoint rows before processing",
    )
    args = parser.parse_args()

    if not args.oracles_dir.is_dir():
        print(f"Missing oracles dir: {args.oracles_dir}", file=sys.stderr)
        return 1
    if not args.matched_goodreads.exists():
        print(f"Missing matched Goodreads file: {args.matched_goodreads}", file=sys.stderr)
        return 1

    scene_books = [
        SceneBookRow(scene=book.scene, book_id=book.book_id, title=book.title, author=book.author)
        for book in load_scene_books(args.oracles_dir)
        if book.book_id
    ]
    if not scene_books:
        print(f"No scene bookIds found in {args.oracles_dir}", file=sys.stderr)
        return 1

    gr_index = load_matched_goodreads_index(args.matched_goodreads)
    checkpoint = init_checkpoint(args.checkpoint_db)
    if args.retry_failed:
        cleared = clear_failed_checkpoint_rows(checkpoint)
        print(f"Cleared {cleared} failed checkpoint rows", file=sys.stderr)

    playwright_client: PlaywrightGoodreadsClient | None = None
    if args.engine == "playwright":
        playwright_client = PlaywrightGoodreadsClient(
            pause_min=args.pause_min,
            pause_max=args.pause_max,
        )

    resolved: dict[str, GoodreadsLookup] = {}
    processed = 0
    try:
        for book in scene_books:
            cached = load_checkpoint_row(checkpoint, book.book_id)
            if cached is not None:
                resolved[book.book_id] = cached
                continue
            if args.limit is not None and processed >= args.limit:
                continue

            lookup = resolve_book(
                book,
                gr_index,
                engine=args.engine,
                playwright_client=playwright_client,
            )
            status = STATUS_DONE if lookup.ratings_count is not None else STATUS_FAILED
            save_checkpoint_row(checkpoint, book, lookup, status=status)
            resolved[book.book_id] = lookup
            processed += 1
            print(
                f"[{processed}] {book.book_id}: ratings={lookup.ratings_count} "
                f"source={lookup.source} gr_id={lookup.goodreads_book_id}",
                file=sys.stderr,
            )
    finally:
        if playwright_client is not None:
            playwright_client.close()
        checkpoint.close()

    report_rows: list[dict] = []
    for book in scene_books:
        lookup = resolved.get(book.book_id) or GoodreadsLookup(source="missing")
        report_rows.append(
            {
                "scene": book.scene,
                "bookId": book.book_id,
                "title": book.title,
                "author": book.author,
                "goodreadsBookId": lookup.goodreads_book_id,
                "goodreadsTitle": lookup.goodreads_title,
                "avgRating": lookup.avg_rating,
                "ratingsCount": lookup.ratings_count,
                "popularityRank": None,
                "source": lookup.source,
                "matchScore": lookup.match_score,
                "error": lookup.error,
            }
        )

    assign_popularity_ranks(report_rows)
    write_report(args.output, report_rows, oracles_dir=args.oracles_dir, checkpoint_db=args.checkpoint_db)

    with_ratings = sum(1 for row in report_rows if row.get("ratingsCount") is not None)
    print(
        f"Wrote {args.output} ({len(report_rows)} books, {with_ratings} with ratingsCount)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
