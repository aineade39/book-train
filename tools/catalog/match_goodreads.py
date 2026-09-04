#!/usr/bin/env python3
"""Match scraped Goodreads books against the Open Library catalog (`full.sqlite`).

Two stages:

1. Parse + dedupe raw Goodreads scrape output (`tools/scrape_goodreads_lists.py`'s
   JSONL, one file per list_id, written by the `list_show` harness profile's
   parallel-array `extract` records) into one row per distinct Goodreads book_id,
   merging list-appearance/rating signals across every seed list a book showed up on.
   `--book-show-api` also contributes wholly new rows here, not just an ISBN
   overlay: a book_show_api success record (`legacy_id`/`isbn13`/`title`/`author`
   all present) for a `book_id` that never appeared on any seeded list_show list
   becomes a `list_appearances=0` row instead of being silently dropped — see
   `load_book_show_api_books`. This population was previously invisible to
   scoring/matching entirely (~12k `book_id`s, ~10.6k of them recoverable this
   way — see docs/BOOK_CATALOG.md's coverage-gap note).
2. Join each book to `full.sqlite`. Default path is the title+author bibliographic
   matcher in `bibliographic_join.py` (list_show title/author + OL `books` only —
   not the on-device OCR 90/8 `token_set_ratio` policy). Harvested ISBNs from
   `--book-show-api` / `--book-show-dir` are an overlay on top of that join:
   an ISBN hit is authoritative; a miss falls through to title+author. Every
   book added in step 1 above already carries an `isbn13`, so in practice these
   resolve via the ISBN overlay with no new bibliographic-matching risk.

Output is gzipped JSONL, one row per Goodreads book, carrying the match outcome plus
the raw shelf signals (avg_rating, ratings_count, list_appearances, ...) that
`build_ios_en_from_goodreads.py` turns into a `shelf_score` inside a *scratch copy* of
full.sqlite. This script only reads `full.sqlite` — it never writes to it.

Also writes `genre_tags.json` alongside the matched output: a tag -> display-name
mapping for every `genre` in `goodreads_seed_lists.yaml`, so each matched book's
`genres` field (the union of seed-list tags it appeared under, e.g. `["fantasy",
"romance"]`) can be rendered with a human-readable name. These are Listopia-list-
derived tags, NOT Goodreads' own per-book `bookGenres` (a separate, not-yet-built
`book_show` scrape) — see docs/BOOK_CATALOG.md.

Usage:
    python tools/catalog/match_goodreads.py --ol-db /path/to/full.sqlite \\
        --out matched_goodreads.jsonl.gz
    python tools/catalog/match_goodreads.py --ol-db /path/to/full.sqlite \\
        --book-show-dir <catalog_goodreads('book_show')> --out matched_goodreads.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bibliographic_join import (  # noqa: E402
    OLCandidate,
    TitleAuthorBlockIndex,
    candidates_from_ol_rows,
    enrich_candidate,
    evaluate_adversarial_pairs,
    evaluate_title_author_against_isbn,
    fold_for,
    load_adversarial_pairs,
    match_title_author,
    strip_series_suffix,
    title_lookup_keys,
)
from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

if TYPE_CHECKING:
    # Type-hint-only: tools.catalog.consolidate_popularity_signals imports
    # tools.catalog.extract_remaining_ids, which imports *this* module at its
    # own top level (for parse_book_id) -- importing it here for real would
    # be a circular import. run() below imports it lazily, after this module
    # has finished loading, to avoid that.
    from tools.catalog.consolidate_popularity_signals import ConsolidatedSignal

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install rapidfuzz: pip install rapidfuzz") from exc

SEED_LISTS_PATH = Path(__file__).resolve().parent / "goodreads_seed_lists.yaml"
DEFAULT_ADVERSARIAL_PAIRS_PATH = Path(__file__).resolve().parent / "matcher_adversarial_pairs.yaml"

# Legacy OCR-style thresholds. Kept because scene-popularity / oracle
# helpers still import them; the GR↔OL catalog join uses
# bibliographic_join.TITLE_ACCEPT / TITLE_MARGIN instead.
ACCEPT_THRESHOLD = 90.0
MARGIN_THRESHOLD = 8.0

_BOOK_URL_RE = re.compile(r"^/book/show/(\d+)")
_RATING_TEXT_RE = re.compile(r"([\d.]+)\s+avg rating\s*[-\u2013\u2014]+\s*([\d,]+)\s+ratings?")
_SCORE_TEXT_RE = re.compile(r"score:\s*([\d,]+)")
_VOTE_TEXT_RE = re.compile(r"([\d,]+)\s+people voted")
def parse_book_id(book_url: str | None) -> int | None:
    if not book_url:
        return None
    m = _BOOK_URL_RE.match(book_url)
    return int(m.group(1)) if m else None


def parse_rating_text(text: str | None) -> tuple[float | None, int | None]:
    """'4.55 avg rating — 745,415 ratings' -> (4.55, 745415)."""
    if not text:
        return None, None
    m = _RATING_TEXT_RE.search(text)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2).replace(",", ""))


def _parse_int_with_commas(text: str | None, pattern: re.Pattern[str]) -> int | None:
    if not text:
        return None
    m = pattern.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_score_text(text: str | None) -> int | None:
    """'score: 42,463' -> 42463."""
    return _parse_int_with_commas(text, _SCORE_TEXT_RE)


def parse_vote_text(text: str | None) -> int | None:
    """'430 people voted' -> 430."""
    return _parse_int_with_commas(text, _VOTE_TEXT_RE)


def author_block_key(author: str) -> str:
    """Loose blocking key: normalize, then drop every non-alphanumeric char
    so 'J.R.R. Tolkien' and 'J. R. R. Tolkien' land in the same bucket
    despite differing only in spacing/periods. Scoring still uses the
    properly normalized string; this key only limits how many OL candidates
    get fuzzy-scored per Goodreads book.
    """
    return re.sub(r"[^a-z0-9]", "", normalize_for_search(author))


@dataclass
class GoodreadsBook:
    book_id: int
    title: str
    author: str
    avg_rating: float | None = None
    ratings_count: int | None = None
    list_appearances: int = 0
    list_score_sum: int = 0
    vote_sum: int = 0
    genres: set[str] = field(default_factory=set)
    isbn13: str | None = None  # only set via an optional book_show merge


_LIST_SHOW_FIELDS = ("book_urls", "titles", "authors", "rating_texts", "score_texts", "vote_texts")


class SchemaError(ValueError):
    """Raised when a raw list_show JSONL record is missing a required field.

    This is the field contract between scrape-harness's list_show.yaml
    profile and this module (see that profile's "FIELD CONTRACT" header
    comment, and docs/BOOK_CATALOG.md's "scrape-harness dependency"
    section). It exists because this exact mismatch has happened before:
    a list_show.yaml edit silently dropped titles/authors/rating_texts,
    and this module kept running, quietly producing null ratings and
    failed fuzzy matches instead of failing loudly at the point of the
    actual problem.
    """


def validate_list_show_schema(raw_dir: Path) -> None:
    """Fail fast if any `<list_id>.jsonl` under `raw_dir` is missing one of
    `_LIST_SHOW_FIELDS`. Checks the first non-empty record of every file
    (cheap — one line each) rather than every record, since a field-contract
    break is a profile-wide change, not a per-record one.
    """
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                missing = [field for field in _LIST_SHOW_FIELDS if field not in record]
                if missing:
                    raise SchemaError(
                        f"{path}: record is missing required list_show field(s) {missing}. "
                        "This usually means sites/_templates/goodreads/profiles/list_show.yaml "
                        "was changed without updating match_goodreads.py's _LIST_SHOW_FIELDS "
                        "(or vice versa) — see that profile's FIELD CONTRACT comment and "
                        "book-train/docs/BOOK_CATALOG.md's 'scrape-harness dependency' section."
                    )
                break  # only the first record per file needs checking


def _zip_list_show_record(record: dict) -> Iterator[dict]:
    """One `extract` step on a list_show page yields one page-record of
    parallel arrays (see sites/_templates/goodreads/profiles/list_show.yaml's
    header comment for why the harness works this way). Zips defensively:
    uses the shortest array's length and warns on stderr rather than
    raising, since a length mismatch here means a markup change worth
    knowing about, not a reason to abort an otherwise-fine scrape.
    """
    lengths = {k: len(record.get(k) or []) for k in _LIST_SHOW_FIELDS}
    n = min(lengths.values()) if lengths else 0
    if n and len(set(lengths.values())) > 1:
        print(
            f"[match_goodreads] WARNING: field length mismatch in a list_show record "
            f"(lengths={lengths}); truncating to shortest ({n})",
            file=sys.stderr,
        )
    for i in range(n):
        yield {k: (record.get(k) or [])[i] if i < len(record.get(k) or []) else None for k in _LIST_SHOW_FIELDS}


def parse_list_show_file(path: Path, genre: str) -> Iterator[GoodreadsBook]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for row in _zip_list_show_record(record):
                book_id = parse_book_id(row.get("book_urls"))
                title = row.get("titles")
                author = row.get("authors")
                if book_id is None or not title or not author:
                    continue
                avg_rating, ratings_count = parse_rating_text(row.get("rating_texts"))
                yield GoodreadsBook(
                    book_id=book_id,
                    title=title.strip(),
                    author=author.strip(),
                    avg_rating=avg_rating,
                    ratings_count=ratings_count,
                    list_appearances=1,
                    list_score_sum=parse_score_text(row.get("score_texts")) or 0,
                    vote_sum=parse_vote_text(row.get("vote_texts")) or 0,
                    genres={genre} if genre else set(),
                )


def dedupe_books(rows: Iterable[GoodreadsBook]) -> dict[int, GoodreadsBook]:
    """A book on multiple seed lists (e.g. LOTR on both "Best Fantasy" and
    "Best SF&F") appears once per list; merge into one row per book_id,
    summing the per-list signal fields and unioning genres. Title/author/
    rating describe the book, not the list, so are kept from whichever
    occurrence set them first (they should already agree across lists).
    """
    by_id: dict[int, GoodreadsBook] = {}
    for row in rows:
        existing = by_id.get(row.book_id)
        if existing is None:
            by_id[row.book_id] = row
            continue
        existing.list_appearances += row.list_appearances
        existing.list_score_sum += row.list_score_sum
        existing.vote_sum += row.vote_sum
        existing.genres |= row.genres
        if existing.avg_rating is None and row.avg_rating is not None:
            existing.avg_rating = row.avg_rating
        if existing.ratings_count is None and row.ratings_count is not None:
            existing.ratings_count = row.ratings_count
    return by_id


def load_goodreads_books(raw_dir: Path, seed_lists: dict[int, str] | None = None) -> dict[int, GoodreadsBook]:
    """Parse every `<list_id>.jsonl` under `raw_dir` (written by
    tools/scrape_goodreads_lists.py into tools.paths.catalog_goodreads('raw'))
    and dedupe by Goodreads book_id. `seed_lists` (list_id -> genre) is
    optional metadata for the output `genres` field.
    """
    all_rows: list[GoodreadsBook] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        try:
            list_id = int(path.stem)
        except ValueError:
            continue
        genre = (seed_lists or {}).get(list_id, "")
        all_rows.extend(parse_list_show_file(path, genre))
    return dedupe_books(all_rows)


@dataclass(frozen=True)
class SeedListMeta:
    list_id: int
    slug: str
    genre: str
    short_label: str
    list_title: str


def load_seed_metadata(seed_lists_path: Path) -> dict[int, SeedListMeta]:
    """Reads `goodreads_seed_lists.yaml` -> {list_id: SeedListMeta}.

    `short_label`/`list_title` are optional per-entry fields (see the seed
    yaml's header comment); when omitted, they fall back to a title-cased
    genre tag and an underscore-to-space'd slug respectively, so existing
    entries need no manual edits to work with `build_genre_tag_mapping`.
    """
    if not seed_lists_path.exists():
        return {}
    import yaml

    data = yaml.safe_load(seed_lists_path.read_text(encoding="utf-8")) or {}
    out: dict[int, SeedListMeta] = {}
    for row in data.get("lists", []):
        list_id = int(row["list_id"])
        slug = str(row.get("slug", ""))
        genre = str(row.get("genre", ""))
        short_label = str(row.get("short_label") or (genre.replace("_", " ").title() if genre else slug))
        list_title = str(row.get("list_title") or slug.replace("_", " "))
        out[list_id] = SeedListMeta(list_id=list_id, slug=slug, genre=genre, short_label=short_label, list_title=list_title)
    return out


def build_genre_tag_mapping(meta: Iterable[SeedListMeta]) -> dict[str, dict]:
    """Tag -> display-name metadata, for `genre_tags.json`. If two seed
    lists ever share a `genre` tag, the first one (by seed-yaml order) wins
    and a warning is printed — today every tag in the seed yaml is unique,
    but nothing enforces that going forward.
    """
    mapping: dict[str, dict] = {}
    for m in meta:
        if not m.genre:
            continue
        if m.genre in mapping:
            print(
                f"[match_goodreads] WARNING: genre tag '{m.genre}' is used by both list "
                f"{mapping[m.genre]['list_id']} and list {m.list_id} — keeping the first "
                f"(seed-yaml order); tag-to-name mapping only supports one list per tag today",
                file=sys.stderr,
            )
            continue
        mapping[m.genre] = {
            "short_label": m.short_label,
            "list_title": m.list_title,
            "slug": m.slug,
            "list_id": m.list_id,
        }
    return mapping


# --- book_show (optional ISBN merge) ----------------------------------------


def parse_book_show_next_data(next_data_json: str) -> dict | None:
    """Walk the Next.js `__NEXT_DATA__` Apollo cache embedded in a book_show
    scrape (see sites/_templates/goodreads/profiles/book_show.yaml) and pull
    out the fields this module cares about. Returns None on any shape
    mismatch rather than raising — Goodreads controls this shape, not us,
    and a handful of unparseable book_show pages shouldn't abort a batch.
    """
    try:
        data = json.loads(next_data_json)
        apollo = data["props"]["pageProps"]["apolloState"]
        book_key = next(k for k in apollo if k.startswith("Book:"))
        book = apollo[book_key]
        details = book.get("details") or {}
        stats = _book_show_work_stats(apollo)
        return {
            "book_id": book.get("legacyId"),
            "title": book.get("title"),
            "isbn13": details.get("isbn13"),
            "language": (details.get("language") or {}).get("name"),
            "avg_rating": stats.get("averageRating"),
            "ratings_count": stats.get("ratingsCount"),
        }
    except (KeyError, StopIteration, json.JSONDecodeError, TypeError):
        return None


def _book_show_work_stats(apollo: dict) -> dict:
    """Pull `stats` from the first `Work:` entry in a book_show Apollo cache."""
    for key, value in apollo.items():
        if key.startswith("Work:") and isinstance(value, dict):
            stats = value.get("stats")
            if isinstance(stats, dict):
                return stats
    return {}


def parse_book_show_html(html: str) -> dict | None:
    """Parse a live or scraped Goodreads book_show HTML page."""
    match = re.search(r"<script id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>", html, re.DOTALL)
    if not match:
        return None
    return parse_book_show_next_data(match.group(1))


def load_book_show_isbns(book_show_dir: Path | None) -> dict[int, str]:
    """Optional merge: `<book_id>.jsonl` files under `book_show_dir` (from
    running the `book_show` profile for books that need ISBN
    disambiguation) -> {book_id: isbn13}. Returns {} if missing/None.
    """
    if not book_show_dir or not book_show_dir.exists():
        return {}
    out: dict[int, str] = {}
    for path in sorted(book_show_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = parse_book_show_next_data(record.get("next_data_json") or "")
            if parsed and parsed.get("book_id") and parsed.get("isbn13"):
                out[int(parsed["book_id"])] = str(parsed["isbn13"])
    return out


# --- book_show_api (optional ISBN merge, direct fields) ----------------------


def load_book_show_api_isbns(path: Path | None) -> dict[int, str]:
    """Optional merge: a single batch JSONL file (from running the
    `book_show_api` profile — see sites/_templates/goodreads/profiles/
    book_show_api.yaml and tools/catalog/extract_remaining_ids.py) ->
    {legacy_id: isbn13}. Returns {} if missing/None.

    Unlike `load_book_show_isbns` (one `.jsonl` per book, `next_data_json`
    blob that needs Apollo-cache parsing), `book_show_api` writes one JSONL
    with `legacy_id` and `isbn13` as direct fields already extracted via
    JMESPath at scrape time — no blob parsing needed here.
    """
    if not path or not path.exists():
        return {}
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        book_id = record.get("legacy_id")
        isbn13 = record.get("isbn13")
        if book_id is not None and isbn13:
            try:
                out[int(book_id)] = str(isbn13)
            except (TypeError, ValueError):
                continue
    return out


def load_book_show_api_books(path: Path | None) -> dict[int, GoodreadsBook]:
    """Full `GoodreadsBook` rows straight from a `book_show_api` batch JSONL --
    for the population no list_show scrape ever surfaces at all: a book_id
    can be a real, popular Goodreads book we already have `book_show_api`
    signal for, yet never have a `list_show` row for at all if it never
    appeared on one of the 21 seeded Listopia lists (which were chosen to
    get broad coverage, not because absence from them means anything about
    popularity). Measured against a real scrape: ~12k such `book_id`s,
    ~10.6k of them recoverable this way (see docs/BOOK_CATALOG.md's
    coverage-gap note, including a 2.9M-rated book that was previously
    invisible to scoring/matching entirely).

    Only records with `legacy_id` + `isbn13` + `title` + `author` all
    present are returned. `isbn13` presence specifically matters here: it
    means these resolve via the existing ISBN overlay (`_attach_isbns` /
    `match_book`) with no new bibliographic-matching risk -- the point is
    coverage, not exercising the fuzzy matcher on an untested population.
    Records missing any of those fields (mostly `_scrape_warning:
    incomplete_record` rows, which never carry a title at all -- see
    `consolidate_popularity_signals.py`) are skipped; recovering those is a
    separate, harder, smaller problem, not handled here.

    `isbn13` is deliberately left unset on the returned rows: the existing
    `_attach_isbns` pass (already reading this same file) sets it for every
    book, old or new, so there's exactly one code path for that. `genres`
    is deliberately left empty, not populated from this record's own
    `genres` field -- this module's output `genres` field means list_show
    seed-list tags (see module docstring), not Goodreads' own per-book
    `bookGenres`, and a book with zero list appearances legitimately has
    zero list-derived tags.
    """
    out: dict[int, GoodreadsBook] = {}
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        book_id = record.get("legacy_id")
        isbn13 = record.get("isbn13")
        title = record.get("title")
        author = record.get("author")
        if book_id is None or not isbn13 or not title or not author:
            continue
        try:
            book_id = int(book_id)
        except (TypeError, ValueError):
            continue
        avg_rating = record.get("average_rating")
        ratings_count = record.get("ratings_count")
        out[book_id] = GoodreadsBook(
            book_id=book_id,
            title=str(title).strip(),
            author=str(author).strip(),
            avg_rating=float(avg_rating) if isinstance(avg_rating, (int, float)) else None,
            ratings_count=int(ratings_count) if isinstance(ratings_count, (int, float)) else None,
            list_appearances=0,
        )
    return out


# --- OL candidate index -----------------------------------------------------


def load_ol_candidates(conn: sqlite3.Connection) -> list[OLCandidate]:
    """Reads Sources/SpineCatalog/BookCatalog.swift's `books` table.

    Fixture-sized catalogs only. Live ``full.sqlite`` must use
    ``load_ol_candidates_for_titles`` — a full-table fetch is tens of GB.
    """
    cur = conn.execute(_OL_SELECT)
    return candidates_from_ol_rows(cur.fetchall())


def load_ol_candidates_for_titles(conn: sqlite3.Connection, titles: Iterable[str]) -> list[OLCandidate]:
    """OL works whose ``titleNormalized`` is one of the GR title lookup keys.

    One batched ``IN`` scan instead of a query per Goodreads book.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for title in titles:
        for key in title_lookup_keys(title):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    rows: list[tuple] = []
    for offset in range(0, len(keys), 400):
        chunk = keys[offset : offset + 400]
        placeholders = ",".join("?" * len(chunk))
        rows.extend(conn.execute(f"{_OL_SELECT} WHERE titleNormalized IN ({placeholders})", chunk).fetchall())
    return candidates_from_ol_rows(rows)


def load_ol_isbn_index(conn: sqlite3.Connection, isbn13s: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Reads the `book_isbns` many-to-many index (an isbn13 can map to more
    than one workKey in rare OL data-quality cases; see BookCatalog.lookupISBN).

    Pass ``isbn13s`` to load only those keys (the live GR harvest is ~50k;
    the full table is ~31M and must not be pulled into Python).
    """
    index: dict[str, list[str]] = defaultdict(list)
    wanted = [i for i in (isbn13s or ()) if i]
    if isbn13s is not None:
        for offset in range(0, len(wanted), 400):
            chunk = wanted[offset : offset + 400]
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"SELECT isbn13, workKey FROM book_isbns WHERE isbn13 IN ({placeholders})",
                chunk,
            )
            for isbn13, work_key in cur:
                index[isbn13].append(work_key)
        return dict(index)
    cur = conn.execute("SELECT isbn13, workKey FROM book_isbns")
    for isbn13, work_key in cur.fetchall():
        index[isbn13].append(work_key)
    return dict(index)


_OL_SELECT = "SELECT workKey, title, author, titleNormalized, authorNormalized, editionCount FROM books"


# Kept as the public name tests / older call sites used. Implementation is
# the title+author Fellegi–Sunter block, not the old full-author compact key.
AuthorBlockIndex = TitleAuthorBlockIndex


# --- Matching ----------------------------------------------------------------


@dataclass
class MatchResult:
    book_id: int
    method: str  # "isbn" | "title_author" | "ambiguous" | "unmatched"
    work_key: str | None = None
    score: float | None = None
    margin: float | None = None


def _score_candidate(title_norm: str, author_norm: str, candidate: OLCandidate) -> float:
    """Legacy OCR-style blend. Kept for scene-popularity / oracle helpers
    that compare mashed strings; the GR↔OL catalog join no longer uses it."""
    title_score = fuzz.token_set_ratio(title_norm, candidate.title_normalized)
    author_score = fuzz.token_set_ratio(author_norm, candidate.author_normalized)
    return 0.6 * title_score + 0.4 * author_score


def match_book(
    book: GoodreadsBook,
    author_index: TitleAuthorBlockIndex,
    isbn_index: dict[str, list[str]],
    *,
    use_isbn: bool = True,
) -> MatchResult:
    """ISBN overlay, then title+author. Pass ``use_isbn=False`` to test the
    bibliographic matcher in isolation (the expert holdout protocol)."""
    if use_isbn and book.isbn13:
        work_keys = isbn_index.get(book.isbn13)
        if work_keys:
            # >1 distinct work for one ISBN is a rare OL data-quality case
            # (see BookCatalog.lookupISBN's doc comment) — take the first
            # deterministically rather than guessing further.
            return MatchResult(book_id=book.book_id, method="isbn", work_key=sorted(work_keys)[0], score=100.0)

    ta = match_title_author(book.title, book.author, author_index)
    return MatchResult(
        book_id=book.book_id,
        method=ta.method,
        work_key=ta.work_key,
        score=ta.score,
        margin=ta.margin,
    )


# --- Shelf score -------------------------------------------------------------


@dataclass(frozen=True)
class ShelfScoreWeights:
    """Weights for `compute_shelf_score`'s four additive terms. Must sum to
    1.0 (checked in `__post_init__`) so the score stays comparable across
    weightings when swept -- see `docs/BOOK_CATALOG.md`'s "Rebalancing
    compute_shelf_score" section for the empirical case behind the default.
    """

    rating: float
    ratings_count: float
    list_: float
    edition: float

    def __post_init__(self) -> None:
        total = self.rating + self.ratings_count + self.list_ + self.edition
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ShelfScoreWeights must sum to 1.0, got {total}")


# 2b-1 rebalance (2026-09-03): `avg_rating` demoted from 0.35 to 0.05 (was the
# *largest* weight despite correlating -0.155 with true rating volume across
# 48,647 books measured against consolidated_signals -- mass-market
# bestsellers draw more mixed reviews than niche books rated only by fans, so
# weighting "liked by whoever rated it" above "how many people encountered
# it" actively worked against the ranking goal). `ratings_count` promoted
# 0.30 -> 0.60 to absorb that weight, since it's the strongest true signal.
# `list` and `edition` are unchanged at 0.20/0.15 -- no evidence either was
# broken; changing them without measurement is exactly the kind of untested
# reweighting this stage is trying to avoid.
DEFAULT_SHELF_SCORE_WEIGHTS = ShelfScoreWeights(rating=0.05, ratings_count=0.60, list_=0.20, edition=0.15)

# Number of seed lists `goodreads_seed_lists.yaml` defined the last time this
# default was checked (2026-09-03). `run()` always passes the *live* count
# (`len(seed_meta)`) instead of this constant -- it exists only for
# standalone/test callers, so it can go stale without silently corrupting a
# real run the way the previous hardcoded "11" did (see docs/BOOK_CATALOG.md).
_DEFAULT_TOTAL_SEED_LISTS = 21


def compute_shelf_score(
    book: GoodreadsBook,
    edition_count: int,
    signal: "ConsolidatedSignal | None" = None,
    *,
    total_seed_lists: int = _DEFAULT_TOTAL_SEED_LISTS,
    weights: ShelfScoreWeights = DEFAULT_SHELF_SCORE_WEIGHTS,
) -> float:
    """Composite 'likely on a bookcase today' signal: Goodreads rating
    + rating volume, how many of our seed lists surfaced the book, and OL
    edition count (a rough availability/recency proxy in the absence of
    real publication-date data at this stage — see module docstring).
    Typically 0..~1, but the `ratings_count` term is deliberately uncapped
    (see below) so a handful of mega-bestsellers can exceed 1.0 rather than
    tying with every other book past an arbitrary ceiling.

    `signal` is an optional
    tools.catalog.consolidate_popularity_signals.ConsolidatedSignal for this
    book_id. It only fills in `avg_rating`/`ratings_count` when the
    list_show-derived `book` fields are missing (a per-row rating-text parse
    gap) — it never overrides a real list_show value, so scores for books
    that already have a list signal are unchanged by passing `signal`.

    `total_seed_lists` should be `len(seed_meta)` from the caller's real
    `load_seed_metadata()` result -- the actual number of Listopia lists a
    book could have appeared on. Callers that hardcode a stale constant
    here silently under-normalize `list_term` (this happened once already:
    normalized against 11 while 21 lists were live, so only 24 of 50,070
    scored rows -- exactly the most cross-list-popular books -- ever hit
    the ceiling, concentrating mis-ranking at the very top of the catalog).

    `weights` defaults to `DEFAULT_SHELF_SCORE_WEIGHTS`; see that constant's
    comment for the empirical case behind the current split, and pass a
    different `ShelfScoreWeights` to sweep alternatives (`fold_for`-gated
    tuning-fold-only, see docs/BOOK_CATALOG.md).
    """
    avg_rating = book.avg_rating
    ratings_count = book.ratings_count
    if signal is not None:
        if avg_rating is None:
            avg_rating = signal.api_avg_rating
        if ratings_count is None:
            ratings_count = signal.api_ratings_count

    rating_term = (avg_rating or 0) / 5.0
    # Deliberately uncapped: a 1.0 ceiling here made every book past ~1M
    # ratings indistinguishable on this term, which is exactly the top of
    # the catalog a 50k-capped selection cares most about getting right.
    ratings_count_term = math.log10((ratings_count or 0) + 1) / 6.0  # ~1.0 at ~1M ratings, grows slowly past it
    # max(..., 1): a caller with zero known seed lists (empty/missing
    # goodreads_seed_lists.yaml) still gets a defined, non-crashing result
    # rather than a log10(1)=0 divide-by-zero.
    list_term = math.log10(book.list_appearances + 1) / math.log10(max(total_seed_lists, 1) + 1)
    edition_term = math.log10(edition_count + 1) / 4.0  # ~0..1 up to ~10k editions

    return (
        weights.rating * min(rating_term, 1.0)
        + weights.ratings_count * ratings_count_term
        + weights.list_ * min(list_term, 1.0)
        + weights.edition * min(edition_term, 1.0)
    )


# --- Orchestration -----------------------------------------------------------


def _attach_isbns(
    books: dict[int, GoodreadsBook],
    *,
    book_show_dir: Path | None,
    book_show_api_path: Path | None,
) -> None:
    # book_show_api (direct fields, no blob parsing) takes precedence over
    # the older book_show (Apollo-cache blob) merge when both are supplied.
    for book_id, isbn13 in load_book_show_isbns(book_show_dir).items():
        if book_id in books:
            books[book_id].isbn13 = isbn13
    for book_id, isbn13 in load_book_show_api_isbns(book_show_api_path).items():
        if book_id in books:
            books[book_id].isbn13 = isbn13


def run(
    raw_dir: Path,
    ol_db_path: Path,
    out_path: Path,
    *,
    book_show_dir: Path | None = None,
    book_show_api_path: Path | None = None,
    seed_lists_path: Path = SEED_LISTS_PATH,
    genre_tags_out_path: Path | None = None,
    use_isbn: bool = True,
    consolidated_signals_path: Path | None = None,
) -> dict[str, int]:
    validate_list_show_schema(raw_dir)

    seed_meta = load_seed_metadata(seed_lists_path)
    seed_genres = {list_id: m.genre for list_id, m in seed_meta.items()}
    books = load_goodreads_books(raw_dir, seed_genres)

    # book_show_api-only books (never on any seeded list) -- see
    # load_book_show_api_books's docstring. Added regardless of `use_isbn`
    # for the same reason --skip-isbn already applies uniformly to every
    # book's harvested ISBN, not selectively by source: `use_isbn` alone
    # decides whether match_book actually uses each book's isbn13 below.
    book_show_api_only_count = 0
    for book_id, book in load_book_show_api_books(book_show_api_path).items():
        if book_id not in books:
            books[book_id] = book
            book_show_api_only_count += 1
    if book_show_api_only_count:
        print(
            f"[match_goodreads] added {book_show_api_only_count} book_show_api-only books "
            "(never on a seeded list_show list)",
            flush=True,
        )

    # Local import (not at module top level): see the TYPE_CHECKING import
    # near the top of this file for why a real top-level import here would
    # be circular. By call time this module has fully finished loading, so
    # the cycle doesn't apply.
    signal_by_book_id: dict[int, "ConsolidatedSignal"] = {}
    if consolidated_signals_path is not None:
        from tools.catalog.consolidate_popularity_signals import load_consolidated_signals

        signal_by_book_id = load_consolidated_signals(consolidated_signals_path)
        print(f"[match_goodreads] loaded {len(signal_by_book_id)} consolidated signal rows", flush=True)

    genre_tags_out_path = genre_tags_out_path or out_path.parent / "genre_tags.json"
    genre_tags_out_path.parent.mkdir(parents=True, exist_ok=True)
    genre_tags_out_path.write_text(
        json.dumps(build_genre_tag_mapping(seed_meta.values()), indent=2, sort_keys=True), encoding="utf-8"
    )

    if use_isbn:
        _attach_isbns(books, book_show_dir=book_show_dir, book_show_api_path=book_show_api_path)

    conn = sqlite3.connect(f"file:{ol_db_path}?mode=ro", uri=True)
    try:
        isbn_index = (
            load_ol_isbn_index(conn, (b.isbn13 for b in books.values() if b.isbn13)) if use_isbn else {}
        )
        print(f"[match_goodreads] loaded {len(isbn_index)} ISBN→work hits", flush=True)
        candidates = load_ol_candidates_for_titles(conn, (b.title for b in books.values()))
        print(f"[match_goodreads] loaded {len(candidates)} OL title-probe hits", flush=True)
        edition_count_by_work = {c.work_key: c.edition_count for c in candidates}
        if use_isbn:
            missing = {wk for keys in isbn_index.values() for wk in keys if wk not in edition_count_by_work}
            for wk in missing:
                row = conn.execute("SELECT editionCount FROM books WHERE workKey = ? LIMIT 1", (wk,)).fetchone()
                edition_count_by_work[wk] = (row[0] or 0) if row else 0
        author_index = TitleAuthorBlockIndex(candidates)
        print(
            f"[match_goodreads] {len(books)} Goodreads books "
            f"(isbn_overlay={'on' if use_isbn else 'off'})",
            flush=True,
        )
    finally:
        conn.close()

    total_seed_lists = len(seed_meta)
    counts: dict[str, int] = defaultdict(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for i, book in enumerate(books.values(), start=1):
            result = match_book(book, author_index, isbn_index, use_isbn=use_isbn)
            if i % 5000 == 0 or i == 1:
                print(f"[match_goodreads] scored {i}/{len(books)}", flush=True)
            counts[result.method] += 1
            edition_count = edition_count_by_work.get(result.work_key, 0) if result.work_key else 0
            out_row = {
                "goodreads_book_id": book.book_id,
                "title": book.title,
                "author": book.author,
                "avg_rating": book.avg_rating,
                "ratings_count": book.ratings_count,
                "list_appearances": book.list_appearances,
                "list_score_sum": book.list_score_sum,
                "vote_sum": book.vote_sum,
                "genres": sorted(book.genres),
                "match_method": result.method,
                "work_key": result.work_key,
                "match_score": result.score,
                "match_margin": result.margin,
                "isbn13": book.isbn13,
                # Persisted so a shelf-score reweight sweep (see
                # docs/BOOK_CATALOG.md's "Rebalancing compute_shelf_score")
                # can recompute shelf_score straight from this JSONL, with
                # no OL db / re-matching needed for weight-only changes.
                "edition_count": edition_count,
                # Computed even when unmatched (edition_count=0 in that
                # case) — build_ios_en_from_goodreads.py's gap-fill step
                # needs a real shelf_score for exactly the popular-but-
                # unmatched books it's designed to pick up.
                "shelf_score": compute_shelf_score(
                    book, edition_count, signal_by_book_id.get(book.book_id), total_seed_lists=total_seed_lists
                ),
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    result = dict(counts)
    result["book_show_api_only"] = book_show_api_only_count
    return result


# --- Persisted eval reports (Phase 2a) --------------------------------------


def git_head_sha(repo: Path = _REPO) -> str:
    """Short (8-char) git SHA of HEAD, or "unknown" if git/the repo isn't
    available. Never raises -- a baseline eval run shouldn't fail over
    metadata it can degrade gracefully without."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8] or "unknown"
    except Exception:
        return "unknown"


def matcher_eval_default_path(base_dir: Path, sha: str, timestamp: datetime) -> Path:
    """``catalog_goodreads('matcher_eval/<UTC timestamp>_<short sha>.json')`` --
    the naming convention every persisted eval report should use, so runs
    sort chronologically on disk and are traceable to the commit that
    produced them."""
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return base_dir / "matcher_eval" / f"{stamp}_{sha}.json"


def write_eval_report(path: Path, report: dict[str, object], *, sha: str, timestamp: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"git_sha": sha, "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), **report}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_isbn_holdout_eval(
    raw_dir: Path,
    ol_db_path: Path,
    *,
    book_show_dir: Path | None = None,
    book_show_api_path: Path | None = None,
    seed_lists_path: Path = SEED_LISTS_PATH,
    fold: str = "all",
    fold_salt: str = "gr-ol-v1",
    adversarial_pairs_path: Path | None = None,
) -> dict[str, object]:
    """Score title+author against harvested ISBNs that hit ``book_isbns``.

    The bibliographic matcher never sees the ISBN. Gold is the OL work the
    ISBN already joins to. This is the holdout protocol bibliographic
    record-linkage papers use when a high-confidence identifier is
    available on a subset of pairs.

    ``fold`` restricts the gold set to ``bibliographic_join.fold_for``'s
    "tuning" or "validation" split ("all" — the default — uses every gold
    pair, matching this function's behavior before fold support existed).
    See that function's docstring for why the tuning fold is safe to
    iterate against repeatedly and the validation fold is not.

    ``adversarial_pairs_path`` (if given) folds
    ``bibliographic_join.evaluate_adversarial_pairs``'s ``false_merges``/
    ``total`` into this same report, scored against the same OL candidate
    index used for the ISBN-holdout — a *negative*-set precision check the
    positive-only ISBN-gold set can't provide on its own.
    """
    validate_list_show_schema(raw_dir)
    seed_meta = load_seed_metadata(seed_lists_path)
    seed_genres = {list_id: m.genre for list_id, m in seed_meta.items()}
    books = load_goodreads_books(raw_dir, seed_genres)
    _attach_isbns(books, book_show_dir=book_show_dir, book_show_api_path=book_show_api_path)

    conn = sqlite3.connect(f"file:{ol_db_path}?mode=ro", uri=True)
    try:
        isbn_index = load_ol_isbn_index(conn, (b.isbn13 for b in books.values() if b.isbn13))
        candidates = load_ol_candidates_for_titles(conn, (b.title for b in books.values()))
        work_title_author = {c.work_key: (c.title, c.author) for c in candidates}
        missing = {wk for keys in isbn_index.values() for wk in keys if wk not in work_title_author}
        for wk in missing:
            row = conn.execute("SELECT title, author FROM books WHERE workKey = ? LIMIT 1", (wk,)).fetchone()
            if row:
                work_title_author[wk] = (row[0], row[1])
    finally:
        conn.close()

    index = TitleAuthorBlockIndex(candidates)
    gold_books = [
        (b.book_id, b.title, b.author, b.isbn13)
        for b in books.values()
        if b.isbn13 and b.isbn13 in isbn_index
    ]
    if fold != "all":
        gold_books = [gb for gb in gold_books if fold_for(gb[0], fold_salt) == fold]
    report = evaluate_title_author_against_isbn(
        gold_books, index, isbn_index, work_title_author=work_title_author
    )
    report.pop("rows", None)
    report["fold"] = fold

    if adversarial_pairs_path is not None:
        pairs = load_adversarial_pairs(adversarial_pairs_path)
        adversarial_report = evaluate_adversarial_pairs(pairs, index)
        report["false_merges"] = adversarial_report["false_merges"]
        report["adversarial_total"] = adversarial_report["total"]
        report["adversarial_false_merge_pairs"] = adversarial_report["false_merge_pairs"]

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ol-db", type=Path, required=True, help="Path to full.sqlite (read-only)")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument("--book-show-dir", type=Path, default=None, help="Optional supplementary book_show scrape output")
    parser.add_argument(
        "--book-show-api",
        type=Path,
        default=None,
        help=(
            "Optional supplementary book_show_api batch JSONL (legacy_id + isbn13 direct "
            "fields; see tools/catalog/extract_remaining_ids.py). Takes precedence over "
            "--book-show-dir for any book_id present in both."
        ),
    )
    parser.add_argument("--seed-lists", type=Path, default=SEED_LISTS_PATH)
    parser.add_argument(
        "--consolidated-signals",
        type=Path,
        default=None,
        help=(
            "Optional tools/catalog/consolidate_popularity_signals.py output "
            "(catalog_goodreads('consolidated_signals.jsonl.gz')). Fills gaps in "
            "avg_rating/ratings_count for shelf_score only; never overrides a real "
            "list_show value. Off by default -- output is unchanged unless passed."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Default: catalog_goodreads('matched_goodreads.jsonl.gz')")
    parser.add_argument(
        "--genre-tags-out", type=Path, default=None, help="Default: catalog_goodreads('genre_tags.json')"
    )
    parser.add_argument(
        "--skip-isbn",
        action="store_true",
        help="Bibliographic join only (ignore harvested ISBNs). Use to test title+author in isolation.",
    )
    parser.add_argument(
        "--eval-isbn-holdout",
        action="store_true",
        help=(
            "Score title+author against harvested ISBNs that hit OL book_isbns; "
            "print recall/conflict. Does not write matched_goodreads.jsonl.gz."
        ),
    )
    parser.add_argument(
        "--fold",
        choices=("all", "tuning", "validation"),
        default="all",
        help=(
            "With --eval-isbn-holdout: restrict to bibliographic_join.fold_for's "
            "tuning or validation split (default: all gold pairs -- iterate against "
            "'tuning', touch 'validation' only at the start/end of a tuning run)."
        ),
    )
    parser.add_argument(
        "--adversarial-pairs",
        type=Path,
        default=DEFAULT_ADVERSARIAL_PAIRS_PATH,
        help=(
            "With --eval-isbn-holdout: negative-set YAML (must_not_share_work_key "
            "pairs) folded into the report as false_merges/adversarial_total. "
            f"Default: {DEFAULT_ADVERSARIAL_PAIRS_PATH.name}. Pass a nonexistent "
            "path to skip."
        ),
    )
    parser.add_argument(
        "--eval-out",
        type=str,
        default=None,
        help=(
            "With --eval-isbn-holdout: persist the report as JSON (git sha + UTC "
            "timestamp included). Pass 'auto' for the default "
            "catalog_goodreads('matcher_eval/<timestamp>_<short_sha>.json') naming, "
            "or an explicit path. Omitted by default -- print-only, no disk writes."
        ),
    )
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or catalog_goodreads("raw")
    out_path = args.out or catalog_goodreads("matched_goodreads.jsonl.gz")
    genre_tags_out_path = args.genre_tags_out or catalog_goodreads("genre_tags.json")

    if args.eval_isbn_holdout:
        report = run_isbn_holdout_eval(
            raw_dir,
            args.ol_db,
            book_show_dir=args.book_show_dir,
            book_show_api_path=args.book_show_api,
            seed_lists_path=args.seed_lists,
            fold=args.fold,
            adversarial_pairs_path=args.adversarial_pairs,
        )
        print("[match_goodreads] ISBN-holdout eval:", json.dumps(report, indent=2, sort_keys=True))
        if args.eval_out:
            sha = git_head_sha()
            timestamp = datetime.now(timezone.utc)
            eval_out_path = (
                matcher_eval_default_path(catalog_goodreads(), sha, timestamp)
                if args.eval_out == "auto"
                else Path(args.eval_out)
            )
            write_eval_report(eval_out_path, report, sha=sha, timestamp=timestamp)
            print(f"[match_goodreads] wrote {eval_out_path}")
        return 0

    counts = run(
        raw_dir,
        args.ol_db,
        out_path,
        book_show_dir=args.book_show_dir,
        book_show_api_path=args.book_show_api,
        seed_lists_path=args.seed_lists,
        genre_tags_out_path=genre_tags_out_path,
        use_isbn=not args.skip_isbn,
        consolidated_signals_path=args.consolidated_signals,
    )
    print("[match_goodreads] match method counts:", json.dumps(counts, indent=2, sort_keys=True))
    print(f"[match_goodreads] wrote {out_path}")
    print(f"[match_goodreads] wrote {genre_tags_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
