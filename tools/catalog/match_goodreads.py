#!/usr/bin/env python3
"""Match scraped Goodreads books against the Open Library catalog (`full.sqlite`).

Two stages:

1. Parse + dedupe raw Goodreads scrape output (`tools/scrape_goodreads_lists.py`'s
   JSONL, one file per list_id, written by the `list_show` harness profile's
   parallel-array `extract` records) into one row per distinct Goodreads book_id,
   merging list-appearance/rating signals across every seed list a book showed up on.
2. Match each book against `full.sqlite`'s `books`/`book_isbns` tables: exact ISBN
   match first (only available for books that also got a supplementary `book_show`
   scrape — Listopia list pages don't carry ISBNs), else a title+author fuzzy match
   using the same `token_set_ratio` scorer and 90/8 accept-threshold/margin policy as
   `Sources/SpineMatching/AcceptPolicy.swift`, blocked by a normalized author key so
   this stays fast against a multi-million-row catalog instead of comparing every
   Goodreads book against every OL work.

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
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import normalize_for_search  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install rapidfuzz: pip install rapidfuzz") from exc

SEED_LISTS_PATH = Path(__file__).resolve().parent / "goodreads_seed_lists.yaml"

# Mirrors Sources/SpineMatching/AcceptPolicy.swift's defaults so a Goodreads
# work is only auto-linked to an OL work under the same bar the on-device
# spine matcher uses for OCR reranking.
ACCEPT_THRESHOLD = 90.0
MARGIN_THRESHOLD = 8.0

_BOOK_URL_RE = re.compile(r"^/book/show/(\d+)")
_RATING_TEXT_RE = re.compile(r"([\d.]+)\s+avg rating\s*[-\u2013\u2014]+\s*([\d,]+)\s+ratings?")
_SCORE_TEXT_RE = re.compile(r"score:\s*([\d,]+)")
_VOTE_TEXT_RE = re.compile(r"([\d,]+)\s+people voted")
_SERIES_SUFFIX_RE = re.compile(r"\s*\([^()]*#[^()]*\)\s*$")


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


def strip_series_suffix(title: str) -> str:
    """Listopia titles often carry a trailing series annotation ("The Lord of
    the Rings (The Lord of the Rings, #1-3)") that OL work titles don't —
    only strips a trailing parenthetical if it contains '#' so a legitimate
    subtitle in parens survives untouched."""
    return _SERIES_SUFFIX_RE.sub("", title).strip()


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
        return {
            "book_id": book.get("legacyId"),
            "title": book.get("title"),
            "isbn13": details.get("isbn13"),
            "language": (details.get("language") or {}).get("name"),
        }
    except (KeyError, StopIteration, json.JSONDecodeError, TypeError):
        return None


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


# --- OL candidate index -----------------------------------------------------


@dataclass(frozen=True)
class OLCandidate:
    work_key: str
    title: str
    author: str
    title_normalized: str
    author_normalized: str
    edition_count: int


def load_ol_candidates(conn: sqlite3.Connection) -> list[OLCandidate]:
    """Reads Sources/SpineCatalog/BookCatalog.swift's `books` table."""
    cur = conn.execute("SELECT workKey, title, author, titleNormalized, authorNormalized, editionCount FROM books")
    return [
        OLCandidate(
            work_key=row[0],
            title=row[1],
            author=row[2],
            title_normalized=row[3],
            author_normalized=row[4],
            edition_count=row[5] or 0,
        )
        for row in cur.fetchall()
    ]


def load_ol_isbn_index(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Reads the `book_isbns` many-to-many index (an isbn13 can map to more
    than one workKey in rare OL data-quality cases; see BookCatalog.lookupISBN)."""
    cur = conn.execute("SELECT isbn13, workKey FROM book_isbns")
    index: dict[str, list[str]] = defaultdict(list)
    for isbn13, work_key in cur.fetchall():
        index[isbn13].append(work_key)
    return dict(index)


class AuthorBlockIndex:
    """Groups OL candidates by `author_block_key` so fuzzy scoring only ever
    runs within one author's small candidate set, not the whole multi-
    million-row catalog."""

    def __init__(self, candidates: Iterable[OLCandidate]) -> None:
        self._by_block: dict[str, list[OLCandidate]] = defaultdict(list)
        for c in candidates:
            self._by_block[author_block_key(c.author)].append(c)

    def candidates_for(self, author: str) -> list[OLCandidate]:
        return self._by_block.get(author_block_key(author), [])


# --- Matching ----------------------------------------------------------------


@dataclass
class MatchResult:
    book_id: int
    method: str  # "isbn" | "fuzzy" | "ambiguous" | "unmatched"
    work_key: str | None = None
    score: float | None = None
    margin: float | None = None


def _score_candidate(title_norm: str, author_norm: str, candidate: OLCandidate) -> float:
    title_score = fuzz.token_set_ratio(title_norm, candidate.title_normalized)
    author_score = fuzz.token_set_ratio(author_norm, candidate.author_normalized)
    # Title-weighted, in the spirit of (but not copying the exact OCR-blob
    # weights of) Sources/SpineMatching/FieldAwareScore.swift: title is the
    # more distinctive field for a clean title+author join like this one —
    # there's no OCR blob term here since both sides are cleanly typed text.
    return 0.6 * title_score + 0.4 * author_score


def match_book(book: GoodreadsBook, author_index: AuthorBlockIndex, isbn_index: dict[str, list[str]]) -> MatchResult:
    if book.isbn13:
        work_keys = isbn_index.get(book.isbn13)
        if work_keys:
            # >1 distinct work for one ISBN is a rare OL data-quality case
            # (see BookCatalog.lookupISBN's doc comment) — take the first
            # deterministically rather than guessing further.
            return MatchResult(book_id=book.book_id, method="isbn", work_key=sorted(work_keys)[0], score=100.0)

    title_norm = normalize_for_search(strip_series_suffix(book.title))
    author_norm = normalize_for_search(book.author)
    candidates = author_index.candidates_for(book.author)
    if not candidates:
        return MatchResult(book_id=book.book_id, method="unmatched")

    scored = [(c, _score_candidate(title_norm, author_norm, c)) for c in candidates]

    # Dedupe to the best score per workKey before the margin test — mirrors
    # AcceptPolicy.decideWithMargin's rationale: different editions of the
    # same work must not compete against each other as "the runner-up".
    best_per_work: dict[str, tuple[OLCandidate, float]] = {}
    for c, score in scored:
        existing = best_per_work.get(c.work_key)
        if existing is None or existing[1] < score:
            best_per_work[c.work_key] = (c, score)
    ranked = sorted(best_per_work.values(), key=lambda cs: cs[1], reverse=True)

    top_candidate, top_score = ranked[0]
    margin = (top_score - ranked[1][1]) if len(ranked) > 1 else None
    if top_score >= ACCEPT_THRESHOLD and (margin is None or margin >= MARGIN_THRESHOLD):
        return MatchResult(
            book_id=book.book_id, method="fuzzy", work_key=top_candidate.work_key, score=top_score, margin=margin
        )
    return MatchResult(book_id=book.book_id, method="ambiguous", score=top_score, margin=margin)


# --- Shelf score -------------------------------------------------------------


def compute_shelf_score(book: GoodreadsBook, edition_count: int) -> float:
    """Composite 0..~1 'likely on a bookcase today' signal: Goodreads rating
    + rating volume, how many of our seed lists surfaced the book, and OL
    edition count (a rough availability/recency proxy in the absence of
    real publication-date data at this stage — see module docstring).

    Weights are a reasonable starting point, NOT tuned against a real
    scrape's distribution — build_ios_en_from_goodreads.py is what actually
    reranks `popularityRank` from this; revisit these once real output
    exists to look at.
    """
    rating_term = (book.avg_rating or 0) / 5.0
    ratings_count_term = math.log10((book.ratings_count or 0) + 1) / 6.0  # ~0..1 up to ~1M ratings
    list_term = math.log10(book.list_appearances + 1) / math.log10(11)  # 0..1 across ~10 seed lists
    edition_term = math.log10(edition_count + 1) / 4.0  # ~0..1 up to ~10k editions

    return (
        0.35 * min(rating_term, 1.0)
        + 0.30 * min(ratings_count_term, 1.0)
        + 0.20 * min(list_term, 1.0)
        + 0.15 * min(edition_term, 1.0)
    )


# --- Orchestration -----------------------------------------------------------


def run(
    raw_dir: Path,
    ol_db_path: Path,
    out_path: Path,
    *,
    book_show_dir: Path | None = None,
    seed_lists_path: Path = SEED_LISTS_PATH,
    genre_tags_out_path: Path | None = None,
) -> dict[str, int]:
    seed_meta = load_seed_metadata(seed_lists_path)
    seed_genres = {list_id: m.genre for list_id, m in seed_meta.items()}
    books = load_goodreads_books(raw_dir, seed_genres)

    genre_tags_out_path = genre_tags_out_path or out_path.parent / "genre_tags.json"
    genre_tags_out_path.parent.mkdir(parents=True, exist_ok=True)
    genre_tags_out_path.write_text(
        json.dumps(build_genre_tag_mapping(seed_meta.values()), indent=2, sort_keys=True), encoding="utf-8"
    )

    for book_id, isbn13 in load_book_show_isbns(book_show_dir).items():
        if book_id in books:
            books[book_id].isbn13 = isbn13

    conn = sqlite3.connect(ol_db_path)
    try:
        candidates = load_ol_candidates(conn)
        isbn_index = load_ol_isbn_index(conn)
    finally:
        conn.close()

    author_index = AuthorBlockIndex(candidates)
    edition_count_by_work = {c.work_key: c.edition_count for c in candidates}

    counts: dict[str, int] = defaultdict(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for book in books.values():
            result = match_book(book, author_index, isbn_index)
            counts[result.method] += 1
            edition_count = edition_count_by_work.get(result.work_key, 0) if result.work_key else 0
            row = {
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
                # Computed even when unmatched (edition_count=0 in that
                # case) — build_ios_en_from_goodreads.py's gap-fill step
                # needs a real shelf_score for exactly the popular-but-
                # unmatched books it's designed to pick up.
                "shelf_score": compute_shelf_score(book, edition_count),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ol-db", type=Path, required=True, help="Path to full.sqlite (read-only)")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument("--book-show-dir", type=Path, default=None, help="Optional supplementary book_show scrape output")
    parser.add_argument("--seed-lists", type=Path, default=SEED_LISTS_PATH)
    parser.add_argument("--out", type=Path, default=None, help="Default: catalog_goodreads('matched_goodreads.jsonl.gz')")
    parser.add_argument(
        "--genre-tags-out", type=Path, default=None, help="Default: catalog_goodreads('genre_tags.json')"
    )
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or catalog_goodreads("raw")
    out_path = args.out or catalog_goodreads("matched_goodreads.jsonl.gz")
    genre_tags_out_path = args.genre_tags_out or catalog_goodreads("genre_tags.json")

    counts = run(
        raw_dir,
        args.ol_db,
        out_path,
        book_show_dir=args.book_show_dir,
        seed_lists_path=args.seed_lists,
        genre_tags_out_path=genre_tags_out_path,
    )
    print("[match_goodreads] match method counts:", json.dumps(counts, indent=2, sort_keys=True))
    print(f"[match_goodreads] wrote {out_path}")
    print(f"[match_goodreads] wrote {genre_tags_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
