#!/usr/bin/env python3
"""Shared `list_show` popularity-signal parsing.

Extracted so two consumers stay on the exact same parsing logic instead of
independently duplicating (and potentially drifting on) it:

- `tools/catalog/analyze_goodreads_lists.py` -- list overlap / top-N
  popularity coverage analysis.
- `tools/catalog/extract_remaining_ids.py` -- orders `ids_remaining.txt` so
  the `book_show_api` scrape fetches popular books first.

Deliberately NOT importing from `tools.catalog.match_goodreads`: that module
hard-requires `rapidfuzz` for its OL fuzzy-matching stage, which parsing a
`list_show` row for popularity signals has no use for. The regexes below
mirror `match_goodreads.py`'s `_RATING_TEXT_RE`/`_SCORE_TEXT_RE`/
`_VOTE_TEXT_RE`/`_BOOK_URL_RE` -- keep them in sync if that profile's markup
selectors ever change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_BOOK_URL_RE = re.compile(r"^/book/show/(\d+)")
_RATING_TEXT_RE = re.compile(r"([\d.]+)\s+avg rating\s*[-\u2013\u2014]+\s*([\d,]+)\s+ratings?")
_SCORE_TEXT_RE = re.compile(r"score:\s*([\d,]+)")
_VOTE_TEXT_RE = re.compile(r"([\d,]+)\s+people voted")
_LIST_SHOW_FIELDS = ("book_urls", "rating_texts", "score_texts", "vote_texts")


def _parse_book_id(book_url: str | None) -> int | None:
    if not book_url:
        return None
    m = _BOOK_URL_RE.match(book_url)
    return int(m.group(1)) if m else None


def _parse_ratings_count(rating_text: str | None) -> int:
    """'4.55 avg rating — 745,415 ratings' -> 745415 (0 if unparseable)."""
    if not rating_text:
        return 0
    m = _RATING_TEXT_RE.search(rating_text)
    return int(m.group(2).replace(",", "")) if m else 0


def _parse_int_with_commas(text: str | None, pattern: re.Pattern[str]) -> int:
    if not text:
        return 0
    m = pattern.search(text)
    return int(m.group(1).replace(",", "")) if m else 0


@dataclass(frozen=True)
class BookPopularity:
    """Aggregated cross-list popularity signals for one Goodreads book_id.

    `ratings_count` is the max seen across lists -- a book's Goodreads
    rating count is a property of the book, not the list, so a stale/partial
    row on one list shouldn't shadow a better reading from another. The rest
    sum across every list row the book appeared in. See
    `popularity_sort_key` for how these combine into a scrape-queue order.
    """

    ratings_count: int = 0
    list_appearances: int = 0
    list_score_sum: int = 0
    list_vote_sum: int = 0


def iter_list_show_books(path: Path) -> Iterator[tuple[int, int, int, int]]:
    """Yields `(book_id, ratings_count, list_score, list_vote)` for every
    book row in a `list_show` raw JSONL file, mirroring
    `match_goodreads._zip_list_show_record`'s parallel-array zip but only
    extracting the fields a popularity signal needs."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lengths = {k: len(record.get(k) or []) for k in _LIST_SHOW_FIELDS}
            n = min(lengths.values()) if lengths else 0
            book_urls = record.get("book_urls") or []
            rating_texts = record.get("rating_texts") or []
            score_texts = record.get("score_texts") or []
            vote_texts = record.get("vote_texts") or []
            for i in range(n):
                book_id = _parse_book_id(book_urls[i] if i < len(book_urls) else None)
                if book_id is None:
                    continue
                ratings_count = _parse_ratings_count(rating_texts[i] if i < len(rating_texts) else None)
                list_score = _parse_int_with_commas(score_texts[i] if i < len(score_texts) else None, _SCORE_TEXT_RE)
                list_vote = _parse_int_with_commas(vote_texts[i] if i < len(vote_texts) else None, _VOTE_TEXT_RE)
                yield book_id, ratings_count, list_score, list_vote


def aggregate_popularity(raw_dir: Path) -> dict[int, BookPopularity]:
    """Scans every `<list_id>.jsonl` under `raw_dir` and aggregates
    per-book_id popularity across all of them (see `BookPopularity` for the
    per-field aggregation rule)."""
    ratings_count: dict[int, int] = {}
    list_appearances: dict[int, int] = {}
    list_score_sum: dict[int, int] = {}
    list_vote_sum: dict[int, int] = {}

    if not raw_dir.exists():
        return {}

    for path in sorted(raw_dir.glob("*.jsonl")):
        for book_id, rc, score, vote in iter_list_show_books(path):
            ratings_count[book_id] = max(ratings_count.get(book_id, 0), rc)
            list_appearances[book_id] = list_appearances.get(book_id, 0) + 1
            list_score_sum[book_id] = list_score_sum.get(book_id, 0) + score
            list_vote_sum[book_id] = list_vote_sum.get(book_id, 0) + vote

    return {
        book_id: BookPopularity(
            ratings_count=ratings_count[book_id],
            list_appearances=list_appearances[book_id],
            list_score_sum=list_score_sum[book_id],
            list_vote_sum=list_vote_sum[book_id],
        )
        for book_id in ratings_count
    }


def popularity_sort_key(book_id: int, pop: BookPopularity) -> tuple[int, int, int, int]:
    """Descending `ratings_count`, then `list_appearances`, then
    `list_score_sum`, with `book_id` as a final deterministic (ascending)
    tie-break. `list_vote_sum` is tracked on `BookPopularity` but not part of
    this key -- `list_score_sum` already captures Listopia list-vote mass."""
    return (-pop.ratings_count, -pop.list_appearances, -pop.list_score_sum, book_id)
