#!/usr/bin/env python3
"""Consolidate every Goodreads popularity/genre signal already on disk into
one per-`book_id` table.

Three sources, all already scraped under `tools.paths.catalog_goodreads()`,
merged on the Goodreads numeric `book_id`:

1. `raw/<list_id>.jsonl` (`list_show`) via
   `tools.catalog.list_show_popularity.aggregate_popularity`:
   `ratings_count`, `list_appearances`, `list_score_sum`, `list_vote_sum`.

2. `book_show_api*.jsonl` success records (`legacy_id` present): `isbn13`,
   `average_rating`, `ratings_count`, `text_reviews_count`, `genres` --
   direct JMESPath fields already flat in the file, no blob parsing (see
   `tools.catalog.match_goodreads.load_book_show_api_isbns`).

3. `book_show_api*.jsonl` `_scrape_warning: incomplete_record` rows: the
   edition-level `Book:` Apollo object (which carries `isbn13`/`legacy_id`/
   `title`) never made it into the cache for these, but the `Work:`/
   `Contributor:` objects (`genres`/ratings/`author`) usually did -- see
   `sites/goodreads/profiles/book_show_api.yaml`'s per-field JMESPath
   filters. `book_id` is recovered from `_url` via
   `tools.catalog.extract_remaining_ids.parse_book_id_from_warning_url`,
   the only way to attribute this signal since `legacy_id` is null here.
   `title` is never present on these rows -- confirmed empirically against
   the live scrape (0/12,112 `incomplete_record` rows had `title` set) --
   which is why nothing here ever stores a title for this source.

A book_show_api *success* for a given `book_id` always wins over a leftover
`incomplete_record` row for that same `book_id` (e.g. an early failed
attempt followed by a later success across retries/shards) -- see
`collect_book_show_api_signals`.

Usage:
    python tools/catalog/consolidate_popularity_signals.py \\
        --raw-dir <catalog_goodreads('raw')> \\
        --out <catalog_goodreads('consolidated_signals.jsonl.gz')>
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.extract_remaining_ids import (  # noqa: E402
    default_book_show_api_paths,
    parse_book_id_from_warning_url,
)
from tools.catalog.list_show_popularity import aggregate_popularity  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_OUT_NAME = "consolidated_signals.jsonl.gz"


@dataclass(frozen=True)
class ConsolidatedSignal:
    book_id: int
    list_ratings_count: int = 0
    list_appearances: int = 0
    list_score_sum: int = 0
    list_vote_sum: int = 0
    api_isbn13: str | None = None
    api_avg_rating: float | None = None
    api_ratings_count: int | None = None
    api_text_reviews_count: int | None = None
    genres: tuple[str, ...] = ()
    has_list_signal: bool = False
    has_api_isbn: bool = False
    has_api_partial: bool = False  # incomplete_record recovery


@dataclass
class _ApiFields:
    """Intermediate, book_show_api-only fields before the list_show merge."""

    isbn13: str | None = None
    avg_rating: float | None = None
    ratings_count: int | None = None
    text_reviews_count: int | None = None
    genres: tuple[str, ...] = field(default_factory=tuple)
    has_isbn: bool = False
    has_partial: bool = False


def _iter_book_show_api_records(paths: Iterable[Path]) -> Iterator[dict]:
    """Yields every JSON record across every path, oldest-mtime first --
    same ordering convention as `tools.catalog.merge_book_show_api.merge_records`,
    so that if the same `book_id` has more than one `incomplete_record`
    attempt (never a success), the most recent attempt's partial fields win.
    A real success always wins regardless of ordering -- see
    `collect_book_show_api_signals`.
    """
    existing = [p for p in paths if p.exists()]
    for path in sorted(existing, key=lambda p: p.stat().st_mtime):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _api_fields_from_success(record: dict) -> _ApiFields:
    genres = tuple(g for g in (record.get("genres") or []) if g)
    return _ApiFields(
        isbn13=record.get("isbn13"),
        avg_rating=record.get("average_rating"),
        ratings_count=record.get("ratings_count"),
        text_reviews_count=record.get("text_reviews_count"),
        genres=genres,
        has_isbn=bool(record.get("isbn13")),
        has_partial=False,
    )


def _api_fields_from_incomplete_record(record: dict) -> _ApiFields:
    """`_scrape_warning: incomplete_record` rows never carry `isbn13` (nor
    `legacy_id`/`title`) -- see the module docstring. `has_partial` is only
    set when at least one of the Work-/Contributor-level fields actually
    came through; a warning row that recovered nothing useful should not
    look any different from a book with no book_show_api signal at all.
    """
    genres = tuple(g for g in (record.get("genres") or []) if g)
    avg_rating = record.get("average_rating")
    ratings_count = record.get("ratings_count")
    text_reviews_count = record.get("text_reviews_count")
    has_partial = bool(genres) or avg_rating is not None or ratings_count is not None or text_reviews_count is not None
    return _ApiFields(
        isbn13=None,
        avg_rating=avg_rating,
        ratings_count=ratings_count,
        text_reviews_count=text_reviews_count,
        genres=genres,
        has_isbn=False,
        has_partial=has_partial,
    )


def collect_book_show_api_signals(paths: Iterable[Path]) -> dict[int, _ApiFields]:
    """One `_ApiFields` per Goodreads `book_id`, merged across every shard
    in `paths`. A success (`legacy_id` present) always wins over a leftover
    `incomplete_record` warning for the same `book_id`, regardless of file
    processing order -- a book that ever got its ISBN should never fall
    back to warning-only fields just because a shard re-records an old
    failed attempt.
    """
    by_id: dict[int, _ApiFields] = {}
    for record in _iter_book_show_api_records(paths):
        legacy_id = record.get("legacy_id")
        if legacy_id is not None:
            try:
                book_id = int(legacy_id)
            except (TypeError, ValueError):
                continue
            by_id[book_id] = _api_fields_from_success(record)
            continue
        if record.get("_scrape_warning") != "incomplete_record":
            continue
        book_id = parse_book_id_from_warning_url(record.get("_url"))
        if book_id is None:
            continue
        if by_id.get(book_id, _ApiFields()).has_isbn:
            continue  # a real success for this book_id already won
        by_id[book_id] = _api_fields_from_incomplete_record(record)
    return by_id


def consolidate(raw_dir: Path, book_show_api_paths: Iterable[Path]) -> dict[int, ConsolidatedSignal]:
    """Merge list_show popularity + book_show_api signals, keyed on
    Goodreads `book_id`. The two sources never write the same field
    (`list_*` vs `api_*`), so this is a plain union: a book present in only
    one source keeps the other group's fields at their dataclass defaults.
    """
    list_pop = aggregate_popularity(raw_dir)
    api_fields = collect_book_show_api_signals(book_show_api_paths)

    book_ids = set(list_pop) | set(api_fields)
    out: dict[int, ConsolidatedSignal] = {}
    for book_id in book_ids:
        pop = list_pop.get(book_id)
        api = api_fields.get(book_id)
        out[book_id] = ConsolidatedSignal(
            book_id=book_id,
            list_ratings_count=pop.ratings_count if pop else 0,
            list_appearances=pop.list_appearances if pop else 0,
            list_score_sum=pop.list_score_sum if pop else 0,
            list_vote_sum=pop.list_vote_sum if pop else 0,
            api_isbn13=api.isbn13 if api else None,
            api_avg_rating=api.avg_rating if api else None,
            api_ratings_count=api.ratings_count if api else None,
            api_text_reviews_count=api.text_reviews_count if api else None,
            genres=api.genres if api else (),
            has_list_signal=pop is not None,
            has_api_isbn=bool(api and api.has_isbn),
            has_api_partial=bool(api and api.has_partial and not api.has_isbn),
        )
    return out


def write_consolidated_signals(path: Path, signals: dict[int, ConsolidatedSignal]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for book_id in sorted(signals):
            row = asdict(signals[book_id])
            row["genres"] = list(row["genres"])
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def load_consolidated_signals(path: Path) -> dict[int, ConsolidatedSignal]:
    """Reads back a file written by `write_consolidated_signals`. Returns
    `{}` if `path` doesn't exist, matching this module's other loaders
    (`collect_book_show_api_signals` on a missing shard, `aggregate_popularity`
    on a missing raw dir) rather than raising.
    """
    if not path.exists():
        return {}
    out: dict[int, ConsolidatedSignal] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["genres"] = tuple(row.get("genres") or ())
            out[row["book_id"]] = ConsolidatedSignal(**row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument(
        "--book-show-api",
        type=Path,
        action="append",
        default=None,
        help=(
            "book_show_api JSONL shard (repeatable). Default: every "
            "catalog_goodreads('book_show_api*.jsonl') shard."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help=f"Default: catalog_goodreads('{DEFAULT_OUT_NAME}')")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or catalog_goodreads("raw")
    book_show_api_paths = args.book_show_api or default_book_show_api_paths(catalog_goodreads())
    out_path = args.out or catalog_goodreads(DEFAULT_OUT_NAME)

    signals = consolidate(raw_dir, book_show_api_paths)
    n_written = write_consolidated_signals(out_path, signals)

    n_list = sum(1 for s in signals.values() if s.has_list_signal)
    n_isbn = sum(1 for s in signals.values() if s.has_api_isbn)
    n_partial = sum(1 for s in signals.values() if s.has_api_partial)
    print(
        f"[consolidate_popularity_signals] {n_written:,} books total "
        f"({n_list:,} with list signal, {n_isbn:,} with api isbn, {n_partial:,} with api-partial-only)",
        flush=True,
    )
    print(f"[consolidate_popularity_signals] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
