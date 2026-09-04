#!/usr/bin/env python3
"""Compute which Goodreads book IDs still need a book_show_api fetch.

Bridges `tools/scrape_goodreads_lists.py`'s raw list_show output (JSONL under
`catalog_goodreads("raw")`, one file per list_id) and a batch `book_show_api`
run: writes `ids_remaining.txt`, ready to pass straight to

    scrape-harness scrape goodreads --profile book_show_api \\
        --set-from-file book_id=ids_remaining.txt \\
        --out <catalog_goodreads('book_show_api.jsonl')>

Safe to re-run any number of times, at any point in the pipeline — before all
lists have finished scraping, after adding new lists, or to resume an
interrupted book_show_api run. Each run only asks for whatever book_ids are
in the raw scrape but not yet in book_show_api.jsonl.

Cost note: rather than re-parsing the (potentially 100k+ record) JSONL every
run, a lightweight sidecar `fetched_ids.txt` (one legacy_id per line) is
maintained alongside `book_show_api.jsonl`. Reading that sidecar back is a
plain-text scan; the full JSONL is only parsed once, to bootstrap the sidecar
the first time it doesn't exist or to append IDs newly written since the
sidecar was last updated.

Give-up tracking: some books genuinely have no ISBN13 in Goodreads' data, so
book_show_api can never produce a legacy_id/isbn13 for them — every attempt
comes back as a `_scrape_warning: incomplete_record`. Once a book accumulates
`--give-up-after` (default 3) such warning-only attempts without ever
succeeding, it's excluded from `ids_remaining.txt` and recorded in a second
sidecar, `book_show_api_gave_up.txt` (same format as `fetched_ids.txt`), so
the batch scrape stops retrying it forever. This is permanent until a human
deletes the id (or the whole file) to force a retry; a book that ever does
produce a legacy_id is never given up regardless of prior warnings.

Queue ordering: within each of never-tried/retry buckets, `ids_remaining.txt`
is popularity-ordered (highest `ratings_count` first, using signals already
present in the `list_show` raw scrape — see `tools.catalog.list_show_popularity`)
so the scrape fetches the books most likely to matter first rather than in an
arbitrary order. Never-tried IDs stay ahead of retries. Incomplete IDs are
withheld for `--retry-cooldown-hours` (default 24) via
`book_show_api_retry_after.json` so a just-failed book is not re-requested on
the next chunk. A third sidecar, `book_popularity.json`, caches the
aggregation the same way `fetched_ids.txt` caches fetched IDs. Pass
`--order shuffle` to fall back to the old random never-tried ordering (kept
as a debugging escape hatch only).

Usage:
    python tools/catalog/extract_remaining_ids.py
    python tools/catalog/extract_remaining_ids.py --raw-dir <dir> --out ids_remaining.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.list_show_popularity import (  # noqa: E402
    BookPopularity,
    aggregate_popularity,
    popularity_sort_key,
)
from tools.catalog.match_goodreads import parse_book_id  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_OUT_NAME = "ids_remaining.txt"
DEFAULT_FETCHED_SIDECAR_NAME = "fetched_ids.txt"
DEFAULT_GAVE_UP_SIDECAR_NAME = "book_show_api_gave_up.txt"
DEFAULT_POPULARITY_SIDECAR_NAME = "book_popularity.json"
DEFAULT_RETRY_AFTER_SIDECAR_NAME = "book_show_api_retry_after.json"
DEFAULT_GIVE_UP_AFTER = 3
DEFAULT_RETRY_COOLDOWN_HOURS = 24.0
BOOK_SHOW_API_GLOB = "book_show_api*.jsonl"
ORDER_POPULARITY = "popularity"
ORDER_SHUFFLE = "shuffle"

# book_show_api warning records only carry `_url`, the resolved Next-data
# endpoint (e.g. ".../_next/data/<build_id>/book/show/33.json") — this is
# specific to that profile's next_build.endpoint_pattern, not a general
# Goodreads URL shape (contrast with match_goodreads.parse_book_id, which
# parses the `/book/show/33.LOTR`-style URLs from list_show scrapes).
_WARNING_URL_RE = re.compile(r"/book/show/(\d+)\.json")


def parse_book_id_from_warning_url(url: str | None) -> int | None:
    if not url:
        return None
    m = _WARNING_URL_RE.search(url)
    return int(m.group(1)) if m else None


def extract_raw_ids(raw_dir: Path) -> set[int]:
    """Scan every `<list_id>.jsonl` under `raw_dir` (list_show output) and
    return the set of distinct Goodreads book IDs across all lists."""
    ids: set[int] = set()
    if not raw_dir.exists():
        return ids
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for url in record.get("book_urls") or []:
                    book_id = parse_book_id(url)
                    if book_id is not None:
                        ids.add(book_id)
    return ids


def _extract_legacy_ids_from_jsonl(path: Path) -> set[int]:
    """Read every record's `legacy_id` from a book_show_api batch JSONL.
    Records without a usable legacy_id (e.g. a _scrape_warning record) are
    skipped rather than aborting the whole read."""
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            legacy_id = record.get("legacy_id")
            if legacy_id is not None:
                try:
                    ids.add(int(legacy_id))
                except (TypeError, ValueError):
                    continue
    return ids


def _extract_legacy_ids_from_jsonls(paths: Iterable[Path]) -> set[int]:
    """Union of `_extract_legacy_ids_from_jsonl` across multiple shard files —
    a session-chunked or manually-split book_show_api run can leave more than
    one JSONL (e.g. `book_show_api.jsonl` plus a stray `book_show_api.batch2.jsonl`
    from an interrupted session); none of them should silently strand progress."""
    ids: set[int] = set()
    for path in paths:
        ids |= _extract_legacy_ids_from_jsonl(path)
    return ids


def default_book_show_api_paths(catalog_dir: Path) -> list[Path]:
    """All `book_show_api*.jsonl` shards in `catalog_dir`, sorted for determinism."""
    return sorted(catalog_dir.glob(BOOK_SHOW_API_GLOB))


def write_fetched_sidecar(path: Path, ids: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def load_fetched_sidecar(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(line))
        except ValueError:
            continue
    return ids


def load_fetched_ids(book_show_api_paths: Path | Iterable[Path], sidecar_path: Path) -> set[int]:
    """Return the set of already-fetched legacy_ids, using `sidecar_path` as
    a fast cache. Bootstraps (or re-syncs) the sidecar from the full JSONL(s)
    whenever the sidecar is missing or older than the newest JSONL — this is
    the only case that pays the cost of a full JSONL scan.

    `book_show_api_paths` may be a single `Path` (the common case, and what
    the pre-multi-shard version of this function accepted) or an iterable of
    paths to union across several shard files.
    """
    paths = [book_show_api_paths] if isinstance(book_show_api_paths, Path) else list(book_show_api_paths)
    paths = [p for p in paths if p.exists()]
    if not paths:
        return set()

    newest_mtime = max(p.stat().st_mtime for p in paths)
    needs_rebuild = not sidecar_path.exists() or sidecar_path.stat().st_mtime < newest_mtime
    if needs_rebuild:
        ids = _extract_legacy_ids_from_jsonls(paths)
        write_fetched_sidecar(sidecar_path, ids)
        return ids

    return load_fetched_sidecar(sidecar_path)


def count_warning_only_attempts(paths: Iterable[Path]) -> dict[int, int]:
    """Tally, per book_id, how many `_scrape_warning` attempts a book has
    accumulated across every shard — a book genuinely lacking an ISBN13 will
    always land here (see book_show_api.yaml's `required_fields`) rather than
    ever producing a `legacy_id`. A raw per-attempt line counts as 1; a
    merge_book_show_api.py-aggregated record's `_attempt_count` is honored so
    counts survive a manual merge."""
    counts: dict[int, int] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_scrape_warning" not in record:
                    continue
                book_id = parse_book_id_from_warning_url(record.get("_url"))
                if book_id is None:
                    continue
                counts[book_id] = counts.get(book_id, 0) + int(record.get("_attempt_count", 1))
    return counts


def compute_gave_up(warning_attempts: dict[int, int], fetched_ids: set[int], give_up_after: int) -> set[int]:
    """A book that ever succeeded (has a `legacy_id` in `fetched_ids`) is
    never given up, no matter how many warnings it accumulated earlier —
    success always overrides prior soft failures."""
    return {
        book_id
        for book_id, count in warning_attempts.items()
        if count >= give_up_after and book_id not in fetched_ids
    }


def load_gave_up_ids(
    book_show_api_paths: Path | Iterable[Path],
    sidecar_path: Path,
    fetched_ids: set[int],
    give_up_after: int = DEFAULT_GIVE_UP_AFTER,
) -> set[int]:
    """Same staleness-cache pattern as `load_fetched_ids`: rebuilds from the
    full JSONL(s) only when `sidecar_path` is missing or older than the
    newest shard, otherwise trusts the cached sidecar as-is."""
    paths = [book_show_api_paths] if isinstance(book_show_api_paths, Path) else list(book_show_api_paths)
    paths = [p for p in paths if p.exists()]
    if not paths:
        return set()

    newest_mtime = max(p.stat().st_mtime for p in paths)
    needs_rebuild = not sidecar_path.exists() or sidecar_path.stat().st_mtime < newest_mtime
    if needs_rebuild:
        attempts = count_warning_only_attempts(paths)
        gave_up = compute_gave_up(attempts, fetched_ids, give_up_after)
        write_fetched_sidecar(sidecar_path, gave_up)
        return gave_up

    return load_fetched_sidecar(sidecar_path)


def write_popularity_sidecar(path: Path, popularity: dict[int, BookPopularity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "books": {
            str(book_id): {
                "ratings_count": pop.ratings_count,
                "list_appearances": pop.list_appearances,
                "list_score_sum": pop.list_score_sum,
                "list_vote_sum": pop.list_vote_sum,
            }
            for book_id, pop in popularity.items()
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_popularity_sidecar(path: Path) -> dict[int, BookPopularity]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    popularity: dict[int, BookPopularity] = {}
    for book_id_str, fields in (data.get("books") or {}).items():
        try:
            book_id = int(book_id_str)
        except ValueError:
            continue
        popularity[book_id] = BookPopularity(
            ratings_count=int(fields.get("ratings_count", 0)),
            list_appearances=int(fields.get("list_appearances", 0)),
            list_score_sum=int(fields.get("list_score_sum", 0)),
            list_vote_sum=int(fields.get("list_vote_sum", 0)),
        )
    return popularity


def load_popularity(raw_dir: Path, sidecar_path: Path) -> dict[int, BookPopularity]:
    """Same staleness-cache pattern as `load_fetched_ids`: rebuilds the
    popularity aggregation from every `raw_dir/*.jsonl` only when
    `sidecar_path` is missing or older than the newest raw file, otherwise
    trusts the cached sidecar — avoids re-scanning every `list_show` row on
    every loop iteration."""
    raw_paths = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    if not raw_paths:
        return load_popularity_sidecar(sidecar_path)

    newest_mtime = max(p.stat().st_mtime for p in raw_paths)
    needs_rebuild = not sidecar_path.exists() or sidecar_path.stat().st_mtime < newest_mtime
    if not needs_rebuild:
        # A sidecar newer than raw/ but with zero entries (e.g. interrupted write of
        # {"books": {}}) poisons queue ordering — treat as corrupt and rebuild.
        if not load_popularity_sidecar(sidecar_path):
            needs_rebuild = True
    if needs_rebuild:
        popularity = aggregate_popularity(raw_dir)
        write_popularity_sidecar(sidecar_path, popularity)
        return popularity

    return load_popularity_sidecar(sidecar_path)


def load_attempted_ids(book_show_api_paths: Path | Iterable[Path]) -> set[int]:
    """Book IDs with at least one `_scrape_warning` row whose `_url` parses to
    a book id. Older silent-null rows (partial Apollo cache, no `legacy_id`)
    carry no book id in the JSONL and are not counted — those books may still
    land in the never-tried bucket until a warning row exists."""
    paths = [book_show_api_paths] if isinstance(book_show_api_paths, Path) else list(book_show_api_paths)
    return set(count_warning_only_attempts(paths).keys())


def write_retry_after_sidecar(path: Path, retries: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retries": {
            str(book_id): {
                "attempts": int(entry["attempts"]),
                "retry_after": entry["retry_after"],
            }
            for book_id, entry in retries.items()
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_retry_after_sidecar(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    retries: dict[int, dict] = {}
    for book_id_str, fields in (data.get("retries") or {}).items():
        try:
            book_id = int(book_id_str)
        except ValueError:
            continue
        if not isinstance(fields, dict) or "retry_after" not in fields:
            continue
        retries[book_id] = {
            "attempts": int(fields.get("attempts", 0)),
            "retry_after": str(fields["retry_after"]),
        }
    return retries


def load_cooling_ids(sidecar_path: Path, now: datetime | None = None) -> set[int]:
    """Book IDs whose stored `retry_after` is still in the future."""
    now = now or datetime.now(timezone.utc)
    cooling: set[int] = set()
    for book_id, entry in load_retry_after_sidecar(sidecar_path).items():
        retry_dt = _parse_retry_after(entry.get("retry_after"))
        if retry_dt is not None and now < retry_dt:
            cooling.add(book_id)
    return cooling


def _parse_retry_after(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def update_retry_after(
    warning_attempts: dict[int, int],
    sidecar_path: Path,
    cooldown_hours: float,
    *,
    now: datetime | None = None,
    fetched_ids: set[int] = frozenset(),
) -> set[int]:
    """Stamp `retry_after` when a book's warning-only attempt count increases.

    JSONL warning rows have no timestamps; comparing the current count to the
    sidecar's stored `attempts` is the durable signal. Returns IDs still in
    cooldown. Fetched books are dropped from the sidecar.
    """
    now = now or datetime.now(timezone.utc)
    existing = load_retry_after_sidecar(sidecar_path)
    updated: dict[int, dict] = {}
    cooling: set[int] = set()
    cooldown = timedelta(hours=cooldown_hours)

    for book_id, count in warning_attempts.items():
        if book_id in fetched_ids:
            continue
        prev = existing.get(book_id)
        if prev is None or count > int(prev.get("attempts", 0)):
            entry = {
                "attempts": count,
                "retry_after": (now + cooldown).isoformat(timespec="seconds"),
            }
        else:
            entry = prev
        updated[book_id] = entry
        retry_dt = _parse_retry_after(entry.get("retry_after"))
        if retry_dt is not None and now < retry_dt:
            cooling.add(book_id)

    write_retry_after_sidecar(sidecar_path, updated)
    return cooling


def order_remaining_ids(remaining: set[int], attempted_ids: set[int], popularity: dict[int, BookPopularity]) -> list[int]:
    """Never-tried IDs first, then prior soft-failure retries — both buckets
    sorted by `popularity_sort_key` (highest `ratings_count` first; a book
    with no popularity signal sorts to the tail of its bucket). Scrape-harness
    walks `--set-from-file` top-to-bottom, so this fetches the books most
    likely to matter first instead of in an arbitrary order."""

    def sort_key(book_id: int) -> tuple[int, int, int, int]:
        return popularity_sort_key(book_id, popularity.get(book_id, BookPopularity()))

    never_tried = sorted(remaining - attempted_ids, key=sort_key)
    retries = sorted(remaining & attempted_ids, key=sort_key)
    return never_tried + retries


def order_remaining_ids_shuffled(remaining: set[int], attempted_ids: set[int]) -> list[int]:
    """Legacy random never-tried ordering, numeric-sorted retries — kept as a
    `--order shuffle` debugging escape hatch only. Scrape-harness walks
    `--set-from-file` top-to-bottom; a numeric sort on never-tried IDs
    clusters early low-id partial-response books at the head and looks like
    a dead session even when ~80%+ of never-tried IDs succeed, which is why
    this is no longer the default (see `order_remaining_ids`)."""
    never_tried = list(remaining - attempted_ids)
    random.shuffle(never_tried)
    retries = sorted(remaining & attempted_ids)
    return never_tried + retries


def compute_remaining(raw_ids: set[int], fetched_ids: set[int], gave_up_ids: set[int] = frozenset()) -> set[int]:
    return raw_ids - fetched_ids - gave_up_ids


@dataclass(frozen=True)
class RemainingQueue:
    """One remaining-ISBN queue: remaining set, popularity order, and the
    inputs that produced them. `main()`, `diagnose_catalog`, and
    `repair_catalog` all use this instead of each recomputing the same
    subtraction/order steps."""

    remaining: set[int]
    attempted: set[int]
    popularity: dict[int, BookPopularity]
    ordered: list[int]
    raw_ids: set[int] = field(default_factory=set)
    fetched_ids: set[int] = field(default_factory=set)
    gave_up_ids: set[int] = field(default_factory=set)
    cooling_ids: set[int] = field(default_factory=set)
    warning_attempts: dict[int, int] = field(default_factory=dict)


def _popularity_for_queue(raw_dir: Path, popularity_sidecar: Path, *, persist: bool) -> dict[int, BookPopularity]:
    if persist:
        return load_popularity(raw_dir, popularity_sidecar)
    cached = load_popularity_sidecar(popularity_sidecar)
    if cached:
        return cached
    return aggregate_popularity(raw_dir) if raw_dir.exists() else {}


def build_remaining_queue(
    raw_dir: Path,
    book_show_api_paths: Path | Iterable[Path],
    *,
    fetched_sidecar: Path,
    gave_up_sidecar: Path,
    popularity_sidecar: Path,
    retry_after_sidecar: Path,
    give_up_after: int = DEFAULT_GIVE_UP_AFTER,
    retry_cooldown_hours: float = DEFAULT_RETRY_COOLDOWN_HOURS,
    order: str = ORDER_POPULARITY,
    persist: bool = True,
    now: datetime | None = None,
) -> RemainingQueue:
    """Fetched / gave-up / cooling / popularity-ordered remaining IDs.

    `persist=True` (CLI rebuild, catalog repair) writes retry-after and
    popularity sidecars. `persist=False` (diagnosis) reads the same
    remaining set a persist would produce without stamping cooldown or
    rebuilding a corrupt popularity sidecar — otherwise diagnose-then-repair
    would see a just-rewritten sidecar and skip the repair.
    """
    fetched_ids = load_fetched_ids(book_show_api_paths, fetched_sidecar)
    gave_up_ids = load_gave_up_ids(book_show_api_paths, gave_up_sidecar, fetched_ids, give_up_after)
    warning_attempts = count_warning_only_attempts(book_show_api_paths)
    if persist:
        cooling_ids = update_retry_after(
            warning_attempts,
            retry_after_sidecar,
            retry_cooldown_hours,
            now=now,
            fetched_ids=fetched_ids,
        )
    else:
        cooling_ids = load_cooling_ids(retry_after_sidecar, now=now)
    raw_ids = extract_raw_ids(raw_dir)
    remaining = compute_remaining(raw_ids, fetched_ids, gave_up_ids) - cooling_ids
    attempted = set(warning_attempts.keys())
    if order == ORDER_SHUFFLE:
        popularity: dict[int, BookPopularity] = {}
        ordered = order_remaining_ids_shuffled(remaining, attempted)
    else:
        popularity = _popularity_for_queue(raw_dir, popularity_sidecar, persist=persist)
        ordered = order_remaining_ids(remaining, attempted, popularity)
    return RemainingQueue(
        remaining=remaining,
        attempted=attempted,
        popularity=popularity,
        ordered=ordered,
        raw_ids=raw_ids,
        fetched_ids=fetched_ids,
        gave_up_ids=gave_up_ids,
        cooling_ids=cooling_ids,
        warning_attempts=warning_attempts,
    )


def write_ids_remaining(path: Path, ids: set[int] | Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(ids) if isinstance(ids, set) else list(ids)
    path.write_text("\n".join(str(i) for i in ordered) + ("\n" if ordered else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument(
        "--book-show-api",
        type=Path,
        action="append",
        default=None,
        help=(
            "book_show_api JSONL path (repeatable). Default: every "
            f"catalog_goodreads('{BOOK_SHOW_API_GLOB}') shard, so stray/interrupted "
            "session files are never dropped from progress tracking."
        ),
    )
    parser.add_argument(
        "--fetched-sidecar", type=Path, default=None, help="Default: catalog_goodreads('fetched_ids.txt')"
    )
    parser.add_argument(
        "--gave-up-sidecar", type=Path, default=None, help=f"Default: catalog_goodreads('{DEFAULT_GAVE_UP_SIDECAR_NAME}')"
    )
    parser.add_argument(
        "--give-up-after",
        type=int,
        default=DEFAULT_GIVE_UP_AFTER,
        help=(
            "Warning-only attempts (no legacy_id ever seen) before excluding a book id from "
            f"ids_remaining.txt as likely lacking an ISBN13 (default: {DEFAULT_GIVE_UP_AFTER})"
        ),
    )
    parser.add_argument(
        "--popularity-sidecar",
        type=Path,
        default=None,
        help=f"Default: catalog_goodreads('{DEFAULT_POPULARITY_SIDECAR_NAME}')",
    )
    parser.add_argument(
        "--retry-after-sidecar",
        type=Path,
        default=None,
        help=f"Default: catalog_goodreads('{DEFAULT_RETRY_AFTER_SIDECAR_NAME}')",
    )
    parser.add_argument(
        "--retry-cooldown-hours",
        type=float,
        default=DEFAULT_RETRY_COOLDOWN_HOURS,
        help=(
            "Hours to withhold a book id after its warning-only attempt count increases "
            f"(default: {DEFAULT_RETRY_COOLDOWN_HOURS:g})"
        ),
    )
    parser.add_argument(
        "--order",
        choices=(ORDER_POPULARITY, ORDER_SHUFFLE),
        default=ORDER_POPULARITY,
        help=(
            f"Queue order within each never-tried/retry bucket (default: {ORDER_POPULARITY}). "
            f"'{ORDER_SHUFFLE}' is a debugging escape hatch that restores the old random never-tried order."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help=f"Default: catalog_goodreads('{DEFAULT_OUT_NAME}')")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or catalog_goodreads("raw")
    book_show_api_paths = args.book_show_api or default_book_show_api_paths(catalog_goodreads())
    sidecar_path = args.fetched_sidecar or catalog_goodreads(DEFAULT_FETCHED_SIDECAR_NAME)
    gave_up_sidecar_path = args.gave_up_sidecar or catalog_goodreads(DEFAULT_GAVE_UP_SIDECAR_NAME)
    popularity_sidecar_path = args.popularity_sidecar or catalog_goodreads(DEFAULT_POPULARITY_SIDECAR_NAME)
    retry_after_sidecar_path = args.retry_after_sidecar or catalog_goodreads(DEFAULT_RETRY_AFTER_SIDECAR_NAME)
    out_path = args.out or catalog_goodreads(DEFAULT_OUT_NAME)

    queue = build_remaining_queue(
        raw_dir,
        book_show_api_paths,
        fetched_sidecar=sidecar_path,
        gave_up_sidecar=gave_up_sidecar_path,
        popularity_sidecar=popularity_sidecar_path,
        retry_after_sidecar=retry_after_sidecar_path,
        give_up_after=args.give_up_after,
        retry_cooldown_hours=args.retry_cooldown_hours,
        order=args.order,
    )

    ratings_suffix = ""
    if args.order != ORDER_SHUFFLE and queue.remaining:
        with_ratings = sum(1 for bid in queue.remaining if queue.popularity.get(bid, BookPopularity()).ratings_count > 0)
        ratings_suffix = (
            f" / {with_ratings:,} with ratings_count>0 ({with_ratings / len(queue.remaining):.0%})"
        )

    never_tried_count = len(queue.remaining - queue.attempted)
    retry_count = len(queue.remaining & queue.attempted)

    write_ids_remaining(out_path, queue.ordered)

    top_remaining_suffix = f" / top remaining: book_id={queue.ordered[0]}" if queue.ordered else ""
    print(
        f"[extract_remaining_ids] {len(queue.remaining):,} remaining / "
        f"{len(queue.raw_ids):,} total / {len(queue.fetched_ids):,} already fetched / "
        f"{len(queue.gave_up_ids):,} gave up (no isbn13 after {args.give_up_after}+ attempts) / "
        f"{len(queue.cooling_ids):,} cooling ({args.retry_cooldown_hours:g}h) / "
        f"{never_tried_count:,} never tried / {retry_count:,} prior soft-failure retries"
        f"{ratings_suffix}{top_remaining_suffix}"
    )
    print(f"[extract_remaining_ids] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
