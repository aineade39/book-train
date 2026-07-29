#!/usr/bin/env python3
"""Analyze scraped Goodreads Listopia lists for overlap and popularity coverage.

Answers: which lists are worth scraping further, and which are redundant?
Read-only by default — only `--apply-deprecations` writes anything, and only
to a separate `seed_list_overrides.yaml` sidecar. This script never touches
`goodreads_seed_lists.yaml` itself: that file's value is its hand-verified
provenance (see its own header comment), and a programmatic YAML round-trip
would silently strip that.

Only lists marked `done` in the scrape checkpoint (see
`tools/scrape_goodreads_lists.py`'s `Checkpoint`) are evaluated for
deprecation. The harness already scrapes a list to pagination exhaustion
(or a very generous `max_pages` ceiling) before marking it `done`, so that
status is a more reliable "fully scraped" signal than the seed file's
`book_count_observed`, which is explicitly documented there as an
approximate, growing-over-time snapshot.

Two-outcome rule engine per `done` list — deliberately simple (see
docs/BOOK_CATALOG.md for why a wider taxonomy isn't worth it here):

  1. Anchor exemption    -- `list_type: anchor` is always `keep`, never
                             evaluated further. Meant for the single broad
                             popularity list (e.g. "Best Books Ever").
  2. Subset              -- more than `SUBSET_CONTAINMENT_THRESHOLD` of this
                             list's books already appear in one other single
                             `done` list -> `deprecate`.
  3. Overlap + low value -- Jaccard similarity with another `done` list
                             exceeds `OVERLAP_JACCARD_THRESHOLD` *and* this
                             list contributes fewer than
                             `LOW_MARGINAL_GAIN_THRESHOLD` new books to a
                             shared top-N-by-`ratings_count` corpus ->
                             `deprecate`. High overlap alone is not enough —
                             a list can overlap heavily with another and
                             still add plenty of unique popular books.
  4. Default             -- `keep`.

Lists not yet `done` (pending/error/challenged) are excluded from evaluation
entirely; the checkpoint's own retry logic already handles resuming them —
this tool has nothing useful to say about a list that isn't finished yet.

Usage:
    python tools/catalog/analyze_goodreads_lists.py
    python tools/catalog/analyze_goodreads_lists.py --top-n 1000
    python tools/catalog/analyze_goodreads_lists.py --apply-deprecations
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.paths import catalog_goodreads  # noqa: E402

# Deliberately NOT importing from tools.catalog.match_goodreads: that module
# hard-requires rapidfuzz for its OL fuzzy-matching stage, which this
# overlap/coverage analysis has no use for. The tiny bit of list_show parsing
# needed here (book id + ratings_count per row) is duplicated below instead,
# keeping this tool's only dependency PyYAML — same footprint as
# tools/scrape_goodreads_lists.py.
_BOOK_URL_RE = re.compile(r"^/book/show/(\d+)")
_RATING_TEXT_RE = re.compile(r"([\d.]+)\s+avg rating\s*[-\u2013\u2014]+\s*([\d,]+)\s+ratings?")
_LIST_SHOW_FIELDS = ("book_urls", "titles", "authors", "rating_texts")


def _parse_book_id(book_url: str | None) -> int | None:
    if not book_url:
        return None
    m = _BOOK_URL_RE.match(book_url)
    return int(m.group(1)) if m else None


def _parse_ratings_count(rating_text: str | None) -> int | None:
    """'4.55 avg rating — 745,415 ratings' -> 745415."""
    if not rating_text:
        return None
    m = _RATING_TEXT_RE.search(rating_text)
    return int(m.group(2).replace(",", "")) if m else None


def _iter_list_show_rows(path: Path) -> Iterator[tuple[int, int]]:
    """Yields (book_id, ratings_count) for every book row in a `list_show`
    raw JSONL file, mirroring `match_goodreads._zip_list_show_record` but
    only extracting the two fields overlap/coverage analysis needs."""
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
            for i in range(n):
                book_id = _parse_book_id(book_urls[i] if i < len(book_urls) else None)
                if book_id is None:
                    continue
                ratings_count = _parse_ratings_count(rating_texts[i] if i < len(rating_texts) else None)
                yield book_id, ratings_count or 0

SEED_LISTS_PATH = Path(__file__).resolve().parent / "goodreads_seed_lists.yaml"

DEFAULT_TOP_N = 1000
MARGINAL_GAIN_TOP_N = 5000
SUBSET_CONTAINMENT_THRESHOLD = 0.7
OVERLAP_JACCARD_THRESHOLD = 0.35
LOW_MARGINAL_GAIN_THRESHOLD = 50

OUTCOME_KEEP = "keep"
OUTCOME_DEPRECATE = "deprecate"

# Mirrors tools.scrape_goodreads_lists.STATUS_DONE — duplicated as a plain
# string (not imported) so this analysis tool has no hard dependency on the
# scraper module, only on the checkpoint DB's on-disk schema.
DONE_STATUS = "done"

LIST_TYPE_ANCHOR = "anchor"
LIST_TYPE_GENRE = "genre"


@dataclass(frozen=True)
class SeedListInfo:
    list_id: int
    slug: str
    genre: str
    list_type: str  # "anchor" | "genre" (default "genre" when omitted)


def load_seed_lists(path: Path) -> dict[int, SeedListInfo]:
    import yaml

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[int, SeedListInfo] = {}
    for row in data.get("lists", []):
        list_id = int(row["list_id"])
        out[list_id] = SeedListInfo(
            list_id=list_id,
            slug=str(row.get("slug", "")),
            genre=str(row.get("genre", "")),
            list_type=str(row.get("list_type") or LIST_TYPE_GENRE),
        )
    return out


def load_checkpoint_status(db_path: Path) -> dict[int, str]:
    """Reads the scrape checkpoint DB -> {list_id: status}. Returns {} if the
    checkpoint doesn't exist yet (e.g. before the first scrape session)."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT list_id, status FROM goodreads_lists")
        return {int(row[0]): str(row[1]) for row in cur.fetchall()}
    finally:
        conn.close()


@dataclass
class ListData:
    list_id: int
    book_ids: set[int] = field(default_factory=set)
    ratings: dict[int, int] = field(default_factory=dict)  # book_id -> best-seen ratings_count


def load_list_data(raw_dir: Path, list_id: int) -> ListData | None:
    """Parses `raw_dir/<list_id>.jsonl`'s `list_show` rows into book ids +
    best-seen ratings_count per book."""
    path = raw_dir / f"{list_id}.jsonl"
    if not path.exists():
        return None
    data = ListData(list_id=list_id)
    for book_id, ratings_count in _iter_list_show_rows(path):
        data.book_ids.add(book_id)
        data.ratings[book_id] = max(data.ratings.get(book_id, 0), ratings_count)
    return data


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def containment(a: set[int], b: set[int]) -> float:
    """Fraction of `a`'s books that also appear in `b`. Asymmetric on
    purpose — a small list wholly inside a big one should be flagged even
    though the reverse containment is tiny."""
    return len(a & b) / len(a) if a else 0.0


def top_n_book_ids(ratings_by_book: dict[int, int], n: int) -> set[int]:
    ranked = sorted(ratings_by_book.items(), key=lambda kv: kv[1], reverse=True)
    return {book_id for book_id, _rating in ranked[:n]}


def merge_ratings(lists: dict[int, ListData]) -> dict[int, int]:
    merged: dict[int, int] = {}
    for data in lists.values():
        for book_id, rating in data.ratings.items():
            merged[book_id] = max(merged.get(book_id, 0), rating)
    return merged


def marginal_gain(list_id: int, done_lists: dict[int, ListData], target: set[int]) -> int:
    """New books this list alone adds to `target` (e.g. the top-5000 by
    ratings_count) beyond what every other `done` list already covers."""
    this_hits = done_lists[list_id].book_ids & target
    others: set[int] = set()
    for other_id, data in done_lists.items():
        if other_id != list_id:
            others |= data.book_ids
    return len(this_hits - others)


def greedy_set_cover(done_lists: dict[int, ListData], target: set[int], seed: dict[int, SeedListInfo]) -> list[dict]:
    """Order to add `done` lists in to cover `target` fastest — informational
    only, not consumed by the rule engine."""
    remaining = set(target)
    candidates = dict(done_lists)
    order: list[dict] = []
    while remaining and candidates:
        best_id = max(candidates, key=lambda lid: len(candidates[lid].book_ids & remaining))
        gain = len(candidates[best_id].book_ids & remaining)
        if gain == 0:
            break
        info = seed.get(best_id)
        order.append({"list_id": best_id, "slug": info.slug if info else str(best_id), "new_coverage": gain})
        remaining -= candidates[best_id].book_ids
        del candidates[best_id]
    return order


@dataclass
class Recommendation:
    list_id: int
    slug: str
    outcome: str
    reasons: list[str]
    metrics: dict[str, object]


def evaluate_lists(
    seed: dict[int, SeedListInfo],
    checkpoint_status: dict[int, str],
    done_lists: dict[int, ListData],
    *,
    top_n: int,
    marginal_gain_top_n: int = MARGINAL_GAIN_TOP_N,
) -> tuple[list[Recommendation], list[int]]:
    """Returns (recommendations for evaluated lists, list_ids not yet evaluated).

    `marginal_gain_top_n` is exposed as a parameter (rather than only ever
    using the `MARGINAL_GAIN_TOP_N` module constant) so tests can exercise
    the marginal-gain rule against small fixtures without needing thousands
    of fake book ids.
    """
    all_ratings = merge_ratings(done_lists)
    top_n_ids = top_n_book_ids(all_ratings, top_n)
    top_marginal_ids = top_n_book_ids(all_ratings, marginal_gain_top_n)

    recommendations: list[Recommendation] = []
    not_evaluated: list[int] = []

    for list_id, info in sorted(seed.items()):
        if checkpoint_status.get(list_id) != DONE_STATUS or list_id not in done_lists:
            not_evaluated.append(list_id)
            continue

        data = done_lists[list_id]
        size = len(data.book_ids)
        top_n_coverage = len(data.book_ids & top_n_ids)
        base_metrics: dict[str, object] = {"size": size, f"top{top_n}_coverage": top_n_coverage}

        if info.list_type == LIST_TYPE_ANCHOR:
            recommendations.append(
                Recommendation(
                    list_id=list_id,
                    slug=info.slug,
                    outcome=OUTCOME_KEEP,
                    reasons=["anchor: always kept, never evaluated for deprecation"],
                    metrics=base_metrics,
                )
            )
            continue

        # Rule 2 -- subset.
        best_containment = 0.0
        best_containment_other: int | None = None
        for other_id, other_data in done_lists.items():
            if other_id == list_id:
                continue
            c = containment(data.book_ids, other_data.book_ids)
            if c > best_containment:
                best_containment, best_containment_other = c, other_id

        if best_containment > SUBSET_CONTAINMENT_THRESHOLD and best_containment_other is not None:
            other_info = seed.get(best_containment_other)
            other_label = other_info.slug if other_info else str(best_containment_other)
            recommendations.append(
                Recommendation(
                    list_id=list_id,
                    slug=info.slug,
                    outcome=OUTCOME_DEPRECATE,
                    reasons=[
                        f"subset: {best_containment:.0%} of its {size} books already in "
                        f"list {best_containment_other} ({other_label})"
                    ],
                    metrics={
                        **base_metrics,
                        "max_containment": round(best_containment, 3),
                        "max_containment_other_list_id": best_containment_other,
                    },
                )
            )
            continue

        # Rule 3 -- overlap + low marginal value (both conditions required).
        max_jaccard = 0.0
        max_jaccard_other: int | None = None
        for other_id, other_data in done_lists.items():
            if other_id == list_id:
                continue
            j = jaccard(data.book_ids, other_data.book_ids)
            if j > max_jaccard:
                max_jaccard, max_jaccard_other = j, other_id

        gain = marginal_gain(list_id, done_lists, top_marginal_ids)
        metrics_with_overlap = {
            **base_metrics,
            "max_jaccard": round(max_jaccard, 3),
            "max_jaccard_other_list_id": max_jaccard_other,
            f"marginal_gain_top{marginal_gain_top_n}": gain,
        }

        if (
            max_jaccard > OVERLAP_JACCARD_THRESHOLD
            and gain < LOW_MARGINAL_GAIN_THRESHOLD
            and max_jaccard_other is not None
        ):
            other_info = seed.get(max_jaccard_other)
            other_label = other_info.slug if other_info else str(max_jaccard_other)
            recommendations.append(
                Recommendation(
                    list_id=list_id,
                    slug=info.slug,
                    outcome=OUTCOME_DEPRECATE,
                    reasons=[
                        f"overlap: jaccard {max_jaccard:.3f} with list {max_jaccard_other} ({other_label}), "
                        f"only {gain} new books in the top {marginal_gain_top_n} by ratings_count"
                    ],
                    metrics=metrics_with_overlap,
                )
            )
            continue

        recommendations.append(
            Recommendation(
                list_id=list_id,
                slug=info.slug,
                outcome=OUTCOME_KEEP,
                reasons=["no rule matched"],
                metrics=metrics_with_overlap,
            )
        )

    return recommendations, not_evaluated


def build_report(
    seed: dict[int, SeedListInfo],
    checkpoint_status: dict[int, str],
    done_lists: dict[int, ListData],
    *,
    top_n: int,
) -> dict:
    recommendations, not_evaluated = evaluate_lists(seed, checkpoint_status, done_lists, top_n=top_n)
    all_ratings = merge_ratings(done_lists)
    top_n_ids = top_n_book_ids(all_ratings, top_n)
    unique_books = set()
    for data in done_lists.values():
        unique_books |= data.book_ids

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated": sorted(done_lists.keys()),
        "not_yet_evaluated": sorted(not_evaluated),
        "corpus": {
            "done_lists": len(done_lists),
            "unique_books": len(unique_books),
            f"top{top_n}_covered": len(top_n_ids & unique_books),
        },
        f"greedy_set_cover_top{top_n}": greedy_set_cover(done_lists, top_n_ids, seed),
        "recommendations": [asdict(r) for r in recommendations],
    }


def print_summary(report: dict) -> None:
    print(f"[analyze] evaluated {len(report['evaluated'])} done list(s); "
          f"{len(report['not_yet_evaluated'])} not yet done: {report['not_yet_evaluated']}")
    print(f"[analyze] corpus: {json.dumps(report['corpus'])}")
    for rec in report["recommendations"]:
        reason = "; ".join(rec["reasons"])
        print(f"[analyze]   list {rec['list_id']:>7} ({rec['slug']}): {rec['outcome']} — {reason}")


def load_overrides(path: Path) -> dict[int, dict]:
    """Reads `seed_list_overrides.yaml` -> {list_id: entry_dict}. Missing
    file means no overrides exist yet — never an error."""
    import yaml

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides") or {}
    return {int(list_id): dict(entry) for list_id, entry in overrides.items()}


def apply_deprecations(path: Path, recommendations: list[Recommendation]) -> int:
    """Adds a `curation_status: deprecated` entry for every recommended
    deprecation not already present in the overrides file. Never overwrites
    an existing entry — if a human has already set one (e.g. to manually
    keep a list the rules would otherwise flag), that choice is respected
    and never silently clobbered by a later run."""
    import yaml

    existing = load_overrides(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    added = 0
    for rec in recommendations:
        if rec.outcome != OUTCOME_DEPRECATE or rec.list_id in existing:
            continue
        existing[rec.list_id] = {
            "curation_status": "deprecated",
            "reason": "; ".join(rec.reasons),
            "set_at": now,
        }
        added += 1

    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Curation overrides for Goodreads Listopia seed lists.\n"
            "# Written by: python tools/catalog/analyze_goodreads_lists.py --apply-deprecations\n"
            "# Lists not listed here are treated as active. Safe to hand-edit — entries here\n"
            "# are never overwritten by a later --apply-deprecations run, only added to.\n"
            "# `curation_status: deprecated` only stops FUTURE scraping of that list (via\n"
            "# scrape_goodreads_lists.py --skip-deprecated); it never removes or excludes\n"
            "# already-scraped data from matching.\n"
        )
        body = yaml.safe_dump(
            {"overrides": {list_id: entry for list_id, entry in sorted(existing.items())}},
            sort_keys=False,
            default_flow_style=False,
        )
        path.write_text(header + "\n" + body, encoding="utf-8")

    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-lists", type=Path, default=SEED_LISTS_PATH)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument(
        "--checkpoint-db", type=Path, default=None, help="Default: catalog_goodreads('checkpoint.sqlite')"
    )
    parser.add_argument("--out", type=Path, default=None, help="Default: catalog_goodreads('list_recommendations.json')")
    parser.add_argument(
        "--overrides", type=Path, default=None, help="Default: catalog_goodreads('seed_list_overrides.yaml')"
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--apply-deprecations",
        action="store_true",
        help="Write new deprecate recommendations to --overrides. Off by default (report-only).",
    )
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or catalog_goodreads("raw")
    checkpoint_db = args.checkpoint_db or catalog_goodreads("checkpoint.sqlite")
    out_path = args.out or catalog_goodreads("list_recommendations.json")
    overrides_path = args.overrides or catalog_goodreads("seed_list_overrides.yaml")

    seed = load_seed_lists(args.seed_lists)
    checkpoint_status = load_checkpoint_status(checkpoint_db)

    done_lists: dict[int, ListData] = {}
    for list_id in seed:
        if checkpoint_status.get(list_id) == DONE_STATUS:
            data = load_list_data(raw_dir, list_id)
            if data is not None:
                done_lists[list_id] = data

    report = build_report(seed, checkpoint_status, done_lists, top_n=args.top_n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"[analyze] wrote {out_path}")
    print_summary(report)

    if args.apply_deprecations:
        recommendations, _not_evaluated = evaluate_lists(seed, checkpoint_status, done_lists, top_n=args.top_n)
        added = apply_deprecations(overrides_path, recommendations)
        print(f"[analyze] --apply-deprecations: added {added} new deprecation(s) to {overrides_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
