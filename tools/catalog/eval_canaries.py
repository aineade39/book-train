#!/usr/bin/env python3
"""Zero-label canaries (Phase 2f): catch over-merging on the books the
ISBN-gold set can't see, using no labels at all.

The ISBN-gold set (``match_goodreads.py --eval-isbn-holdout``) covers a
minority of books, so precision on everything else is currently
unmeasurable by that eval alone. These three checks run over the full
``matched_goodreads.jsonl.gz`` and need no ground truth:

1. ``suspicious_duplicate_targets`` -- distinct GR ``book_id``s that landed
   on the same OL ``workKey`` via ``title_author``, where their own
   title/author don't actually look like the same book. Legitimate reasons
   two GR books share a ``workKey`` (an OL duplicate-work collapse via the
   edition-count tie-break, a series omnibus) look identical or
   author-compatible; anything else is the strongest available over-merge
   alarm on the ~20k books with no ISBN to check against.
2. ``gap_fill_candidates`` -- mirrors
   ``build_ios_en_from_goodreads.gap_fill_unmatched``'s insert gate exactly
   (without touching a database). Should *fall* as matching improves; a
   drop with no corresponding rise in verified matches means books are
   being absorbed somewhere wrong.
3. ``match_method_distribution`` -- raw counts per ``match_method``, plus a
   delta against a baseline report if given. A change that moves thousands
   of books between ``unmatched`` and ``title_author`` in one step deserves
   manual scrutiny regardless of what the recall number says.

None of these gate on their own (see ``tools/catalog/acceptance_gate.py``
for the reusable accept/reject check that consumes their output alongside
the ISBN-holdout and adversarial-pairs numbers).

Usage:
    python tools/catalog/eval_canaries.py \\
        --matched <catalog_goodreads('matched_goodreads.jsonl.gz')> \\
        [--baseline <prior report.json>] [--out report.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bibliographic_join import names_compatible, title_core  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

# Mirrors build_ios_en_from_goodreads.DEFAULT_GAP_FILL_MIN_RATINGS_COUNT --
# duplicated rather than imported so this eval script has no dependency on
# that orchestration module (which touches a real sqlite scratch copy).
DEFAULT_GAP_FILL_MIN_RATINGS_COUNT = 1_000
MAX_SUSPICIOUS_EXAMPLES = 25


def load_matched_rows(path: Path) -> list[dict]:
    """Reads ``match_goodreads.py``'s gzipped-JSONL output."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def suspicious_duplicate_targets(rows: Iterable[dict]) -> dict[str, object]:
    by_work: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        work_key = row.get("work_key")
        if work_key and row.get("match_method") == "title_author":
            by_work[work_key].append(row)

    suspicious: list[dict[str, object]] = []
    legitimate = 0
    for work_key, group in by_work.items():
        if len(group) < 2:
            continue
        cores = {title_core(r["title"]) for r in group}
        all_compatible = all(
            names_compatible(group[i]["author"], group[j]["author"])
            for i in range(len(group))
            for j in range(i + 1, len(group))
        )
        if len(cores) > 1 or not all_compatible:
            suspicious.append(
                {
                    "work_key": work_key,
                    "book_ids": [r.get("goodreads_book_id") for r in group],
                    "titles": sorted({r["title"] for r in group}),
                    "authors": sorted({r["author"] for r in group}),
                }
            )
        else:
            legitimate += 1

    return {
        "suspicious_duplicate_targets": len(suspicious),
        "legitimate_duplicate_targets": legitimate,
        "suspicious_duplicate_examples": suspicious[:MAX_SUSPICIOUS_EXAMPLES],
    }


def count_gap_fill_candidates(rows: Iterable[dict], *, min_ratings_count: int = DEFAULT_GAP_FILL_MIN_RATINGS_COUNT) -> int:
    """Replicates ``build_ios_en_from_goodreads.gap_fill_unmatched``'s
    insert gate exactly: no ``work_key``, ``match_method`` in
    ``{"unmatched", "ambiguous"}``, ``ratings_count`` at or above the floor,
    and both ``title``/``author`` present."""
    count = 0
    for row in rows:
        if row.get("work_key") or row.get("match_method") not in ("unmatched", "ambiguous"):
            continue
        if (row.get("ratings_count") or 0) < min_ratings_count:
            continue
        if not row.get("title") or not row.get("author"):
            continue
        count += 1
    return count


def match_method_distribution(rows: Iterable[dict]) -> dict[str, int]:
    return dict(Counter(row.get("match_method") for row in rows))


def distribution_delta(baseline: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    keys = set(baseline) | set(current)
    return {k: current.get(k, 0) - baseline.get(k, 0) for k in sorted(keys, key=lambda k: (k is None, k))}


def evaluate_canaries(
    rows: list[dict],
    *,
    baseline: dict[str, object] | None = None,
    min_ratings_count: int = DEFAULT_GAP_FILL_MIN_RATINGS_COUNT,
) -> dict[str, object]:
    report: dict[str, object] = {
        **suspicious_duplicate_targets(rows),
        "gap_fill_candidates": count_gap_fill_candidates(rows, min_ratings_count=min_ratings_count),
        "match_method_distribution": match_method_distribution(rows),
    }
    if baseline:
        report["gap_fill_candidates_delta"] = report["gap_fill_candidates"] - baseline.get("gap_fill_candidates", 0)
        report["match_method_distribution_delta"] = distribution_delta(
            baseline.get("match_method_distribution", {}), report["match_method_distribution"]
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--matched", type=Path, default=None, help="Default: catalog_goodreads('matched_goodreads.jsonl.gz')"
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="Optional prior eval_canaries.py --out report, for the deltas"
    )
    parser.add_argument(
        "--gap-fill-min-ratings-count", type=int, default=DEFAULT_GAP_FILL_MIN_RATINGS_COUNT
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional: write the report as JSON")
    args = parser.parse_args(argv)

    matched_path = args.matched or catalog_goodreads("matched_goodreads.jsonl.gz")
    rows = load_matched_rows(matched_path)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None

    report = evaluate_canaries(rows, baseline=baseline, min_ratings_count=args.gap_fill_min_ratings_count)
    # Examples are for human review only -- keep the printed/console summary
    # short by omitting them; --out keeps the full report for later reading.
    summary = {k: v for k, v in report.items() if k != "suspicious_duplicate_examples"}
    print("[eval_canaries]", json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[eval_canaries] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
