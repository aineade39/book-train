#!/usr/bin/env python3
"""Ranking eval for the GR<->OL popularity signal (Phase 2e).

Linkage (does a book attach to the right OL work?) is necessary but not
sufficient: the on-device catalog's shipped 50k slice is decided by *rank*,
via ``compute_shelf_score`` (``match_goodreads.py``) and
``rerank_popularity`` (``build_ios_en_from_goodreads.py``). Nothing measured
that until this script -- see the plan's Goals section
(``.cursor/plans/iterative_gr-ol_matcher_loop_dfcdd41a.plan.md``).

Reference signal: ``book_show_api``'s own ``ratings_count`` -- the
authoritative Goodreads popularity number, recovered for ~40k books by
``tools/catalog/consolidate_popularity_signals.py`` (successes plus the
``incomplete_record`` rows). This is "the strongest available signal" the
goal refers to; ``list_show``'s ``ratings_count`` (what ``shelf_score``
itself is partly built from) is a page-scraped approximation of the same
number, not an independent check.

Inputs:
- ``matched_goodreads.jsonl.gz`` (``match_goodreads.py``'s output):
  ``goodreads_book_id``, ``shelf_score``.
- ``consolidated_signals.jsonl.gz`` (``consolidate_popularity_signals.py``'s
  output): ``api_ratings_count``.

Only books present in both, with ``api_ratings_count > 0``, are scored.

Same fold discipline as ``bibliographic_join.fold_for``: any weight tuning
should score against the tuning fold; the validation fold is for a final,
un-tuned-against check only -- otherwise weight tuning Goodharts this
correlation number the same way repeated matcher tuning Goodharts recall.

Usage:
    python tools/catalog/eval_popularity_ranking.py \\
        --matched <catalog_goodreads('matched_goodreads.jsonl.gz')> \\
        --consolidated-signals <catalog_goodreads('consolidated_signals.jsonl.gz')> \\
        [--fold tuning|validation|all] [--out report.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bibliographic_join import fold_for  # noqa: E402
from tools.catalog.consolidate_popularity_signals import load_consolidated_signals  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_TOP_KS = (1_000, 10_000, 50_000)
DEFAULT_WORST_DISPLACEMENTS = 20


def load_shelf_scores(path: Path) -> dict[int, float]:
    """Reads ``goodreads_book_id`` -> ``shelf_score`` from a
    ``match_goodreads.py`` output file."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            book_id = row.get("goodreads_book_id")
            shelf_score = row.get("shelf_score")
            if book_id is not None and shelf_score is not None:
                out[int(book_id)] = float(shelf_score)
    return out


def load_reference_ratings_counts(consolidated_signals_path: Path) -> dict[int, int]:
    """Reads ``api_ratings_count`` (the reference signal) from a
    ``consolidate_popularity_signals.py`` output file."""
    signals = load_consolidated_signals(consolidated_signals_path)
    return {book_id: sig.api_ratings_count for book_id, sig in signals.items() if sig.api_ratings_count}


def _fractional_ranks(values: list[float]) -> list[float]:
    """Standard Spearman tie handling: tied values share the mean of the
    ranks they'd occupy (1-indexed)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation with no scipy dependency: Pearson
    correlation of fractional ranks, computed via numpy (already a project
    dependency)."""
    x = list(x)
    y = list(y)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = np.asarray(_fractional_ranks(x), dtype=float)
    ry = np.asarray(_fractional_ranks(y), dtype=float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def top_k_overlap(ranked_a: list[int], ranked_b: list[int], k: int) -> float:
    """Fraction of ``ranked_b``'s top-``k`` that ``ranked_a``'s top-``k``
    also contains -- 1.0 means the two top-``k`` sets are identical."""
    a = set(ranked_a[:k])
    b = set(ranked_b[:k])
    if not b:
        return 0.0
    return len(a & b) / len(b)


def worst_rank_displacements(
    book_ids: list[int],
    shelf_rank: dict[int, int],
    reference_rank: dict[int, int],
    limit: int = DEFAULT_WORST_DISPLACEMENTS,
) -> list[dict[str, object]]:
    displacements = [(abs(shelf_rank[b] - reference_rank[b]), b) for b in book_ids]
    displacements.sort(reverse=True)
    return [
        {
            "book_id": book_id,
            "shelf_rank": shelf_rank[book_id],
            "reference_rank": reference_rank[book_id],
            "displacement": displacement,
        }
        for displacement, book_id in displacements[:limit]
    ]


def evaluate_ranking(
    shelf_scores: dict[int, float],
    reference_ratings_counts: dict[int, int],
    *,
    fold: str = "all",
    fold_salt: str = "gr-ol-v1",
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    worst_displacements_limit: int = DEFAULT_WORST_DISPLACEMENTS,
) -> dict[str, object]:
    book_ids = sorted(set(shelf_scores) & set(reference_ratings_counts))
    if fold != "all":
        book_ids = [b for b in book_ids if fold_for(b, fold_salt) == fold]

    n = len(book_ids)
    if n < 2:
        return {
            "n": n,
            "fold": fold,
            "spearman": 0.0,
            "top_k_overlap": {f"top_{k}": 0.0 for k in top_ks},
            "worst_displacements": [],
        }

    shelf_vals = [shelf_scores[b] for b in book_ids]
    ref_vals = [float(reference_ratings_counts[b]) for b in book_ids]
    spearman = spearman_correlation(shelf_vals, ref_vals)

    by_shelf = sorted(book_ids, key=lambda b: (-shelf_scores[b], b))
    by_reference = sorted(book_ids, key=lambda b: (-reference_ratings_counts[b], b))
    shelf_rank = {b: i + 1 for i, b in enumerate(by_shelf)}
    reference_rank = {b: i + 1 for i, b in enumerate(by_reference)}

    return {
        "n": n,
        "fold": fold,
        "spearman": spearman,
        "top_k_overlap": {f"top_{k}": top_k_overlap(by_shelf, by_reference, k) for k in top_ks},
        "worst_displacements": worst_rank_displacements(
            book_ids, shelf_rank, reference_rank, limit=worst_displacements_limit
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--matched", type=Path, default=None, help="Default: catalog_goodreads('matched_goodreads.jsonl.gz')"
    )
    parser.add_argument(
        "--consolidated-signals",
        type=Path,
        default=None,
        help="Default: catalog_goodreads('consolidated_signals.jsonl.gz')",
    )
    parser.add_argument("--fold", choices=("all", "tuning", "validation"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="Optional: write the report as JSON")
    args = parser.parse_args(argv)

    matched_path = args.matched or catalog_goodreads("matched_goodreads.jsonl.gz")
    signals_path = args.consolidated_signals or catalog_goodreads("consolidated_signals.jsonl.gz")

    shelf_scores = load_shelf_scores(matched_path)
    reference = load_reference_ratings_counts(signals_path)
    report = evaluate_ranking(shelf_scores, reference, fold=args.fold)

    print("[eval_popularity_ranking]", json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[eval_popularity_ranking] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
