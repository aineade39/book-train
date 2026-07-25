#!/usr/bin/env python3
"""Stream Open Library dumps → title,author,workKey CSV (one row per work).

See docs/BOOK_CATALOG.md §Phase 1.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import build_work_rows, filter_work_rows  # noqa: E402
from tools.paths import catalog_raw_ol  # noqa: E402

DEFAULT_DUMPS = {
    "editions": "ol_dump_editions_latest.txt.gz",
    "works": "ol_dump_works_latest.txt.gz",
    "authors": "ol_dump_authors_latest.txt.gz",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--raw-dir", type=Path, default=None, help="OL dump directory")
    parser.add_argument("--editions", type=Path, default=None)
    parser.add_argument("--works", type=Path, default=None)
    parser.add_argument("--authors", type=Path, default=None)
    parser.add_argument("--languages", nargs="*", default=["eng"], help="Edition language filter")
    parser.add_argument("--max-works", type=int, default=250_000)
    parser.add_argument("--min-editions", type=int, default=2)
    args = parser.parse_args()

    raw = args.raw_dir or catalog_raw_ol()
    editions = args.editions or raw / DEFAULT_DUMPS["editions"]
    works = args.works or raw / DEFAULT_DUMPS["works"]
    authors = args.authors or raw / DEFAULT_DUMPS["authors"]
    for path in (editions, works, authors):
        if not path.exists():
            print(f"Missing dump: {path}", file=sys.stderr)
            return 1

    all_rows = build_work_rows(
        editions_path=editions,
        works_path=works,
        authors_path=authors,
        min_editions=args.min_editions,
    )
    rows = filter_work_rows(
        all_rows,
        languages=args.languages,
        max_works=args.max_works,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["title", "author", "isbn", "workKey"])
        for row in rows:
            writer.writerow([row.title, row.author, row.isbn13 or "", row.work_key])

    print(f"Wrote {len(rows)} works to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
