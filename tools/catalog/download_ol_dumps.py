#!/usr/bin/env python3
"""Download Open Library bulk dumps into $BOOK_SPINES_DATA/raw/open-library/."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.derived_meta import write_derived_source  # noqa: E402
from tools.paths import catalog_raw_ol  # noqa: E402

# Stable "latest" aliases on openlibrary.org/data/
DUMP_URLS = {
    "editions": "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz",
    "works": "https://openlibrary.org/data/ol_dump_works_latest.txt.gz",
    "authors": "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz",
}

LOCAL_NAMES = {
    "editions": "ol_dump_editions_latest.txt.gz",
    "works": "ol_dump_works_latest.txt.gz",
    "authors": "ol_dump_authors_latest.txt.gz",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "book-train-catalog/1.0"})
    if dest.exists():
        req.add_header("If-Modified-Since", datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
        print(f"Downloaded {dest.name} ({dest.stat().st_size} bytes)", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and dest.exists():
            print(f"Not modified: {dest.name}", file=sys.stderr)
            if tmp.exists():
                tmp.unlink()
            return
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or catalog_raw_ol()
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for key, url in DUMP_URLS.items():
        dest = out_dir / LOCAL_NAMES[key]
        download(url, dest)
        hashes[key] = sha256_file(dest)

    write_derived_source(
        out_dir,
        derived_id="open-library",
        title="Open Library bulk dumps",
        sources=[url for url in DUMP_URLS.values()],
        script="tools/catalog/download_ol_dumps.py",
        flags={"out_dir": str(out_dir), **{f"sha256_{k}": v[:16] for k, v in hashes.items()}},
        notes="Raw CC0 dumps for catalog build. Rebuildable.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
