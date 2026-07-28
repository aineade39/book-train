"""Canonical data-root paths for book-spines (see DATA.md)."""

from __future__ import annotations

import os
from pathlib import Path


def book_spines_data() -> Path:
    raw = os.environ.get("BOOK_SPINES_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "ml" / "book-spines"


def raw_dir(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("raw", *parts)


def derived_dir(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("derived", *parts)


def runs_dir(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("runs", *parts)


def models_production(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("models", "production", *parts)


def models_candidates(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("models", "candidates", *parts)


def eval_dir(*parts: str | Path) -> Path:
    return book_spines_data().joinpath("eval", *parts)


def catalog_raw_ol(*parts: str | Path) -> Path:
    """Open Library bulk dumps (see docs/BOOK_CATALOG.md)."""
    return raw_dir("open-library", *parts)


def catalog_dir(*parts: str | Path) -> Path:
    """Derived book-catalog SQLite outputs and intermediate cache."""
    return derived_dir("book-catalog", *parts)


def catalog_intermediate(*parts: str | Path) -> Path:
    return catalog_dir("intermediate", *parts)


def catalog_goodreads(*parts: str | Path) -> Path:
    """Scraped Goodreads Listopia/book-page JSONL + the scrape checkpoint DB.

    Rebuildable from scratch by re-running tools/scrape_goodreads_lists.py
    (checkpointed, so a rebuild only re-fetches what isn't already `done`).
    """
    return catalog_dir("goodreads", *parts)
