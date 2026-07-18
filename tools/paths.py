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
