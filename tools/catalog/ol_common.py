"""Shared Open Library dump parsing for catalog build scripts."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def ol_key_tail(key: str | None) -> str | None:
    if not key:
        return None
    return key.rsplit("/", 1)[-1]


def normalize_language(code: str | None) -> str | None:
    if not code:
        return None
    code = code.strip().lower()
    if code.startswith("/languages/"):
        code = code.rsplit("/", 1)[-1]
    return code or None


def isbn10_checksum(digits: str) -> bool:
    if len(digits) != 10:
        return False
    total = 0
    for i, ch in enumerate(digits):
        if i == 9 and ch == "X":
            val = 10
        elif ch.isdigit():
            val = int(ch)
        else:
            return False
        total += (10 - i) * val
    return total % 11 == 0


def isbn13_checksum(digits: str) -> bool:
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
    return (10 - (total % 10)) % 10 == int(digits[12])


def isbn10_to_13(digits10: str) -> str | None:
    if not isbn10_checksum(digits10):
        return None
    core = "978" + digits10[:9]
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
    check = (10 - (total % 10)) % 10
    return core + str(check)


def normalize_isbn13(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"[^0-9Xx]", "", raw).upper()
    if len(digits) == 13 and isbn13_checksum(digits):
        return digits
    if len(digits) == 10:
        return isbn10_to_13(digits)
    return None


def parse_ol_record(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        return json.loads(line)
    parts = line.split("\t", 4)
    if len(parts) < 5:
        return None
    rec_type, key, _revision, _timestamp, json_blob = parts
    data = json.loads(json_blob)
    if "key" not in data:
        data["key"] = key
    data["_ol_type"] = rec_type
    return data


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".jsonl.gz") or path.name.endswith(".txt.gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            try:
                record = parse_ol_record(line)
            except json.JSONDecodeError:
                continue
            if record is not None:
                yield record


def write_jsonl_gz(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            count += 1
    return count


@dataclass
class EditionAgg:
    edition_count: int = 0
    languages: set[str] = field(default_factory=set)
    isbns: set[str] = field(default_factory=set)


@dataclass
class WorkRow:
    work_key: str
    title: str
    author: str
    edition_count: int
    languages: set[str]
    isbn13: str | None
    popularity_rank: int = 0


def author_key_from_work(row: dict[str, Any]) -> str | None:
    authors = row.get("authors") or []
    if not authors or not isinstance(authors[0], dict):
        return None
    first = authors[0]
    if isinstance(first.get("key"), str):
        return first["key"]
    nested = first.get("author")
    if isinstance(nested, dict) and isinstance(nested.get("key"), str):
        return nested["key"]
    return None


def load_authors(authors_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in stream_jsonl(authors_path):
        key = row.get("key")
        name = row.get("name")
        if isinstance(key, str) and isinstance(name, str) and name.strip():
            names[key] = name.strip()
    return names


def aggregate_editions(editions_path: Path) -> dict[str, EditionAgg]:
    by_work: dict[str, EditionAgg] = defaultdict(EditionAgg)
    for row in stream_jsonl(editions_path):
        works = row.get("works") or []
        work_key = None
        if works and isinstance(works[0], dict):
            work_key = works[0].get("key")
        if not isinstance(work_key, str) or not work_key.startswith("/works/"):
            continue
        agg = by_work[work_key]
        agg.edition_count += 1
        for lang in row.get("languages") or []:
            if isinstance(lang, dict):
                code = normalize_language(lang.get("key"))
            else:
                code = normalize_language(str(lang))
            if code:
                agg.languages.add(code)
        for raw in (row.get("isbn_13") or []) + (row.get("isbn_10") or []):
            if not isinstance(raw, str):
                continue
            isbn = normalize_isbn13(raw)
            if isbn:
                agg.isbns.add(isbn)
    return dict(by_work)


def build_work_rows(
    *,
    editions_path: Path,
    works_path: Path,
    authors_path: Path,
    min_editions: int = 1,
) -> list[WorkRow]:
    edition_agg = aggregate_editions(editions_path)
    author_names = load_authors(authors_path)
    rows: list[WorkRow] = []

    for row in stream_jsonl(works_path):
        work_key = row.get("key")
        if not isinstance(work_key, str) or not work_key.startswith("/works/"):
            continue
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        author_key = author_key_from_work(row)
        author = author_names.get(author_key or "", "").strip()
        if not author:
            continue
        agg = edition_agg.get(work_key)
        if not agg or agg.edition_count < min_editions:
            continue
        if not agg.isbns and agg.edition_count < 1:
            continue
        isbn13 = sorted(agg.isbns)[0] if agg.isbns else None
        rows.append(
            WorkRow(
                work_key=work_key,
                title=title.strip(),
                author=author,
                edition_count=agg.edition_count,
                languages=set(agg.languages),
                isbn13=isbn13,
            )
        )

    rows.sort(key=lambda w: (-w.edition_count, -len(w.isbn13 or ""), w.work_key))
    for rank, work in enumerate(rows, start=1):
        work.popularity_rank = rank
    return rows


def filter_work_rows(
    rows: list[WorkRow],
    *,
    languages: list[str] | None,
    max_works: int | None,
) -> list[WorkRow]:
    langs = {normalize_language(x) for x in (languages or []) if normalize_language(x)}
    filtered: list[WorkRow] = []
    for row in rows:
        if langs and not (row.languages & langs):
            continue
        filtered.append(row)
    if max_works is not None:
        filtered = filtered[:max_works]
    return filtered


def load_profiles(profiles_path: Path) -> dict[str, dict[str, Any]]:
    json_path = profiles_path.with_suffix(".json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data["profiles"]
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Install PyYAML or use {json_path.name} next to {profiles_path.name}"
        ) from exc
    data = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    return data["profiles"]
