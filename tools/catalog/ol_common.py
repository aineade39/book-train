"""Shared Open Library dump parsing for catalog build scripts."""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


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


# Python port of Sources/SpineMatching/Normalization.swift normalizeForSearch.
# Keep the two in sync; see that file's header comment for the matching
# rationale (fuzzy rerank tolerates variation, this only strips *decorative*
# punctuation). Used here so Goodreads title/author strings are normalized
# identically to how the iOS app normalizes catalog rows and OCR queries,
# which is what match_goodreads.py's fuzzy fallback relies on.
#
# Caveat: Swift's `.folding(options: [.caseInsensitive, .diacriticInsensitive])`
# is ICU-backed and not guaranteed bit-for-bit identical to this NFKD +
# combining-mark-strip + lowercase approximation for every Unicode edge case
# (rare scripts, some ligatures). Matches for all common Latin-script
# title/author text, which is effectively all of this catalog.
_DECORATIVE_PUNCTUATION = frozenset(
    '"\u201c\u201d'
    "()[]{}"
    "!?;:,"
    "*#@\u2013\u2014/\\_~`^|<>=+"
    # "." joins "/" and "\\" above (2j-5b): GR and OL disagree on the
    # separator in date-shaped titles -- "11.22.63" (GR) vs "11/22/63" (OL,
    # 49 editions, confirmed via direct full.sqlite lookup) never collided
    # because "/" was already dropped here but "." wasn't. Checked against
    # author-initial strings ("J.R.R. Tolkien" -> already normalized via a
    # separate path, `split_people`, which tokenizes on whitespace after
    # this function runs -- dropping "." here just removes the now-redundant
    # punctuation between initials, still leaving a same-tokens result) and
    # abbreviations ("U.S.A." -> "usa", "Mr. Smith" -> "mr smith" -- both
    # already had a following space so dropping "." doesn't fuse words).
    "."
)

# U+2018/U+2019 (curly single quotes) are ambiguous: sometimes a scare-quote
# pair (drop, like the double-quote curly forms above), but in practice
# overwhelmingly an apostrophe -- "Assassin's", "She's", "O'Brien" -- almost
# always typed/rendered as U+2019 in scraped web text (Goodreads) while OL's
# own catalog strings mix straight and curly. Stripping them (the previous
# behavior, matching the double-quote curlies) silently diverged from the
# straight apostrophe U+0027, which `_DECORATIVE_PUNCTUATION` has always
# deliberately kept (see `test_keeps_meaningful_marks` / Swift's
# `testStripsDecorativePunctuationButKeepsMeaningfulMarks` -- apostrophes are
# meaningful, not decorative): "Assassin's Blade" and "Assassin's Blade" (one
# straight, one curly -- otherwise byte-identical) normalized to two
# different strings, so `titleNormalized`/`title_core` lookups silently
# missed a real OL row for any GR title using the glyph the OL row didn't.
# Folding both curly forms to the straight apostrophe first (not stripping
# either) fixes the mismatch while keeping the "apostrophes are meaningful"
# property for both glyphs equally.
_CURLY_APOSTROPHE_TO_STRAIGHT = {"\u2018": "'", "\u2019": "'"}

# 2j-5c: GR and OL disagree on "&" vs "and" in the same title in at least one
# observed pair (Carissa Broadbent, "The Serpent & the Wings of Night" (OL,
# 14 editions) / "...and the Wings of Night" (GR)) -- a token substitution,
# not a single-character fold like the apostrophe case above, so it can't
# reuse that dict (one input char -> multiple output chars). Folds
# unconditionally in both directions' favor (GR and OL each use both forms
# somewhere in their own data) by always expanding to the spelled-out form;
# existing surrounding whitespace (almost always present around "&" in real
# titles) is untouched, so spacing comes out identical to a native "and".
# `split_people` (author name splitting) already treats "&" as an
# and-conjunction on the *raw*, pre-normalization string, so this doesn't
# change author-splitting behavior -- just makes the normalized text of an
# author string that happens to literally contain "&" consistent with that
# existing convention.


def _fold_case_and_diacritics(raw: str) -> str:
    decomposed = unicodedata.normalize("NFKD", raw)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.lower()


def normalize_for_search(raw: str) -> str:
    """Lowercase, diacritic-fold, collapse whitespace, strip decorative
    punctuation and invisible Unicode format characters, fold curly
    apostrophes to straight and "&" to "and"."""
    folded = _fold_case_and_diacritics(raw)

    out_chars: list[str] = []
    last_was_space = False
    for ch in folded:
        if ch.isspace():
            if not last_was_space and out_chars:
                out_chars.append(" ")
            last_was_space = True
            continue
        last_was_space = False
        # 2j-5a: Unicode category "Cf" ("format") covers zero-width space
        # (U+200B), zero-width non/joiner (U+200C/U+200D), the BOM
        # (U+FEFF), and similar invisible characters that render as
        # nothing but are neither whitespace (`isspace()` is False for
        # these) nor in the fixed decorative-punctuation set above --
        # confirmed via a real GR title, "The \u200bCrown of Gilded Bones",
        # where a stray ZWSP after the real space before "Crown" silently
        # broke the match. Dropped like decorative punctuation (not
        # replaced with a space) since these characters are never a
        # word-separator stand-in themselves -- see the ZWSP case above,
        # where the real separating space was already a distinct character.
        if ch in _DECORATIVE_PUNCTUATION or unicodedata.category(ch) == "Cf":
            continue
        if ch == "&":
            out_chars.append("and")
            continue
        ch = _CURLY_APOSTROPHE_TO_STRAIGHT.get(ch, ch)
        out_chars.append(ch)

    result = "".join(out_chars)
    if result.endswith(" "):
        result = result[:-1]
    return result


def search_tokens(normalized: str) -> list[str]:
    """Whitespace-separated tokens of an already-normalized string."""
    return normalized.split(" ") if normalized else []


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


def author_keys_from_work(row: dict[str, Any]) -> list[str]:
    """Every author key credited on a work, in OL's listed order (first =
    primary). OL works can credit multiple authors (co-authored books);
    the old `author_key_from_work` kept only `authors[0]`, silently
    dropping every co-author's name from `authorNormalized` and losing
    real match signal for that slice of the catalog -- see the "Catalog
    match quality" plan's co-author fix."""
    keys: list[str] = []
    for entry in row.get("authors") or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str):
            keys.append(key)
            continue
        nested = entry.get("author")
        if isinstance(nested, dict) and isinstance(nested.get("key"), str):
            keys.append(nested["key"])
    return keys


def join_author_names(names: Sequence[str]) -> str:
    """Joins multiple author display names into one string, preserving OL
    order (first = primary). Any reasonable multi-author punctuation works
    here -- `CustomWordsBuilder.individualWords` (Swift) splits a display
    author string on `,`/`&`/` and ` regardless of exactly how it was
    joined, so this only needs to read naturally, not round-trip exactly."""
    cleaned = [n.strip() for n in names if n and n.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


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
        author_keys = author_keys_from_work(row)
        author = join_author_names([author_names[k] for k in author_keys if k in author_names])
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
