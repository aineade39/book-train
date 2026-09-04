"""Title+author join of Goodreads list rows to Open Library works.

ISBN is a separate, later overlay (see ``match_goodreads.match_book``). This
module is the non-ISBN bibliographic matcher: it uses only title and author
from Listopia ``list_show`` plus the OL ``books`` table.

Why this exists
---------------
The previous GR↔OL join reused the on-device OCR accept policy
(``token_set_ratio``, 90/8). That scorer is correct for a mashed spine blob
against a clean catalog title. It is the wrong scorer for two clean titles:

- ``token_set_ratio`` saturates at 100 when one title's tokens are a subset
  of the other, so "The Hunger Games" ties its own companion/guide at
  ceiling and the margin test fails. The current matched file marks the
  most popular Goodreads books ``ambiguous`` for this reason.
- Author blocking on the *full* compact author string misses the OL
  data-quality pattern where translators/illustrators are credited as
  co-authors ("Kathryn Stockett, Álvaro Abella Villar, and …") and the
  catalog-inverted form ("Fitch, Janet").

Publisher is the standard third bibliographic key. ``list_show`` does not
carry it and ``full.sqlite``'s ``books`` table does not store it, so it
cannot participate in this join. Author last-name compatibility is the
disambiguator for same-title different-works (Meyer vs Koontz *Twilight*).
Edition count is the tie-break among OL duplicate works of the same
identity (the ~9% identical ``(titleNormalized, authorNormalized)``
pattern documented in ``docs/BOOK_CATALOG.md``).

Evaluation
----------
Experts evaluate a bibliographic linker against a high-confidence
identifier holdout, not against itself. Harvested GR ISBNs that hit
``book_isbns`` are that gold set: run this matcher *without* ISBN and
score recall / conflict. See ``evaluate_title_author_against_isbn``.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz

from tools.catalog.ol_common import normalize_for_search

# Identity / strong-accept. token_sort_ratio, not token_set_ratio — length-
# sensitive, so a short title cannot tie a longer companion at 100.
TITLE_ACCEPT = 90.0
TITLE_MARGIN = 5.0

_SERIES_SUFFIX_RE = re.compile(r"\s*\([^()]*#[^())]*\)?\s*$")
_AUTHOR_SPLIT_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
_AUTHOR_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"})
_LEADING_ARTICLES = frozenset({"the", "a", "an"})
_TITLE_BLOCK_STOP = frozenset(
    {"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with", "from", "by", "at"}
)
_NAME_STOP = frozenset({"the", "a", "an", "and", "or", "von", "van", "de", "del", "da", "di", "la", "le"})

_ONES = (
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def strip_series_suffix(title: str) -> str:
    """Drop a trailing series annotation: Listopia's "Title (Series, #1)"
    convention, or free-text "Title (Series Name Book 1)" / "Title (Book 1)"
    annotations that use a volume keyword instead of "#" (2j-5d -- OL itself
    uses this convention on some works, e.g. "The Heir (The Selection #4)"
    already stripped fine via the ``#`` path below, but "Women on Top 2
    (The Dud Wimpole Saga Book 1)" has no ``#`` at all and silently kept its
    full parenthetical, which never matched GR's shorter title).

    Also strips a truncated unclosed paren that still contains ``#``, which
    shows up when list_show titles are cut off mid-annotation.
    """
    stripped = _SERIES_SUFFIX_RE.sub("", title)
    if stripped == title:
        stripped = _SERIES_SUFFIX_FREE_TEXT_RE.sub("", title)
    return stripped.strip()


def number_to_words(n: int) -> str:
    """Year-style wording: 1984 -> 'nineteen eighty four'."""
    if n < 0:
        return ""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return f"{_TENS[tens]} {_ONES[ones]}".strip()
    if 1100 <= n <= 1999:
        return f"{number_to_words(n // 100)} {number_to_words(n % 100)}".strip()
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest:
            return f"{_ONES[hundreds]} hundred {number_to_words(rest)}".strip()
        return f"{_ONES[hundreds]} hundred"
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        if rest:
            return f"{number_to_words(thousands)} thousand {number_to_words(rest)}".strip()
        return f"{number_to_words(thousands)} thousand"
    return str(n)


_WORD_TO_SMALL_NUMBER = {word: n for n, word in enumerate(_ONES) if word} | {
    word: 10 * n for n, word in enumerate(_TENS) if word
}


_ROMAN_NUMERALS = (
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
)
_ROMAN_TO_NUMBER = {r: n + 1 for n, r in enumerate(_ROMAN_NUMERALS)}

# 2j-5d: the free-text half of `strip_series_suffix` -- requires the token
# right after the volume keyword to actually *be* a number (digit,
# single-word spelled-out cardinal, or roman numeral I-XV), not just any
# word. Without that constraint, a trailing paren like "(Book of the
# Year)" would false-positive-match on "book" + the following word "of"
# and get wrongly stripped as if it were a volume annotation.
_NUMBER_WORD_ALTERNATION = "|".join(
    sorted({w for w in _ONES if w} | {w for w in _TENS if w} | set(_ROMAN_NUMERALS), key=len, reverse=True)
)
_SERIES_SUFFIX_FREE_TEXT_RE = re.compile(
    rf"\s*\([^()]*?\b(?:book|volume|vol\.?|part|bk\.?)\s+(?:\d+|{_NUMBER_WORD_ALTERNATION})\b[^()]*\)\s*$",
    re.IGNORECASE,
)


def _standalone_numbers(core: str) -> set[int]:
    """Small cardinal numbers (digit, spelled-out, or roman numeral)
    appearing as their own token in a title core -- "volume 2", "part
    three", "volume ii" -- not digits embedded in a larger token or
    year-scale numbers (see `titles_identity` for whole-title year matching
    like "1984"). Includes bare "i" as roman numeral 1 despite the
    collision with the pronoun "I": a title core is a noun phrase, not a
    sentence, so a standalone "i" here is almost always "Volume I", and the
    failure mode this guards against (two different specific volumes/parts
    silently merging) is worse than the rare false rejection of two
    genuinely-different, unrelated titles that each happen to contain a
    solo "i"/other number -- rejecting a match neither of those would have
    likely passed on other grounds anyway.
    """
    out: set[int] = set()
    for tok in core.split():
        if tok.isdigit() and 1 <= int(tok) <= 99:
            out.add(int(tok))
        elif tok in _WORD_TO_SMALL_NUMBER:
            out.add(_WORD_TO_SMALL_NUMBER[tok])
        elif tok in _ROMAN_TO_NUMBER:
            out.add(_ROMAN_TO_NUMBER[tok])
    return out


_VOLUME_KEYWORDS = frozenset({"volume", "vol", "part", "book", "chapter", "bk"})


def _has_volume_marker(text: str) -> bool:
    """True when `text` (typically a discarded colon-subtitle tail) names a
    specific volume/part -- a keyword ("volume", "part", ...) or a small
    standalone number/spelled-number/roman-numeral ("Volume II" -> caught
    by the keyword and the numeral; "Year One" -> caught by the number)."""
    normalized = normalize_for_search(text).replace("-", " ")
    tokens = normalized.split()
    if _VOLUME_KEYWORDS.intersection(tokens):
        return True
    return bool(_standalone_numbers(normalized))


def title_core(title: str) -> str:
    """Search-normalized title with series suffix, subtitle, and leading article removed.

    The colon check must run on `raw` -- before `normalize_for_search` --
    because that normalizer strips ":" as decorative punctuation. Checking
    on the already-normalized string (the previous bug here) means the
    split never fires for any colon-subtitle title ("Atomic Habits: An Easy
    & Proven Way..."), which is the single most common subtitle convention:
    the GR title keeps the subtitle, OL's own title often doesn't, so the
    two never became `titles_identity`-equal and the SQL title-probe (whose
    keys come from this function) never even pulled OL's short-titled
    record into the candidate pool -- silently `unmatched`, not `ambiguous`.
    The " or " check (for "Title: Or, Alternate Subtitle") stays on the
    normalized string deliberately: without the comma stripped first, " or"
    is followed by "," not a space, so it wouldn't match on raw text either.

    The head must have >= 2 words to be used. A colon is also the
    "Brand: Specific Title" convention ("Batman: Knightfall, Part Three:
    Knightsend", "Ghostbusters, Volume 4: Who Ya Gonna Call") -- common for
    comics/franchise tie-ins -- where the head is a single-word franchise
    name, not a complete title, and the *distinguishing* content (which
    specific volume/story) is what comes after the colon. A single-word
    head is too generic to be a safe identity key on its own (it would
    exact-match every volume/story sharing that franchise name against
    whatever OL work happens to be titled just that word); requiring >= 2
    words filters out "Batman" while keeping "Atomic Habits", "Guns, Germs,
    and Steel", etc. Titles with a genuinely single-word *main* title and a
    subtitle ("Sapiens: A Brief History of Humankind") fall back to the
    unstripped core here -- an accepted, narrower gap versus the false-merge
    risk of stripping them.

    Also don't strip when the discarded tail itself carries a volume/part
    marker -- "The Chronicles of Amber: Volume II (...#3-5)" has a
    substantial 4-word head ("The Chronicles of Amber") that passes the
    check above, but "Volume II" is the *only* thing distinguishing it from
    "Volume I": stripping it collapses both onto the same bare series title
    (and whatever generic, volume-less OL work happens to be titled that).
    """
    raw = strip_series_suffix(title)
    if ":" in raw:
        head, _, tail = raw.partition(":")
        head = head.strip()
        if head and len(head.split()) >= 2 and not _has_volume_marker(tail):
            raw = head
    normalized = normalize_for_search(raw).replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if " or " in normalized:
        head = normalized.split(" or ", 1)[0].strip()
        if head:
            normalized = head
    tokens = normalized.split()
    while tokens and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def title_lookup_keys(title: str) -> list[str]:
    """Exact ``titleNormalized`` values to probe in OL (indexed equality).

    Includes the raw normalized title (what ``books.titleNormalized`` stores),
    the article-stripped core, the core with ``the``/``a`` restored, and
    year-word aliases so ``1984`` finds ``Nineteen Eighty-Four``.
    """
    raw = strip_series_suffix(title)
    stored = normalize_for_search(raw)
    core = title_core(title)
    keys: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)

    add(stored)
    add(core)
    add("the " + core)
    add("a " + core)
    add(core.replace(" ", "-"))
    if core.isdigit() and 1 <= int(core) <= 2100:
        words = number_to_words(int(core))
        add(words)
        add(words.replace(" ", "-"))
        # 1984 → nineteen eighty-four (OL keeps the hyphen in normalize_for_search)
        parts = words.split()
        if len(parts) >= 2:
            add(f"{parts[0]} {parts[1]}-" + " ".join(parts[2:]) if len(parts) > 2 else f"{parts[0]} {parts[1]}")
            if len(parts) == 3:
                add(f"{parts[0]} {parts[1]}-{parts[2]}")
    return keys


def title_prefix_query(title: str) -> str | None:
    """Indexed prefix for a fuzzy fallback when exact title probes miss.

    Only used when the prefix is long enough to stay selective
    (``hunger games``, not ``help``).
    """
    tokens = title_block_tokens(title_core(title))
    if len(tokens) >= 2:
        prefix = " ".join(tokens[:2])
        return prefix if len(prefix) >= 8 else None
    if tokens and len(tokens[0]) >= 8:
        return tokens[0]
    return None


def title_block_tokens(core: str) -> list[str]:
    tokens = [t for t in core.split() if t not in _TITLE_BLOCK_STOP and len(t) > 1]
    return tokens[:2]


def titles_identity(left: str, right: str) -> bool:
    """True when two title cores name the same work, including 1984 ↔ nineteen eighty four."""
    a = re.sub(r"\s+", " ", left.replace("-", " ")).strip()
    b = re.sub(r"\s+", " ", right.replace("-", " ")).strip()
    if a == b:
        return True
    if a.isdigit() and 1 <= int(a) <= 2100 and number_to_words(int(a)) == b:
        return True
    if b.isdigit() and 1 <= int(b) <= 2100 and number_to_words(int(b)) == a:
        return True
    return False


def title_score(gr_core: str, ol_core: str) -> float:
    if titles_identity(gr_core, ol_core):
        return 100.0
    gr_numbers = _standalone_numbers(gr_core)
    ol_numbers = _standalone_numbers(ol_core)
    if gr_numbers and ol_numbers and gr_numbers.isdisjoint(ol_numbers):
        # "Volume 2" vs "Volume 3", "Part Two" vs "Part Three": a lexically
        # near-identical title with a *different* small standalone number is
        # a different specific installment, not a near-miss of the same
        # one -- token_sort_ratio treats one differing digit among many
        # shared tokens as a trivial edit and scores it high regardless.
        return 0.0
    return float(fuzz.token_sort_ratio(gr_core, ol_core))


@dataclass(frozen=True, slots=True)
class PersonName:
    display: str
    tokens: tuple[str, ...]
    family: str
    inverted: bool


def _significant_tokens(normalized: str) -> list[str]:
    out: list[str] = []
    for tok in normalized.split():
        if tok in _AUTHOR_SUFFIXES or tok in _NAME_STOP:
            continue
        if len(tok) == 1:
            continue
        out.append(tok)
    return out


def parse_inverted_person(chunk: str) -> PersonName | None:
    """'Fitch, Janet' / 'Tolkien, J. R. R.' → family=fitch/tolkien."""
    if "," not in chunk:
        return None
    last, first = chunk.split(",", 1)
    last_norm = normalize_for_search(last)
    first_norm = normalize_for_search(first)
    family_tokens = _significant_tokens(last_norm)
    if len(family_tokens) != 1:
        return None
    given = _significant_tokens(first_norm)
    tokens = tuple(family_tokens + given)
    return PersonName(display=chunk.strip(), tokens=tokens, family=family_tokens[0], inverted=True)


def parse_western_person(chunk: str) -> PersonName | None:
    normalized = normalize_for_search(chunk)
    tokens = tuple(_significant_tokens(normalized))
    if not tokens:
        raw = tuple(t for t in normalized.split() if t and t not in _AUTHOR_SUFFIXES)
        if not raw:
            return None
        tokens = raw
    return PersonName(display=chunk.strip(), tokens=tokens, family=tokens[-1], inverted=False)


def split_people(author: str) -> list[PersonName]:
    """Split a display author string into people.

    Comma rules follow catalog practice, not naive CSV:
    - ``and`` / ``&`` → multi-author, then split remaining commas.
    - two or more commas and no ``and`` → multi-author list.
    - exactly one comma → catalog-inverted ``Last, First``.
    - else → one Western-order person.
    """
    raw = (author or "").strip()
    if not raw:
        return []

    multi = bool(_AUTHOR_SPLIT_AND_RE.search(raw) or "&" in raw)
    if multi:
        chunks = _AUTHOR_SPLIT_AND_RE.split(raw.replace("&", ","))
        people: list[PersonName] = []
        for chunk in chunks:
            for part in chunk.split(","):
                part = part.strip()
                if not part:
                    continue
                person = parse_western_person(part)
                if person:
                    people.append(person)
        return people

    if raw.count(",") >= 2:
        people = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            person = parse_western_person(part)
            if person:
                people.append(person)
        return people

    if raw.count(",") == 1:
        inverted = parse_inverted_person(raw)
        if inverted:
            return [inverted]

    person = parse_western_person(raw)
    return [person] if person else []


def family_block_keys(person: PersonName) -> list[str]:
    """Blocking keys for one person (Fellegi–Sunter family-name blocks).

    Always the Western/inverted family name. Two-token names also emit the
    other token (CJK / order-flipped Latin). Three-or-more-token names emit
    middle tokens of length >= 4 (maiden / unused middle family names).
    """
    keys = [person.family]
    if len(person.tokens) == 2:
        other = person.tokens[0] if person.tokens[1] == person.family else person.tokens[1]
        if other and other not in keys:
            keys.append(other)
    elif len(person.tokens) >= 3:
        for tok in person.tokens[:-1]:
            if len(tok) >= 4 and tok not in keys:
                keys.append(tok)
    return keys


def names_compatible(gr_author: str, ol_author: str) -> bool:
    """Hard author gate. Last-name overlap, 2-token order flip, or maiden containment."""
    gr_people = split_people(gr_author)
    ol_people = split_people(ol_author)
    if not gr_people or not ol_people:
        return False

    gr_lasts = {p.family for p in gr_people if p.family}
    ol_lasts = {p.family for p in ol_people if p.family}
    if gr_lasts & ol_lasts:
        return True

    for g in gr_people:
        if len(g.tokens) == 2:
            gset = set(g.tokens)
            for o in ol_people:
                if len(o.tokens) == 2 and set(o.tokens) == gset:
                    return True

    ol_tokens = {tok for p in ol_people for tok in p.tokens}
    gr_tokens = {tok for p in gr_people for tok in p.tokens}
    if gr_lasts & ol_tokens or ol_lasts & gr_tokens:
        return True
    return False


def author_score(gr_author: str, ol_author: str) -> float:
    if names_compatible(gr_author, ol_author):
        return 100.0
    return float(fuzz.token_sort_ratio(normalize_for_search(gr_author), normalize_for_search(ol_author)))


@dataclass(frozen=True, slots=True)
class OLCandidate:
    work_key: str
    title: str
    author: str
    title_normalized: str
    author_normalized: str
    edition_count: int
    title_core: str = ""
    family_keys: tuple[str, ...] = ()


def enrich_candidate(candidate: OLCandidate) -> OLCandidate:
    core = title_core(candidate.title) or candidate.title_normalized.replace("-", " ")
    people = split_people(candidate.author)
    keys: list[str] = []
    seen: set[str] = set()
    for person in people:
        for key in family_block_keys(person):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return OLCandidate(
        work_key=candidate.work_key,
        title=candidate.title,
        author=candidate.author,
        title_normalized=candidate.title_normalized,
        author_normalized=candidate.author_normalized,
        edition_count=candidate.edition_count,
        title_core=core,
        family_keys=tuple(keys),
    )


def block_keys_for(title: str, author: str) -> list[str]:
    core = title_core(title)
    title_tokens = title_block_tokens(core)
    if core.isdigit() and 1 <= int(core) <= 2100:
        words = number_to_words(int(core)).split()
        if words and words[0] not in title_tokens:
            title_tokens = [*title_tokens, words[0]]
    if not title_tokens:
        title_tokens = [t for t in core.split() if t][:1] or ["_"]
    people = split_people(author)
    families: list[str] = []
    seen: set[str] = set()
    for person in people:
        for key in family_block_keys(person):
            if key not in seen:
                seen.add(key)
                families.append(key)
    if not families:
        compact = re.sub(r"[^a-z0-9]", "", normalize_for_search(author))
        if compact:
            families.append(compact)
    return [f"{fam}|{tok}" for fam in families for tok in title_tokens[:2]]


class TitleAuthorBlockIndex:
    """Fellegi–Sunter block: family-name token × first content title token."""

    def __init__(self, candidates: Iterable[OLCandidate]) -> None:
        self._by_block: dict[str, list[OLCandidate]] = defaultdict(list)
        for raw in candidates:
            cand = raw if raw.title_core else enrich_candidate(raw)
            keys = (
                [f"{fam}|{tok}" for fam in cand.family_keys for tok in (title_block_tokens(cand.title_core) or ["_"])[:2]]
                if cand.family_keys
                else block_keys_for(cand.title, cand.author)
            )
            if cand.title_core.isdigit() and 1 <= int(cand.title_core) <= 2100:
                word = number_to_words(int(cand.title_core)).split()[:1]
                extra = [f"{fam}|{word[0]}" for fam in cand.family_keys if word]
                keys = list(dict.fromkeys([*keys, *extra]))
            for key in keys:
                self._by_block[key].append(cand)

    def candidates_for(self, title: str, author: str) -> list[OLCandidate]:
        seen: dict[str, OLCandidate] = {}
        for key in block_keys_for(title, author):
            for cand in self._by_block.get(key, ()):
                seen[cand.work_key] = cand
        return list(seen.values())


def candidates_from_ol_rows(rows: Iterable[tuple]) -> list[OLCandidate]:
    """``books`` SELECT rows → enriched candidates (shared by SQL + memory paths)."""
    out: list[OLCandidate] = []
    seen: set[str] = set()
    for row in rows:
        work_key = row[0]
        if work_key in seen:
            continue
        seen.add(work_key)
        out.append(
            enrich_candidate(
                OLCandidate(
                    work_key=work_key,
                    title=row[1],
                    author=row[2],
                    title_normalized=row[3],
                    author_normalized=row[4],
                    edition_count=row[5] or 0,
                )
            )
        )
    return out


@dataclass
class TitleAuthorMatch:
    method: str  # "title_author" | "ambiguous" | "unmatched"
    work_key: str | None = None
    score: float | None = None
    margin: float | None = None


def match_title_author(title: str, author: str, index: TitleAuthorBlockIndex) -> TitleAuthorMatch:
    """Join one GR title+author to an OL work. Never reads ISBN."""
    candidates = index.candidates_for(title, author)
    if not candidates:
        return TitleAuthorMatch(method="unmatched")

    gr_core = title_core(title)
    compatible = [c for c in candidates if names_compatible(author, c.author)]
    if not compatible:
        return TitleAuthorMatch(method="unmatched")

    identity = [c for c in compatible if titles_identity(gr_core, c.title_core)]
    if identity:
        winner = max(identity, key=lambda c: (c.edition_count, c.work_key))
        return TitleAuthorMatch(method="title_author", work_key=winner.work_key, score=100.0, margin=None)

    scored: list[tuple[OLCandidate, float]] = []
    for cand in compatible:
        scored.append((cand, title_score(gr_core, cand.title_core)))

    best_per_title: dict[str, tuple[OLCandidate, float]] = {}
    for cand, score in scored:
        existing = best_per_title.get(cand.title_core)
        if existing is None or cand.edition_count > existing[0].edition_count or (
            cand.edition_count == existing[0].edition_count and score > existing[1]
        ):
            best_per_title[cand.title_core] = (cand, score)

    ranked = sorted(best_per_title.values(), key=lambda cs: (cs[1], cs[0].edition_count), reverse=True)
    top_candidate, top_score = ranked[0]
    margin = (top_score - ranked[1][1]) if len(ranked) > 1 else None
    if top_score >= TITLE_ACCEPT and (margin is None or margin >= TITLE_MARGIN):
        blended = 0.7 * top_score + 0.3 * 100.0
        return TitleAuthorMatch(
            method="title_author",
            work_key=top_candidate.work_key,
            score=blended,
            margin=margin,
        )
    return TitleAuthorMatch(method="ambiguous", score=top_score, margin=margin)


# --- Tuning / validation fold split -----------------------------------------
#
# Repeatedly tuning the matcher against the same ISBN-gold set eventually
# fits noise in that specific set (Goodhart's law for record-linkage; see
# Splink's evaluation guide and the Fellegi-Sunter active-learning
# literature). `fold_for` gives every book_id a fixed, reproducible
# tuning/validation assignment with no state to persist: the tuning fold is
# used every iteration, the validation fold only at the start and end of a
# run (see docs/BOOK_CATALOG.md).

_DEFAULT_FOLD_SALT = "gr-ol-v1"
_DEFAULT_VALIDATION_FRACTION = 0.2


def fold_for(book_id: int, salt: str = _DEFAULT_FOLD_SALT, *, validation_fraction: float = _DEFAULT_VALIDATION_FRACTION) -> str:
    """Deterministic ~80/20 tuning/validation split, keyed only on `book_id`.

    A cryptographic hash of `book_id` is uncorrelated with any other
    property of the book (popularity, title, author) by construction, so
    this already avoids the specific pitfall a *naive* split risks here:
    the ISBN-gold set is popularity-ordered by scrape design
    (`extract_remaining_ids.py`'s queue fetches highest-`ratings_count`
    books first), so slicing it by *position* (first 80% / last 20%) would
    skew the validation fold toward obscure tail books. A hash-based split
    has no such position dependence -- see `fold_split_stats` for an
    empirical check that the split stays balanced across popularity
    deciles too.
    """
    digest = hashlib.sha256(f"{salt}:{book_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_fraction else "tuning"


def fold_split_stats(
    ratings_count_by_book_id: dict[int, int], salt: str = _DEFAULT_FOLD_SALT
) -> dict[str, object]:
    """Empirical balance check for `fold_for`: buckets `book_id`s into
    `list_ratings_count` deciles and reports the validation-fold fraction
    per decile. Diagnostic only -- `fold_for` needs no popularity input to
    already be balanced; this just proves it, since it is the exact
    property a naive positional split would have violated.
    """
    if not ratings_count_by_book_id:
        return {"deciles": [], "overall_validation_fraction": 0.0}

    ordered = sorted(ratings_count_by_book_id.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(ordered)
    decile_size = max(1, -(-n // 10))  # ceil(n / 10)

    deciles: list[dict[str, object]] = []
    total_validation = 0
    for decile_idx in range(0, n, decile_size):
        chunk = ordered[decile_idx : decile_idx + decile_size]
        folds = [fold_for(book_id, salt) for book_id, _ in chunk]
        validation_count = sum(1 for f in folds if f == "validation")
        total_validation += validation_count
        deciles.append(
            {
                "decile": len(deciles),
                "n": len(chunk),
                "validation_fraction": (validation_count / len(chunk)) if chunk else 0.0,
            }
        )
    return {"deciles": deciles, "overall_validation_fraction": total_validation / n}


# --- Adversarial (known-distinct-books) precision set -----------------------
#
# The ISBN-gold set is entirely *positive* pairs -- it can't surface a false
# merge between two genuinely different books unless one happens to collide.
# This is a small, hand-curated *negative* set: pairs the matcher must never
# resolve to the same OL workKey. See tools/catalog/matcher_adversarial_pairs.yaml.


@dataclass(frozen=True)
class AdversarialPair:
    title_a: str
    author_a: str
    title_b: str
    author_b: str
    note: str = ""


def load_adversarial_pairs(path: Path) -> list[AdversarialPair]:
    if not path.exists():
        return []
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs: list[AdversarialPair] = []
    for row in data.get("pairs", []):
        pairs.append(
            AdversarialPair(
                title_a=row["title_a"],
                author_a=row["author_a"],
                title_b=row["title_b"],
                author_b=row["author_b"],
                note=str(row.get("note", "")).strip(),
            )
        )
    return pairs


def evaluate_adversarial_pairs(pairs: Iterable[AdversarialPair], index: TitleAuthorBlockIndex) -> dict[str, object]:
    """Each pair is two titles/authors known to be different books. A false
    merge is when *both* sides accept a title_author match and land on the
    *same* OL workKey -- i.e. the matcher, run against the real candidate
    index, collapsed two different books into one identity.

    Two accepted matches landing on *different* keys is fine even if one or
    both differ from whatever the "textbook" answer would be; this check
    only cares about the negative-set guarantee (`must_not_share_work_key`),
    not linkage correctness, which the ISBN-gold set already measures.
    """
    pairs = list(pairs)
    false_merge_pairs: list[dict[str, object]] = []
    for pair in pairs:
        match_a = match_title_author(pair.title_a, pair.author_a, index)
        match_b = match_title_author(pair.title_b, pair.author_b, index)
        if (
            match_a.method == "title_author"
            and match_b.method == "title_author"
            and match_a.work_key == match_b.work_key
        ):
            false_merge_pairs.append(
                {
                    "title_a": pair.title_a,
                    "author_a": pair.author_a,
                    "title_b": pair.title_b,
                    "author_b": pair.author_b,
                    "work_key": match_a.work_key,
                    "note": pair.note,
                }
            )
    return {
        "total": len(pairs),
        "false_merges": len(false_merge_pairs),
        "false_merge_pairs": false_merge_pairs,
    }


@dataclass
class IsbnHoldoutRow:
    book_id: int
    isbn13: str
    gold_work_key: str
    predicted_work_key: str | None
    method: str
    same_work: bool
    identity_equivalent: bool


def evaluate_title_author_against_isbn(
    books: Iterable[tuple[int, str, str, str]],
    index: TitleAuthorBlockIndex,
    isbn_index: dict[str, list[str]],
    *,
    work_title_author: dict[str, tuple[str, str]] | None = None,
    filter_unreliable_gold: bool = True,
) -> dict[str, object]:
    """ISBN-holdout eval as recommended for bibliographic linkers.

    ``books`` is ``(book_id, title, author, isbn13)``. Gold is the first
    OL ``workKey`` for that ISBN. The matcher is run with title+author
    only. ``identity_equivalent`` is true when the predicted work shares
    title-core identity + compatible author with the gold work — the
    right answer when OL split one work across several ``workKey``s.

    ``filter_unreliable_gold`` (default on) drops a gold pair when the
    gold work's own author isn't ``names_compatible`` with the querying
    book's own author -- an objective signal the *gold* row is bad (a
    pre-registration placeholder like "Untitled", "To Be Announced" as
    author, or a genuine OL ISBN->work mis-linkage), independent of
    anything the matcher predicts. Discovered when fixing `title_core`'s
    colon-subtitle bug: recovering previously-`unmatched` long-tail books
    (where OL's own ISBN->work linkage is noisiest -- pre-release ISBNs,
    Bible-style ISBN reuse) surfaced a ~93%-placeholder spike in raw
    `conflict`, none of it a real matcher regression (`false_merges`
    stayed 0). Without this filter, *any* future recall gain into that
    same long tail looks like a precision regression by construction --
    exactly the misleading-loss-function failure mode this pipeline's
    guardrails exist to catch. Excluded pairs are reported separately
    (`gold_quality_excluded`), never silently dropped.
    """
    rows: list[IsbnHoldoutRow] = []
    gold_quality_excluded = 0
    for book_id, title, author, isbn13 in books:
        gold_keys = isbn_index.get(isbn13)
        if not gold_keys:
            continue
        gold = sorted(gold_keys)[0]
        if filter_unreliable_gold and work_title_author:
            gold_ta = work_title_author.get(gold)
            if gold_ta and not names_compatible(author, gold_ta[1]):
                gold_quality_excluded += 1
                continue
        predicted = match_title_author(title, author, index)
        same = predicted.work_key == gold
        identity = False
        if predicted.work_key and work_title_author:
            gold_ta = work_title_author.get(gold)
            pred_ta = work_title_author.get(predicted.work_key)
            if gold_ta and pred_ta:
                identity = titles_identity(title_core(gold_ta[0]), title_core(pred_ta[0])) and names_compatible(
                    gold_ta[1], pred_ta[1]
                )
        rows.append(
            IsbnHoldoutRow(
                book_id=book_id,
                isbn13=isbn13,
                gold_work_key=gold,
                predicted_work_key=predicted.work_key,
                method=predicted.method,
                same_work=same,
                identity_equivalent=same or identity,
            )
        )

    n = len(rows)
    same = sum(1 for r in rows if r.same_work)
    ident = sum(1 for r in rows if r.identity_equivalent)
    conflict = sum(1 for r in rows if r.predicted_work_key and not r.identity_equivalent)
    missed = sum(1 for r in rows if r.predicted_work_key is None)
    return {
        "gold_pairs": n,
        "gold_quality_excluded": gold_quality_excluded,
        "exact_work_key": same,
        "identity_equivalent": ident,
        "conflict": conflict,
        "missed": missed,
        "exact_recall": (same / n) if n else 0.0,
        "identity_recall": (ident / n) if n else 0.0,
        "precision": ((n - conflict) / n) if n else 0.0,
        "rows": rows,
    }


# --- Residual label set (Stage 2g): hand-verified ground truth for the
# population no other signal here can check -----------------------------
#
# The ISBN-gold set only covers books with a harvested ISBN that hits OL.
# The adversarial set only covers hand-picked known-distinct pairs, not
# real matcher output. Neither says anything about a *specific*
# unmatched/ambiguous/low-margin/suspicious-duplicate book actually seen in
# a real run -- exactly the population where the matcher is most likely to
# be silently wrong, and the only place a human's judgment is the strongest
# available signal. tools/catalog/sample_matcher_residual_candidates.py
# picks *what* a human should look at, from real matched_goodreads.jsonl.gz
# output; a human then fills in `verdict` (never this code -- see that
# script's module docstring) in matcher_residual_labels.yaml. Only entries
# with a `verdict` set are ground truth here; everything else is an
# unreviewed candidate.

_RESIDUAL_VERDICTS = frozenset({"correct", "wrong", "no_ol_match", "unsure"})


@dataclass(frozen=True)
class ResidualLabel:
    goodreads_book_id: int
    sample_reason: str
    verdict: str  # "correct" | "wrong" | "no_ol_match" | "unsure"
    candidate_work_key: str | None = None
    corrected_work_key: str | None = None
    notes: str = ""

    @property
    def truth_work_key(self) -> str | None:
        """The human-verified correct ``workKey``, or ``None`` when there
        isn't one to check a prediction against (``no_ol_match`` -- this GR
        book genuinely has no OL entry; ``unsure`` -- reviewer couldn't
        tell; ``wrong`` with no ``corrected_work_key`` -- reviewer knows the
        candidate is wrong but doesn't know the right answer either)."""
        if self.verdict == "correct":
            return self.candidate_work_key
        if self.verdict == "wrong" and self.corrected_work_key:
            return self.corrected_work_key
        return None


def load_residual_labels(path: Path) -> list[ResidualLabel]:
    """Reads ``matcher_residual_labels.yaml``. Entries with no ``verdict``
    (the state every freshly-sampled candidate starts in) are skipped --
    unreviewed, not ground truth. Raises ``ValueError`` on a ``verdict``
    outside ``_RESIDUAL_VERDICTS`` -- a typo here should fail loudly, not
    silently drop a hand-reviewed label.
    """
    if not path.exists():
        return []
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels: list[ResidualLabel] = []
    for row in data.get("entries", []):
        verdict = row.get("verdict")
        if not verdict:
            continue
        verdict = str(verdict)
        if verdict not in _RESIDUAL_VERDICTS:
            raise ValueError(
                f"matcher_residual_labels.yaml: goodreads_book_id={row.get('goodreads_book_id')} has "
                f"verdict={verdict!r}, not one of {sorted(_RESIDUAL_VERDICTS)}"
            )
        labels.append(
            ResidualLabel(
                goodreads_book_id=int(row["goodreads_book_id"]),
                sample_reason=str(row.get("sample_reason", "")),
                verdict=verdict,
                candidate_work_key=row.get("candidate_work_key"),
                corrected_work_key=row.get("corrected_work_key"),
                notes=str(row.get("notes", "")),
            )
        )
    return labels


def evaluate_residual_labels(
    labels: Iterable[ResidualLabel], predicted_by_book_id: dict[int, str | None]
) -> dict[str, object]:
    """Checks the *current* matcher's predictions against this hand-verified
    residual set. ``predicted_by_book_id`` is ``goodreads_book_id ->
    predicted work_key`` (e.g. read straight off ``matched_goodreads.jsonl.gz``).

    Unlike the ISBN-gold eval, this covers exactly the population that eval
    can't reach: no ISBN, and specifically the unmatched/ambiguous/
    low-margin/suspicious-duplicate cases most likely to be silently wrong.
    It is also the smallest, most expensive, and most trustworthy signal
    this pipeline has -- treat every `mismatch` here as a real bug report,
    not noise to average away.
    """
    labels = list(labels)
    reviewed = len(labels)
    no_ol_match = sum(1 for l in labels if l.verdict == "no_ol_match")
    unsure = sum(1 for l in labels if l.verdict == "unsure")
    checkable = [l for l in labels if l.truth_work_key is not None]

    correct = 0
    mismatches: list[dict[str, object]] = []
    for label in checkable:
        predicted = predicted_by_book_id.get(label.goodreads_book_id)
        if predicted == label.truth_work_key:
            correct += 1
        else:
            mismatches.append(
                {
                    "goodreads_book_id": label.goodreads_book_id,
                    "sample_reason": label.sample_reason,
                    "truth_work_key": label.truth_work_key,
                    "predicted_work_key": predicted,
                }
            )

    return {
        "reviewed": reviewed,
        "checkable": len(checkable),
        "no_ol_match": no_ol_match,
        "unsure": unsure,
        "correct": correct,
        "accuracy": (correct / len(checkable)) if checkable else 0.0,
        "mismatches": mismatches,
    }
