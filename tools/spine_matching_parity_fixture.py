#!/usr/bin/env python3
"""RapidFuzz + normalization reference fixture for the opt-in Swift/Python
parity test (`Tests/SpineMatchingTests/ParityIntegrationTests.swift`).

Computes, for a fixed list of string pairs, RapidFuzz's `ratio` /
`token_sort_ratio` / `token_set_ratio` (the exact functions
`Sources/SpineMatching/FuzzyMatch.swift` claims "RapidFuzz-equivalent"
parity with) plus a small independent Python re-implementation of
`normalizeForSearch` (lowercase + Unicode NFKD-fold + collapse whitespace +
strip decorative punctuation), and prints both as JSON on stdout.

`wRatio` is intentionally excluded: `Sources/SpineMatching/FuzzyMatch.swift`
documents a deliberate deviation from upstream `WRatio` (it drops the
partial_ratio branch entirely to avoid the same ceiling-saturation
RapidFuzz's `partial_token_set_ratio` has), so it has no upstream function to
be "parity" with.

Not part of the production CLI pipeline -- invoked only by the Swift test.
Requires `rapidfuzz` (`pip install rapidfuzz`); not part of the training/
export venvs' normal dependency set, so this fails loudly rather than
silently if it's missing.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata

try:
    from rapidfuzz import fuzz
except ImportError:
    print(
        "rapidfuzz not installed in this venv. `pip install rapidfuzz`",
        file=sys.stderr,
    )
    raise SystemExit(1)

# Kept numerically in sync with the decorative-punctuation set in
# `Sources/SpineMatching/Normalization.swift`'s `isDecorativePunctuation`.
# U+2018/U+2019 (curly single quotes) are deliberately absent here -- as of
# 2i they're folded to the straight apostrophe, not stripped (see
# `_CURLY_APOSTROPHE_TO_STRAIGHT` below), same as the Swift/`ol_common.py`
# production code.
_DECORATIVE_PUNCTUATION = set(
    "\"\u201c\u201d()[]{}!?;:,.*#@\u2013\u2014/\\_~`^|<>=+"
)

# 2i: fold both curly single-quote glyphs to the straight ASCII apostrophe
# (not strip either) -- see `Normalization.swift`'s `curlyApostropheToStraight`.
_CURLY_APOSTROPHE_TO_STRAIGHT = {"\u2018": "'", "\u2019": "'"}


def normalize_for_search(raw: str) -> str:
    """Independent Python re-implementation of
    `Sources/SpineMatching/Normalization.swift`'s `normalizeForSearch`, for
    cross-language parity checking (not the production normalizer)."""
    folded = unicodedata.normalize("NFKD", raw)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.casefold()

    out = []
    last_was_space = False
    for ch in folded:
        if ch.isspace():
            if not last_was_space and out:
                out.append(" ")
            last_was_space = True
            continue
        last_was_space = False
        # 2j-5a: Unicode format characters (zero-width space, BOM, etc.) are
        # invisible but not whitespace -- drop them like decorative
        # punctuation.
        if unicodedata.category(ch) == "Cf":
            continue
        if ch in _DECORATIVE_PUNCTUATION:
            continue
        if ch == "&":
            # 2j-5c: token substitution, not a single-character fold.
            out.append("and")
            continue
        out.append(_CURLY_APOSTROPHE_TO_STRAIGHT.get(ch, ch))
    result = "".join(out)
    return result[:-1] if result.endswith(" ") else result


# Pairs chosen to exercise: identical strings, pure reordering, subset
# tokens (token_set_ratio's documented ceiling case), leftovers on both
# sides (must land below ceiling), and a realistic mashed-OCR-blob case.
FUZZY_PAIRS = [
    ("abc", "abc"),
    ("abc", "xyz"),
    ("abcd", "abc"),
    ("kitten", "sitting"),
    ("great gatsby the", "the great gatsby"),
    ("dune messiah", "dune messiah"),
    ("dune", "dune messiah"),
    ("the great gatsby", "great gatsby a novel by f scott fitzgerald"),
    ("project hail mary andy weir ballantine", "project hail mary andy weir"),
]

NORMALIZE_INPUTS = [
    "Café DU MONDE",
    "The   Great\nGatsby",
    "O'Brien: \"The Things They Carried\"",
    "Jean-Paul Sartre",
    "AT&T Vol. 2",
    "",
    "PROJECT HAIL MARY ANDY WEIR BALLANTINE",
]


def main() -> int:
    payload = {
        "fuzzy": [
            {
                "a": a,
                "b": b,
                "ratio": fuzz.ratio(a, b),
                "token_sort_ratio": fuzz.token_sort_ratio(a, b),
                "token_set_ratio": fuzz.token_set_ratio(a, b),
            }
            for a, b in FUZZY_PAIRS
        ],
        "normalization": [
            {"input": s, "normalized": normalize_for_search(s)} for s in NORMALIZE_INPUTS
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
