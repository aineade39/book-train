import Foundation

// Light search-form normalization for OCR strings and catalog fields, per
// docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching: lowercase, Unicode fold,
// collapse whitespace, strip *decorative* punctuation only. Deliberately
// does **not** delete whitespace or all punctuation — the fuzzy rerank
// (`tokenSetRatio`/`wRatio`) is what tolerates variation, not aggressive
// string mangling here (an explicit anti-pattern in the spec).

/// Search-form normalization: case/diacritic-insensitive, whitespace
/// -collapsed, decorative-punctuation-stripped. Applied identically to OCR
/// query strings and catalog title/author fields before FTS5 retrieval and
/// fuzzy rerank, so both sides of the comparison are in the same form.
public func normalizeForSearch(_ raw: String) -> String {
    // Unicode fold: case-insensitive, diacritic-insensitive, using the
    // locale-independent comparison folding (matches é/E/e etc.).
    let folded = raw.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)

    var out = String.UnicodeScalarView()
    var lastWasSpace = false
    for scalar in folded.unicodeScalars {
        if CharacterSet.whitespacesAndNewlines.contains(scalar) {
            if !lastWasSpace && !out.isEmpty { out.append(" ") }
            lastWasSpace = true
            continue
        }
        lastWasSpace = false
        // 2j-5a: Unicode general category "format" (Cf) covers zero-width
        // space (U+200B), zero-width non/joiner (U+200C/U+200D), the BOM
        // (U+FEFF), and similar invisible characters that render as
        // nothing but are neither whitespace (not caught above) nor in the
        // fixed decorative-punctuation set below -- confirmed via a real
        // Goodreads title, "The \u{200B}Crown of Gilded Bones", where a
        // stray ZWSP after the real space before "Crown" silently broke
        // the match. Dropped like decorative punctuation (not replaced
        // with a space) since these characters are never themselves a
        // word-separator stand-in -- the ZWSP case above already had a
        // real separating space as a distinct scalar.
        if scalar.properties.generalCategory == .format { continue }
        if isDecorativePunctuation(scalar) { continue }
        if scalar == "&" {
            // 2j-5c: token substitution, not a single-scalar fold (see
            // `curlyApostropheToStraight` below) -- GR and OL disagree on
            // "&" vs "and" in the same title in at least one observed pair
            // (Carissa Broadbent, "The Serpent & the Wings of Night" (OL) /
            // "...and the Wings of Night" (GR)). Existing surrounding
            // whitespace (almost always present around "&" in real titles)
            // is untouched, so spacing comes out identical to a native
            // "and". This side has no separate raw-string multi-author
            // splitter to interact with (that logic is Python-only, in
            // `tools/catalog/bibliographic_join.py`'s `split_people`, used
            // only at catalog-build time) -- this fold is unconditional
            // here.
            out.append("a")
            out.append("n")
            out.append("d")
            continue
        }
        out.append(curlyApostropheToStraight(scalar))
    }
    var result = String(out)
    if result.hasSuffix(" ") { result.removeLast() }
    return result
}

/// Punctuation that carries no search signal and is safe to drop outright
/// (quotes, brackets, most ASCII punctuation). Deliberately keeps
/// alphanumerics, whitespace, and a couple of marks (`-`, `'`) that can be
/// meaningful inside titles/author names (e.g. "Jean-Paul", "O'Brien") — the
/// fuzzy scorer downstream tolerates any residual mismatch, so this only
/// needs to strip clearly decorative noise. `&` is folded to "and" above,
/// not listed here. `.` used to be kept for cases like "Vol. 2", but 2j-5b
/// found GR/OL disagree on `.` vs `/` as a date-title separator ("11.22.63"
/// vs "11/22/63", OL's own canonical form, 49 editions) — `/` was already
/// dropped here, so `.` joins it for consistency; checked against
/// author-initial strings ("J.R.R. Tolkien" -> "jrr tolkien") and
/// abbreviations ("U.S.A." -> "usa") for regressions -- none found, since
/// both already have a following space or no meaningful separator role.
private func isDecorativePunctuation(_ scalar: Unicode.Scalar) -> Bool {
    switch scalar {
    case "\"", "\u{201C}", "\u{201D}",
         "(", ")", "[", "]", "{", "}",
         "!", "?", ";", ":", ",", ".",
         "*", "#", "@", "\u{2013}", "\u{2014}", "/", "\\", "_", "~", "`", "^", "|", "<", ">", "=", "+":
        return true
    default:
        return false
    }
}

/// U+2018/U+2019 (curly single quotes) used to be grouped with the curly
/// double-quotes above and stripped outright, but that's the single most
/// common apostrophe glyph in real-world text (most web/scraped sources
/// render "O'Brien" with U+2019, not the ASCII U+0027 this function
/// deliberately keeps — see `isDecorativePunctuation`'s doc comment).
/// Stripping one glyph and keeping the other meant the exact same word
/// normalized two different ways depending purely on which apostrophe
/// character the source used, silently breaking `titleNormalized` /
/// `title_core` lookups across the straight/curly boundary. Folding both
/// curly forms to the straight apostrophe here fixes the mismatch while
/// keeping "apostrophes are meaningful, not decorative" true for both.
/// Kept in sync with `tools/catalog/ol_common.py`'s Python port.
private func curlyApostropheToStraight(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
    switch scalar {
    case "\u{2018}", "\u{2019}":
        return "'"
    default:
        return scalar
    }
}

/// Whitespace-separated tokens of a normalized string (used by the fuzzy
/// scorer's token-set operations).
public func searchTokens(_ normalized: String) -> [String] {
    normalized.split(separator: " ").map(String.init)
}
