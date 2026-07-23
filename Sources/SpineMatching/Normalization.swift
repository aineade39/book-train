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
        if isDecorativePunctuation(scalar) { continue }
        out.append(scalar)
    }
    var result = String(out)
    if result.hasSuffix(" ") { result.removeLast() }
    return result
}

/// Punctuation that carries no search signal and is safe to drop outright
/// (quotes, brackets, most ASCII punctuation). Deliberately keeps
/// alphanumerics, whitespace, and a few marks (`-`, `'`, `&`, `.`) that can
/// be meaningful inside titles/author names (e.g. "Jean-Paul", "O'Brien",
/// "AT&T", "Vol. 2") — the fuzzy scorer downstream tolerates any residual
/// mismatch, so this only needs to strip clearly decorative noise.
private func isDecorativePunctuation(_ scalar: Unicode.Scalar) -> Bool {
    switch scalar {
    case "\"", "\u{201C}", "\u{201D}", "\u{2018}", "\u{2019}",
         "(", ")", "[", "]", "{", "}",
         "!", "?", ";", ":", ",",
         "*", "#", "@", "\u{2013}", "\u{2014}", "/", "\\", "_", "~", "`", "^", "|", "<", ">", "=", "+":
        return true
    default:
        return false
    }
}

/// Whitespace-separated tokens of a normalized string (used by the fuzzy
/// scorer's token-set operations).
public func searchTokens(_ normalized: String) -> [String] {
    normalized.split(separator: " ").map(String.init)
}
