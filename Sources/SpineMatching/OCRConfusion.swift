import Foundation

// OCR confusion-pair expansion per the locked "Book ID OCR gains" plan
// §A (OCR n-best + confusion -> retrieval only):
//
//   "Only on normalized tokens length 4-18 from kept C1/C2/C3 with source
//   confidence < 0.94. One edit from {O<->0, I<->l<->1, S<->5, Z<->2,
//   B<->8, G<->6, rn<->m, cl<->d, vv<->w}; <=2 expansions per token,
//   mapping order as listed."
//
// Expansions widen the FTS *retrieval* token pool only -- rerank and
// AcceptPolicy always operate on canonical (unexpanded) text per the
// plan's "Scope" row.
public enum OCRConfusion {
    /// Ordered confusion rules -- each pair substitutes in either
    /// direction. Single-character rules are tried before the
    /// multi-character ones (`rn`/`cl`/`vv`), matching the plan's listed
    /// mapping order.
    static let rules: [(String, String)] = [
        ("o", "0"),
        ("i", "l"),
        ("l", "1"),
        ("s", "5"),
        ("z", "2"),
        ("b", "8"),
        ("g", "6"),
        ("rn", "m"),
        ("cl", "d"),
        ("vv", "w"),
    ]

    public static let minTokenLength = 4
    public static let maxTokenLength = 18
    public static let maxConfidenceForExpansion: Float = 0.94
    public static let maxExpansionsPerToken = 2

    /// Generates up to `maxExpansionsPerToken` single-edit confusion
    /// variants of `token` (already search-normalized, lowercase). Returns
    /// `[]` when `token`'s length or `confidence` falls outside the gate
    /// -- callers should still keep the original token, this only adds
    /// *additional* retrieval variants.
    public static func expand(token: String, confidence: Float) -> [String] {
        guard (minTokenLength...maxTokenLength).contains(token.count) else { return [] }
        guard confidence < maxConfidenceForExpansion else { return [] }

        var variants: [String] = []
        for (a, b) in rules {
            if variants.count >= maxExpansionsPerToken { break }
            if let range = token.range(of: a) {
                var variant = token
                variant.replaceSubrange(range, with: b)
                if variant != token, !variants.contains(variant) { variants.append(variant) }
            }
            if variants.count >= maxExpansionsPerToken { break }
            if let range = token.range(of: b) {
                var variant = token
                variant.replaceSubrange(range, with: a)
                if variant != token, !variants.contains(variant) { variants.append(variant) }
            }
        }
        return Array(variants.prefix(maxExpansionsPerToken))
    }
}
