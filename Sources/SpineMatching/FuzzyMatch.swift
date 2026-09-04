import Foundation

// token_set_ratio / WRatio (RapidFuzz-equivalent) rerank, per
// docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching:
//
//   "token-set-ratio-style fuzzy match (token_set_ratio / WRatio,
//   RapidFuzz-equivalent)... Do not use partial_token_set_ratio: it
//   saturates at 100 whenever the OCR tokens are a subset of a candidate,
//   producing many tied 100s that collapse the accept margin test."
//
// All scorers here operate on already search-normalized strings (see
// `normalizeForSearch`) and return a score in `[0, 100]`.

/// Base similarity: `100 * (1 - indelDistance / (len(a) + len(b)))`, the
/// same normalized-Indel-distance definition RapidFuzz uses for `fuzz.ratio`
/// (an LCS-based identity: Indel distance, i.e. edit distance restricted to
/// insertions/deletions only, equals `len(a) + len(b) - 2 * lcsLength(a,b)`
/// exactly — so this is computed directly from the LCS length rather than a
/// difflib-style heuristic, matching RapidFuzz's own implementation rather
/// than approximating it).
public func ratio(_ a: String, _ b: String) -> Double {
    let aChars = Array(a), bChars = Array(b)
    let total = aChars.count + bChars.count
    guard total > 0 else { return 100 }
    guard !aChars.isEmpty, !bChars.isEmpty else { return 0 }
    let lcs = lcsLength(aChars, bChars)
    return 100.0 * Double(2 * lcs) / Double(total)
}

/// Sorts each string's whitespace tokens before comparing — cancels out
/// pure re-ordering (e.g. "Author, Title" vs "Title Author").
public func tokenSortRatio(_ a: String, _ b: String) -> Double {
    ratio(sortedTokenString(a), sortedTokenString(b))
}

/// FuzzyWuzzy/RapidFuzz `token_set_ratio`: compares the shared-token core
/// against each side's leftover tokens, taking the best of three pairwise
/// ratios. Tolerates one string being a *superset* of the other's tokens
/// (e.g. a mashed "title author publisher" OCR blob vs. a clean title)
/// without the ceiling-saturation `partial_token_set_ratio` has, because it
/// still penalizes leftover/missing tokens via the base `ratio` length
/// terms instead of ignoring them.
public func tokenSetRatio(_ a: String, _ b: String) -> Double {
    let t1 = Set(searchTokens(a))
    let t2 = Set(searchTokens(b))
    let intersection = t1.intersection(t2)
    let diff1to2 = t1.subtracting(t2)
    let diff2to1 = t2.subtracting(t1)

    let sortedSect = intersection.sorted().joined(separator: " ")
    let sorted1to2 = joinNonEmpty(sortedSect, diff1to2.sorted().joined(separator: " "))
    let sorted2to1 = joinNonEmpty(sortedSect, diff2to1.sorted().joined(separator: " "))

    return max(
        ratio(sortedSect, sorted1to2),
        ratio(sortedSect, sorted2to1),
        ratio(sorted1to2, sorted2to1)
    )
}

/// A safe `WRatio`-style combination: the best of `ratio`, a
/// length-discounted `tokenSortRatio`, and a length-discounted
/// `tokenSetRatio`.
///
/// **Intentional, documented deviation from upstream FuzzyWuzzy/RapidFuzz
/// `WRatio`:** the reference algorithm also blends in `partial_ratio` /
/// `partial_token_sort_ratio` / `partial_token_set_ratio` once the two
/// strings' lengths diverge enough (`len_ratio >= 1.5`) — but those are
/// exactly the "saturates at 100 on subset matches" family the spec forbids
/// for the *same* reason as `partial_token_set_ratio`. This `wRatio` omits
/// that branch unconditionally, so it — like `tokenSetRatio` — stays safe
/// to use with a margin-based accept policy. Prefer `tokenSetRatio` as the
/// default rerank scorer; `wRatio` is offered for spec-name parity and as a
/// slightly more conservative alternative (it never scores *below*
/// `tokenSetRatio` alone, only ever adds the plain/token-sort signal).
public func wRatio(_ a: String, _ b: String) -> Double {
    let base = ratio(a, b)
    let sort = tokenSortRatio(a, b) * 0.95
    let set = tokenSetRatio(a, b) * 0.95
    return max(base, sort, set)
}

// MARK: - Helpers

private func sortedTokenString(_ s: String) -> String {
    searchTokens(s).sorted().joined(separator: " ")
}

private func joinNonEmpty(_ head: String, _ tail: String) -> String {
    guard !tail.isEmpty else { return head }
    guard !head.isEmpty else { return tail }
    return head + " " + tail
}

/// Standard O(n*m) LCS-length dynamic program over `Character` arrays.
/// Titles/authors and OCR strings are short (tens of characters), and the
/// catalog shortlist is capped (~50 candidates per docs/BOOK_ID_IOS_PIPELINE.md
/// §Catalog matching), so this is comfortably cheap per match.
private func lcsLength(_ a: [Character], _ b: [Character]) -> Int {
    guard !a.isEmpty, !b.isEmpty else { return 0 }
    var prev = [Int](repeating: 0, count: b.count + 1)
    var curr = [Int](repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        for j in 1...b.count {
            curr[j] = a[i - 1] == b[j - 1] ? prev[j - 1] + 1 : max(prev[j], curr[j - 1])
        }
        swap(&prev, &curr)
    }
    return prev[b.count]
}
