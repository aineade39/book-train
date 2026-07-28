import Foundation

// Field-aware rerank per the locked "Book ID OCR gains" plan §B, superseding
// the flat `0.75 tokenSetRatio + 0.20 wRatio` blend (see plan
// §Reconciliations #2): title/author/blob are scored against their own
// catalog fields (not one mashed comparison), then a small popularity
// prior breaks residual ties. `AcceptPolicy` (unchanged, 90/8) still
// consumes the resulting `finalScore` exactly as it did the old score.

public enum FieldAwareScore {
    /// Reference catalog size the popularity term's log-scale is
    /// calibrated against (the `ios_en` profile's `--max-works` ceiling).
    public static let defaultCatalogSize = 250_000

    /// `0-100` popularity contribution: `0` with no rank (unranked/unknown
    /// popularity), decaying log-linearly from `100` at rank `1` to `0` at
    /// `rank == catalogSize`.
    public static func popularityScore(rank: Int?, catalogSize: Int = defaultCatalogSize) -> Double {
        guard let rank, rank >= 1, catalogSize > 1 else { return 0 }
        let value = 100 * (1 - log(Double(rank)) / log(Double(catalogSize)))
        return min(max(value, 0), 100)
    }

    /// `0-100` final rerank score: `0.95` field-aware similarity + `0.05`
    /// popularity. Title/author/blob are all computed against
    /// search-normalized catalog fields, matching how `titleQuery`/
    /// `authorQuery`/`allNormalizedOCR` are themselves normalized.
    ///
    /// - Parameter titleQueryIsFallback: carried through from the caller's
    ///   `SpineRoleQueries.titleQueryIsFallback` for telemetry/future use;
    ///   *not* currently used to reweight the field blend -- see
    ///   `coverageAdjustedScore`'s doc comment for why a discount/reweight
    ///   keyed off query-vs-candidate size was tried and reverted.
    public static func finalScore(
        titleQuery: String,
        authorQuery: String,
        allNormalizedOCR: String,
        candidateTitle: String,
        candidateAuthor: String,
        candidateSearchableText: String,
        popularityRank: Int?,
        catalogSize: Int = defaultCatalogSize,
        titleQueryIsFallback: Bool = false
    ) -> Double {
        let titleScore = tokenSetRatio(titleQuery, normalizeForSearch(candidateTitle))
        let authorScore = tokenSetRatio(authorQuery, normalizeForSearch(candidateAuthor))
        let blobScore = tokenSetRatio(allNormalizedOCR, normalizeForSearch(candidateSearchableText))

        let fieldScore: Double
        if authorQuery.trimmingCharacters(in: .whitespaces).isEmpty {
            fieldScore = 0.66 * titleScore + 0.34 * blobScore
        } else {
            fieldScore = 0.52 * titleScore + 0.28 * authorScore + 0.20 * blobScore
        }

        let popularity = popularityScore(rank: popularityRank, catalogSize: catalogSize)
        return 0.95 * fieldScore + 0.05 * popularity
    }

    // Tried-and-reverted (see git history for the full implementation): a
    // "coverage-adjusted" variant of `tokenSetRatio`, discounted whenever a
    // single-token candidate (a bare first name, publisher, or country
    // name -- "Wiley", "Beowulf") fully explained a long OCR query and
    // saturated at 100, to stop it from beating a real, longer title that
    // can't reach that ceiling. Measured against all 5 oracle scenes
    // end-to-end (not just the hand-picked failure examples that motivated
    // it), this was a *net regression*: this catalog's domain includes
    // many legitimately single-word correct titles (travel guides titled
    // by bare country name -- "Spain", "Sweden", "Portugal" -- plus
    // one-word academic titles like "Physics", "Democracy"), and the
    // discount punished those just as hard as the genuinely-generic wrong
    // candidates it was meant to catch, with no lexical signal available
    // to tell the two apart. Top-shown-candidate oracle-hit rate across
    // all paired spines dropped from 62.6% to 48.5% with this enabled.
    // The query-collapse fix (`SpineRoleQueries.canonicalQuery`'s
    // fallback, see `titleQueryIsFallback`) already resolves the original
    // motivating case (Beowulf/"The Invention of Nature") on its own,
    // since a correctly-assembled, non-collapsed title query is a *bigger*
    // token set than a single stray line and no longer ties a generic
    // single-word candidate at the same ceiling. A viable future version
    // of this idea would need a *non-lexical* signal (e.g. corroborating
    // author agreement, or popularity) to distinguish the two cases this
    // couldn't.
}
