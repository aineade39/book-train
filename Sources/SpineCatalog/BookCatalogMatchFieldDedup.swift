import Foundation

// Defense-in-depth for catalogs not built via the OL ETL's build-time
// dedup (`CatalogOLBuild.buildFromIntermediate`'s `seenMatchFieldKeys`) --
// CSV imports (`tools/catalog/ol_to_csv.py` -> `swift run catalog-build
// <csv>`), hand-built/test catalogs (direct `BookCatalog.insert`/
// `bulkInsert` calls), and any catalog installed on-device before an app
// update ships a rebuilt one. Collapses rows sharing normalized match
// fields (`CatalogCandidate.matchFieldKey`) but different `workKey`s
// before they consume shortlist slots or create a false tied-distinct-
// work margin failure in `SpineMatching.AcceptPolicy` (which stays
// workKey-scoped -- see its own dedup comment).
extension BookCatalog {
    /// Collapses `candidates` down to at most one *work* (`workKey`) per
    /// unique `matchFieldKey` (`titleNormalized + "|" + authorNormalized`)
    /// -- rows sharing both `matchFieldKey` *and* `workKey` (legitimate
    /// multiple editions of the same work) are left untouched; only rows
    /// whose `matchFieldKey` collides *across different* `workKey`s get
    /// collapsed down to the one canonical work.
    ///
    /// Canonical work selection per colliding group: lowest `popularityRank`
    /// wins (`nil` -- no OL popularity signal, e.g. CSV/synthetic rows --
    /// loses ties against any real rank); ties (including nil vs. nil)
    /// broken by best retrieval rank (lower is better). Surviving groups
    /// are ordered by their best retrieval rank, then optionally capped --
    /// capping *after* dedup so a run of duplicates can't waste shortlist
    /// slots that should go to distinct works.
    ///
    /// - Parameters:
    ///   - candidates: candidates to dedupe. When `rankedBy` is nil, this
    ///     array's order is used as the rank (array index == rank) -- the
    ///     right choice for `columnRetrieve`/`shortReadFallback` results,
    ///     which are already `ORDER BY rank`/relevance but don't carry an
    ///     explicit per-candidate rank value.
    ///   - rankedBy: explicit `(candidate, bestRank)` pairs when the
    ///     caller already has per-candidate ranks (e.g.
    ///     `retrieveRoleAware`'s cross-pass `bestRank` union). Takes
    ///     precedence over `candidates`'s array order when provided;
    ///     `candidates` itself is ignored in that case.
    ///   - cap: optional result-size cap, applied after dedup.
    static func dedupeByMatchFields(
        _ candidates: [CatalogCandidate],
        rankedBy: [(candidate: CatalogCandidate, bestRank: Int)]? = nil,
        cap: Int? = nil
    ) -> [CatalogCandidate] {
        let ranked = rankedBy ?? candidates.enumerated().map { (candidate: $1, bestRank: $0) }

        // Pick the canonical *workKey* per matchFieldKey group -- not the
        // canonical row. Two rows can legitimately share both
        // `matchFieldKey` and `workKey` (multiple editions of the same
        // work, e.g. different ISBNs of the same title/author) and must
        // both survive; only rows whose `matchFieldKey` collides *across*
        // different `workKey`s (the OL data-quality pattern this guards
        // against) get collapsed down to the one canonical work.
        var canonicalWorkKey: [String: (workKey: String, popularityRank: Int?, bestRank: Int)] = [:]
        for entry in ranked {
            let key = entry.candidate.matchFieldKey
            let contender = (workKey: entry.candidate.workKey, popularityRank: entry.candidate.popularityRank, bestRank: entry.bestRank)
            guard let incumbent = canonicalWorkKey[key] else {
                canonicalWorkKey[key] = contender
                continue
            }
            if isBetterWork(contender, than: incumbent) {
                canonicalWorkKey[key] = contender
            }
        }

        let survivors = ranked.filter { entry in
            canonicalWorkKey[entry.candidate.matchFieldKey]?.workKey == entry.candidate.workKey
        }
        let deduped = survivors.sorted { $0.bestRank < $1.bestRank }.map(\.candidate)
        guard let cap else { return deduped }
        return Array(deduped.prefix(cap))
    }

    /// True if `contender` should replace `incumbent` as the canonical
    /// workKey for a shared `matchFieldKey` group. Lower `popularityRank`
    /// wins (`nil` -- no OL popularity signal -- loses ties against any
    /// real rank); ties (including nil vs. nil) broken by best retrieval
    /// rank (lower is better).
    private static func isBetterWork(
        _ contender: (workKey: String, popularityRank: Int?, bestRank: Int),
        than incumbent: (workKey: String, popularityRank: Int?, bestRank: Int)
    ) -> Bool {
        switch (contender.popularityRank, incumbent.popularityRank) {
        case let (challenger?, champion?):
            if challenger != champion { return challenger < champion }
        case (nil, .some):
            return false
        case (.some, nil):
            return true
        case (nil, nil):
            break
        }
        return contender.bestRank < incumbent.bestRank
    }
}
