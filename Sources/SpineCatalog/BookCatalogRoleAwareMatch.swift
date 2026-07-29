import Foundation
import SpineMatching

// Single shared retrieve -> field-aware rerank -> accept entry point per
// the locked "Book ID OCR gains" plan §H ("spine-id: shared package APIs
// for ISBN, role retrieve/rerank, accept; CLI and app call the same
// code") -- both `SpineIdentificationPipeline` (app) and `spine-id` (CLI)
// call this one function instead of each re-deriving
// `retrieveRoleAware` + `FieldAwareScore.finalScore` + `AcceptPolicy`
// wiring themselves.
extension BookCatalog {
    /// One spine's `SpineRoleQueries` (geometry-derived or FM-derived, see
    /// `SpineRoleQueryBuilder.build(fmTitle:fmAuthor:)`) -> role-aware FTS
    /// shortlist -> `FieldAwareScore` rerank -> `AcceptPolicy` decision,
    /// with the winning margin surfaced for telemetry (§rerank-telemetry:
    /// "emit margin"; `AcceptPolicy`'s own 90/8 rule is unchanged either
    /// way).
    ///
    /// - Important: An empty shortlist is a normal outcome and surfaces as
    ///   `AcceptOutcome.noMatch` -- not thrown. A GRDB/SQLite failure (a
    ///   corrupt DB, disk I/O error, etc.) *does* throw here and callers
    ///   must let it propagate (`try`, not `try?`); silently converting it
    ///   into `.noMatch` is indistinguishable from a genuine miss and hides
    ///   real failures from telemetry and the UI.
    public func matchRoleAware(
        _ queries: SpineRoleQueries,
        acceptPolicy: AcceptPolicy = AcceptPolicy(),
        catalogSize: Int = FieldAwareScore.defaultCatalogSize,
        shortlistCap: Int = 160
    ) throws -> AcceptOutcome<CatalogCandidate> {
        let candidates = try retrieveRoleAware(queries, shortlistCap: shortlistCap)
        let scored = candidates.map { candidate in
            ScoredCandidate(
                candidate: candidate,
                score: FieldAwareScore.finalScore(
                    titleQuery: queries.titleQuery,
                    authorQuery: queries.authorQuery,
                    allNormalizedOCR: queries.allNormalizedOCR,
                    candidateTitle: candidate.title,
                    candidateAuthor: candidate.author,
                    candidateSearchableText: candidate.searchableText,
                    popularityRank: candidate.popularityRank,
                    catalogSize: catalogSize,
                    titleQueryIsFallback: queries.titleQueryIsFallback
                )
            )
        }
        return acceptPolicy.decideWithMargin(scored)
    }
}
