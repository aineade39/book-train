import Foundation
import GRDB
import SpineMatching

// 3-pass, field-scoped FTS retrieval per the locked "Book ID OCR gains"
// plan §B ("FTS (replaces single mashed OR for match path)") -- replaces
// the single mashed-OR `retrieveCandidates` query for the app/CLI match
// path (that method stays as-is for CSV-only/simple catalogs and its own
// test coverage). Token pools already carry n-best/confusion variants and
// the top-8/6/8 cuts (`SpineRoleQueryBuilder`); this file only does the
// SQL side: per-pass column-scoped MATCH, union-by-work-id keeping the
// best rank across passes, and the empty-shortlist fallbacks.
extension BookCatalog {
    /// Below this many candidates already found by the title/author
    /// passes, the general (unscoped, most expensive) pass still runs to
    /// widen coverage; at or above it, the pool is considered healthy
    /// enough that the general pass's cost isn't worth paying.
    static let generalPassSkipThreshold = 20

    /// Runs the title/author/general passes, unions their results by
    /// catalog row id (keeping each row's best cross-pass rank), and caps
    /// the result at `shortlistCap`. Falls back to a wider unscoped pool,
    /// then a short-read LIKE scan, before returning `[]` (never throws --
    /// an empty shortlist is a normal `AcceptPolicy.noMatch`, not an
    /// error).
    public func retrieveRoleAware(_ queries: SpineRoleQueries, shortlistCap: Int = 160) throws -> [CatalogCandidate] {
        var ranked: [Int64: (candidate: CatalogCandidate, bestRank: Int)] = [:]
        func merge(_ candidates: [CatalogCandidate]) {
            for (rank, candidate) in candidates.enumerated() {
                if let existing = ranked[candidate.id], existing.bestRank <= rank { continue }
                ranked[candidate.id] = (candidate, rank)
            }
        }

        if !queries.titleTokens.isEmpty {
            merge(try columnRetrieve(tokens: queries.titleTokens.map(\.token), columns: ["titleNormalized"], limit: 80))
        }
        if !queries.authorTokens.isEmpty {
            merge(try columnRetrieve(tokens: queries.authorTokens.map(\.token), columns: ["authorNormalized"], limit: 50))
        }
        // The general pass is unscoped (both columns), so it can't
        // benefit from column-scoped MATCH the way title/author passes
        // do -- it's the most expensive of the three on a large catalog.
        // Its purpose is widening coverage when title/author came up
        // thin; skip it once they've already produced a healthy pool
        // (`Self.generalPassSkipThreshold`) rather than always paying for
        // a wider query that's largely redundant at that point.
        if !queries.generalTokens.isEmpty, ranked.count < Self.generalPassSkipThreshold {
            merge(try columnRetrieve(tokens: queries.generalTokens.map(\.token), columns: nil, limit: 30))
        }

        if !ranked.isEmpty {
            return ranked.values.sorted { $0.bestRank < $1.bestRank }.prefix(shortlistCap).map(\.candidate)
        }

        // Empty-shortlist fallback #1: wider (top-12) unscoped pool, no
        // per-pass LIMIT-80/50/30 narrowing.
        if !queries.fallbackTokens.isEmpty {
            let fallback = try columnRetrieve(tokens: queries.fallbackTokens.map(\.token), columns: nil, limit: 100)
            if !fallback.isEmpty { return fallback }
        }

        // Empty-shortlist fallback #2: short-read LIKE, only meaningful
        // when the best available token was too short to have been
        // trigram-tokenizable in the first place.
        if let shortest = queries.fallbackTokens.map(\.token).min(by: { $0.count < $1.count }),
           shortest.count < Self.shortReadThreshold {
            return try shortReadFallback(query: shortest, limit: 50)
        }

        return []
    }

    private func columnRetrieve(tokens: [String], columns: [String]?, limit: Int) throws -> [CatalogCandidate] {
        guard !tokens.isEmpty else { return [] }
        // Drop near-zero-information stopwords ("the", "and", ...) before
        // querying -- a common token like "the" can match millions of
        // rows in a large catalog, and every row matched is a row `ORDER
        // BY rank` must score. Never let this empty the token list: if
        // every token happens to be a stopword (rare, but possible for a
        // very short/degenerate OCR read), fall back to the unfiltered
        // set rather than sending a query that can't match anything.
        let stopwordFiltered = tokens.filter { !englishStopwords.contains($0) }
        let effectiveTokens = stopwordFiltered.isEmpty ? tokens : stopwordFiltered
        return try dbQueue.read { db in
            let phrases = effectiveTokens.map { "\"\($0.replacingOccurrences(of: "\"", with: "\"\""))\"" }
            let matchExpression: String
            if let columns, !columns.isEmpty {
                matchExpression = phrases
                    .flatMap { phrase in columns.map { "\($0):\(phrase)" } }
                    .joined(separator: " OR ")
            } else {
                matchExpression = phrases.joined(separator: " OR ")
            }
            // `rank`, not `bm25(books_fts)`: identical bm25 score (verified
            // byte-identical top results on a 39M-row catalog), but `rank`
            // is FTS5's special-cased top-K path (bounded heap) instead of
            // an opaque scalar function call that forces scoring every
            // matching row before LIMIT can apply -- ~10-20x faster on a
            // large catalog with a common token. See `BookCatalog.swift`'s
            // `trigramRetrieve` for the same note.
            let sql = """
                SELECT books.* FROM books_fts
                JOIN books ON books.id = books_fts.rowid
                WHERE books_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """
            let rows = try BookRecord.fetchAll(db, sql: sql, arguments: [matchExpression, limit])
            // Defensive verification against the FTS5 trigram tokenizer's
            // own false positives -- unlike `retrieveCandidates`'s
            // deliberately loose substring LIKE check (kept as-is for the
            // CSV-only path), this is *whole-word* matching: a raw
            // substring check would let a short leaked token like "wulf"
            // verify against an unrelated word like "beowulf" that merely
            // contains it, letting completely wrong candidates through.
            let verified = rows.filter { record in
                let titleWords = Set(searchTokens(record.titleNormalized))
                let authorWords = Set(searchTokens(record.authorNormalized))
                return effectiveTokens.contains { token in
                    let matchesTitle = titleWords.contains(token)
                    let matchesAuthor = authorWords.contains(token)
                    guard let columns, !columns.isEmpty else { return matchesTitle || matchesAuthor }
                    return (columns.contains("titleNormalized") && matchesTitle)
                        || (columns.contains("authorNormalized") && matchesAuthor)
                }
            }
            return verified.map(CatalogCandidate.init)
        }
    }
}
