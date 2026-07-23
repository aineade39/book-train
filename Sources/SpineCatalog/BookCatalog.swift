import Foundation
import GRDB
import SpineMatching

// FTS5 trigram catalog per docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching:
//
//   "Stage 1 -- retrieval: FTS5 virtual table with tokenize='trigram' and
//   detail='none', backed by a content table (not contentless FTS) for
//   LIKE verification. Return ~50 candidates in milliseconds. Trigram
//   needs >= 3 characters, so for very short reads ... fall back to a
//   prefix / LIKE query on the content table instead of returning zero
//   candidates."
//
// Stage 2 (fuzzy rerank + accept policy) lives in `SpineMatching` and is
// deliberately *not* duplicated here — `retrieveCandidates` only narrows
// the shortlist; ordering it is the caller's job (see `book-match` /
// `spine-id` CLIs and docs/BOOK_ID_IOS_PIPELINE.md's layered pipeline).

public enum SpineCatalogError: Error, CustomStringConvertible {
    case emptyQuery
    public var description: String {
        switch self {
        case .emptyQuery: return "query is empty after normalization"
        }
    }
}

public final class BookCatalog {
    public let dbQueue: DatabaseQueue

    /// Trigram tokens need >= 3 characters to exist at all; below that,
    /// `retrieveCandidates` falls back to a prefix/LIKE scan instead of the
    /// FTS5 index.
    public static let shortReadThreshold = 3

    public init(path: String) throws {
        dbQueue = try DatabaseQueue(path: path)
        try Self.migrator.migrate(dbQueue)
    }

    public init(dbQueue: DatabaseQueue) throws {
        self.dbQueue = dbQueue
        try Self.migrator.migrate(dbQueue)
    }

    /// In-memory catalog — the normal choice for tests and the `book-match`
    /// CLI's one-shot runs.
    public static func inMemory() throws -> BookCatalog {
        try BookCatalog(dbQueue: try DatabaseQueue())
    }

    private static var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()
        migrator.registerMigration("v1_books_and_fts") { db in
            try db.create(table: "books") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("workKey", .text).notNull().indexed()
                t.column("title", .text).notNull()
                t.column("author", .text).notNull()
                t.column("isbn", .text)
                t.column("titleNormalized", .text).notNull().indexed()
                t.column("authorNormalized", .text).notNull().indexed()
            }
            try db.create(virtualTable: "books_fts", using: FTS5()) { t in
                // Substring retrieval (not edit-distance search) -- see
                // §Catalog matching. case_sensitive=0 is already SQLite's
                // trigram default; spelled out for self-documentation.
                t.tokenizer = FTS5TokenizerDescriptor(components: ["trigram", "case_sensitive", "0"])
                //
                // Intentional, documented deviation from the spec's literal
                // `detail='none'`: SQLite's trigram substring matching is
                // implemented via *phrase* queries (adjacent-trigram
                // sequences), and phrase queries require `detail=full`
                // ("fts5: phrase queries are not supported (detail!=full)")
                // -- `detail='none'`/`'column'` only support unordered
                // bag-of-tokens matching, which would silently degrade
                // trigram matching from "contains this substring" to
                // "contains these three characters somewhere", defeating
                // the entire reason to use the trigram tokenizer. Leaving
                // `detail` at its default (`full`) here; Stage 2
                // (`SpineMatching` fuzzy rerank) still does the real
                // ranking, so this only affects Stage-1 index size/speed,
                // not correctness of anything downstream.
                //
                // External content table (not contentless): required for
                // the LIKE-verification step and for returning full rows.
                t.synchronize(withTable: "books")
                t.column("titleNormalized")
                t.column("authorNormalized")
            }
        }
        return migrator
    }

    /// Total row count -- cheap catalog-size/sanity check for callers (e.g.
    /// an app surfacing "N books in your catalog") that don't want to reach
    /// past `SpineCatalog` into raw GRDB/SQL themselves.
    public func countBooks() throws -> Int {
        try dbQueue.read { db in try BookRecord.fetchCount(db) }
    }

    // MARK: - Writes

    @discardableResult
    public func insert(title: String, author: String, isbn: String? = nil, workKey: String? = nil) throws -> Int64 {
        var record = BookRecord(
            workKey: workKey ?? Self.defaultWorkKey(title: title, author: author),
            title: title, author: author, isbn: isbn
        )
        try dbQueue.write { db in try record.insert(db) }
        return record.id!
    }

    /// Default work key when the caller doesn't have a real work
    /// identifier (e.g. no OpenLibrary/ISBN-work mapping available):
    /// normalized title + normalized author. Distinct editions of the same
    /// book sharing a title/author will collapse under this key, matching
    /// §Data model's edition-dedup requirement.
    public static func defaultWorkKey(title: String, author: String) -> String {
        normalizeForSearch(title) + "|" + normalizeForSearch(author)
    }

    // MARK: - Stage 1 retrieval

    /// Returns up to `limit` **unranked** candidates plausibly matching
    /// `rawQuery` (title/author/mashed OCR blob). Empty input throws
    /// rather than silently returning everything.
    ///
    /// A spine OCR blob commonly mashes title + author + publisher into one
    /// string (§Catalog matching), and title/author are separate columns
    /// here -- so retrieval can't be "the whole query is a substring of one
    /// column" (that would only ever match single-field queries). Instead
    /// each `query` word of >= 3 chars becomes its own quoted trigram
    /// phrase, OR'd together: any row containing *any* query word as a
    /// verbatim substring in either field enters the Stage-1 shortlist.
    /// This is intentionally broad/cheap -- Stage 2 (`SpineMatching`
    /// `tokenSetRatio`/`wRatio`) does the real discrimination.
    public func retrieveCandidates(forQuery rawQuery: String, limit: Int = 50) throws -> [CatalogCandidate] {
        let query = normalizeForSearch(rawQuery)
        guard !query.isEmpty else { throw SpineCatalogError.emptyQuery }

        let trigrammableTokens = searchTokens(query).filter { $0.count >= Self.shortReadThreshold }
        guard query.count >= Self.shortReadThreshold, !trigrammableTokens.isEmpty else {
            return try shortReadFallback(query: query, limit: limit)
        }
        return try trigramRetrieve(tokens: trigrammableTokens, limit: limit)
    }

    private func trigramRetrieve(tokens: [String], limit: Int) throws -> [CatalogCandidate] {
        try dbQueue.read { db in
            // Each token is a quoted phrase (SQLite's documented
            // "does this column contain this substring" shape for the
            // trigram tokenizer); tokens are OR'd, not AND'd, so a mashed
            // OCR blob only needs one clean word to enter the shortlist.
            let matchExpression = tokens
                .map { "\"\($0.replacingOccurrences(of: "\"", with: "\"\""))\"" }
                .joined(separator: " OR ")
            let sql = """
                SELECT books.* FROM books_fts
                JOIN books ON books.id = books_fts.rowid
                WHERE books_fts MATCH ?
                LIMIT ?
                """
            let rows = try BookRecord.fetchAll(db, sql: sql, arguments: [matchExpression, limit])
            // Defensive LIKE-style verification against the content table,
            // per spec -- guards against any tokenizer/collation edge case
            // producing a candidate whose fields don't actually contain any
            // of the query tokens.
            let verified = rows.filter { record in
                tokens.contains { record.titleNormalized.contains($0) || record.authorNormalized.contains($0) }
            }
            return verified.map(CatalogCandidate.init)
        }
    }

    /// Trigram needs >= 3 characters to tokenize at all (SQLite's trigram
    /// tokenizer simply produces no tokens below that), so short reads
    /// (initials, one-word titles under 3 chars) go straight to a
    /// prefix/`LIKE` scan of the content table instead of returning zero
    /// candidates.
    private func shortReadFallback(query: String, limit: Int) throws -> [CatalogCandidate] {
        try dbQueue.read { db in
            let prefixPattern = "\(query)%"
            let wordPrefixPattern = "% \(query)%"
            let rows = try BookRecord
                .filter(
                    Column("titleNormalized").like(prefixPattern)
                        || Column("authorNormalized").like(prefixPattern)
                        || Column("titleNormalized").like(wordPrefixPattern)
                        || Column("authorNormalized").like(wordPrefixPattern)
                )
                .limit(limit)
                .fetchAll(db)
            return rows.map(CatalogCandidate.init)
        }
    }
}
