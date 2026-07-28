import Foundation
import GRDB
import SpineMatching

// Local book catalog: SQLite via GRDB, per
// docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching / §Data model —
// "store both display and search-normalized fields."

/// One catalog row (an *edition* — see `workKey`). This is the FTS5
/// `content` table: the authoritative store, synchronized one-way into
/// `books_fts` by GRDB-managed triggers (`FTS5TableDefinition.synchronize`).
public struct BookRecord: Codable, FetchableRecord, MutablePersistableRecord {
    public var id: Int64?
    /// Groups editions of the same underlying work so the accept policy's
    /// margin test isn't defeated by a book's own reprints (see
    /// `SpineMatching.RankableCandidate` / `AcceptPolicy`). Defaults to
    /// `normalizedTitle|normalizedAuthor` (see `BookCatalog.defaultWorkKey`)
    /// but callers may supply a real work identifier (e.g. an OpenLibrary
    /// work key) when known.
    public var workKey: String
    public var title: String
    public var author: String
    public var isbn: String?
    /// Search-normalized (see `normalizeForSearch`) mirrors of `title`/
    /// `author` — these, not the display fields, are what `books_fts`
    /// indexes and what `LIKE` verification/fallback compares against.
    public var titleNormalized: String
    public var authorNormalized: String
    /// Lower is more popular (OL edition-count rank). Optional — used for subset
    /// rebuild ordering, not for matching (see docs/BOOK_CATALOG.md).
    public var popularityRank: Int?
    public var editionCount: Int?

    public static let databaseTableName = "books"

    public init(
        id: Int64? = nil,
        workKey: String,
        title: String,
        author: String,
        isbn: String? = nil,
        popularityRank: Int? = nil,
        editionCount: Int? = nil
    ) {
        self.id = id
        self.workKey = workKey
        self.title = title
        self.author = author
        self.isbn = isbn
        self.titleNormalized = normalizeForSearch(title)
        self.authorNormalized = normalizeForSearch(author)
        self.popularityRank = popularityRank
        self.editionCount = editionCount
    }

    public mutating func didInsert(_ inserted: InsertionSuccess) {
        id = inserted.rowID
    }
}

/// A retrieval-stage shortlist entry — the shape `SpineMatching`'s fuzzy
/// rerank and `AcceptPolicy` consume. Deliberately does not expose
/// `titleNormalized`/`authorNormalized` (those are catalog-internal).
public struct CatalogCandidate: RankableCandidate, Equatable {
    public let id: Int64
    public let workKey: String
    public let title: String
    public let author: String
    public let isbn: String?
    /// Lower is more popular (OL edition-count rank) -- feeds the
    /// field-aware rerank's small popularity term (`SpineMatching.FieldAwareScore`).
    /// `nil` for catalogs without OL metadata (CSV imports, tests).
    public let popularityRank: Int?

    /// Concatenated display text, matching the "mashed title+author+
    /// publisher strings" shape a spine OCR blob is compared against.
    public var searchableText: String { "\(title) \(author)" }

    init(_ record: BookRecord) {
        id = record.id ?? 0
        workKey = record.workKey
        title = record.title
        author = record.author
        isbn = record.isbn
        popularityRank = record.popularityRank
    }
}
