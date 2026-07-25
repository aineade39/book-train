import XCTest

@testable import SpineCatalog
@testable import SpineMatching

final class BookCatalogTests: XCTestCase {
    private func makeCatalog() throws -> BookCatalog {
        try BookCatalog.inMemory()
    }

    private func seedSampleBooks(_ catalog: BookCatalog) throws {
        try catalog.insert(title: "Dune", author: "Frank Herbert", isbn: "9780441013593")
        try catalog.insert(title: "Dune Messiah", author: "Frank Herbert", isbn: "9780441172696")
        try catalog.insert(title: "Project Hail Mary", author: "Andy Weir", isbn: "9780593135204")
        try catalog.insert(title: "The Martian", author: "Andy Weir", isbn: "9780553418026")
        try catalog.insert(title: "Gone Girl", author: "Gillian Flynn", isbn: "9780307588371")
        // Two editions of the same work -- same title/author, different ISBN.
        try catalog.insert(title: "Gone Girl", author: "Gillian Flynn", isbn: "9780307588388")
    }

    // MARK: - Basic retrieval (trigram, >= 3 chars)

    func testRetrievesExactTitleMatch() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        let results = try catalog.retrieveCandidates(forQuery: "Dune Messiah")
        XCTAssertTrue(results.contains { $0.title == "Dune Messiah" })
    }

    func testRetrievesSubstringOfTitle() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        // "hail mary" is a substring of "Project Hail Mary" but not a prefix.
        let results = try catalog.retrieveCandidates(forQuery: "hail mary")
        XCTAssertTrue(results.contains { $0.title == "Project Hail Mary" })
    }

    func testRetrievesByAuthorSubstring() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        let results = try catalog.retrieveCandidates(forQuery: "herbert")
        XCTAssertTrue(results.contains { $0.author == "Frank Herbert" })
        XCTAssertGreaterThanOrEqual(results.count, 2) // Dune + Dune Messiah
    }

    func testRetrievesMashedOCRBlobContainingCleanSubstring() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        // A spine crop OCR blob mashes title+author+publisher together;
        // retrieval only needs *a* matching substring window, not an exact
        // full-string match.
        let results = try catalog.retrieveCandidates(forQuery: "PROJECT HAIL MARY ANDY WEIR BALLANTINE")
        XCTAssertTrue(results.contains { $0.title == "Project Hail Mary" })
    }

    func testNoMatchReturnsEmptyNotError() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        let results = try catalog.retrieveCandidates(forQuery: "nonexistent book title xyz")
        XCTAssertTrue(results.isEmpty)
    }

    func testEmptyQueryThrows() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        XCTAssertThrowsError(try catalog.retrieveCandidates(forQuery: "   "))
    }

    // MARK: - Short-read fallback (< 3 chars)

    func testShortReadFallsBackToPrefixLikeInsteadOfEmpty() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        // "du" is below the trigram threshold; must not silently return zero.
        let results = try catalog.retrieveCandidates(forQuery: "du")
        XCTAssertTrue(results.contains { $0.title == "Dune" })
    }

    func testShortReadMatchesWordPrefixNotJustStringPrefix() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        // "he" is a prefix of "Herbert" (a second word), not of any title.
        let results = try catalog.retrieveCandidates(forQuery: "he")
        XCTAssertTrue(results.contains { $0.author == "Frank Herbert" })
    }

    func testCountBooksReflectsInsertedRows() throws {
        let catalog = try makeCatalog()
        XCTAssertEqual(try catalog.countBooks(), 0)
        try seedSampleBooks(catalog)
        XCTAssertEqual(try catalog.countBooks(), 6)
    }

    // MARK: - Edition grouping

    func testSameTitleAuthorEditionsShareWorkKey() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)
        let results = try catalog.retrieveCandidates(forQuery: "gone girl")
        let goneGirls = results.filter { $0.title == "Gone Girl" }
        XCTAssertEqual(goneGirls.count, 2, "expected both editions to be retrievable")
        XCTAssertEqual(Set(goneGirls.map(\.workKey)).count, 1, "editions of the same work should share a workKey")
    }

    // MARK: - End-to-end with SpineMatching (Stage 1 + Stage 2 + accept policy)

    func testEndToEndRetrieveRerankAndAccept() throws {
        let catalog = try makeCatalog()
        try seedSampleBooks(catalog)

        let ocrBlob = "PROJECT HAIL MARY ANDY WEIR BALLANTINE BOOKS"
        let normalizedQuery = normalizeForSearch(ocrBlob)
        let candidates = try catalog.retrieveCandidates(forQuery: ocrBlob)
        XCTAssertFalse(candidates.isEmpty)

        let scored = candidates.map {
            ScoredCandidate(candidate: $0, score: tokenSetRatio(normalizedQuery, normalizeForSearch($0.searchableText)))
        }
        let policy = AcceptPolicy(acceptThreshold: 85, marginThreshold: 8, topN: 5)
        guard case .autoAccept(let winner) = policy.decide(scored) else {
            return XCTFail("expected a confident auto-accept for a clean OCR blob")
        }
        XCTAssertEqual(winner.candidate.title, "Project Hail Mary")
    }

    func testBulkInsertPreservesFTS() throws {
        let catalog = try makeCatalog()
        let rows = [
            BookCatalog.WorkInsert(workKey: "/works/OL1W", title: "Dune", author: "Frank Herbert", popularityRank: 1, editionCount: 2),
            BookCatalog.WorkInsert(workKey: "/works/OL2W", title: "Dune Messiah", author: "Frank Herbert", popularityRank: 2, editionCount: 1),
        ]
        XCTAssertEqual(try catalog.bulkInsert(rows), 2)
        let results = try catalog.retrieveCandidates(forQuery: "herbert")
        XCTAssertGreaterThanOrEqual(results.count, 2)
    }

    /// Common OR-tokens must not starve distinctive title words out of the
    /// shortlist — regression for unranked `LIMIT 50` FTS on ios_en.
    func testBM25PrefersDistinctiveTitleOverCommonTokenFlood() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "The Frogs Wore Red Suspenders", author: "Jack Prelutsky")
        // Flood the catalog with books that match the common OCR token "red".
        for i in 0..<80 {
            try catalog.insert(title: "Red Book \(i)", author: "Someone \(i)")
        }
        let ocr = "SKY MATHERS THE FROGS WORE RED SUSPENDERS HARPERTROPHY"
        let results = try catalog.retrieveCandidates(forQuery: ocr, limit: 50)
        XCTAssertTrue(
            results.contains { $0.title == "The Frogs Wore Red Suspenders" },
            "expected BM25-ranked shortlist to include the distinctive title; got: \(results.prefix(5).map(\.title))"
        )
        XCTAssertEqual(results.first?.title, "The Frogs Wore Red Suspenders")
    }
}
