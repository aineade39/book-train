import CoreGraphics
import XCTest

@testable import SpineCatalog
@testable import SpineMatching

final class BookCatalogISBNAndRoleRetrievalTests: XCTestCase {
    private func makeCatalog() throws -> BookCatalog {
        try BookCatalog.inMemory()
    }

    // MARK: - book_isbns (§D)

    func testLookupISBNReturnsTheAssociatedWork() throws {
        let catalog = try makeCatalog()
        let id = try catalog.insert(title: "Dune", author: "Frank Herbert", workKey: "/works/OL1W")
        try catalog.insertISBNs([(isbn13: "9780441013593", workKey: "/works/OL1W")])

        let results = try catalog.lookupISBN("9780441013593")
        XCTAssertEqual(results.map(\.id), [id])
        XCTAssertEqual(results.first?.title, "Dune")
    }

    func testLookupISBNWithNoMatchReturnsEmpty() throws {
        let catalog = try makeCatalog()
        XCTAssertTrue(try catalog.lookupISBN("9780441013593").isEmpty)
    }

    func testLookupISBNSharedAcrossMultipleWorksReturnsAll() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Book A", author: "Author A", workKey: "/works/OLA")
        try catalog.insert(title: "Book B", author: "Author B", workKey: "/works/OLB")
        // Rare OL data-quality case: same ISBN mapped to two distinct works.
        try catalog.insertISBNs([
            (isbn13: "9780441013593", workKey: "/works/OLA"),
            (isbn13: "9780441013593", workKey: "/works/OLB"),
        ])
        let results = try catalog.lookupISBN("9780441013593")
        XCTAssertEqual(Set(results.map(\.title)), ["Book A", "Book B"])
    }

    func testInsertISBNsIsIdempotent() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Dune", author: "Frank Herbert", workKey: "/works/OL1W")
        try catalog.insertISBNs([(isbn13: "9780441013593", workKey: "/works/OL1W")])
        try catalog.insertISBNs([(isbn13: "9780441013593", workKey: "/works/OL1W")])
        XCTAssertEqual(try catalog.lookupISBN("9780441013593").count, 1)
    }

    // MARK: - custom_words (§C)

    func testCustomWordsRoundTrip() throws {
        let catalog = try makeCatalog()
        try catalog.insertCustomWords(["Herbert", "Weir", "Flynn"])
        XCTAssertEqual(try catalog.customWords(), ["Flynn", "Herbert", "Weir"])
    }

    func testInsertCustomWordsReplacesPreviousContents() throws {
        let catalog = try makeCatalog()
        try catalog.insertCustomWords(["Old"])
        try catalog.insertCustomWords(["New"])
        XCTAssertEqual(try catalog.customWords(), ["New"])
    }

    // MARK: - Role-aware retrieval (§B)

    private func line(text: String, top: Double, height: Double, cropHeight: Double = 300) -> SpineTextLine {
        SpineTextLine(
            text: text, confidence: 0.9,
            topLeft: CGPoint(x: 2, y: top), topRight: CGPoint(x: 38, y: top),
            bottomRight: CGPoint(x: 38, y: top + height), bottomLeft: CGPoint(x: 2, y: top + height),
            cropWidth: 40, cropHeight: cropHeight
        )
    }

    func testRoleAwareRetrievalFindsCandidateFromTitleLine() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Project Hail Mary", author: "Andy Weir")
        try catalog.insert(title: "Dune", author: "Frank Herbert")

        let title = line(text: "Project Hail Mary", top: 126, height: 48)
        let author = line(text: "By Andy Weir", top: 200, height: 20)
        let queries = SpineRoleQueryBuilder.build(lines: [title, author])

        let results = try catalog.retrieveRoleAware(queries)
        XCTAssertTrue(results.contains { $0.title == "Project Hail Mary" })
    }

    func testRoleAwareRetrievalEmptyShortlistFallsBackToUnscopedPool() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "The Frogs Wore Red Suspenders", author: "Jack Prelutsky")

        // Title-only pool won't match (wrong words); general/fallback pool
        // should still surface the right row from the author-ish line.
        let odd = line(text: "Suspenders Prelutsky", top: 126, height: 48)
        let queries = SpineRoleQueryBuilder.build(lines: [odd])
        let results = try catalog.retrieveRoleAware(queries)
        XCTAssertTrue(results.contains { $0.title == "The Frogs Wore Red Suspenders" })
    }

    func testRoleAwareRetrievalReturnsEmptyNotErrorOnNoMatch() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Dune", author: "Frank Herbert")
        let line = line(text: "Completely Unrelated Zzyzx Nonexistent", top: 126, height: 48)
        let queries = SpineRoleQueryBuilder.build(lines: [line])
        XCTAssertTrue(try catalog.retrieveRoleAware(queries).isEmpty)
    }

    // MARK: - matchRoleAware (§B rerank, §H shared CLI/app match path)

    /// Builds `SpineRoleQueries` directly (bypassing geometry/role-scoring
    /// entirely) so these tests exercise `matchRoleAware`'s retrieve ->
    /// rerank -> accept wiring without depending on precisely-tuned
    /// title/author band geometry (that's `SpineRoleScoringTests`'/
    /// `SpineRoleQueriesTests`' job) -- equivalent to what
    /// `SpineRoleQueryBuilder.build(fmTitle:fmAuthor:)` already produces
    /// for a clean, unambiguous title+author read.
    private func roleQueries(title: String, author: String) -> SpineRoleQueries {
        SpineRoleQueryBuilder.build(fmTitle: title, fmAuthor: author)
    }

    func testMatchRoleAwareAutoAcceptsClearTitleAndAuthorMatch() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Project Hail Mary", author: "Andy Weir")
        try catalog.insert(title: "Artemis", author: "Andy Weir")

        let outcome = try catalog.matchRoleAware(roleQueries(title: "Project Hail Mary", author: "Andy Weir"))
        guard case .autoAccept(let winner) = outcome.decision else {
            return XCTFail("expected .autoAccept, got \(outcome.decision)")
        }
        XCTAssertEqual(winner.candidate.title, "Project Hail Mary")
        // "Artemis" shares the author but not the title, so it's a genuine
        // (lower-scoring) distinct-work runner-up -- margin must be
        // reported, and must clear the policy's own bar for auto-accept
        // to have happened at all.
        XCTAssertNotNil(outcome.margin)
        XCTAssertGreaterThanOrEqual(outcome.margin ?? 0, AcceptPolicy().marginThreshold)
    }

    func testMatchRoleAwareIsAmbiguousBetweenSimilarlyScoredWorks() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Dune", author: "Frank Herbert")
        try catalog.insert(title: "Dune Messiah", author: "Frank Herbert")

        // "Frank Herbert" alone (no title read) scores close to both Dune
        // books -- neither clears the margin bar over the other.
        let outcome = try catalog.matchRoleAware(roleQueries(title: "", author: "Frank Herbert"))
        guard case .ambiguous = outcome.decision else {
            return XCTFail("expected .ambiguous, got \(outcome.decision)")
        }
        XCTAssertNotNil(outcome.margin, "two genuinely competing works should still report a (small) margin")
    }

    func testMatchRoleAwareNoMatchOnEmptyCatalog() throws {
        let catalog = try makeCatalog()
        let title = line(text: "Project Hail Mary", top: 126, height: 48)
        let outcome = try catalog.matchRoleAware(SpineRoleQueryBuilder.build(lines: [title]))
        guard case .noMatch = outcome.decision else { return XCTFail("expected .noMatch") }
        XCTAssertNil(outcome.margin)
    }

    /// A genuine DB failure must throw, not look like `.noMatch` -- callers
    /// (e.g. `SpineIdentificationEngine.matchDecision`) rely on this to
    /// distinguish "no match found" from "the catalog is broken" and must
    /// not swallow it with `try?`. Deleting the underlying file wouldn't
    /// reliably reproduce a failure here: POSIX keeps an already-open file
    /// descriptor (and `BookCatalog`'s `mmap_size` pragma) readable past
    /// unlink, so dropping the FTS shadow table instead guarantees a real
    /// "no such table" error on the next query, independent of OS/filesystem
    /// behavior.
    func testMatchRoleAwareThrowsOnDatabaseErrorInsteadOfReturningNoMatch() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Project Hail Mary", author: "Andy Weir")
        try catalog.dbQueue.write { db in try db.execute(sql: "DROP TABLE books_fts") }

        XCTAssertThrowsError(
            try catalog.matchRoleAware(roleQueries(title: "Project Hail Mary", author: "Andy Weir"))
        )
    }

    /// §G: FM escalation output must flow through the exact same
    /// retrieve/rerank/accept path as geometry-derived queries.
    func testMatchRoleAwareAcceptsFMBuiltQueries() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Project Hail Mary", author: "Andy Weir")
        try catalog.insert(title: "Dune", author: "Frank Herbert")

        let queries = SpineRoleQueryBuilder.build(fmTitle: "Project Hail Mary", fmAuthor: "Andy Weir")
        let outcome = try catalog.matchRoleAware(queries)
        guard case .autoAccept(let winner) = outcome.decision else {
            return XCTFail("expected .autoAccept, got \(outcome.decision)")
        }
        XCTAssertEqual(winner.candidate.title, "Project Hail Mary")
    }

    // MARK: - Match-field dedup, defense-in-depth ("Catalog match quality" plan §Part C)

    /// Distinct `workKey`s sharing normalized match fields (differing
    /// punctuation, not just casing -- `normalizeForSearch` already covers
    /// case) is the OL data-quality pattern build-time dedup collapses at
    /// catalog-build time; this is the runtime safety net for catalogs
    /// that don't go through that ETL (e.g. this in-memory test catalog).
    func testRetrieveRoleAwareDedupesSameMatchFields() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Dune", author: "Frank Herbert", workKey: "/works/OLA")
        try catalog.insert(title: "Dune!", author: "Frank Herbert", workKey: "/works/OLB")

        let results = try catalog.retrieveRoleAware(roleQueries(title: "Dune", author: "Frank Herbert"))
        XCTAssertEqual(results.count, 1, "distinct workKeys sharing normalized match fields should collapse to one candidate")
    }

    /// Regression guard: distinct titles sharing an author must not be
    /// collapsed just because they share one normalized field.
    func testRetrieveRoleAwareKeepsDistinctMatchFields() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Dune", author: "Frank Herbert")
        try catalog.insert(title: "Dune Messiah", author: "Frank Herbert")

        let results = try catalog.retrieveRoleAware(roleQueries(title: "", author: "Frank Herbert"))
        XCTAssertEqual(
            Set(results.map(\.title)), ["Dune", "Dune Messiah"],
            "distinct titles must not be collapsed by match-field dedup"
        )
    }

    func testMatchFieldDedupPrefersLowerPopularityRank() throws {
        let catalog = try makeCatalog()
        _ = try catalog.bulkInsert([
            BookCatalog.WorkInsert(workKey: "/works/OLA", title: "Dune", author: "Frank Herbert", popularityRank: 500),
            BookCatalog.WorkInsert(workKey: "/works/OLB", title: "Dune!", author: "Frank Herbert", popularityRank: 5),
        ])

        let results = try catalog.retrieveRoleAware(roleQueries(title: "Dune", author: "Frank Herbert"))
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.workKey, "/works/OLB", "lower (more popular) popularityRank should win the dedup group")
    }

    /// Without upstream match-field dedup, two duplicate-`workKey` rows
    /// with near-identical scores would tie in `AcceptPolicy.bestPerWork`
    /// (which is workKey-scoped, so it sees them as two distinct works)
    /// and block the margin test -- `.ambiguous` instead of `.autoAccept`,
    /// even though there's really only one candidate work here plus one
    /// clearly weaker unrelated runner-up. Dedup applied in
    /// `retrieveRoleAware`, *before* scoring, fixes this.
    func testMatchRoleAwareAutoAcceptsWhenMatchFieldDupWouldHaveBlockedMargin() throws {
        let catalog = try makeCatalog()
        _ = try catalog.bulkInsert([
            BookCatalog.WorkInsert(workKey: "/works/OLA", title: "Dune", author: "Frank Herbert", popularityRank: 1),
            BookCatalog.WorkInsert(workKey: "/works/OLB", title: "Dune!", author: "Frank Herbert", popularityRank: 2),
        ])
        // Shares the author (so the author-scoped pass retrieves it into
        // the shortlist at all -- an unrelated title+author wouldn't be
        // retrieved by this query, and so couldn't have blocked the
        // margin test either) but a completely different title, so it
        // scores far behind the "Dune" duplicate group once reranked --
        // same shape as `testMatchRoleAwareAutoAcceptsClearTitleAndAuthorMatch`'s
        // "Artemis" runner-up.
        try catalog.insert(title: "The Godmakers", author: "Frank Herbert", workKey: "/works/OLC")

        let outcome = try catalog.matchRoleAware(roleQueries(title: "Dune", author: "Frank Herbert"))
        guard case .autoAccept(let winner) = outcome.decision else {
            return XCTFail("expected .autoAccept, got \(outcome.decision)")
        }
        XCTAssertEqual(winner.candidate.workKey, "/works/OLA")
        XCTAssertNotNil(outcome.margin)
        XCTAssertGreaterThanOrEqual(outcome.margin ?? 0, AcceptPolicy().marginThreshold)
    }
}
