import CoreGraphics
import XCTest

@testable import SpineMatching

final class SpineRoleQueriesTests: XCTestCase {
    private let cropWidth = 80.0
    private let cropHeight = 300.0

    private func line(
        text: String, confidence: Float, top: Double, height: Double,
        alternates: [SpineTextAlternate] = []
    ) -> SpineTextLine {
        SpineTextLine(
            text: text, confidence: confidence,
            topLeft: CGPoint(x: 2, y: top), topRight: CGPoint(x: cropWidth - 2, y: top),
            bottomRight: CGPoint(x: cropWidth - 2, y: top + height), bottomLeft: CGPoint(x: 2, y: top + height),
            cropWidth: cropWidth, cropHeight: cropHeight, alternates: alternates
        )
    }

    func testEmptyLinesProducesEmptyQueries() {
        let queries = SpineRoleQueryBuilder.build(lines: [])
        XCTAssertEqual(queries.titleQuery, "")
        XCTAssertEqual(queries.authorQuery, "")
        XCTAssertTrue(queries.titleTokens.isEmpty)
    }

    func testTitleAndAuthorLinesProduceDistinctQueries() {
        let title = line(text: "Project Hail Mary", confidence: 0.9, top: 0, height: 30)
        let author = line(text: "By Andy Weir", confidence: 0.9, top: 99, height: 12)
        let queries = SpineRoleQueryBuilder.build(lines: [title, author])

        XCTAssertTrue(queries.titleQuery.contains("project"))
        XCTAssertTrue(queries.authorQuery.contains("weir"))
        XCTAssertFalse(queries.titleTokens.isEmpty)
        XCTAssertFalse(queries.authorTokens.isEmpty)
    }

    func testAllNormalizedOCRJoinsAllLinesInOrder() {
        let title = line(text: "Project Hail Mary", confidence: 0.9, top: 0, height: 30)
        let author = line(text: "By Andy Weir", confidence: 0.9, top: 99, height: 12)
        let queries = SpineRoleQueryBuilder.build(lines: [title, author])
        XCTAssertEqual(queries.allNormalizedOCR, "project hail mary by andy weir")
    }

    func testAlternatesContributeRetrievalOnlyTokensNotInCanonicalQuery() {
        // A gated alternate reading with a token that never appears in the
        // canonical text -- must show up in a token pool but never in the
        // canonical titleQuery string.
        let alt = SpineTextAlternate(text: "Prqject Hail Mary", confidence: 0.7)
        let title = line(text: "Project Hail Mary", confidence: 0.9, top: 0, height: 30, alternates: [alt])
        let queries = SpineRoleQueryBuilder.build(lines: [title])

        XCTAssertFalse(queries.titleQuery.contains("prqject"))
        XCTAssertTrue(queries.titleTokens.contains { $0.token == "prqject" })
    }

    func testConfusionExpansionOfCanonicalTokenEntersPool() {
        // "d0nut" (digit-for-letter confusion) should expand to "donut"
        // and enter the pools, subject to length/confidence gating.
        let title = line(text: "The D0nut Shop", confidence: 0.5, top: 0, height: 30)
        let queries = SpineRoleQueryBuilder.build(lines: [title])
        XCTAssertTrue(queries.titleTokens.contains { $0.token == "donut" })
    }

    func testManyExpandableTokensStayWithinDocumentedPoolCaps() {
        // Many distinct low-confidence 4-18-char tokens, each expandable --
        // regression for unbounded growth / hangs in the expansion-budget
        // bookkeeping; every pool must still respect its documented top-N cap.
        let text = (0..<40).map { "abcd\($0)e" }.joined(separator: " ")
        let title = line(text: text, confidence: 0.1, top: 0, height: 30)
        let queries = SpineRoleQueryBuilder.build(lines: [title])

        XCTAssertLessThanOrEqual(queries.titleTokens.count, 8)
        XCTAssertLessThanOrEqual(queries.authorTokens.count, 6)
        XCTAssertLessThanOrEqual(queries.generalTokens.count, 8)
        XCTAssertLessThanOrEqual(queries.fallbackTokens.count, 12)
    }

    func testRolesAmbiguousWhenNoLineStandsOutAsTitle() {
        // A single short, low, ambiguous-height line -- doesn't clearly
        // read as a title or an author.
        let vague = line(text: "Vintage", confidence: 0.6, top: 150, height: 10)
        let queries = SpineRoleQueryBuilder.build(lines: [vague])
        XCTAssertTrue(queries.rolesAmbiguous)
    }

    func testSingleLineClearingBothThresholdsOnlyQualifiesForTitleWhenTitleDominates() {
        // Regression for a real-photo failure: a single OCR line (e.g. a
        // two-word title stacked/placed such that its height/position sit
        // between the title and author geometry bands) can clear *both*
        // `titleQueryThreshold` (0.35) and `authorQueryThreshold` (0.40) at
        // once. h=0.08, p=0.4 -> title~0.51, author~0.43 here -- title is
        // higher but author still clears its own floor. Before the
        // author-dominance guard, this line's own words leaked into
        // `authorQuery` too, diluting `authorScore` against the real
        // catalog author with completely unrelated words. Only `titleQuery`
        // should claim this line.
        let line = line(text: "MOON UTAH", confidence: 0.95, top: 108, height: 24)
        let queries = SpineRoleQueryBuilder.build(lines: [line])

        XCTAssertTrue(queries.titleQuery.contains("moon"))
        XCTAssertTrue(queries.titleQuery.contains("utah"))
        XCTAssertEqual(queries.authorQuery, "", "a line title-dominates its own author score and must not double as the author query")
    }

    func testRolesNotAmbiguousWithAClearTitleLine() {
        // relativeHeight = 54/300 = 0.18 (past the author band's upper
        // edge at 0.17, so author gets no height credit) at mid-spine
        // (clear of the "other" position bands near the top/bottom edges).
        let title = line(text: "Project Hail Mary", confidence: 0.95, top: 123, height: 54)
        let queries = SpineRoleQueryBuilder.build(lines: [title])
        XCTAssertFalse(queries.rolesAmbiguous)
    }

    // MARK: - build(fmTitle:fmAuthor:) -- §G FM escalation, text-only in/out

    func testFMBuilderProducesNormalizedTitleAndAuthorQueries() {
        let queries = SpineRoleQueryBuilder.build(fmTitle: "Project Hail Mary", fmAuthor: "Andy Weir")
        XCTAssertEqual(queries.titleQuery, "project hail mary")
        XCTAssertEqual(queries.authorQuery, "andy weir")
        XCTAssertEqual(queries.allNormalizedOCR, "project hail mary andy weir")
        XCTAssertFalse(queries.rolesAmbiguous, "FM already resolved the ambiguity that triggered escalation")
    }

    func testFMBuilderWithEmptyAuthorOmitsItFromTheBlob() {
        let queries = SpineRoleQueryBuilder.build(fmTitle: "Dune", fmAuthor: "")
        XCTAssertEqual(queries.authorQuery, "")
        XCTAssertEqual(queries.allNormalizedOCR, "dune")
        XCTAssertTrue(queries.authorTokens.isEmpty)
    }

    func testFMBuilderTokensFeedTheSameRetrievalPools() {
        let queries = SpineRoleQueryBuilder.build(fmTitle: "Project Hail Mary", fmAuthor: "Andy Weir")
        XCTAssertTrue(queries.titleTokens.contains { $0.token == "project" })
        XCTAssertTrue(queries.authorTokens.contains { $0.token == "weir" })
        XCTAssertTrue(queries.generalTokens.contains { $0.token == "weir" })
    }
}
