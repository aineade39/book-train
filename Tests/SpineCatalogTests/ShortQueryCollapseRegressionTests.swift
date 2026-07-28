import CoreGraphics
import XCTest

@testable import SpineCatalog
@testable import SpineMatching

/// Regression coverage for the "short-query collapse" failure pattern found
/// across the `spine-id-scenes-advisory` eval's `ambiguous-matches.html`
/// report: a multi-line vertical spine where no single OCR line clears the
/// title-role threshold used to collapse `titleQuery` down to one arbitrary
/// line (`SpineRoleQueries.canonicalQuery`'s old `fallbackToMax` behavior).
///
/// A second mechanism (a coverage/specificity discount in
/// `FieldAwareScore`, meant to stop a short generic candidate from tying a
/// correct longer title via `tokenSetRatio`'s subset-match ceiling) was
/// also implemented and tested here, but reverted after measuring a net
/// regression across all 5 real oracle scenes -- see the "tried-and-
/// reverted" comment on `FieldAwareScore.finalScore` for the full
/// writeup. It turned out the query-collapse fix below was sufficient on
/// its own for the motivating case.
final class ShortQueryCollapseRegressionTests: XCTestCase {
    private func makeCatalog() throws -> BookCatalog {
        try BookCatalog.inMemory()
    }

    private func line(text: String, top: Double, height: Double, cropHeight: Double) -> SpineTextLine {
        SpineTextLine(
            text: text, confidence: 0.9,
            topLeft: CGPoint(x: 2, y: top), topRight: CGPoint(x: 38, y: top),
            bottomRight: CGPoint(x: 38, y: top + height), bottomLeft: CGPoint(x: 2, y: top + height),
            cropWidth: 40, cropHeight: cropHeight
        )
    }

    // MARK: - Fix 1 + Fix 4, end-to-end: "The Invention of Nature" vs. "Beowulf"

    /// The flagship case: Vision reads a vertical spine reading (top to
    /// bottom) "The" / "INVENTION" / "NATURE" / "Andrea" / "Wulf" /
    /// "Vintag" as six separate single-word lines, none of which clears
    /// the 0.35 title-role threshold on its own. Before the fix,
    /// `canonicalQuery`'s fallback collapsed `titleQuery` to the single
    /// highest-scoring line -- "wulf" -- which then retrieved and
    /// (via `tokenSetRatio`'s subset-match ceiling) out-scored the correct
    /// book with "Beowulf" and "Wulf the Saxon". After the fix, every line
    /// that leans title-over-author *and* isn't itself `other`-dominant
    /// gets joined; at this synthetic geometry's line height, "The" and
    /// "INVENTION" happen to score `other`-dominant (short, near-top lines
    /// sit close to that role's position band) and get filtered back out,
    /// leaving `titleQuery` as just "nature" -- still enough on its own
    /// (a full-subset match against "The Invention of Nature", and a weak
    /// one against "Beowulf"/"Wulf the Saxon") for the correct book to win
    /// by a wide, unambiguous margin. This is also confirmed against the
    /// real production spine in the `spine-id-scenes-advisory` oracle eval
    /// (`bedroom1`), which flips from ambiguous/"Beowulf" to
    /// auto-accept/"The Invention of Nature" with this fix.
    func testInventionOfNatureBeatsBeowulfDespiteEveryLineReadingAsASingleWord() throws {
        let cropHeight = 520.0
        let lines = [
            line(text: "The", top: 0.10 * cropHeight, height: 23, cropHeight: cropHeight),
            line(text: "INVENTION", top: 0.20 * cropHeight, height: 23, cropHeight: cropHeight),
            line(text: "NATURE", top: 0.29 * cropHeight, height: 23, cropHeight: cropHeight),
            line(text: "Andrea", top: 0.55 * cropHeight, height: 23, cropHeight: cropHeight),
            line(text: "Wulf", top: 0.70 * cropHeight, height: 23, cropHeight: cropHeight),
            line(text: "Vintag", top: 0.86 * cropHeight, height: 23, cropHeight: cropHeight),
        ]

        let queries = SpineRoleQueryBuilder.build(lines: lines)
        XCTAssertTrue(queries.titleQueryIsFallback, "no single line should clear the title threshold here")
        XCTAssertTrue(queries.titleQuery.contains("nature"), "titleQuery was '\(queries.titleQuery)'")
        XCTAssertFalse(
            queries.titleQuery == "wulf",
            "titleQuery must not collapse to the single stray author-name line"
        )

        let catalog = try makeCatalog()
        try catalog.insert(title: "The Invention of Nature", author: "Andrea Wulf", workKey: "/works/invention")
        try catalog.insert(title: "Beowulf", author: "Unknown", workKey: "/works/beowulf")
        try catalog.insert(title: "Wulf the Saxon", author: "Henry William Herbert", workKey: "/works/wulfsaxon")

        let outcome = try catalog.matchRoleAware(queries)
        let winner: CatalogCandidate
        switch outcome.decision {
        case .autoAccept(let scored):
            winner = scored.candidate
        case .ambiguous(let topCandidates):
            guard let top = topCandidates.first else { return XCTFail("expected at least one candidate") }
            winner = top.candidate
        case .noMatch:
            return XCTFail("expected a match, got .noMatch")
        }
        XCTAssertEqual(winner.title, "The Invention of Nature", "wrong winner: \(winner.title)")
    }

    // MARK: - Word-boundary retrieval verification

    /// `"wulf"` must not retrieve `"Beowulf"` via `columnRetrieve`'s
    /// defensive verification -- a raw substring check would let it
    /// through even though `"wulf"` never appears as its own word in
    /// `"beowulf"`.
    func testWulfTokenDoesNotRetrieveBeowulfViaSubstring() throws {
        let catalog = try makeCatalog()
        try catalog.insert(title: "Beowulf", author: "Unknown", workKey: "/works/beowulf")
        try catalog.insert(title: "Wulf the Saxon", author: "Henry William Herbert", workKey: "/works/wulfsaxon")

        let queries = SpineRoleQueryBuilder.build(fmTitle: "Wulf", fmAuthor: "")
        let results = try catalog.retrieveRoleAware(queries)

        XCTAssertFalse(results.contains { $0.title == "Beowulf" }, "\"wulf\" leaked into \"Beowulf\" via substring match")
        XCTAssertTrue(results.contains { $0.title == "Wulf the Saxon" }, "a genuine whole-word match must still retrieve")
    }
}
