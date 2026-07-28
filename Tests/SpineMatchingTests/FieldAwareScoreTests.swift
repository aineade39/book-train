import XCTest

@testable import SpineMatching

final class FieldAwareScoreTests: XCTestCase {
    func testPopularityScoreIsHighestAtRankOne() {
        let rank1 = FieldAwareScore.popularityScore(rank: 1, catalogSize: 250_000)
        XCTAssertEqual(rank1, 100, accuracy: 0.01)
    }

    func testPopularityScoreDecaysTowardZeroAtCatalogSize() {
        let rankMax = FieldAwareScore.popularityScore(rank: 250_000, catalogSize: 250_000)
        XCTAssertEqual(rankMax, 0, accuracy: 0.01)
    }

    func testPopularityScoreIsZeroWithNoRank() {
        XCTAssertEqual(FieldAwareScore.popularityScore(rank: nil), 0)
    }

    func testPopularityScoreIsMonotonicallyDecreasing() {
        let a = FieldAwareScore.popularityScore(rank: 100, catalogSize: 250_000)
        let b = FieldAwareScore.popularityScore(rank: 10_000, catalogSize: 250_000)
        XCTAssertGreaterThan(a, b)
    }

    func testFinalScoreForCleanTitleAuthorMatchIsHigh() {
        let score = FieldAwareScore.finalScore(
            titleQuery: "project hail mary", authorQuery: "andy weir",
            allNormalizedOCR: "project hail mary andy weir ballantine",
            candidateTitle: "Project Hail Mary", candidateAuthor: "Andy Weir",
            candidateSearchableText: "Project Hail Mary Andy Weir",
            popularityRank: 500
        )
        XCTAssertGreaterThan(score, 90)
    }

    func testFinalScoreWithoutAuthorQueryUsesTitleBlobBlendOnly() {
        let withAuthor = FieldAwareScore.finalScore(
            titleQuery: "dune", authorQuery: "frank herbert",
            allNormalizedOCR: "dune frank herbert",
            candidateTitle: "Dune", candidateAuthor: "Someone Else",
            candidateSearchableText: "Dune Someone Else", popularityRank: nil
        )
        let withoutAuthor = FieldAwareScore.finalScore(
            titleQuery: "dune", authorQuery: "",
            allNormalizedOCR: "dune frank herbert",
            candidateTitle: "Dune", candidateAuthor: "Someone Else",
            candidateSearchableText: "Dune Someone Else", popularityRank: nil
        )
        // Dropping a mismatched author from the blend should never hurt
        // the score.
        XCTAssertGreaterThanOrEqual(withoutAuthor, withAuthor)
    }

    func testMorePopularCandidateScoresHigherAllElseEqual() {
        let popular = FieldAwareScore.finalScore(
            titleQuery: "dune", authorQuery: "", allNormalizedOCR: "dune",
            candidateTitle: "Dune", candidateAuthor: "Frank Herbert",
            candidateSearchableText: "Dune Frank Herbert", popularityRank: 1
        )
        let obscure = FieldAwareScore.finalScore(
            titleQuery: "dune", authorQuery: "", allNormalizedOCR: "dune",
            candidateTitle: "Dune", candidateAuthor: "Frank Herbert",
            candidateSearchableText: "Dune Frank Herbert", popularityRank: 200_000
        )
        XCTAssertGreaterThan(popular, obscure)
    }
}
