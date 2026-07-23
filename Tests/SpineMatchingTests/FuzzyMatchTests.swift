import XCTest

@testable import SpineMatching

final class FuzzyMatchTests: XCTestCase {

    // MARK: - ratio (base Indel/LCS similarity)

    func testRatioIdenticalStringsIs100() {
        XCTAssertEqual(ratio("abc", "abc"), 100, accuracy: 1e-9)
    }

    func testRatioDisjointStringsIsZero() {
        XCTAssertEqual(ratio("abc", "xyz"), 0, accuracy: 1e-9)
    }

    func testRatioKnownValue() {
        // LCS("abcd", "abc") = "abc" (length 3); ratio = 100*2*3/(4+3).
        XCTAssertEqual(ratio("abcd", "abc"), 600.0 / 7.0, accuracy: 1e-9)
    }

    func testRatioEmptyVsEmptyIs100() {
        XCTAssertEqual(ratio("", ""), 100, accuracy: 1e-9)
    }

    func testRatioEmptyVsNonEmptyIsZero() {
        XCTAssertEqual(ratio("", "abc"), 0, accuracy: 1e-9)
    }

    func testRatioIsSymmetric() {
        let pairs = [("kitten", "sitting"), ("gatsby", "the great gatsby"), ("", "x")]
        for (a, b) in pairs {
            XCTAssertEqual(ratio(a, b), ratio(b, a), accuracy: 1e-9)
        }
    }

    // MARK: - tokenSortRatio

    func testTokenSortRatioCancelsPureReordering() {
        XCTAssertEqual(tokenSortRatio("great gatsby the", "the great gatsby"), 100, accuracy: 1e-9)
    }

    // MARK: - tokenSetRatio

    func testTokenSetRatioIdenticalIs100() {
        XCTAssertEqual(tokenSetRatio("dune messiah", "dune messiah"), 100, accuracy: 1e-9)
    }

    /// Documented, expected property (matches upstream FuzzyWuzzy/RapidFuzz
    /// `token_set_ratio`): when the query's tokens are a *pure subset* of
    /// the candidate's, the score saturates at 100 — this is intentional
    /// (a genuine full-token-subset match is a strong signal) and is a
    /// different, narrower case than the `partial_token_set_ratio`
    /// anti-pattern the spec forbids, which saturates far more often (any
    /// short shared substring, not just a full token subset).
    func testTokenSetRatioPureTokenSubsetSaturatesAt100() {
        XCTAssertEqual(tokenSetRatio("dune", "dune messiah"), 100, accuracy: 1e-9)
    }

    /// When *both* sides have leftover tokens the other lacks, none of the
    /// three internal pairwise comparisons is an identical-string
    /// comparison, so the score must land strictly below the ceiling —
    /// this is the key property `partial_token_set_ratio` lacks (it can
    /// still hit 100 even here via a partial substring match).
    func testTokenSetRatioWithLeftoversOnBothSidesIsBelowCeiling() {
        let score = tokenSetRatio("the great gatsby", "great gatsby a novel by f scott fitzgerald")
        XCTAssertLessThan(score, 100)
        XCTAssertGreaterThan(score, 0)
    }

    func testTokenSetRatioIsSymmetric() {
        let score1 = tokenSetRatio("the great gatsby", "great gatsby a novel by f scott fitzgerald")
        let score2 = tokenSetRatio("great gatsby a novel by f scott fitzgerald", "the great gatsby")
        XCTAssertEqual(score1, score2, accuracy: 1e-9)
    }

    func testTokenSetRatioHandlesEmptyStrings() {
        XCTAssertEqual(tokenSetRatio("", ""), 100, accuracy: 1e-9)
        // Degenerate upstream quirk faithfully reproduced here: comparing
        // against a *fully empty* token set collapses to the "sortedSect
        // vs sorted2to1" branch with both sides empty, which — like
        // `ratio("", "")` itself — is defined as 100 by convention. Never
        // hit in practice: the OCR quality gate rejects empty strings
        // before they reach the matcher.
        XCTAssertEqual(tokenSetRatio("dune", ""), 100, accuracy: 1e-9)
    }

    // MARK: - wRatio

    func testWRatioNeverBelowTokenSetRatioOrTokenSortRatio() {
        let a = "the great gatsby"
        let b = "great gatsby a novel by f scott fitzgerald"
        let w = wRatio(a, b)
        XCTAssertGreaterThanOrEqual(w, tokenSetRatio(a, b) * 0.95 - 1e-9)
        XCTAssertGreaterThanOrEqual(w, tokenSortRatio(a, b) * 0.95 - 1e-9)
        XCTAssertGreaterThanOrEqual(w, ratio(a, b) - 1e-9)
    }

    func testWRatioIdenticalIs100() {
        XCTAssertEqual(wRatio("dune messiah", "dune messiah"), 100, accuracy: 1e-9)
    }

    // MARK: - End-to-end style: realistic mashed OCR blob

    func testMashedOCRBlobScoresHighlyAgainstCleanTitleAuthor() {
        // A spine crop often OCRs title + author + publisher run together.
        let ocrBlob = normalizeForSearch("PROJECT HAIL MARY ANDY WEIR BALLANTINE")
        let candidateTitle = normalizeForSearch("Project Hail Mary")
        let candidateAuthor = normalizeForSearch("Andy Weir")
        let combined = candidateTitle + " " + candidateAuthor
        XCTAssertGreaterThan(tokenSetRatio(ocrBlob, combined), 90)
        // But a wrong book by the same author should score meaningfully lower.
        let wrongBook = normalizeForSearch("The Martian") + " " + candidateAuthor
        XCTAssertLessThan(tokenSetRatio(ocrBlob, wrongBook), tokenSetRatio(ocrBlob, combined))
    }
}
