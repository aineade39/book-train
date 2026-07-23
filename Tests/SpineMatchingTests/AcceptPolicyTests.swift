import XCTest

@testable import SpineMatching

private struct FakeBook: RankableCandidate {
    let workKey: String
    let title: String
}

final class AcceptPolicyTests: XCTestCase {
    let policy = AcceptPolicy(acceptThreshold: 90, marginThreshold: 8, topN: 5)

    func testNoMatchOnEmptyCandidates() {
        let decision = policy.decide([ScoredCandidate<FakeBook>]())
        guard case .noMatch = decision else { return XCTFail("expected .noMatch") }
    }

    func testAutoAcceptsHighScoreWithClearMargin() {
        let candidates = [
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune"), score: 96),
            ScoredCandidate(candidate: FakeBook(workKey: "dune-messiah", title: "Dune Messiah"), score: 60),
        ]
        guard case .autoAccept(let winner) = policy.decide(candidates) else {
            return XCTFail("expected .autoAccept")
        }
        XCTAssertEqual(winner.candidate.workKey, "dune")
    }

    func testAmbiguousWhenBelowThresholdEvenWithMargin() {
        let candidates = [
            ScoredCandidate(candidate: FakeBook(workKey: "a", title: "A"), score: 70),
            ScoredCandidate(candidate: FakeBook(workKey: "b", title: "B"), score: 20),
        ]
        guard case .ambiguous = policy.decide(candidates) else { return XCTFail("expected .ambiguous") }
    }

    func testAmbiguousWhenAboveThresholdButNoMargin() {
        let candidates = [
            ScoredCandidate(candidate: FakeBook(workKey: "a", title: "A"), score: 95),
            ScoredCandidate(candidate: FakeBook(workKey: "b", title: "B"), score: 92),
        ]
        guard case .ambiguous(let top) = policy.decide(candidates) else { return XCTFail("expected .ambiguous") }
        XCTAssertEqual(top.first?.candidate.workKey, "a")
    }

    /// The core "editions" requirement: multiple editions of the *same*
    /// work must not block auto-accept just because they're technically
    /// separate rows — they collapse to one entry, and the real runner-up
    /// is the next *different* work.
    func testEditionsOfSameWorkDoNotBlockAutoAccept() {
        let candidates = [
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune (Mass Market)"), score: 96),
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune (Hardcover 40th Anniversary)"), score: 95),
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune (Paperback)"), score: 94),
            ScoredCandidate(candidate: FakeBook(workKey: "dune-messiah", title: "Dune Messiah"), score: 55),
        ]
        guard case .autoAccept(let winner) = policy.decide(candidates) else {
            return XCTFail("expected .autoAccept despite same-work duplicates")
        }
        XCTAssertEqual(winner.candidate.workKey, "dune")
        XCTAssertEqual(winner.score, 96)
    }

    /// Without dedup, three near-tied editions of one book could look like
    /// "ambiguous" (small margin between #1 and #2) even though there is
    /// really only one plausible *work* in the shortlist.
    func testWithoutEditionDedupWouldHaveLookedAmbiguous() {
        let candidates = [
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune (Mass Market)"), score: 96),
            ScoredCandidate(candidate: FakeBook(workKey: "dune", title: "Dune (Hardcover)"), score: 95),
        ]
        // Best-per-work dedup means only one "dune" entry remains, so with
        // no other distinct work present the runner-up is -infinity and
        // this still auto-accepts.
        guard case .autoAccept = policy.decide(candidates) else {
            return XCTFail("expected .autoAccept — same-work near-ties must not create a false margin failure")
        }
    }

    func testTopNLimitsAmbiguousCandidateList() {
        let smallTopNPolicy = AcceptPolicy(acceptThreshold: 90, marginThreshold: 8, topN: 2)
        let candidates = (0..<10).map {
            ScoredCandidate(candidate: FakeBook(workKey: "book-\($0)", title: "Book \($0)"), score: Double(50 + $0))
        }
        guard case .ambiguous(let top) = smallTopNPolicy.decide(candidates) else {
            return XCTFail("expected .ambiguous")
        }
        XCTAssertEqual(top.count, 2)
        XCTAssertEqual(top.first?.candidate.workKey, "book-9")
    }
}
