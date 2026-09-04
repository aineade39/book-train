import XCTest

@testable import SpineReasoning

/// Pure logic, no `FoundationModels`/OS-version dependency -- runs
/// everywhere `SpineCoreTests` etc. do, unlike the gated
/// `SpineReasoningServiceTests`.
final class FMEscalationPolicyTests: XCTestCase {
    // MARK: - isHardCase

    func testIsHardCaseIsTrueInsideTheMarginalBandAndFalseOutsideIt() {
        let policy = FMEscalationPolicy(marginalScoreRange: 0.35...0.65, maxEscalationFraction: 0.05, minEscalations: 1)
        XCTAssertFalse(policy.isHardCase(qualityScore: 0.1), "clearly-bad reads have no text worth cleaning up")
        XCTAssertTrue(policy.isHardCase(qualityScore: 0.35))
        XCTAssertTrue(policy.isHardCase(qualityScore: 0.5))
        XCTAssertTrue(policy.isHardCase(qualityScore: 0.65))
        XCTAssertFalse(policy.isHardCase(qualityScore: 0.9), "clearly-good reads are already trustworthy without FM")
    }

    // MARK: - escalationBudget

    func testEscalationBudgetIsZeroForZeroOrNegativeSpineCount() {
        let policy = FMEscalationPolicy.default
        XCTAssertEqual(policy.escalationBudget(forSpineCount: 0), 0)
        XCTAssertEqual(policy.escalationBudget(forSpineCount: -1), 0)
    }

    func testEscalationBudgetRoundsUpFivePercentAndAppliesTheMinimumFloor() {
        let policy = FMEscalationPolicy(marginalScoreRange: 0...1, maxEscalationFraction: 0.05, minEscalations: 1)
        XCTAssertEqual(policy.escalationBudget(forSpineCount: 1), 1, "floor applies even to a single-spine capture")
        XCTAssertEqual(policy.escalationBudget(forSpineCount: 20), 1)
        XCTAssertEqual(policy.escalationBudget(forSpineCount: 21), 2, "5% of 21 rounds up from 1.05")
        XCTAssertEqual(policy.escalationBudget(forSpineCount: 100), 5)
    }

    // MARK: - selectForEscalation

    func testSelectForEscalationPicksOnlyHardCasesWorstScoreFirstUpToTheBudget() {
        let policy = FMEscalationPolicy(marginalScoreRange: 0.3...0.7, maxEscalationFraction: 1.0, minEscalations: 0)
        // Budget = ceil(1.0 * 5) = 5, but only 3 of these 5 scores are
        // hard cases at all, so the cap never actually binds here.
        let scores: [(id: Int, qualityScore: Double)] = [
            (1, 0.9),  // clearly good -- never a hard case
            (2, 0.5),  // hard case
            (3, 0.1),  // clearly bad -- never a hard case
            (4, 0.65), // hard case
            (5, 0.31), // hard case, worst of the three
        ]
        let selected = policy.selectForEscalation(scores: scores)
        XCTAssertEqual(selected, Set([2, 4, 5]))
    }

    func testSelectForEscalationCapsAtTheBudgetPreferringTheWorstScores() {
        let policy = FMEscalationPolicy(marginalScoreRange: 0...1, maxEscalationFraction: 0.05, minEscalations: 1)
        // 20 spines -> budget 1 (floor); all are hard cases (range is
        // 0...1), so only the single worst-scoring one should be picked.
        let scores: [(id: Int, qualityScore: Double)] = (0..<20).map { i in
            (i, Double(i) / 20.0) // scores 0.0, 0.05, ..., 0.95 -- id 0 is worst.
        }
        let selected = policy.selectForEscalation(scores: scores)
        XCTAssertEqual(selected, Set([0]))
    }

    func testSelectForEscalationReturnsEmptySetWhenNoScoresAreHardCases() {
        let policy = FMEscalationPolicy(marginalScoreRange: 0.4...0.6, maxEscalationFraction: 1.0, minEscalations: 1)
        let scores: [(id: Int, qualityScore: Double)] = [(1, 0.9), (2, 0.1)]
        XCTAssertTrue(policy.selectForEscalation(scores: scores).isEmpty)
    }

    func testSelectForEscalationReturnsEmptySetForEmptyInput() {
        let policy = FMEscalationPolicy.default
        let scores: [(id: Int, qualityScore: Double)] = []
        XCTAssertTrue(policy.selectForEscalation(scores: scores).isEmpty)
    }
}
