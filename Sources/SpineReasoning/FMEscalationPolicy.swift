import Foundation

// Rate-limiting policy for FM escalation per docs/BOOK_ID_IOS_PIPELINE.md
// §Non-functional requirements: "FM escalation is seconds per spine and
// must be rate-limited (top ~5% of hard cases)." Deliberately has zero
// dependency on `FoundationModels` (unlike `SpineExtraction`/
// `SpineReasoningService`) so the escalation-selection logic is
// unit-testable on every OS version, not just 26+.

/// Decides which OCR-quality-gated spines in a single capture are worth
/// the multi-second cost of an on-device FM call.
public struct FMEscalationPolicy: Sendable, Equatable {
    /// OCR quality scores in this range are "marginal": not so low that
    /// there's plausibly no legible text to clean up, not so high that
    /// the raw Vision text is already trustworthy enough for fuzzy match.
    /// FM cleanup is most likely to actually change the outcome here.
    public var marginalScoreRange: ClosedRange<Double>
    /// Escalation budget for one capture, as a fraction of *all*
    /// OCR-quality-gated spines (not just hard cases) -- so a dense shelf
    /// with many marginal reads still only pays for a handful of FM round
    /// trips.
    public var maxEscalationFraction: Double
    /// Floor on the budget so a small capture with even one hard case can
    /// still escalate it, despite `maxEscalationFraction` of a small count
    /// rounding down to zero.
    public var minEscalations: Int

    public static let `default` = FMEscalationPolicy()

    public init(
        marginalScoreRange: ClosedRange<Double> = 0.35...0.65,
        maxEscalationFraction: Double = 0.05,
        minEscalations: Int = 1
    ) {
        self.marginalScoreRange = marginalScoreRange
        self.maxEscalationFraction = maxEscalationFraction
        self.minEscalations = minEscalations
    }

    /// `true` when `qualityScore` sits in the marginal band this policy
    /// considers worth escalating.
    public func isHardCase(qualityScore: Double) -> Bool {
        marginalScoreRange.contains(qualityScore)
    }

    /// How many spines out of `spineCount` (total OCR-quality-gated
    /// spines in the capture, not just hard cases) may be escalated to FM.
    /// `0` when `spineCount <= 0` (nothing to escalate).
    public func escalationBudget(forSpineCount spineCount: Int) -> Int {
        guard spineCount > 0 else { return 0 }
        return max(minEscalations, Int((Double(spineCount) * maxEscalationFraction).rounded(.up)))
    }

    /// Picks which of `scores` (a capture's per-spine `id` -> OCR quality
    /// score) to escalate: hard cases only, worst score first (the reads
    /// FM is most likely to actually rescue), capped at
    /// `escalationBudget(forSpineCount:)`.
    public func selectForEscalation<ID: Hashable>(scores: [(id: ID, qualityScore: Double)]) -> Set<ID> {
        let budget = escalationBudget(forSpineCount: scores.count)
        guard budget > 0 else { return [] }
        let picked = scores
            .filter { isHardCase(qualityScore: $0.qualityScore) }
            .sorted { $0.qualityScore < $1.qualityScore }
            .prefix(budget)
        return Set(picked.map(\.id))
    }
}
