import Foundation

// Accept policy per docs/BOOK_ID_IOS_PIPELINE.md §Matching design:
//
//   "Accept policy: score >= T and score - runnerUp >= Delta; otherwise
//   surface UI. The margin test only works if the scorer doesn't tie at
//   ceiling." + §Data model: "title+author maps to many editions. Retrieve
//   /rank at the *work* level (or dedupe editions before applying the
//   margin test) so the runner-up in the accept policy isn't just another
//   printing of the same book."

/// A catalog candidate the accept policy can rank. `workKey` identifies the
/// underlying work (e.g. normalized title+primary-author) so multiple
/// editions of the same book collapse to one entry before the margin test
/// runs — otherwise a book's own paperback/hardback reprint would always be
/// the "runner-up" and the margin test could never pass.
public protocol RankableCandidate {
    var workKey: String { get }
}

public struct ScoredCandidate<T: RankableCandidate> {
    public let candidate: T
    public let score: Double

    public init(candidate: T, score: Double) {
        self.candidate = candidate
        self.score = score
    }
}

public enum AcceptDecision<T: RankableCandidate> {
    /// Score cleared the threshold and margin over the runner-up *work*.
    case autoAccept(ScoredCandidate<T>)
    /// Below threshold/margin, or multiple plausible works — surface top-N
    /// to the user rather than guessing.
    case ambiguous(topCandidates: [ScoredCandidate<T>])
    /// No candidates at all (e.g. empty shortlist).
    case noMatch
}

public struct AcceptPolicy {
    /// `T` — minimum top score to ever auto-accept.
    public var acceptThreshold: Double
    /// `Delta` — minimum lead over the best *distinct-work* runner-up.
    public var marginThreshold: Double
    /// Size of the candidate list surfaced to the user on `.ambiguous`.
    public var topN: Int

    public init(acceptThreshold: Double = 90, marginThreshold: Double = 8, topN: Int = 5) {
        self.acceptThreshold = acceptThreshold
        self.marginThreshold = marginThreshold
        self.topN = topN
    }

    /// Dedupes `scored` to one (best-scoring) entry per `workKey`, then
    /// applies the score+margin test against the best distinct-work
    /// runner-up.
    public func decide<T: RankableCandidate>(_ scored: [ScoredCandidate<T>]) -> AcceptDecision<T> {
        decideWithMargin(scored).decision
    }

    /// Same decision as `decide(_:)`, plus the top-vs-runner-up-work
    /// score gap that drove it (per the locked "Book ID OCR gains" plan
    /// §rerank-telemetry: "emit margin") -- `nil` when there's no
    /// distinct-work runner-up to measure against (a single-work
    /// shortlist, or an empty one). `AcceptPolicy`'s own 90/8 accept rule
    /// is unaffected by whether a caller reads this value.
    public func decideWithMargin<T: RankableCandidate>(_ scored: [ScoredCandidate<T>]) -> AcceptOutcome<T> {
        guard !scored.isEmpty else { return AcceptOutcome(decision: .noMatch, margin: nil) }

        var bestPerWork: [String: ScoredCandidate<T>] = [:]
        for sc in scored {
            if let existing = bestPerWork[sc.candidate.workKey], existing.score >= sc.score { continue }
            bestPerWork[sc.candidate.workKey] = sc
        }
        let ranked = bestPerWork.values.sorted { $0.score > $1.score }
        guard let top = ranked.first else { return AcceptOutcome(decision: .noMatch, margin: nil) }

        let margin: Double? = ranked.count > 1 ? top.score - ranked[1].score : nil
        if top.score >= acceptThreshold && (margin ?? .infinity) >= marginThreshold {
            return AcceptOutcome(decision: .autoAccept(top), margin: margin)
        }
        return AcceptOutcome(decision: .ambiguous(topCandidates: Array(ranked.prefix(topN))), margin: margin)
    }
}

/// `decideWithMargin(_:)`'s result -- the decision plus the score gap
/// that drove it, for telemetry/debugging without re-deriving it from
/// the (already work-deduped, already sorted-away) candidate list.
public struct AcceptOutcome<T: RankableCandidate> {
    public let decision: AcceptDecision<T>
    public let margin: Double?

    public init(decision: AcceptDecision<T>, margin: Double?) {
        self.decision = decision
        self.margin = margin
    }
}
