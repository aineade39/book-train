import Foundation

// OCR quality gate per docs/BOOK_ID_IOS_PIPELINE.md §Text extraction:
// "Quality gate before catalog lookup: joint score from OCR confidence,
// orientation agreement, string plausibility (length, charset, token
// shape), and optional detection confidence. Low scores do not
// auto-match." Also doubles as the per-pass gate that decides whether the
// OCR orientation router's third (.down) pass is worth running.

public struct OCRQualityGateInputs {
    /// Mean Vision candidate confidence across the winning pass's
    /// observations (`0` if there were none at all).
    public var meanConfidence: Float
    /// How much the winning orientation's confidence-sum beat the
    /// runner-up's, normalized to roughly `[0, 1]` (`0` = tied/ambiguous,
    /// `1` = runner-up contributed nothing).
    public var orientationAgreement: Double
    /// The assembled (reading-order) text for the winning pass.
    public var assembledText: String
    /// Upstream spine-detector confidence, if the caller wants it folded
    /// into the joint score (optional per spec).
    public var detectionConfidence: Float?

    public init(
        meanConfidence: Float, orientationAgreement: Double, assembledText: String, detectionConfidence: Float? = nil
    ) {
        self.meanConfidence = meanConfidence
        self.orientationAgreement = orientationAgreement
        self.assembledText = assembledText
        self.detectionConfidence = detectionConfidence
    }
}

public struct OCRQualityGate {
    public var minConfidence: Float
    public var minOrientationAgreement: Double
    public var minTextLength: Int
    public var minLetterFraction: Double
    /// Weight of the optional detection-confidence term relative to the
    /// other three (each implicitly weight `1`). Kept deliberately small:
    /// detection confidence says "there is definitely a spine here", not
    /// "the OCR text is trustworthy" — it must never be able to single-handedly
    /// drag a garbage OCR reading over the accept threshold.
    public var detectionConfidenceWeight: Double
    /// Overall `score(_:)` threshold `passes(_:)` checks against.
    public var acceptScoreThreshold: Double

    public static let `default` = OCRQualityGate()

    public init(
        minConfidence: Float = 0.35,
        minOrientationAgreement: Double = 0.0,
        minTextLength: Int = 3,
        minLetterFraction: Double = 0.4,
        detectionConfidenceWeight: Double = 0.3,
        acceptScoreThreshold: Double = 0.5
    ) {
        self.minConfidence = minConfidence
        self.minOrientationAgreement = minOrientationAgreement
        self.minTextLength = minTextLength
        self.minLetterFraction = minLetterFraction
        self.detectionConfidenceWeight = detectionConfidenceWeight
        self.acceptScoreThreshold = acceptScoreThreshold
    }

    /// Joint score in `[0, 1]`: a weighted mean of confidence, orientation
    /// agreement, string plausibility (each weight `1`), and — when
    /// supplied — detection confidence (weight `detectionConfidenceWeight`).
    public func score(_ inputs: OCRQualityGateInputs) -> Double {
        // Hard fail: no text at all is never acceptable regardless of how
        // confident/agreed-upon the (nonexistent) reading was -- averaging
        // this into the joint score could otherwise let a high-confidence
        // empty pass slip through.
        guard !inputs.assembledText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return 0 }

        var weightedSum = normalizedRatio(Double(inputs.meanConfidence), floor: Double(minConfidence))
            + normalizedRatio(inputs.orientationAgreement, floor: minOrientationAgreement)
            + plausibility(of: inputs.assembledText)
        var totalWeight = 3.0
        if let detectionConfidence = inputs.detectionConfidence {
            weightedSum += Double(detectionConfidence) * detectionConfidenceWeight
            totalWeight += detectionConfidenceWeight
        }
        return weightedSum / totalWeight
    }

    public func passes(_ inputs: OCRQualityGateInputs) -> Bool {
        score(inputs) >= acceptScoreThreshold
    }

    /// String plausibility sub-score: length + charset (has letters) +
    /// token shape (at least one token with >= 2 letters, i.e. not just
    /// stray punctuation/single characters Vision sometimes hallucinates).
    private func plausibility(of text: String) -> Double {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }

        let lengthScore = normalizedRatio(Double(trimmed.count), floor: Double(minTextLength))

        let letters = trimmed.filter(\.isLetter)
        let letterFraction = Double(letters.count) / Double(trimmed.count)
        let charsetScore = normalizedRatio(letterFraction, floor: minLetterFraction)

        let hasWordLikeToken = trimmed
            .split(separator: " ")
            .contains { $0.filter(\.isLetter).count >= 2 }
        let tokenShapeScore: Double = hasWordLikeToken ? 1 : 0

        return (lengthScore + charsetScore + tokenShapeScore) / 3
    }

    /// Maps `value` to `[0, 1]`: `0` at/under `floor`, ramping to `1` at
    /// `2 * floor` (or `1` immediately when `floor <= 0`, i.e. the check is
    /// disabled). Deliberately smooth rather than a hard boolean cutoff, so
    /// the joint score degrades gracefully near the threshold.
    private func normalizedRatio(_ value: Double, floor: Double) -> Double {
        guard floor > 0 else { return 1 }
        guard value > floor else { return 0 }
        return min(1, (value - floor) / floor)
    }
}
