import CoreGraphics
import Foundation
import ImageIO
import SpineCore
import SpineMatching

// OCR orientation router per docs/BOOK_ID_IOS_PIPELINE.md §Text extraction /
// §OCR orientation design: "guided two-pass with gated third pass, not
// brute-force 4-way on every spine":
//
//   1. Aspect ratio picks primary/secondary orientation (.up vs .right).
//   2. Score by confidence sum and plausibility, not observation count.
//   3. Run .down only on fallback (both passes below the quality gate).
//
// This supersedes `bookspines`' `ocrSpine` placeholder, which tries a fixed
// `.up`/`.right` pair and picks the winner by observation *count*
// (`up.count >= rotated.count`) — exactly the anti-pattern step 2 forbids.

/// One OCR pass's raw result (all observations from one orientation).
public struct OCRPassResult {
    public let orientation: CGImagePropertyOrientation
    public let observations: [RecognizedTextObservation]

    public init(orientation: CGImagePropertyOrientation, observations: [RecognizedTextObservation]) {
        self.orientation = orientation
        self.observations = observations
    }

    /// Selection criterion: summed candidate confidence, **not** `.count`.
    public var confidenceSum: Float { observations.reduce(0) { $0 + $1.confidence } }

    public var meanConfidence: Float {
        guard !observations.isEmpty else { return 0 }
        return confidenceSum / Float(observations.count)
    }
}

/// Final routed OCR result for one spine crop.
public struct SpineOCRResult {
    public let winningPass: OCRPassResult
    /// Reading-order-assembled text (see `assembleReadingOrder`), ready for
    /// `SpineMatching.normalizeForSearch` + catalog retrieval.
    public let assembledText: String
    /// Same reading-order-sorted observations as `assembledText`, but kept
    /// structured (crop-local geometry + kept alternates) for
    /// `SpineRoleQueryBuilder` (§B geometry roles / §A n-best confusion).
    public let lines: [SpineTextLine]
    /// Title/author/general retrieval token pools + canonical rerank
    /// query strings, built from `lines` once per winning pass so callers
    /// never have to re-derive them.
    public let roleQueries: SpineRoleQueries
    public let qualityScore: Double
    public let passedQualityGate: Bool
    /// Whether the gated `.down` fallback pass ran (both aspect-guided
    /// passes failed the quality gate).
    public let ranThirdPass: Bool

    public init(
        winningPass: OCRPassResult, assembledText: String, lines: [SpineTextLine] = [],
        roleQueries: SpineRoleQueries = SpineRoleQueryBuilder.build(lines: []),
        qualityScore: Double, passedQualityGate: Bool, ranThirdPass: Bool
    ) {
        self.winningPass = winningPass
        self.assembledText = assembledText
        self.lines = lines
        self.roleQueries = roleQueries
        self.qualityScore = qualityScore
        self.passedQualityGate = passedQualityGate
        self.ranThirdPass = ranThirdPass
    }
}

public struct OCROrientationRouter {
    public var recognizer: TextRecognizer
    public var qualityGate: OCRQualityGate

    public init(recognizer: TextRecognizer, qualityGate: OCRQualityGate = .default) {
        self.recognizer = recognizer
        self.qualityGate = qualityGate
    }

    /// Aspect-guided primary/secondary order: tall crops (spine standing,
    /// title running top-to-bottom) read `.right` first, then `.up`; wide
    /// crops (spine lying flat) read `.up` first, then `.right`.
    public static func orientationOrder(
        width: Double, height: Double
    ) -> (primary: CGImagePropertyOrientation, secondary: CGImagePropertyOrientation) {
        height > width ? (.right, .up) : (.up, .right)
    }

    /// Runs the guided two-pass (+ gated third pass) strategy over `crop`
    /// (an `uprightWarp` output for `detection`) and returns the winning
    /// pass plus its reading-order-assembled text and quality-gate verdict.
    public func recognize(crop: CGImage, detection: OBBDetection) throws -> SpineOCRResult {
        let (primary, secondary) = Self.orientationOrder(width: detection.w, height: detection.h)
        let pass1 = OCRPassResult(
            orientation: primary,
            observations: try recognizer.recognizeText(in: crop, orientation: primary)
        )
        let pass2 = OCRPassResult(
            orientation: secondary,
            observations: try recognizer.recognizeText(in: crop, orientation: secondary)
        )

        if let result = evaluate(candidates: [pass1, pass2], detection: detection, ranThirdPass: false),
           result.passedQualityGate {
            return result
        }

        // Gated fallback: only reached when both aspect-guided passes
        // failed the quality gate.
        let pass3 = OCRPassResult(
            orientation: .down,
            observations: try recognizer.recognizeText(in: crop, orientation: .down)
        )
        // Falls back to the best-of-three even if it still doesn't pass —
        // callers inspect `passedQualityGate` themselves before matching.
        return evaluate(candidates: [pass1, pass2, pass3], detection: detection, ranThirdPass: true)
            ?? SpineOCRResult(
                winningPass: pass3, assembledText: "", qualityScore: 0, passedQualityGate: false, ranThirdPass: true
            )
    }

    private func evaluate(candidates: [OCRPassResult], detection: OBBDetection, ranThirdPass: Bool) -> SpineOCRResult? {
        let ranked = candidates.sorted { $0.confidenceSum > $1.confidenceSum }
        guard let best = ranked.first else { return nil }
        let runnerUp = ranked.count > 1 ? ranked[1] : nil
        let agreement = Self.orientationAgreement(best: best, runnerUp: runnerUp)
        let text = assembleReadingOrder(observations: best.observations, orientation: best.orientation, detection: detection)
        let lines = buildOrderedTextLines(observations: best.observations, orientation: best.orientation, detection: detection)
        let inputs = OCRQualityGateInputs(
            meanConfidence: best.meanConfidence, orientationAgreement: agreement,
            assembledText: text, detectionConfidence: detection.conf
        )
        return SpineOCRResult(
            winningPass: best, assembledText: text, lines: lines, roleQueries: SpineRoleQueryBuilder.build(lines: lines),
            qualityScore: qualityGate.score(inputs), passedQualityGate: qualityGate.passes(inputs),
            ranThirdPass: ranThirdPass
        )
    }

    /// Normalized `[0, 1]` margin of the winning pass's confidence-sum over
    /// the runner-up's (`1` with no runner-up at all).
    private static func orientationAgreement(best: OCRPassResult, runnerUp: OCRPassResult?) -> Double {
        guard let runnerUp else { return 1 }
        let total = best.confidenceSum + runnerUp.confidenceSum
        guard total > 0 else { return 0 }
        return Double((best.confidenceSum - runnerUp.confidenceSum) / total)
    }
}
