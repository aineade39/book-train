import CoreGraphics
import Foundation
import ImageIO
import Vision

// Vision OCR primary path per docs/BOOK_ID_IOS_PIPELINE.md §Text extraction:
// "VNRecognizeTextRequest at .accurate, with language correction and
// locale-appropriate recognitionLanguages."
//
// OCR is behind a protocol (`TextRecognizer`) so `OCROrientationRouter` and
// reading-order assembly are unit-testable on Mac with hand-built
// observations, with no dependency on Vision actually reading anything —
// the router's *decision logic* (which orientation wins, when to run a
// third pass) is what needs coverage, not Vision's recognition accuracy.

/// One kept alternate (non-winning) Vision candidate for an observation,
/// per the locked "Book ID OCR gains" plan §A n-best gate: "C2/C3 only if
/// normalized text != C1, confidence >= 0.65, and C1.conf - Cn.conf <=
/// 0.15." Retrieval-only -- never substitutes for the winning `text`.
public struct RecognizedTextAlternate: Equatable {
    public let text: String
    public let confidence: Float

    public init(text: String, confidence: Float) {
        self.text = text
        self.confidence = confidence
    }
}

/// One recognized text observation from a single OCR pass. Corner fields
/// are Vision's own normalized (`[0, 1]`, bottom-left origin, relative to
/// the "logical upright" image for whatever orientation the pass used)
/// coordinates — pass them to `SpineCore.ocrQuadToScene` together with the
/// same `orientation` and the source `OBBDetection` to place them in scene
/// pixels.
public struct RecognizedTextObservation: Equatable {
    public let text: String
    public let confidence: Float
    public let topLeft: CGPoint
    public let topRight: CGPoint
    public let bottomRight: CGPoint
    public let bottomLeft: CGPoint
    /// Kept C2/C3 candidates (see `RecognizedTextAlternate`); empty for
    /// the vast majority of observations where nothing cleared the gate.
    public let alternates: [RecognizedTextAlternate]

    public init(
        text: String, confidence: Float,
        topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint,
        alternates: [RecognizedTextAlternate] = []
    ) {
        self.text = text
        self.confidence = confidence
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
        self.alternates = alternates
    }
}

public protocol TextRecognizer {
    /// Runs one OCR pass over `image`, telling Vision to treat `orientation`
    /// as "up" for this pass.
    func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation]
}

/// Real `VNRecognizeTextRequest`-backed implementation.
public struct VisionTextRecognizer: TextRecognizer {
    public var usesLanguageCorrection: Bool
    public var recognitionLanguages: [String]
    /// Bundled authors-only lexicon (§C) -- passed straight through to
    /// `VNRecognizeTextRequest.customWords`. Always used with
    /// `usesLanguageCorrection = true` per the locked decision; harmless
    /// (a no-op) if `usesLanguageCorrection` is `false` or the list is
    /// empty.
    public var customWords: [String]

    /// Per §A: request 3 candidates per observation so callers can
    /// consider gated alternates for retrieval; only the top one is ever
    /// used as canonical text.
    static let candidateCount = 3

    public init(usesLanguageCorrection: Bool = true, recognitionLanguages: [String] = [], customWords: [String] = []) {
        self.usesLanguageCorrection = usesLanguageCorrection
        self.recognitionLanguages = recognitionLanguages
        self.customWords = customWords
    }

    public func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = usesLanguageCorrection
        if !recognitionLanguages.isEmpty {
            request.recognitionLanguages = recognitionLanguages
        }
        if !customWords.isEmpty {
            request.customWords = customWords
        }
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        try handler.perform([request])
        return (request.results ?? []).compactMap { observation in
            let candidates = observation.topCandidates(Self.candidateCount)
            guard let top = candidates.first else { return nil }
            let alternates = candidates.dropFirst().compactMap { candidate -> RecognizedTextAlternate? in
                guard candidate.string != top.string else { return nil }
                guard candidate.confidence >= 0.65 else { return nil }
                guard top.confidence - candidate.confidence <= 0.15 else { return nil }
                return RecognizedTextAlternate(text: candidate.string, confidence: candidate.confidence)
            }
            return RecognizedTextObservation(
                text: top.string,
                confidence: top.confidence,
                topLeft: observation.topLeft,
                topRight: observation.topRight,
                bottomRight: observation.bottomRight,
                bottomLeft: observation.bottomLeft,
                alternates: Array(alternates)
            )
        }
    }
}
