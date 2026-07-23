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

    public init(
        text: String, confidence: Float,
        topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint
    ) {
        self.text = text
        self.confidence = confidence
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
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

    public init(usesLanguageCorrection: Bool = true, recognitionLanguages: [String] = []) {
        self.usesLanguageCorrection = usesLanguageCorrection
        self.recognitionLanguages = recognitionLanguages
    }

    public func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = usesLanguageCorrection
        if !recognitionLanguages.isEmpty {
            request.recognitionLanguages = recognitionLanguages
        }
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        try handler.perform([request])
        return (request.results ?? []).compactMap { observation in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            return RecognizedTextObservation(
                text: candidate.string,
                confidence: candidate.confidence,
                topLeft: observation.topLeft,
                topRight: observation.topRight,
                bottomRight: observation.bottomRight,
                bottomLeft: observation.bottomLeft
            )
        }
    }
}
