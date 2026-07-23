import CoreGraphics
import ImageIO

@testable import SpinePerception

/// Deterministic stand-in for Vision, so orientation-router/assembly logic
/// is testable without ever calling into `VNRecognizeTextRequest`.
struct FakeTextRecognizer: TextRecognizer {
    /// Observations to return, keyed by orientation.
    var byOrientation: [CGImagePropertyOrientation: [RecognizedTextObservation]]

    func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation] {
        byOrientation[orientation] ?? []
    }
}

/// A single full-crop observation (`[0,1]` normalized corners spanning the
/// whole image) with the given text/confidence — the common case for a
/// tall spine title running the length of the crop.
func fullCropObservation(text: String, confidence: Float) -> RecognizedTextObservation {
    RecognizedTextObservation(
        text: text, confidence: confidence,
        topLeft: CGPoint(x: 0, y: 1), topRight: CGPoint(x: 1, y: 1),
        bottomRight: CGPoint(x: 1, y: 0), bottomLeft: CGPoint(x: 0, y: 0)
    )
}

/// An observation confined to a normalized horizontal band
/// `[yBottom, yTop]` of the (post-orientation, "logical") image — lets
/// tests place multiple observations at distinct positions along the
/// reading axis.
func bandObservation(text: String, confidence: Float, yBottom: Double, yTop: Double) -> RecognizedTextObservation {
    RecognizedTextObservation(
        text: text, confidence: confidence,
        topLeft: CGPoint(x: 0, y: yTop), topRight: CGPoint(x: 1, y: yTop),
        bottomRight: CGPoint(x: 1, y: yBottom), bottomLeft: CGPoint(x: 0, y: yBottom)
    )
}
