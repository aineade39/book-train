import CoreGraphics
import Foundation

// Crop-local-pixel text line, the input shape `SpineRoleScoring` consumes.
// Deliberately independent of `SpineCore.OBBDetection`/Vision types so
// `SpineMatching` stays dependency-free (per Package.swift's target
// comment) -- `SpinePerception` converts its Vision observations (already
// mapped to crop-local pixels via `SpineCore.visionNormalizedPointToCropPixels`)
// into this shape before handing them to role scoring / retrieval.

/// One alternate (non-winning) Vision candidate for a text line, kept only
/// when it passed the plan's §A alternate-gate (confidence floor + margin
/// vs. the winning candidate, and text that actually differs).
public struct SpineTextAlternate: Equatable {
    public let text: String
    public let confidence: Float

    public init(text: String, confidence: Float) {
        self.text = text
        self.confidence = confidence
    }
}

/// One OCR text line (a single Vision observation) in **crop-local pixel**
/// coordinates (top-left origin, y-down, `[0, cropWidth] x [0, cropHeight]`)
/// -- i.e. the space `SpineCore.visionNormalizedPointToCropPixels` produces,
/// *not* full-scene pixels. Role scoring (title/author/other) is defined
/// entirely in this crop-local space per the locked plan ("Crop-local
/// pixels; upright text left-to-right after winning orientation").
public struct SpineTextLine: Equatable {
    /// Canonical (Vision top-1 / "C1") recognized text for this line.
    public let text: String
    public let confidence: Float
    public let topLeft: CGPoint
    public let topRight: CGPoint
    public let bottomRight: CGPoint
    public let bottomLeft: CGPoint
    public let cropWidth: Double
    public let cropHeight: Double
    /// Kept alternate (C2/C3) candidates for this same line, per §A.
    public let alternates: [SpineTextAlternate]

    public init(
        text: String, confidence: Float,
        topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint,
        cropWidth: Double, cropHeight: Double,
        alternates: [SpineTextAlternate] = []
    ) {
        self.text = text
        self.confidence = confidence
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
        self.cropWidth = cropWidth
        self.cropHeight = cropHeight
        self.alternates = alternates
    }

    var corners: [CGPoint] { [topLeft, topRight, bottomRight, bottomLeft] }
}
