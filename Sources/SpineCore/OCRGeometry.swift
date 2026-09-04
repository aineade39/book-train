import CoreGraphics
import Foundation
import ImageIO

// Vision OCR box -> scene pixels: the **data** boundary GEOMETRY.md calls
// out (not just the UI/draw boundary) — text observations from
// `VNRecognizeTextRequest` on an `uprightWarp` crop must be mapped back to
// scene pixels to associate OCR text with the spine `OBBDetection` it came
// from (e.g. for reading-order assembly, or drawing OCR boxes on the scene
// overlay). Superseded by, and generalizes, the ad hoc normalized-to-pixel
// math in the standalone `ocr.swift` reference script (kept as-is; not a
// dependency of this file).
//
// Only 90°-increment orientations are supported (`.up`, `.right`, `.left`,
// `.down`), matching the OCR orientation router's "90° increments only"
// contract in docs/BOOK_ID_IOS_PIPELINE.md — mirrored/arbitrary orientations
// never occur in this pipeline (the crop itself is never mirrored, and
// `VNImageRequestHandler` is only ever given one of those four).

/// Converts a Vision-normalized point (`[0, 1]`, origin bottom-left, in the
/// "logical upright" space Vision reports when `orientation` was passed to
/// `VNImageRequestHandler`) back into a pixel point in the *original* crop
/// buffer (top-left origin, y-down, `[0, cropWidth] x [0, cropHeight]`) —
/// i.e. undoes exactly the rotation `orientation` implies, so the result is
/// in the same space `cropPixelToScene` expects.
///
/// Derived directly from Apple's `CGImagePropertyOrientation` row/column
/// semantics (e.g. `.right`: "0th row is on the right, 0th column is the
/// top"), not from empirical testing — see inline derivation in code review
/// history / GEOMETRY.md if this ever needs re-deriving.
public func visionNormalizedPointToCropPixels(
    _ normalized: CGPoint,
    orientation: CGImagePropertyOrientation,
    cropWidth: Double,
    cropHeight: Double
) -> CGPoint {
    let swapped = orientation == .left || orientation == .right
    let logicalW = swapped ? cropHeight : cropWidth
    let logicalH = swapped ? cropWidth : cropHeight

    // Normalized (bottom-left, y-up) -> logical pixels (top-left, y-down).
    let lx = Double(normalized.x) * logicalW
    let ly = (1 - Double(normalized.y)) * logicalH

    switch orientation {
    case .up:
        return CGPoint(x: lx, y: ly)
    case .down:
        return CGPoint(x: cropWidth - lx, y: cropHeight - ly)
    case .right:
        return CGPoint(x: ly, y: cropHeight - lx)
    case .left:
        return CGPoint(x: cropWidth - ly, y: lx)
    default:
        // Mirrored orientations never occur in this pipeline (the crop is
        // never mirrored before OCR); fall back to the identity mapping
        // rather than guessing at a flip.
        return CGPoint(x: lx, y: ly)
    }
}

/// Four corners of an OCR text observation, in full-scene pixels — the
/// per-spine analogue of `OBBDetection.corners`. Field names match Vision's
/// `VNRecognizedTextObservation` corner accessors.
public struct OCRTextBoxScene: Equatable {
    public let topLeft: CGPoint
    public let topRight: CGPoint
    public let bottomRight: CGPoint
    public let bottomLeft: CGPoint

    public init(topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint) {
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
    }

    /// Scene-pixel bounding rect of the four corners (axis-aligned; the
    /// quad itself may be rotated — use the corners directly when the
    /// rotation matters, e.g. long-axis reading-order assembly).
    public var boundingBox: CGRect {
        let xs = [topLeft.x, topRight.x, bottomRight.x, bottomLeft.x]
        let ys = [topLeft.y, topRight.y, bottomRight.y, bottomLeft.y]
        return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
    }
}

/// Maps one Vision text observation's four normalized corners — read off an
/// `orientation`-tagged `VNImageRequestHandler` pass over the
/// `uprightWarp(of: detection, ...)` crop — into full-scene pixels.
public func ocrQuadToScene(
    topLeft: CGPoint,
    topRight: CGPoint,
    bottomRight: CGPoint,
    bottomLeft: CGPoint,
    orientation: CGImagePropertyOrientation,
    detection: OBBDetection
) -> OCRTextBoxScene {
    func map(_ p: CGPoint) -> CGPoint {
        let cropPt = visionNormalizedPointToCropPixels(
            p, orientation: orientation, cropWidth: detection.w, cropHeight: detection.h
        )
        return cropPixelToScene(cropPt, detection: detection)
    }
    return OCRTextBoxScene(
        topLeft: map(topLeft), topRight: map(topRight),
        bottomRight: map(bottomRight), bottomLeft: map(bottomLeft)
    )
}
