import CoreGraphics
import Foundation
import ImageIO
import SpineCore
import SpineMatching

// Reading-order assembly per docs/BOOK_ID_IOS_PIPELINE.md §Text extraction:
// "a spine yields several text observations. Before normalization/match,
// concatenate them ordered along the spine's long axis (longAxisAngle()),
// not in raw Vision result order, so title/author tokens stay coherent for
// the fuzzy scorer."

/// Concatenates `observations` (all from one OCR pass over one spine crop)
/// in order along `detection.longAxisAngle()`, using each observation's
/// scene-mapped centroid (via `SpineCore.ocrQuadToScene`) as its position —
/// this is the "OCR text boxes -> scene" data boundary GEOMETRY.md calls
/// out, not just a draw-time concern, because the *order* depends on it.
public func assembleReadingOrder(
    observations: [RecognizedTextObservation],
    orientation: CGImagePropertyOrientation,
    detection: OBBDetection
) -> String {
    guard !observations.isEmpty else { return "" }

    let axis = detection.longAxisAngle()
    let axisX = cos(axis), axisY = sin(axis)

    let positioned = observations.map { obs -> (Double, String) in
        let box = ocrQuadToScene(
            topLeft: obs.topLeft, topRight: obs.topRight,
            bottomRight: obs.bottomRight, bottomLeft: obs.bottomLeft,
            orientation: orientation, detection: detection
        )
        let cx = (box.topLeft.x + box.topRight.x + box.bottomRight.x + box.bottomLeft.x) / 4
        let cy = (box.topLeft.y + box.topRight.y + box.bottomRight.y + box.bottomLeft.y) / 4
        // Projection of the centroid onto the long-axis unit vector.
        let projection = Double(cx) * axisX + Double(cy) * axisY
        return (projection, obs.text)
    }

    return positioned
        .sorted { $0.0 < $1.0 }
        .map(\.1)
        .joined(separator: " ")
}

/// Same long-axis ordering as `assembleReadingOrder`, but returns
/// `SpineMatching.SpineTextLine`s (crop-local pixel geometry, per-line
/// alternates) instead of a flattened string -- the input
/// `SpineRoleQueryBuilder.build(lines:)` (geometry title/author/other
/// roles, §B) consumes. Crop-local pixel corners come from
/// `SpineCore.visionNormalizedPointToCropPixels` (undoes only the pass's
/// `orientation`, not the OBB's scene rotation) -- role scoring is defined
/// relative to the upright crop, not the full scene.
public func buildOrderedTextLines(
    observations: [RecognizedTextObservation],
    orientation: CGImagePropertyOrientation,
    detection: OBBDetection
) -> [SpineTextLine] {
    guard !observations.isEmpty else { return [] }

    let axis = detection.longAxisAngle()
    let axisX = cos(axis), axisY = sin(axis)

    let positioned = observations.map { obs -> (Double, SpineTextLine) in
        let sceneBox = ocrQuadToScene(
            topLeft: obs.topLeft, topRight: obs.topRight,
            bottomRight: obs.bottomRight, bottomLeft: obs.bottomLeft,
            orientation: orientation, detection: detection
        )
        let cx = (sceneBox.topLeft.x + sceneBox.topRight.x + sceneBox.bottomRight.x + sceneBox.bottomLeft.x) / 4
        let cy = (sceneBox.topLeft.y + sceneBox.topRight.y + sceneBox.bottomRight.y + sceneBox.bottomLeft.y) / 4
        let projection = Double(cx) * axisX + Double(cy) * axisY

        func cropPixel(_ p: CGPoint) -> CGPoint {
            visionNormalizedPointToCropPixels(p, orientation: orientation, cropWidth: detection.w, cropHeight: detection.h)
        }
        let line = SpineTextLine(
            text: obs.text, confidence: obs.confidence,
            topLeft: cropPixel(obs.topLeft), topRight: cropPixel(obs.topRight),
            bottomRight: cropPixel(obs.bottomRight), bottomLeft: cropPixel(obs.bottomLeft),
            cropWidth: detection.w, cropHeight: detection.h,
            alternates: obs.alternates.map { SpineTextAlternate(text: $0.text, confidence: $0.confidence) }
        )
        return (projection, line)
    }

    return positioned.sorted { $0.0 < $1.0 }.map(\.1)
}
