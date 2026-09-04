import CoreGraphics
import CoreImage
import Foundation

// Upright, perspective-corrected spine crop — the sole crop primitive for
// OCR per docs/BOOK_ID_IOS_PIPELINE.md ("axis-aligned bounding boxes are out
// of scope"). Moved here from `Sources/bookspines/main.swift`'s
// `uprightCrop` per the GEOMETRY.md boundary table (was "partial"); this is
// now the canonical implementation for both the CLIs and the iOS app.
//
// `cropPixelToScene` is the inverse of this warp in pure scene-pixel space
// (no Core Image, no image height) — the "data boundary" GEOMETRY.md calls
// out for OCR text boxes, which live inside the warped crop and must be
// mapped back to scene pixels to associate text with a spine OBB.

/// Perspective-corrects `detection` to an upright `w x h` crop (undoing the
/// box's own rotation) from the full-scene `image`. Returns `nil` if the
/// crop would be empty/degenerate.
public func uprightWarp(of detection: OBBDetection, in image: CGImage) -> CGImage? {
    let imageH = CGFloat(image.height)
    let ciImage = CIImage(cgImage: image)
    // Pixel space is top-left/y-down; Core Image is bottom-left/y-up.
    let ciCenterX = CGFloat(detection.cx)
    let ciCenterY = imageH - CGFloat(detection.cy)

    var transform = CGAffineTransform(translationX: -ciCenterX, y: -ciCenterY)
    transform = transform.concatenating(CGAffineTransform(rotationAngle: CGFloat(detection.angle)))
    transform = transform.concatenating(CGAffineTransform(translationX: CGFloat(detection.w) / 2, y: CGFloat(detection.h) / 2))

    let transformed = ciImage.transformed(by: transform)
    let cropRect = CGRect(x: 0, y: 0, width: CGFloat(detection.w), height: CGFloat(detection.h)).integral
    guard cropRect.width > 0, cropRect.height > 0 else { return nil }
    let cropped = transformed.cropped(to: cropRect)

    let context = CIContext()
    return context.createCGImage(cropped, from: cropRect)
}

/// Maps a pixel point `p` inside the `uprightWarp(of: detection, ...)` crop
/// (top-left origin, y-down, `[0, detection.w] x [0, detection.h]`) back to
/// full-scene pixels. Pure rotation + translation in scene space — derived
/// algebraically from `uprightWarp`'s Core Image transform, so it needs
/// neither Core Image nor the scene image height (they cancel out).
///
/// This is the inverse of the same rotation `OBBDetection.corners` applies:
/// `corners` maps crop-relative offsets `(±w/2, ±h/2)` to scene via
/// `center + R(angle) * offset`; this function is that formula for an
/// arbitrary interior point instead of just the four corners.
public func cropPixelToScene(_ p: CGPoint, detection: OBBDetection) -> CGPoint {
    let dxLocal = Double(p.x) - detection.w / 2
    let dyLocal = Double(p.y) - detection.h / 2
    let c = cos(detection.angle), s = sin(detection.angle)
    let sx = detection.cx + c * dxLocal - s * dyLocal
    let sy = detection.cy + s * dxLocal + c * dyLocal
    return CGPoint(x: sx, y: sy)
}
