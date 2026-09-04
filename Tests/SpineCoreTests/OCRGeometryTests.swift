import CoreGraphics
import Foundation
import ImageIO
import XCTest

@testable import SpineCore

/// Anchor expectations below are derived directly from Apple's
/// `CGImagePropertyOrientation` row/column documentation (not empirically),
/// using a deliberately non-square crop (`W=100, H=40`) so any accidental
/// width/height swap for `.left`/`.right` would be caught:
///
/// - `.up`:    normalized top-left (0,1) -> buffer pixel top-left    (0,0)
/// - `.down`:  normalized top-left (0,1) -> buffer pixel bottom-right (W,H)
/// - `.right`: normalized top-left (0,1) -> buffer pixel bottom-left  (0,H)
/// - `.left`:  normalized top-left (0,1) -> buffer pixel top-right   (W,0)
final class OCRGeometryTests: XCTestCase {
    let W = 100.0
    let H = 40.0

    func testUpIsIdentity() {
        let p = visionNormalizedPointToCropPixels(CGPoint(x: 0, y: 1), orientation: .up, cropWidth: W, cropHeight: H)
        XCTAssertEqual(Double(p.x), 0, accuracy: 1e-9)
        XCTAssertEqual(Double(p.y), 0, accuracy: 1e-9)
    }

    func testDownIsHalfTurn() {
        let p = visionNormalizedPointToCropPixels(CGPoint(x: 0, y: 1), orientation: .down, cropWidth: W, cropHeight: H)
        XCTAssertEqual(Double(p.x), W, accuracy: 1e-9)
        XCTAssertEqual(Double(p.y), H, accuracy: 1e-9)
    }

    func testRightMapsLogicalTopLeftToBufferBottomLeft() {
        let p = visionNormalizedPointToCropPixels(CGPoint(x: 0, y: 1), orientation: .right, cropWidth: W, cropHeight: H)
        XCTAssertEqual(Double(p.x), 0, accuracy: 1e-9)
        XCTAssertEqual(Double(p.y), H, accuracy: 1e-9)
    }

    func testLeftMapsLogicalTopLeftToBufferTopRight() {
        let p = visionNormalizedPointToCropPixels(CGPoint(x: 0, y: 1), orientation: .left, cropWidth: W, cropHeight: H)
        XCTAssertEqual(Double(p.x), W, accuracy: 1e-9)
        XCTAssertEqual(Double(p.y), 0, accuracy: 1e-9)
    }

    /// Every orientation is a rigid rotation, so the four normalized-square
    /// corners must map onto exactly the four buffer corners, regardless of
    /// which corner goes where.
    func testAllOrientationsMapCornersOntoBufferCornerSet() {
        let bufferCorners: Set<[Int]> = [[0, 0], [Int(W), 0], [Int(W), Int(H)], [0, Int(H)]]
        for orientation: CGImagePropertyOrientation in [.up, .down, .left, .right] {
            let normalizedCorners = [
                CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1),
            ]
            let mapped = Set(normalizedCorners.map { n -> [Int] in
                let p = visionNormalizedPointToCropPixels(n, orientation: orientation, cropWidth: W, cropHeight: H)
                return [Int(p.x.rounded()), Int(p.y.rounded())]
            })
            XCTAssertEqual(mapped, bufferCorners, "orientation \(orientation) did not permute the buffer corners")
        }
    }

    // MARK: - ocrQuadToScene: combined with cropPixelToScene

    func testOcrQuadToSceneUnrotatedUpOrientationMatchesDetectionCorners() {
        let det = OBBDetection(cx: 50, cy: 20, w: W, h: H, angle: 0, conf: 0.9)
        // A full-crop text box (top-left, top-right, bottom-right, bottom-left
        // in Vision's own normalized, bottom-left-origin convention).
        let box = ocrQuadToScene(
            topLeft: CGPoint(x: 0, y: 1),
            topRight: CGPoint(x: 1, y: 1),
            bottomRight: CGPoint(x: 1, y: 0),
            bottomLeft: CGPoint(x: 0, y: 0),
            orientation: .up,
            detection: det
        )
        // .up + angle 0: scene top-left should be the detection's
        // min-x/min-y corner and scene bottom-right the max-x/max-y corner.
        XCTAssertEqual(Double(box.topLeft.x), det.cx - det.w / 2, accuracy: 1e-6)
        XCTAssertEqual(Double(box.topLeft.y), det.cy - det.h / 2, accuracy: 1e-6)
        XCTAssertEqual(Double(box.bottomRight.x), det.cx + det.w / 2, accuracy: 1e-6)
        XCTAssertEqual(Double(box.bottomRight.y), det.cy + det.h / 2, accuracy: 1e-6)
    }

    func testOcrQuadToSceneRotatedDetectionStaysWithinDetectionCornerHull() {
        let det = OBBDetection(cx: 300, cy: 200, w: 90, h: 30, angle: 35 * .pi / 180, conf: 0.9)
        let box = ocrQuadToScene(
            topLeft: CGPoint(x: 0.1, y: 0.9),
            topRight: CGPoint(x: 0.9, y: 0.9),
            bottomRight: CGPoint(x: 0.9, y: 0.1),
            bottomLeft: CGPoint(x: 0.1, y: 0.1),
            orientation: .right,
            detection: det
        )
        // A sub-region of the crop must land strictly inside the detection's
        // own scene quad after mapping.
        for p in [box.topLeft, box.topRight, box.bottomRight, box.bottomLeft] {
            XCTAssertTrue(pointStrictlyInside(Double(p.x), Double(p.y), det.corners), "\(p) fell outside detection corners")
        }
    }
}
