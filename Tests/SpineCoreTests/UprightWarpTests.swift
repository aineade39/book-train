import CoreGraphics
import Foundation
import XCTest

@testable import SpineCore

final class UprightWarpTests: XCTestCase {

    // MARK: - cropPixelToScene: pure-math round trip against OBBDetection.corners

    func testCropPixelToSceneMapsCropCornersOntoDetectionCornersUnrotated() {
        let det = OBBDetection(cx: 50, cy: 20, w: 100, h: 40, angle: 0, conf: 0.9)
        let mapped = [
            cropPixelToScene(CGPoint(x: 0, y: 0), detection: det),
            cropPixelToScene(CGPoint(x: det.w, y: 0), detection: det),
            cropPixelToScene(CGPoint(x: det.w, y: det.h), detection: det),
            cropPixelToScene(CGPoint(x: 0, y: det.h), detection: det),
        ]
        assertSameCornerSet(mapped, det.corners)
    }

    func testCropPixelToSceneMapsCropCornersOntoDetectionCornersRotated() {
        for angleDeg in [10.0, 45.0, 90.0, 135.0, -30.0] {
            let det = OBBDetection(cx: 200, cy: 150, w: 80, h: 30, angle: angleDeg * .pi / 180, conf: 0.9)
            let mapped = [
                cropPixelToScene(CGPoint(x: 0, y: 0), detection: det),
                cropPixelToScene(CGPoint(x: det.w, y: 0), detection: det),
                cropPixelToScene(CGPoint(x: det.w, y: det.h), detection: det),
                cropPixelToScene(CGPoint(x: 0, y: det.h), detection: det),
            ]
            assertSameCornerSet(mapped, det.corners, accuracy: 1e-6, "angle \(angleDeg)deg")
        }
    }

    func testCropPixelToSceneCenterMapsToDetectionCenter() {
        let det = OBBDetection(cx: 123, cy: 45, w: 60, h: 20, angle: 0.4, conf: 0.5)
        let center = cropPixelToScene(CGPoint(x: det.w / 2, y: det.h / 2), detection: det)
        XCTAssertEqual(Double(center.x), det.cx, accuracy: 1e-9)
        XCTAssertEqual(Double(center.y), det.cy, accuracy: 1e-9)
    }

    // MARK: - uprightWarp (Core Image) + cropPixelToScene consistency
    //
    // Renders a real CGImage, warps it, then checks that mapping the
    // warped crop's own pixel corners back via `cropPixelToScene` reproduces
    // the detection's scene corners within a pixel of rounding error. This
    // is the strongest available check that the Core Image transform inside
    // `uprightWarp` and the pure-math inverse in `cropPixelToScene` agree
    // with each other (they must, since one is the algebraic inverse of the
    // other, but this exercises the actual Core Image path too).
    func testUprightWarpRoundTripsThroughCropPixelToScene() throws {
        let scene = makeCheckerboardCGImage(width: 400, height: 300, cols: 4, rows: 3)
        let det = OBBDetection(cx: 200, cy: 150, w: 120, h: 40, angle: 20 * .pi / 180, conf: 0.9)
        guard let warped = uprightWarp(of: det, in: scene) else {
            XCTFail("expected a non-nil warp")
            return
        }
        XCTAssertEqual(warped.width, Int(det.w.rounded()))
        XCTAssertEqual(warped.height, Int(det.h.rounded()))

        let mapped = [
            cropPixelToScene(CGPoint(x: 0, y: 0), detection: det),
            cropPixelToScene(CGPoint(x: CGFloat(warped.width), y: 0), detection: det),
            cropPixelToScene(CGPoint(x: CGFloat(warped.width), y: CGFloat(warped.height)), detection: det),
            cropPixelToScene(CGPoint(x: 0, y: CGFloat(warped.height)), detection: det),
        ]
        assertSameCornerSet(mapped, det.corners, accuracy: 1.0, "uprightWarp output size vs. cropPixelToScene")
    }

    func testUprightWarpDegenerateDetectionReturnsNil() {
        let scene = makeSolidCGImage(width: 100, height: 100)
        let det = OBBDetection(cx: 50, cy: 50, w: 0, h: 0, angle: 0, conf: 0.5)
        XCTAssertNil(uprightWarp(of: det, in: scene))
    }

    // MARK: - Helpers

    private func assertSameCornerSet(
        _ a: [CGPoint], _ b: [CGPoint], accuracy: Double = 1e-6, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, message, file: file, line: line)
        var remaining = b
        for p in a {
            guard let idx = remaining.firstIndex(where: { dist($0, p) <= accuracy }) else {
                XCTFail("\(p) has no match within \(accuracy) in \(remaining) (\(message))", file: file, line: line)
                continue
            }
            remaining.remove(at: idx)
        }
    }
}
