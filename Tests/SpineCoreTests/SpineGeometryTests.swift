import CoreGraphics
import XCTest

@testable import SpineCore

final class SpineGeometryTests: XCTestCase {

    // MARK: - Angles

    func testNormAngleWrapsIntoZeroToPi() {
        XCTAssertEqual(normAngle(0), 0, accuracy: 1e-9)
        XCTAssertEqual(normAngle(.pi), 0, accuracy: 1e-9)
        XCTAssertEqual(normAngle(.pi + 0.1), 0.1, accuracy: 1e-9)
        XCTAssertEqual(normAngle(-0.1), .pi - 0.1, accuracy: 1e-9)
        XCTAssertEqual(normAngle(3 * .pi + 0.2), 0.2, accuracy: 1e-9)
    }

    func testAngleDiffIsSymmetricAndBoundedByHalfPi() {
        XCTAssertEqual(angleDiff(0, 0), 0, accuracy: 1e-9)
        XCTAssertEqual(angleDiff(0.1, 0.1 + .pi), 0, accuracy: 1e-9) // mod-pi identity
        XCTAssertEqual(angleDiff(0, .pi / 2), .pi / 2, accuracy: 1e-9)
        // Near the wraparound boundary the short way round is the true diff.
        XCTAssertEqual(angleDiff(0.05, .pi - 0.05), 0.1, accuracy: 1e-9)
        XCTAssertEqual(angleDiff(1.2, 0.3), angleDiff(0.3, 1.2), accuracy: 1e-12)
    }

    func testCircularMeanOfIdenticalAnglesIsThatAngle() {
        XCTAssertEqual(circularMean([0.7, 0.7, 0.7]), 0.7, accuracy: 1e-9)
    }

    func testCircularMeanAveragesAcrossWraparound() {
        // Two angles straddling 0/pi should average to (near) 0, not pi/2.
        let mean = circularMean([0.05, .pi - 0.05])
        XCTAssertTrue(mean < 0.06 || mean > .pi - 0.06, "expected wraparound-aware mean near 0/pi, got \(mean)")
    }

    func testCircularMeanEmptyIsZero() {
        XCTAssertEqual(circularMean([]), 0)
    }

    // MARK: - OBBDetection corners / spans / long axis

    func testOBBDetectionCornersAxisAlignedAtZeroAngle() {
        let det = OBBDetection(cx: 100, cy: 50, w: 20, h: 10, angle: 0, conf: 0.9)
        let c = det.corners
        XCTAssertEqual(c.count, 4)
        let xs = c.map { Double($0.x) }.sorted()
        let ys = c.map { Double($0.y) }.sorted()
        for (got, want) in zip(xs, [90.0, 90.0, 110.0, 110.0]) {
            XCTAssertEqual(got, want, accuracy: 1e-9)
        }
        for (got, want) in zip(ys, [45.0, 45.0, 55.0, 55.0]) {
            XCTAssertEqual(got, want, accuracy: 1e-9)
        }
    }

    func testOBBDetectionOffsetShiftsCornersButKeepsIdentity() {
        let det = OBBDetection(cx: 10, cy: 10, w: 4, h: 2, angle: 0, conf: 0.5)
        let moved = det.offset(dx: 5, dy: -3)
        XCTAssertEqual(moved.cx, 15, accuracy: 1e-9)
        XCTAssertEqual(moved.cy, 7, accuracy: 1e-9)
        XCTAssertEqual(moved.id, det.id)
    }

    func testLongAxisAngleAccountsForWidthHeightSwap() {
        // w >= h: angle passes through unchanged.
        let upright = OBBDetection(cx: 0, cy: 0, w: 10, h: 40, angle: 0.3, conf: 1)
        XCTAssertEqual(upright.longAxisAngle(), normAngle(0.3 + .pi / 2), accuracy: 1e-9)
        let flat = OBBDetection(cx: 0, cy: 0, w: 40, h: 10, angle: 0.3, conf: 1)
        XCTAssertEqual(flat.longAxisAngle(), normAngle(0.3), accuracy: 1e-9)
    }

    func testSpansMatchCornerExtents() {
        let det = OBBDetection(cx: 0, cy: 0, w: 10, h: 4, angle: .pi / 4, conf: 1)
        let (xLo, xHi) = det.xSpan
        let (yLo, yHi) = det.ySpan
        let xs = det.corners.map { Double($0.x) }
        let ys = det.corners.map { Double($0.y) }
        XCTAssertEqual(xLo, xs.min()!, accuracy: 1e-9)
        XCTAssertEqual(xHi, xs.max()!, accuracy: 1e-9)
        XCTAssertEqual(yLo, ys.min()!, accuracy: 1e-9)
        XCTAssertEqual(yHi, ys.max()!, accuracy: 1e-9)
    }

    // MARK: - Lines

    func testLineIntersectionOfPerpendicularLines() {
        let vertical = Line.vertical(atX: 5)
        let horizontal = Line.horizontal(atY: 3)
        let p = vertical.intersection(with: horizontal)
        XCTAssertNotNil(p)
        XCTAssertPointsEqual(p!, CGPoint(x: 5, y: 3))
    }

    func testParallelLinesDoNotIntersect() {
        let a = Line.horizontal(atY: 0)
        let b = Line.horizontal(atY: 10)
        XCTAssertNil(a.intersection(with: b))
    }

    func testLineThroughTwoPointsSignedSideIsConsistent() {
        let line = Line.through(CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0))
        // Horizontal line through the origin: points above/below should have opposite signs.
        let above = line.signed(CGPoint(x: 5, y: -5))
        let below = line.signed(CGPoint(x: 5, y: 5))
        XCTAssertTrue(above * below < 0)
    }

    func testFitLineRecoversExactLinearPoints() {
        // y = 2x + 1
        let pts = (0..<5).map { CGPoint(x: CGFloat($0), y: CGFloat(2 * $0 + 1)) }
        let line = fitLine(pts)
        for x in stride(from: 0.0, through: 4.0, by: 1.0) {
            XCTAssertEqual(line.yValue(atX: x), 2 * x + 1, accuracy: 1e-6)
        }
    }

    // MARK: - Polygon primitives

    func testPolygonAreaOfUnitSquare() {
        let square = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]
        XCTAssertEqual(polygonArea(square), 1.0, accuracy: 1e-9)
    }

    func testEnsureCCWReversesClockwiseInput() {
        let cw = [CGPoint(x: 0, y: 0), CGPoint(x: 0, y: 1), CGPoint(x: 1, y: 1), CGPoint(x: 1, y: 0)]
        let ccw = ensureCCW(cw)
        // Signed area of the result must be positive (CCW).
        var signedArea = 0.0
        for i in 0..<ccw.count {
            let p = ccw[i], q = ccw[(i + 1) % ccw.count]
            signedArea += Double(p.x * q.y - q.x * p.y)
        }
        XCTAssertTrue(signedArea > 0)
    }

    func testClipPolygonAgainstHalfPlane() {
        let square = [CGPoint(x: 0, y: 0), CGPoint(x: 4, y: 0), CGPoint(x: 4, y: 4), CGPoint(x: 0, y: 4)]
        // Clip to the left half-plane of a vertical line through x=2, CCW edge going "up" keeps x <= 2.
        let clipped = clipPolygon(square, edgeA: CGPoint(x: 2, y: 4), edgeB: CGPoint(x: 2, y: 0))
        XCTAssertEqual(polygonArea(clipped), 8.0, accuracy: 1e-6)
    }

    func testPolyIntersectionAreaOfOverlappingSquares() {
        let a = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10)]
        let b = [CGPoint(x: 5, y: 5), CGPoint(x: 15, y: 5), CGPoint(x: 15, y: 15), CGPoint(x: 5, y: 15)]
        XCTAssertEqual(polyIntersectionArea(a, b), 25.0, accuracy: 1e-6)
    }

    func testPolyIntersectionAreaOfDisjointSquaresIsZero() {
        let a = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]
        let b = [CGPoint(x: 10, y: 10), CGPoint(x: 11, y: 10), CGPoint(x: 11, y: 11), CGPoint(x: 10, y: 11)]
        XCTAssertEqual(polyIntersectionArea(a, b), 0.0, accuracy: 1e-9)
    }

    func testPointInPolygonInsideOutsideAndOnBoundary() {
        let square = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10)]
        XCTAssertTrue(pointInPolygon(5, 5, square))
        XCTAssertFalse(pointInPolygon(50, 50, square))
        XCTAssertTrue(pointInPolygon(0, 5, square)) // boundary counts as inside
    }

    func testPointStrictlyInsideRejectsBoundaryButAcceptsInterior() {
        let square = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10)]
        XCTAssertFalse(pointStrictlyInside(0, 5, square), "boundary point must not be strictly inside")
        XCTAssertFalse(pointStrictlyInside(50, 50, square), "outside point must not be strictly inside")
        XCTAssertTrue(pointStrictlyInside(5, 5, square))
    }

    func testConvexHullOfPointsWithInteriorPoint() {
        let pts = [
            CGPoint(x: 0, y: 0), CGPoint(x: 4, y: 0), CGPoint(x: 4, y: 4), CGPoint(x: 0, y: 4),
            CGPoint(x: 2, y: 2), // interior — must be dropped
        ]
        let hull = convexHull(pts)
        XCTAssertEqual(hull.count, 4)
        XCTAssertEqual(polygonArea(hull), 16.0, accuracy: 1e-9)
        for p in hull {
            XCTAssertFalse(p == CGPoint(x: 2, y: 2))
        }
    }

    func testConvexHullOfCollinearPointsIsTheTwoEndpoints() {
        let pts = (0...4).map { CGPoint(x: CGFloat($0), y: 0) }
        let hull = convexHull(pts)
        XCTAssertEqual(hull.count, 2)
    }

    // MARK: - SAT separator

    func testFullySeparatesDetectsClearSeparationAndOverlap() {
        let line = Line.vertical(atX: 5)
        let left: [CGPoint] = [CGPoint(x: 0, y: 0), CGPoint(x: 2, y: 2)]
        let right: [CGPoint] = [CGPoint(x: 10, y: 0), CGPoint(x: 8, y: 2)]
        XCTAssertTrue(fullySeparates(line, left, right))
        XCTAssertFalse(fullySeparates(line, left, left))
    }

    func testSeamUsableRejectsOutOfImageIntersections() {
        let top = Line.horizontal(atY: 0)
        let bottom = Line.horizontal(atY: 100)
        XCTAssertTrue(seamUsable(.vertical(atX: 50), topLine: top, bottomLine: bottom, imgW: 100, imgH: 100))
        // A near-horizontal line barely misses hitting the vertical band lines within range.
        XCTAssertFalse(seamUsable(.horizontal(atY: 1000), topLine: top, bottomLine: bottom, imgW: 100, imgH: 100))
    }

    func testSatSeparatingLineFindsVerticalSeamBetweenTwoBoxes() {
        let boxA = OBBDetection(cx: 20, cy: 50, w: 20, h: 40, angle: 0, conf: 1).corners
        let boxB = OBBDetection(cx: 80, cy: 50, w: 20, h: 40, angle: 0, conf: 1).corners
        let line = satSeparatingLine(
            cornersA: boxA, cornersB: boxB,
            topLine: .horizontal(atY: 0), bottomLine: .horizontal(atY: 100),
            imgW: 100, imgH: 100, prefer: .vertical
        )
        XCTAssertNotNil(line)
        if let line {
            XCTAssertTrue(fullySeparates(line, boxA, boxB))
        }
    }

    func testSatSeparatingLineReturnsNilForOverlappingBoxes() {
        let boxA = OBBDetection(cx: 50, cy: 50, w: 40, h: 40, angle: 0, conf: 1).corners
        let boxB = OBBDetection(cx: 55, cy: 50, w: 40, h: 40, angle: 0, conf: 1).corners
        let line = satSeparatingLine(
            cornersA: boxA, cornersB: boxB,
            topLine: .horizontal(atY: 0), bottomLine: .horizontal(atY: 100),
            imgW: 100, imgH: 100
        )
        XCTAssertNil(line)
    }

    // MARK: - Rotated IoU

    func testRotatedIoUOfIdenticalBoxesIsOne() {
        let det = OBBDetection(cx: 10, cy: 10, w: 10, h: 6, angle: 0.4, conf: 1)
        XCTAssertEqual(rotatedIoU(det, det), 1.0, accuracy: 1e-6)
    }

    func testRotatedIoUOfDistantBoxesIsZero() {
        let a = OBBDetection(cx: 0, cy: 0, w: 5, h: 5, angle: 0, conf: 1)
        let b = OBBDetection(cx: 1000, cy: 1000, w: 5, h: 5, angle: 0, conf: 1)
        XCTAssertEqual(rotatedIoU(a, b), 0.0, accuracy: 1e-9)
    }

    func testRotatedIoUOfKnownPartialOverlap() {
        // Two axis-aligned 10x10 boxes offset by 5 in x: 5x10 overlap, area 50,
        // union = 100 + 100 - 50 = 150 -> IoU = 1/3.
        let a = OBBDetection(cx: 0, cy: 0, w: 10, h: 10, angle: 0, conf: 1)
        let b = OBBDetection(cx: 5, cy: 0, w: 10, h: 10, angle: 0, conf: 1)
        XCTAssertEqual(rotatedIoU(a, b), 1.0 / 3.0, accuracy: 1e-6)
    }
}
