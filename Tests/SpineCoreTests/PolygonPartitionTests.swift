import CoreGraphics
import XCTest

@testable import SpineCore

/// Stage 3 polygon primitives: exact overlap area for the non-convex
/// staircase pieces the free-form cutter emits, and splitting a polygon
/// with a monotone seam (~/dev/book-id-design/docs/
/// jigsaw-zoom-requirements.md R-2).
final class PolygonPartitionTests: XCTestCase {
    private func pt(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: x, y: y) }

    private func rect(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [CGPoint] {
        [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)]
    }

    /// An L: the unit square with its top-right quarter removed.
    private var lShape: [CGPoint] {
        [pt(0, 0), pt(50, 0), pt(50, 50), pt(100, 50), pt(100, 100), pt(0, 100)]
    }

    // MARK: - Convexity + triangulation

    func testConvexityDetection() {
        XCTAssertTrue(isConvexPolygon(rect(0, 0, 10, 10)))
        XCTAssertFalse(isConvexPolygon(lShape))
    }

    func testTriangulationConservesArea() {
        for polygon in [rect(0, 0, 40, 25), lShape] {
            let triangles = triangulatePolygon(polygon)
            XCTAssertFalse(triangles.isEmpty)
            XCTAssertEqual(triangles.count, dedupeRing(polygon).count - 2)
            let sum = triangles.reduce(0.0) { $0 + polygonArea($1) }
            XCTAssertEqual(sum, polygonArea(polygon), accuracy: 1e-6)
        }
    }

    func testTriangulationHandlesStaircase() {
        var staircase: [CGPoint] = [pt(0, 0)]
        for step in 0..<8 {
            staircase.append(pt(Double(step) * 10, Double(step) * 10 + 10))
            staircase.append(pt(Double(step) * 10 + 10, Double(step) * 10 + 10))
        }
        staircase.append(pt(80, 0))
        let triangles = triangulatePolygon(staircase)
        XCTAssertFalse(triangles.isEmpty, "a monotone staircase must triangulate")
        let sum = triangles.reduce(0.0) { $0 + polygonArea($1) }
        XCTAssertEqual(sum, polygonArea(staircase), accuracy: 1e-6)
    }

    // MARK: - Intersection area

    func testIntersectionAreaMatchesConvexPathOnQuads() {
        let a = rect(0, 0, 10, 10), b = rect(5, 5, 20, 20)
        XCTAssertEqual(simplePolyIntersectionArea(a, b), 25, accuracy: 1e-6)
        XCTAssertEqual(simplePolyIntersectionArea(a, b), polyIntersectionArea(ensureCCW(a), ensureCCW(b)), accuracy: 1e-6)
    }

    func testDisjointPolygonsHaveNoIntersection() {
        XCTAssertEqual(simplePolyIntersectionArea(rect(0, 0, 10, 10), rect(20, 0, 30, 10)), 0, accuracy: 1e-9)
    }

    /// The reason this file exists: `polyIntersectionArea` clips the subject
    /// against the clip polygon's half-planes, which only bounds the polygon
    /// when it is convex. Two interlocking Ls overlap across both diagonal
    /// quarters, and the half-plane clip misses it entirely — an overlap
    /// silently reported as zero is the one failure mode `R2_NO_OVERLAP`
    /// cannot tolerate.
    func testNonConvexOverlapIsExactWhereConvexClipIsNot() {
        // The square minus its bottom-left quarter, versus the L above
        // (square minus its top-right quarter).
        let mirrored = [pt(0, 0), pt(100, 0), pt(100, 100), pt(50, 100), pt(50, 50), pt(0, 50)]
        XCTAssertFalse(isConvexPolygon(mirrored))
        XCTAssertEqual(simplePolyIntersectionArea(lShape, mirrored), 5000, accuracy: 1e-6)
        XCTAssertEqual(polyIntersectionArea(ensureCCW(lShape), ensureCCW(mirrored)), 0, accuracy: 1e-6)
    }

    func testNonConvexPartialOverlapArea() {
        // Spans the L's lower half, which the L fully covers.
        let strip = rect(0, 60, 100, 80)
        XCTAssertEqual(simplePolyIntersectionArea(lShape, strip), 100 * 20, accuracy: 1e-6)
    }

    // MARK: - Letterbox source dimension

    /// The cutter's stop criterion has to agree with the scale the engine
    /// will actually letterbox a piece at, for both placement modes.
    func testLetterboxSourceDimMatchesTheEnginesLetterboxScale() throws {
        let raster = try XCTUnwrap(SceneRaster(cgImage: makeSolidCGImage(width: 400, height: 400)))
        // A diagonal bar: its bounding box and its minimum-area rect have
        // very different long sides, so the two modes must not be conflated.
        let tilted = [pt(20, 20), pt(300, 300), pt(280, 320), pt(0, 40)]
        for rotate in [false, true] {
            let dim = letterboxSourceDim(tilted, rotate: rotate)
            let lb = try XCTUnwrap(letterboxPiece(raster: raster, polygon: tilted, imgsz: 128, rotate: rotate))
            XCTAssertEqual(dim, 128 / lb.scale, accuracy: 1.5, "rotate=\(rotate)")
        }
        XCTAssertEqual(letterboxSourceDim(rect(0, 0, 90, 40), rotate: false), 90, accuracy: 1e-6)
    }

    // MARK: - Monotone seam split

    private func verticalSeam(x: [Double], y: [Double]) -> MonotoneSeam {
        MonotoneSeam(points: zip(x, y).map { pt($0, $1) }, splitAxis: .x)
    }

    func testStraightSeamSplitsRectangleInTwo() throws {
        let region = rect(0, 0, 100, 60)
        let seam = verticalSeam(x: [40, 40], y: [-10, 70])
        let halves = try XCTUnwrap(splitPolygonByMonotoneSeam(region, seam: seam))
        XCTAssertEqual(polygonArea(halves.a) + polygonArea(halves.b), 6000, accuracy: 1e-6)
        let areas = [polygonArea(halves.a), polygonArea(halves.b)].sorted()
        XCTAssertEqual(areas[0], 40 * 60, accuracy: 1e-6)
        XCTAssertEqual(areas[1], 60 * 60, accuracy: 1e-6)
    }

    func testStaircaseSeamConservesAreaAndBendsAroundObstacle() throws {
        let region = rect(0, 0, 100, 100)
        // Steps right in the middle third, the shape a seam takes to weave
        // between two spines that no straight line can separate.
        let seam = verticalSeam(x: [30, 30, 70, 70, 30, 30], y: [-5, 20, 40, 60, 80, 105])
        let halves = try XCTUnwrap(splitPolygonByMonotoneSeam(region, seam: seam))
        XCTAssertEqual(polygonArea(halves.a) + polygonArea(halves.b), 10000, accuracy: 1e-6)
        XCTAssertFalse(isConvexPolygon(halves.a), "a staircase cut yields a non-convex piece")
        XCTAssertEqual(simplePolyIntersectionArea(halves.a, halves.b), 0, accuracy: 4.0)
    }

    func testSeamThatDoesNotCrossTwiceIsRejected() {
        let region = rect(0, 0, 100, 60)
        // Stops inside the region: three crossings are ambiguous, one is a
        // dead end. Either way there is no clean two-sided split.
        let stopsInside = verticalSeam(x: [40, 40], y: [-10, 30])
        XCTAssertNil(splitPolygonByMonotoneSeam(region, seam: stopsInside))

        let entirelyOutside = verticalSeam(x: [140, 140], y: [-10, 70])
        XCTAssertNil(splitPolygonByMonotoneSeam(region, seam: entirelyOutside))
    }

    func testSplittingAnLShapeStaysInsideTheParent() throws {
        let seam = verticalSeam(x: [25, 25], y: [-10, 110])
        let halves = try XCTUnwrap(splitPolygonByMonotoneSeam(lShape, seam: seam))
        XCTAssertEqual(polygonArea(halves.a) + polygonArea(halves.b), polygonArea(lShape), accuracy: 1e-6)
        // Neither half may reach into the L's missing quarter.
        for half in [halves.a, halves.b] {
            XCTAssertFalse(pointInPolygon(75, 25, half), "a child must not leave the parent piece")
        }
    }
}
