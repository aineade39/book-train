import CoreGraphics
import SpineCore
import XCTest
@testable import SpinePipeline

/// Covers the locked "Book ID OCR gains" plan §D "Spine association":
/// "scene-pixel barcode center; attach if inside OBB or distance <=
/// clamp(0.5 * spineShortEdge, 24, 96) px; ties within 5px -> unassociated."
///
/// Note on geometry: for any axis-aligned box, a disk of radius
/// `r = min(halfWidth, halfHeight)` centered at the box's own center is
/// always a *subset* of the box (`dx^2 + dy^2 <= r^2` implies `|dx| <= r`
/// and `|dy| <= r`). Since `attachRadius`'s un-clamped value is exactly
/// `min(halfWidth, halfHeight)`, the "outside the box but within its own
/// radius" fallback can only ever fire once the **24px floor** raises the
/// radius above that box's own half-short-edge -- i.e. for thin spines
/// (short edge < 48px). The tests below use thin boxes for that reason.
final class BarcodeSpineAssociationTests: XCTestCase {
    private func detection(cx: Double, cy: Double, w: Double = 60, h: Double = 300, angle: Double = 0) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angle, conf: 0.9)
    }

    // MARK: - attachRadius

    func testAttachRadiusClampsToLowerBoundForTinyShortEdge() {
        XCTAssertEqual(BarcodeSpineAssociation.attachRadius(shortEdge: 10), 24)
    }

    func testAttachRadiusClampsToUpperBoundForHugeShortEdge() {
        XCTAssertEqual(BarcodeSpineAssociation.attachRadius(shortEdge: 1000), 96)
    }

    func testAttachRadiusIsHalfShortEdgeInMiddleRange() {
        XCTAssertEqual(BarcodeSpineAssociation.attachRadius(shortEdge: 100), 50)
    }

    // MARK: - Containment

    func testPointInsideAxisAlignedOBBAssociatesEvenWithACloserByDistanceButNonContainingDetection() {
        let containing = detection(cx: 100, cy: 100, w: 60, h: 300)
        let distractor = detection(cx: 500, cy: 500, w: 60, h: 300)
        let point = CGPoint(x: 110, y: 100) // inside containing (half-width 30)

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-A", scenePoint: point)], detections: [containing, distractor]
        )
        XCTAssertEqual(result[containing.id], "isbn-A")
        XCTAssertNil(result[distractor.id])
    }

    func testPointInsideRotatedOBBAssociatesViaLocalFrameRotation() {
        // A 60x300 box centered at (200, 200), rotated 90 degrees -- its
        // *scene* footprint is now 300 wide x 60 tall, so a point 100px to
        // the right of center (which would miss the box unrotated) falls
        // inside it once rotation is accounted for.
        let rotated = detection(cx: 200, cy: 200, w: 60, h: 300, angle: .pi / 2)
        let point = CGPoint(x: 300, y: 200) // 100px along scene-x from center

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-rot", scenePoint: point)], detections: [rotated]
        )
        XCTAssertEqual(result[rotated.id], "isbn-rot")
    }

    // MARK: - Nearest-within-radius fallback (thin spines only, see header note)

    func testPointOutsideAThinSpineButWithinTheClampedFloorRadiusStillAssociates() {
        // w=20 -> halfWidth 10, shortEdge 20 -> attachRadius clamps up to
        // 24 (well above the box's own 10px half-width), so a point 15px
        // outside the long edge -- not contained -- still falls inside
        // the clamped-up radius disk.
        let thinSpine = detection(cx: 0, cy: 0, w: 20, h: 300)
        let justOutsideEdge = CGPoint(x: 15, y: 0)

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-near", scenePoint: justOutsideEdge)], detections: [thinSpine]
        )
        XCTAssertEqual(result[thinSpine.id], "isbn-near")
    }

    func testPointBeyondTheClampedFloorRadiusIsUnassociated() {
        let thinSpine = detection(cx: 0, cy: 0, w: 20, h: 300)
        let farPoint = CGPoint(x: 100, y: 0) // outside the 24px clamped radius

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-far", scenePoint: farPoint)], detections: [thinSpine]
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testGenuineTieWithinFivePixelsIsLeftUnassociated() {
        // Two thin spines (radius 24 each) placed so a point sits exactly
        // equidistant (24px) from both centroids, contained by neither.
        let left = detection(cx: 0, cy: 0, w: 20, h: 300)
        let right = detection(cx: 48, cy: 0, w: 20, h: 300)
        let midpoint = CGPoint(x: 24, y: 0)

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-tie", scenePoint: midpoint)], detections: [left, right]
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testUnambiguouslyNearerDetectionWinsOverAFartherSecondCandidate() {
        let near = detection(cx: 0, cy: 0, w: 20, h: 300)
        let far = detection(cx: 1000, cy: 0, w: 20, h: 300)
        let point = CGPoint(x: 15, y: 0) // 15px from `near`'s centroid, ~985px from `far`'s.

        let result = BarcodeSpineAssociation.associate(
            points: [(payload: "isbn-near2", scenePoint: point)], detections: [near, far]
        )
        XCTAssertEqual(result[near.id], "isbn-near2")
        XCTAssertNil(result[far.id])
    }

    // MARK: - Multiple payloads

    func testMultipleBarcodesAssociateIndependentlyToDifferentSpines() {
        let first = detection(cx: 0, cy: 0, w: 60, h: 300)
        let second = detection(cx: 1000, cy: 0, w: 60, h: 300)

        let result = BarcodeSpineAssociation.associate(
            points: [
                (payload: "isbn-1", scenePoint: CGPoint(x: 0, y: 0)),
                (payload: "isbn-2", scenePoint: CGPoint(x: 1000, y: 0)),
            ],
            detections: [first, second]
        )
        XCTAssertEqual(result[first.id], "isbn-1")
        XCTAssertEqual(result[second.id], "isbn-2")
    }
}
