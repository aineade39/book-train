import CoreGraphics
import XCTest

@testable import SpineCore

/// Rule-level tests for `verifyPlan`: a valid tiled plan, each individual
/// R1-R5 failure mode in isolation, allowed shared boundaries, and the S2/S3
/// soft-rule outcomes. All scenes are hand-authored (no model/dataset).
final class LayoutCropRulesTests: XCTestCase {
    let imgW = 100
    let imgH = 100

    private func quad(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [CGPoint] {
        [CGPoint(x: x0, y: y0), CGPoint(x: x1, y: y0), CGPoint(x: x1, y: y1), CGPoint(x: x0, y: y1)]
    }

    private func result(_ results: [RuleResult], _ rule: String) -> RuleResult {
        guard let r = results.first(where: { $0.rule == rule }) else {
            XCTFail("missing rule \(rule)")
            return RuleResult(rule: rule, hard: true, ok: false, detail: "missing")
        }
        return r
    }

    // MARK: - Valid plan

    func testValidTwoColumnPlanPassesAllHardRules() {
        // Two side-by-side crops that exactly tile the image; one det per crop,
        // both fully contained (no cut, no orphan).
        let plans = [
            CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 50, 100), memberIndices: [0]),
            CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(50, 0, 100, 100), memberIndices: [1]),
        ]
        let dets = [
            OBBDetection(cx: 25, cy: 50, w: 10, h: 30, angle: 0, conf: 1),
            OBBDetection(cx: 75, cy: 50, w: 10, h: 30, angle: 0, conf: 1),
        ]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        for r in results where r.hard {
            XCTAssertTrue(r.ok, "\(r.rule) failed: \(r.detail)")
        }
        XCTAssertTrue(hardRulesOK(results))
    }

    func testSharedBoundaryTouchIsAllowedNotFlaggedAsOverlapOrCut() {
        // A det that straddles the shared edge exactly (corner touching, not
        // strictly inside the neighbor) must not trip R2 or R3.
        let plans = [
            CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 50, 100), memberIndices: [0]),
            CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(50, 0, 100, 100), memberIndices: []),
        ]
        // Center at x=25 (owned by crop 0), corner touches x=50 boundary exactly.
        let dets = [OBBDetection(cx: 25, cy: 50, w: 50, h: 20, angle: 0, conf: 1)]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertTrue(result(results, "R2_NO_OVERLAP").ok)
        XCTAssertTrue(result(results, "R3_NO_OBB_CUT").ok, result(results, "R3_NO_OBB_CUT").detail)
    }

    // MARK: - R1 COVER_IMAGE

    func testR1FailsWhenPlansLeaveAGap() {
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 50, 100), memberIndices: [])]
        let results = verifyPlan(dets: [], plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R1_COVER_IMAGE").ok)
    }

    // MARK: - R2 NO_OVERLAP

    func testR2FailsWhenPlansOverlap() {
        let plans = [
            CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 60, 100), memberIndices: []),
            CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(40, 0, 100, 100), memberIndices: []),
        ]
        let results = verifyPlan(dets: [], plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R2_NO_OVERLAP").ok)
    }

    // MARK: - R3 NO_OBB_CUT

    func testR3FailsWhenACropBoundaryCutsThroughAnOBB() {
        let plans = [
            CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 50, 100), memberIndices: [0]),
            CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(50, 0, 100, 100), memberIndices: []),
        ]
        // Det centered at x=45, wide enough (w=30) that its corner (x=60) is
        // strictly inside the neighboring crop -- the boundary cuts it.
        let dets = [OBBDetection(cx: 45, cy: 50, w: 30, h: 20, angle: 0, conf: 1)]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R3_NO_OBB_CUT").ok)
    }

    // MARK: - R4 EVERY_OBB_OWNED

    func testR4FailsWhenADetCenterIsOutsideEveryCrop() {
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 100, 100), memberIndices: [])]
        let dets = [OBBDetection(cx: 500, cy: 500, w: 10, h: 10, angle: 0, conf: 1)]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R4_EVERY_OBB_OWNED").ok)
    }

    // MARK: - R5 QUAD_FINITE

    func testR5FailsWhenAQuadCornerIsFarOutsideTheImage() {
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 100, 200), memberIndices: [])]
        let results = verifyPlan(dets: [], plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R5_QUAD_FINITE").ok)
    }

    func testR5FailsWithNoPlansAtAll() {
        let results = verifyPlan(dets: [], plans: [], imgW: imgW, imgH: imgH)
        XCTAssertEqual(results.count, 1)
        XCTAssertFalse(result(results, "R5_QUAD_FINITE").ok)
        XCTAssertFalse(hardRulesOK(results))
    }

    func testR5FailsOnNonFiniteCorner() {
        let badQuad = [CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 0), CGPoint(x: Double.nan, y: 100), CGPoint(x: 0, y: 100)]
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: badQuad, memberIndices: [])]
        let results = verifyPlan(dets: [], plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertFalse(result(results, "R5_QUAD_FINITE").ok)
    }

    // MARK: - S2 / S3 soft rules

    func testS2ReportsEmptyCellCountButAlwaysOk() {
        let plans = [
            CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 50, 100), memberIndices: []),
            CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(50, 0, 100, 100), memberIndices: [0]),
        ]
        let dets = [OBBDetection(cx: 75, cy: 50, w: 10, h: 30, angle: 0, conf: 1)]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        let s2 = result(results, "S2_EMPTY_CELLS_OK")
        XCTAssertTrue(s2.ok)
        XCTAssertFalse(s2.hard)
        XCTAssertTrue(s2.detail.contains("empty_cells=1"))
    }

    func testS3FlagsOrientationImpurityBeyondDoubleTolerance() {
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 100, 100), memberIndices: [0, 1])]
        // Long-axis angles 0 and pi/2 (mod-pi orthogonal, the maximum possible
        // separation): their circular mean sits exactly halfway, so each
        // member's spread from the mean is 45 deg. With a tight tolerance
        // (2*angleTolDeg=10deg here) that 45 deg spread must trip S3.
        let dets = [
            OBBDetection(cx: 50, cy: 50, w: 40, h: 10, angle: 0, conf: 1),
            OBBDetection(cx: 50, cy: 50, w: 10, h: 40, angle: 0, conf: 1),
        ]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH, angleTolDeg: 5)
        let s3 = result(results, "S3_BLOCK_ORIENTATION_PURITY")
        XCTAssertFalse(s3.ok)
        XCTAssertFalse(s3.hard)
    }

    func testS3PassesWhenMembersShareOrientation() {
        let plans = [CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 100, 100), memberIndices: [0, 1])]
        let dets = [
            OBBDetection(cx: 30, cy: 50, w: 10, h: 40, angle: 0.02, conf: 1),
            OBBDetection(cx: 70, cy: 50, w: 10, h: 40, angle: -0.02, conf: 1),
        ]
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertTrue(result(results, "S3_BLOCK_ORIENTATION_PURITY").ok)
    }

    // MARK: - owningCrop

    func testOwningCropPicksSmallestContainingPolygon() {
        let big = CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 100, 100))
        let small = CropPlan(shelfId: 0, blockId: 1, angleDeg: 0, quad: quad(20, 20, 40, 40))
        let owner = owningCrop(x: 30, y: 30, plans: [big, small])
        XCTAssertEqual(owner?.name, small.name)
    }

    func testOwningCropIsNilWhenNoCropContainsThePoint() {
        let plan = CropPlan(shelfId: 0, blockId: 0, angleDeg: 0, quad: quad(0, 0, 10, 10))
        XCTAssertNil(owningCrop(x: 500, y: 500, plans: [plan]))
    }
}
