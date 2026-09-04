import CoreGraphics
import XCTest

@testable import SpineCore

/// The quad rules generalized to polygons (A-7). Two things must hold: the
/// generalized checker agrees with the frozen quad reference wherever the
/// pieces are quads (so the Python-parity contract survives delegation),
/// and it catches partition violations that only arise once pieces are
/// arbitrary polygons.
final class PartitionRulesTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, w: Double = 40, h: Double = 90, angleDeg: Double = 0) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: 0.9)
    }

    private func quad(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [CGPoint] {
        [CGPoint(x: x0, y: y0), CGPoint(x: x1, y: y0), CGPoint(x: x1, y: y1), CGPoint(x: x0, y: y1)]
    }

    private func plan(_ shelf: Int, _ block: Int, _ q: [CGPoint], members: [Int]) -> CropPlan {
        CropPlan(shelfId: shelf, blockId: block, angleDeg: 0, quad: q, memberIndices: members)
    }

    // MARK: - Delegation stays faithful to the quad reference

    private func assertMatchesQuadReference(
        dets: [OBBDetection], plans: [CropPlan], imgW: Int, imgH: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let generalized = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        let reference = verifyPlanQuadReference(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        XCTAssertEqual(generalized.count, reference.count, file: file, line: line)
        for (g, r) in zip(generalized, reference) {
            XCTAssertEqual(g.rule, r.rule, file: file, line: line)
            XCTAssertEqual(g.hard, r.hard, "\(g.rule) hard flag", file: file, line: line)
            XCTAssertEqual(g.ok, r.ok, "\(g.rule) verdict", file: file, line: line)
            XCTAssertEqual(g.detail, r.detail, "\(g.rule) detail", file: file, line: line)
        }
    }

    func testCleanTilingMatchesQuadReference() {
        let dets = [det(60, 100), det(140, 100), det(260, 100), det(340, 100)]
        let plans = [
            plan(0, 0, quad(0, 0, 200, 400), members: [0, 1]),
            plan(0, 1, quad(200, 0, 400, 400), members: [2, 3]),
        ]
        assertMatchesQuadReference(dets: dets, plans: plans, imgW: 400, imgH: 400)
        XCTAssertTrue(hardRulesOK(verifyPlan(dets: dets, plans: plans, imgW: 400, imgH: 400)))
    }

    func testOverlappingAndUncoveredTilingsMatchQuadReference() {
        let dets = [det(60, 100), det(340, 100)]
        let overlapping = [
            plan(0, 0, quad(0, 0, 260, 400), members: [0]),
            plan(0, 1, quad(200, 0, 400, 400), members: [1]),
        ]
        assertMatchesQuadReference(dets: dets, plans: overlapping, imgW: 400, imgH: 400)
        XCTAssertFalse(hardRulesOK(verifyPlan(dets: dets, plans: overlapping, imgW: 400, imgH: 400)))

        let gapped = [
            plan(0, 0, quad(0, 0, 150, 400), members: [0]),
            plan(0, 1, quad(250, 0, 400, 400), members: [1]),
        ]
        assertMatchesQuadReference(dets: dets, plans: gapped, imgW: 400, imgH: 400)
        XCTAssertFalse(hardRulesOK(verifyPlan(dets: dets, plans: gapped, imgW: 400, imgH: 400)))
    }

    func testCutOBBMatchesQuadReference() {
        // The seam runs straight through the middle spine.
        let dets = [det(200, 100)]
        let plans = [
            plan(0, 0, quad(0, 0, 200, 400), members: []),
            plan(0, 1, quad(200, 0, 400, 400), members: [0]),
        ]
        assertMatchesQuadReference(dets: dets, plans: plans, imgW: 400, imgH: 400)
        let rules = verifyPlan(dets: dets, plans: plans, imgW: 400, imgH: 400)
        XCTAssertFalse(rules.first { $0.rule == "R3_NO_OBB_CUT" }!.ok)
    }

    func testEmptyPlansMatchQuadReference() {
        assertMatchesQuadReference(dets: [], plans: [], imgW: 400, imgH: 400)
    }

    // MARK: - Polygon-only behavior

    func testStaircasePartitionOfARegionPasses() {
        let region = quad(0, 0, 200, 200)
        // Two halves split by a staircase seam through the middle.
        let left: [CGPoint] = [
            CGPoint(x: 0, y: 0), CGPoint(x: 80, y: 0), CGPoint(x: 80, y: 100),
            CGPoint(x: 120, y: 100), CGPoint(x: 120, y: 200), CGPoint(x: 0, y: 200),
        ]
        let right: [CGPoint] = [
            CGPoint(x: 80, y: 0), CGPoint(x: 200, y: 0), CGPoint(x: 200, y: 200),
            CGPoint(x: 120, y: 200), CGPoint(x: 120, y: 100), CGPoint(x: 80, y: 100),
        ]
        let dets = [det(40, 100), det(160, 100)]
        let pieces = [
            PartitionPiece(name: "piece0", polygon: left, memberIndices: [0]),
            PartitionPiece(name: "piece1", polygon: right, memberIndices: [1]),
        ]
        let rules = verifyPartition(dets: dets, pieces: pieces, region: region, imgW: 200, imgH: 200)
        XCTAssertTrue(hardRulesOK(rules), "\(rules.filter { $0.hard && !$0.ok }.map(\.detail))")
    }

    /// Coverage is judged against the region being partitioned, not the whole
    /// image — the invariant that keeps a recursive cut from leaking into a
    /// sibling piece's area.
    func testCoverageIsRelativeToTheRegionNotTheImage() {
        let region = quad(0, 0, 100, 200)
        let pieces = [
            PartitionPiece(name: "piece0", polygon: quad(0, 0, 100, 100)),
            PartitionPiece(name: "piece1", polygon: quad(0, 100, 100, 200)),
        ]
        let inRegion = verifyPartition(dets: [], pieces: pieces, region: region, imgW: 400, imgH: 200)
        XCTAssertTrue(hardRulesOK(inRegion))

        let againstWholeImage = verifyPartition(
            dets: [], pieces: pieces, region: imageRegion(imgW: 400, imgH: 200), imgW: 400, imgH: 200
        )
        XCTAssertFalse(hardRulesOK(againstWholeImage), "the same pieces cover only a quarter of the image")
    }

    func testNonConvexOverlapIsCaught() {
        let region = quad(0, 0, 100, 100)
        let lShape: [CGPoint] = [
            CGPoint(x: 0, y: 0), CGPoint(x: 50, y: 0), CGPoint(x: 50, y: 50),
            CGPoint(x: 100, y: 50), CGPoint(x: 100, y: 100), CGPoint(x: 0, y: 100),
        ]
        let pieces = [
            PartitionPiece(name: "piece0", polygon: lShape),
            // Overlaps the L's lower band rather than sitting in its notch.
            PartitionPiece(name: "piece1", polygon: quad(0, 60, 100, 100)),
        ]
        let rules = verifyPartition(dets: [], pieces: pieces, region: region, imgW: 100, imgH: 100)
        XCTAssertFalse(rules.first { $0.rule == "R2_NO_OVERLAP" }!.ok)
    }

    // MARK: - Run-level invariants

    private func result(leaves: [ZoomLeafInfo]) -> JigsawZoomResult {
        JigsawZoomResult(
            detections: [], firstPassCount: 0, plannedCropCount: leaves.count,
            inferencePasses: leaves.count, leafCount: leaves.count,
            fallbackLeafCount: leaves.filter(\.isFallback).count,
            topLevelDropCount: 0, dedupHitCount: 0, cropQuads: [],
            usedRecursion: leaves.count > 1, leaves: leaves
        )
    }

    func testZoomRunInvariantsPassWhenEveryLeafReachedThreshold() {
        let rules = verifyZoomRun(
            result(leaves: [
                ZoomLeafInfo(depth: 1, scale: 0.97, isFallback: false, detectionCount: 3),
                ZoomLeafInfo(depth: 2, scale: 1.0, isFallback: false, detectionCount: 4),
            ]),
            downsampleThreshold: 0.95
        )
        XCTAssertTrue(hardRulesOK(rules))
        XCTAssertEqual(rules.first { $0.rule == "S4_DEPTH_HISTOGRAM" }?.detail, "d1=1 d2=1")
    }

    func testUnderScaledNonFallbackLeafFailsButA4LeafDoesNot() {
        let bad = verifyZoomRun(
            result(leaves: [ZoomLeafInfo(depth: 1, scale: 0.4, isFallback: false, detectionCount: 1)]),
            downsampleThreshold: 0.95
        )
        XCTAssertFalse(bad.first { $0.rule == "R7_LEAF_SCALE_OK" }!.ok)

        let fallback = verifyZoomRun(
            result(leaves: [ZoomLeafInfo(depth: 1, scale: 0.4, isFallback: true, detectionCount: 1)]),
            downsampleThreshold: 0.95
        )
        XCTAssertTrue(hardRulesOK(fallback), "an A-4 fallback leaf is legal, just logged")
    }

    func testRunWithoutLeavesFailsTermination() {
        XCTAssertFalse(hardRulesOK(verifyZoomRun(result(leaves: []), downsampleThreshold: 0.95)))
    }
}
