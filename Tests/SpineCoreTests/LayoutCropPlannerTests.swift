import CoreGraphics
import XCTest

@testable import SpineCore

/// Planner tests against hand-authored OBB scenes: empty scene, a single
/// shelf/block, gapped columns, an orientation-forced column split, a
/// geometrically-unsplittable interleaved pair (must merge, never cut a
/// book), and oversized-group recursion. Uses a uniform-gray raster
/// throughout so pixel-texture seam search is deterministic (zero energy
/// everywhere -> first-allowed-position tie-break, matching Python's
/// `np.argmin` first-occurrence behavior).
final class LayoutCropPlannerTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, _ w: Double, _ h: Double, angleDeg: Double = 0, conf: Float = 0.9) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: conf)
    }

    private func raster(width: Int, height: Int) -> SceneRaster {
        SceneRaster(cgImage: makeSolidCGImage(width: width, height: height))!
    }

    /// Every det index appears in exactly one crop's `memberIndices`, and
    /// `verifyPlan`'s hard rules all pass -- the baseline every plan must
    /// satisfy regardless of scene content.
    private func assertValidTiling(_ dets: [OBBDetection], _ plans: [CropPlan], imgW: Int, imgH: Int, file: StaticString = #filePath, line: UInt = #line) {
        let results = verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)
        for r in results where r.hard {
            XCTAssertTrue(r.ok, "\(r.rule) failed: \(r.detail)", file: file, line: line)
        }
        var owners = [Int: Int]() // det index -> count of crops claiming it
        for p in plans {
            for m in p.memberIndices { owners[m, default: 0] += 1 }
        }
        for i in 0..<dets.count {
            XCTAssertEqual(owners[i], 1, "det \(i) should be claimed by exactly one crop", file: file, line: line)
        }
    }

    // MARK: - Empty scene

    func testPlanCropsOnEmptySceneReturnsOneFullImageCrop() {
        let imgW = 200, imgH = 100
        let plans = planCrops(dets: [], imgW: imgW, imgH: imgH, raster: raster(width: imgW, height: imgH))
        XCTAssertEqual(plans.count, 1)
        XCTAssertTrue(plans[0].memberIndices.isEmpty)
        let (x0, y0, x1, y1) = plans[0].rect
        XCTAssertEqual(x0, 0, accuracy: 1e-6); XCTAssertEqual(y0, 0, accuracy: 1e-6)
        XCTAssertEqual(x1, Double(imgW), accuracy: 1e-6); XCTAssertEqual(y1, Double(imgH), accuracy: 1e-6)
        assertValidTiling([], plans, imgW: imgW, imgH: imgH)
    }

    // MARK: - Single shelf, single block

    func testPlanCropsSingleShelfSingleBlockOwnsAllMembers() {
        let imgW = 300, imgH = 220
        let dets = [
            det(60, 110, 40, 200), det(120, 110, 40, 200), det(180, 110, 40, 200), det(240, 110, 40, 200),
        ]
        let plans = planCrops(dets: dets, imgW: imgW, imgH: imgH, raster: raster(width: imgW, height: imgH))
        assertValidTiling(dets, plans, imgW: imgW, imgH: imgH)
        let nonEmpty = plans.filter { !$0.memberIndices.isEmpty }
        XCTAssertEqual(nonEmpty.count, 1, "touching same-orientation books on one shelf should stay one block")
        XCTAssertEqual(Set(nonEmpty[0].memberIndices), Set(0..<dets.count))
    }

    // MARK: - Gapped columns

    func testPlanCropsSplitsTwoColumnsSeparatedByAWideGap() {
        let imgW = 900, imgH = 300
        let left = [det(80, 150, 50, 200), det(150, 150, 50, 200), det(220, 150, 50, 200)]
        let right = [det(600, 150, 50, 200), det(670, 150, 50, 200), det(740, 150, 50, 200)]
        let dets = left + right
        let plans = planCrops(dets: dets, imgW: imgW, imgH: imgH, raster: raster(width: imgW, height: imgH))
        assertValidTiling(dets, plans, imgW: imgW, imgH: imgH)

        let leftIdx = Set(0..<left.count)
        let rightIdx = Set(left.count..<dets.count)
        let leftCrop = plans.first { !$0.memberIndices.isEmpty && Set($0.memberIndices).isSubset(of: leftIdx) }
        let rightCrop = plans.first { !$0.memberIndices.isEmpty && Set($0.memberIndices).isSubset(of: rightIdx) }
        XCTAssertNotNil(leftCrop, "left cluster should form its own block")
        XCTAssertNotNil(rightCrop, "right cluster should form its own block")
        XCTAssertNotEqual(leftCrop?.name, rightCrop?.name)
    }

    // MARK: - Orientation-forced split (no gap required)

    func testBuildColumnBlocksSplitsOnOrientationChangeEvenWhenTouching() {
        // Two touching pairs, same x-chain, no whitespace gap between them,
        // but a >25 deg orientation change between the pairs -- orientation
        // alone must start a new block.
        let dets = [
            det(80, 100, 40, 200, angleDeg: 0), det(150, 100, 40, 200, angleDeg: 0),
            det(220, 100, 40, 200, angleDeg: 0), det(290, 100, 40, 200, angleDeg: 0),
        ]
        // Verify the pure gap-chain (no orientation check) would have merged
        // everything, i.e. this scene has no real whitespace gap forcing a split.
        let members = Array(0..<dets.count)
        let sameOrientationBlocks = buildColumnBlocks(dets, members: members, angleTolDeg: 25, colGapPx: 40, minBlockMembers: 2)
        XCTAssertEqual(sameOrientationBlocks.count, 1, "sanity: no gap-based split expected before rotating any member")

        var rotated = dets
        rotated[2] = det(220, 100, 40, 200, angleDeg: 80)
        rotated[3] = det(290, 100, 40, 200, angleDeg: 80)
        let blocks = buildColumnBlocks(rotated, members: members, angleTolDeg: 25, colGapPx: 40, minBlockMembers: 2)
        XCTAssertEqual(blocks.count, 2, "an orientation change beyond tolerance must split the chain even with no gap")
        XCTAssertEqual(blocks[0], [0, 1])
        XCTAssertEqual(blocks[1], [2, 3])
    }

    // MARK: - Unsplittable interleaving

    func testResolveColumnSeamsMergesGeometricallyInterleavedBlocks() {
        // A near-flat rotated pair whose convex hull overlaps in x with an
        // upright neighboring pair -- no straight vertical (or any) line can
        // separate the two blocks' OBB corners without cutting one. The
        // no-cut rule must win: resolveColumnSeams merges them into one.
        let dets = [
            det(80, 400, 50, 200, angleDeg: 0), det(150, 400, 50, 200, angleDeg: 0),
            det(220, 400, 50, 200, angleDeg: 80), det(290, 400, 50, 200, angleDeg: 80),
        ]
        let blocks = [[0, 1], [2, 3]]
        let (merged, vLines) = resolveColumnSeams(
            dets, blocks: blocks, colGapPx: 40,
            topLine: .horizontal(atY: 300), bottomLine: .horizontal(atY: 500),
            imgW: 900, imgH: 800, energy: nil
        )
        XCTAssertEqual(merged.count, 1, "interleaved blocks must merge rather than produce a cutting seam")
        XCTAssertEqual(Set(merged[0]), Set([0, 1, 2, 3]))
        XCTAssertEqual(vLines.count, 2) // just the outer image-edge bounds
    }

    // MARK: - Oversized-group recursion

    func testPlanCropsSplitsAnOversizedShelfBySize() {
        // 8 books on one shelf, each 30px apart (well under colGapPx so they
        // still chain into one column block) but spanning ~690px total --
        // past a tiny maxDim (imgsz 100 * maxCropDimK 1.2 = 120). Must be
        // recursively re-split purely by size; the 30px gaps (bigger than
        // 2 * the 9px forbidden-zone pad around each book) leave genuine
        // pixel-seam room so the size-forced cut never grazes a book.
        let imgW = 900, imgH = 300
        var dets: [OBBDetection] = []
        for i in 0..<8 {
            dets.append(det(60.0 + Double(i) * 90.0, 150, 60, 200))
        }
        let plans = planCrops(
            dets: dets, imgW: imgW, imgH: imgH, raster: raster(width: imgW, height: imgH),
            imgsz: 100, maxCropDimK: 1.2
        )
        assertValidTiling(dets, plans, imgW: imgW, imgH: imgH)
        let nonEmpty = plans.filter { !$0.memberIndices.isEmpty }
        XCTAssertGreaterThan(nonEmpty.count, 1, "an oversized shelf group must be re-split by size")
    }
}
