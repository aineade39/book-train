import CoreGraphics
import XCTest

@testable import SpineCore

/// Stage 3 cutter behavior (~/dev/book-id-design/docs/
/// jigsaw-zoom-requirements.md R-2 / R-6 / A-4 / A-9). Model-free: cutters
/// only consume detections, a raster and an edge-energy map.
final class PieceCutterTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, w: Double = 30, h: Double = 120, angleDeg: Double = 0) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: 0.9)
    }

    private func request(
        dets: [OBBDetection],
        width: Int,
        height: Int,
        polygon: [CGPoint]? = nil,
        scale: Double = 1.0,
        imgsz: Int = 1024,
        threshold: Double = 0.95,
        image: CGImage? = nil
    ) throws -> ZoomCutRequest {
        let cgImage = image ?? makeSolidCGImage(width: width, height: height)
        let raster = try XCTUnwrap(SceneRaster(cgImage: cgImage))
        let energy = try XCTUnwrap(EdgeEnergy(raster: raster))
        return ZoomCutRequest(
            polygon: polygon ?? [
                CGPoint(x: 0, y: 0), CGPoint(x: width, y: 0),
                CGPoint(x: width, y: height), CGPoint(x: 0, y: height),
            ],
            dets: dets, raster: raster, energy: energy,
            width: width, height: height, imgsz: imgsz, scale: scale,
            rotatePieces: false, downsampleThreshold: threshold,
            angleTolDeg: LayoutConstants.defaultAngleTolDeg, depth: 0, debug: false
        )
    }

    /// Four upright spines in a row with clear gaps between them.
    private func spineRow(count: Int, spacing: Double, y: Double) -> [OBBDetection] {
        (0..<count).map { det(spacing * (Double($0) + 0.5), y) }
    }

    private func assertPartition(
        _ pieces: [[CGPoint]], of request: ZoomCutRequest,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let named = pieces.enumerated().map { index, polygon in
            PartitionPiece(
                name: "piece\(index)", polygon: polygon,
                memberIndices: request.dets.indices.filter {
                    pointInPolygon(request.dets[$0].cx, request.dets[$0].cy, polygon)
                }
            )
        }
        let rules = verifyPartition(
            dets: request.dets, pieces: named, region: request.polygon,
            imgW: request.width, imgH: request.height
        )
        let failures = rules.filter { $0.hard && !$0.ok }
        XCTAssertTrue(failures.isEmpty, "\(failures.map { "\($0.rule): \($0.detail)" })", file: file, line: line)
    }

    // MARK: - targetDim

    func testTargetDimIsTheResolutionThreshold() throws {
        // A 5712px photo letterboxed to 1024 shrinks to 0.179; a child piece
        // may then be at most ~193 crop px to letterbox at >= 0.95.
        let r = try request(dets: [], width: 768, height: 1024, scale: 1024.0 / 5712.0)
        XCTAssertEqual(r.targetDim, 1024 * (1024.0 / 5712.0) / 0.95, accuracy: 1e-9)
        XCTAssertEqual(r.targetDim, 193.2, accuracy: 0.1)

        // An already-native piece needs no cut at all.
        let native = try request(dets: [], width: 400, height: 400, scale: 1.0)
        XCTAssertEqual(native.targetDim, 1024 / 0.95, accuracy: 1e-9)
        XCTAssertGreaterThan(native.targetDim, 400)
    }

    // MARK: - v2 free-form cutter

    func testFreeFormCutsToTargetInOneCut() throws {
        // 600x600 crop that must reach 150px pieces: 4x4 = 16 leaves, and
        // recursion inside the cutter means the engine never has to re-detect
        // an intermediate level (R-6 / A-9).
        let dets = spineRow(count: 4, spacing: 150, y: 300)
        let r = try request(dets: dets, width: 600, height: 600, scale: 150.0 * 0.95 / 1024.0)
        XCTAssertEqual(r.targetDim, 150, accuracy: 1e-6)

        let pieces = try XCTUnwrap(FreeFormCutter().cut(r))
        XCTAssertGreaterThan(pieces.count, 4, "one cut should reach the target, not just halve")
        for piece in pieces {
            XCTAssertLessThanOrEqual(
                letterboxSourceDim(piece, rotate: false), r.targetDim + 1,
                "every piece must letterbox at or above the threshold"
            )
        }
        assertPartition(pieces, of: r)
    }

    func testFreeFormReturnsNilWhenPieceAlreadyFits() throws {
        let r = try request(dets: spineRow(count: 3, spacing: 100, y: 200), width: 300, height: 400, scale: 1.0)
        XCTAssertNil(FreeFormCutter().cut(r), "a piece at native resolution is a leaf, not a cut")
    }

    func testFreeFormSeamAvoidsCuttingDetections() throws {
        // Spines wide enough that a naive halfway cut at x=200 would slice the
        // middle one; the seam has to move into a gap instead.
        let dets = [det(60, 200, w: 80, h: 300), det(200, 200, w: 80, h: 300), det(340, 200, w: 80, h: 300)]
        let r = try request(dets: dets, width: 400, height: 400, scale: 190.0 * 0.95 / 1024.0)
        let pieces = try XCTUnwrap(FreeFormCutter().cut(r))
        XCTAssertGreaterThan(pieces.count, 1)
        assertPartition(pieces, of: r)
    }

    /// A-4: interlocked detections leave no legal seam, and the cutter must
    /// say so rather than emit an illegal partition.
    func testFreeFormGivesUpWhenNoLegalSeamExists() throws {
        // One OBB spanning the full width at every candidate row: any vertical
        // seam cuts it, and any horizontal seam cuts the tall one.
        let dets = [
            det(200, 200, w: 396, h: 396),
        ]
        let r = try request(dets: dets, width: 400, height: 400, scale: 100.0 * 0.95 / 1024.0)
        XCTAssertNil(FreeFormCutter().cut(r), "no seam can avoid a piece-filling OBB (A-4)")
    }

    func testFreeFormRespectsPieceBudget() throws {
        let dets = spineRow(count: 8, spacing: 100, y: 400)
        let r = try request(dets: dets, width: 800, height: 800, scale: 60.0 * 0.95 / 1024.0)
        let options = FreeFormCutterOptions(maxPieces: 4)
        let pieces = try XCTUnwrap(FreeFormCutter(options: options).cut(r))
        XCTAssertLessThanOrEqual(pieces.count, 4, "A-9: fragmentation is a cost, capped by maxPieces")
        assertPartition(pieces, of: r)
    }

    /// A non-rectangular parent's children must stay inside it, or sibling
    /// pieces from the parent's own cut would overlap in scene space.
    func testFreeFormCutOfAnLShapedPieceStaysInside() throws {
        let lShape: [CGPoint] = [
            CGPoint(x: 0, y: 0), CGPoint(x: 200, y: 0), CGPoint(x: 200, y: 200),
            CGPoint(x: 400, y: 200), CGPoint(x: 400, y: 400), CGPoint(x: 0, y: 400),
        ]
        let dets = [det(60, 100), det(60, 300), det(300, 300)]
        let r = try request(
            dets: dets, width: 400, height: 400, polygon: lShape,
            scale: 150.0 * 0.95 / 1024.0
        )
        let pieces = try XCTUnwrap(FreeFormCutter().cut(r))
        XCTAssertGreaterThan(pieces.count, 1)
        let areaSum = pieces.reduce(0.0) { $0 + polygonArea($1) }
        XCTAssertEqual(areaSum, polygonArea(lShape), accuracy: 0.02 * polygonArea(lShape))
        for piece in pieces {
            XCTAssertFalse(pointInPolygon(300, 100, piece), "no child may enter the L's missing quarter")
        }
        assertPartition(pieces, of: r)
    }

    // MARK: - v1 adapter parity

    func testPlanCropsCutterMatchesPlanCropsDirectly() throws {
        let dets = spineRow(count: 4, spacing: 100, y: 200)
        let r = try request(dets: dets, width: 400, height: 400, scale: 1.0, imgsz: 200)
        let viaCutter = PlanCropsCutter(maxCropDimK: 1.0).cut(r)
        let direct = planCrops(
            dets: dets, imgW: 400, imgH: 400, raster: r.raster,
            imgsz: 200, maxCropDimK: 1.0
        )
        XCTAssertEqual(viaCutter?.count, direct.count)
        for (a, b) in zip(viaCutter ?? [], direct.map(\.quad)) {
            XCTAssertEqual(a.count, b.count)
            for (p, q) in zip(a, b) { XCTAssertPointsEqual(p, q, accuracy: 1e-9) }
        }
    }

    // MARK: - Blocked mask

    func testBlockedMaskCoversDetectionsAndGaps() {
        let obb = det(100, 100, w: 40, h: 40)
        let mask = BlockedMask(obstacles: [obb.corners], width: 200, height: 200, cell: 2)
        XCTAssertTrue(mask.blocked(100, 100))
        XCTAssertFalse(mask.blocked(160, 100))
        XCTAssertTrue(mask.segmentBlocked(CGPoint(x: 100, y: 0), CGPoint(x: 100, y: 200)))
        XCTAssertFalse(mask.segmentBlocked(CGPoint(x: 160, y: 0), CGPoint(x: 160, y: 200)))
    }

    func testBlockedMaskCatchesThinDetections() {
        // Thinner than one cell: the outline walk, not the cell centers, is
        // what keeps a seam from slipping through a thin spine.
        let thin = det(100, 100, w: 1, h: 80)
        let mask = BlockedMask(obstacles: [thin.corners], width: 200, height: 200, cell: 8)
        XCTAssertTrue(mask.segmentBlocked(CGPoint(x: 100, y: 60), CGPoint(x: 100, y: 140)))
    }
}
