import CoreGraphics
import XCTest

@testable import SpineCore

final class JigsawZoomTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, w: Double = 40, h: Double = 80, angleDeg: Double = 0, conf: Float = 0.9) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: conf)
    }

    private func inference(_ dets: [OBBDetection]) -> InferenceResult {
        InferenceResult(detections: dets, layout: .legacyChannelsFirst, alreadyNMSed: true, inferenceMs: 0)
    }

    private let options = DetectionOptions(confidenceThreshold: 0.15, iouThreshold: 0.45, maxDetections: 500)

    /// Projects a scene-space box onto a letterbox canvas the same way the
    /// engine's inverse (`mapDetFromCrop`) expects: corners through the
    /// placement homography (which carries the downscale), then pad offset.
    private func canvasDet(_ scene: OBBDetection, _ lb: ZoomLetterbox) -> OBBDetection {
        let pts = scene.corners.map { lb.homography.apply($0) }
        let rect = minAreaRect(pts)!
        var w = rect.width, h = rect.height, angle = rect.angleRad
        if w < h {
            swap(&w, &h)
            angle += .pi / 2
        }
        return OBBDetection(
            cx: rect.cx * lb.gain + lb.padX,
            cy: rect.cy * lb.gain + lb.padY,
            w: w * lb.gain, h: h * lb.gain,
            angle: angle, conf: scene.conf
        )
    }

    // MARK: - Letterbox mask

    func testLetterboxMasksPixelsOutsidePolygon() throws {
        let image = makeCheckerboardCGImage(width: 80, height: 40, cols: 8, rows: 4)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        // Triangle whose AABB is the full image, so the crop still contains
        // pixels that the mask must paint pad-gray.
        let poly = [
            CGPoint(x: 0, y: 0), CGPoint(x: 80, y: 0), CGPoint(x: 0, y: 40),
        ]
        let lb = try XCTUnwrap(letterboxPiece(
            raster: raster, polygon: poly, imgsz: 80, padValue: 114, rotate: false
        ))
        let out = try XCTUnwrap(SceneRaster(cgImage: lb.image))
        XCTAssertEqual(lb.gain, 1.0, accuracy: 1e-9)
        let inside = out.rgb(x: Int(lb.padX.rounded()) + 8, y: Int(lb.padY.rounded()) + 4)
        let outside = out.rgb(x: Int(lb.padX.rounded()) + 70, y: Int(lb.padY.rounded()) + 34)
        XCTAssertNotEqual(inside.0, 114, "interior should keep scene pixels")
        XCTAssertEqual(outside.0, 114)
        XCTAssertEqual(outside.1, 114)
        XCTAssertEqual(outside.2, 114)
    }

    func testMapDetFromLetterboxRoundTripsSceneCenter() throws {
        let image = makeSolidCGImage(width: 200, height: 120, gray: 80)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let poly = [
            CGPoint(x: 0, y: 0), CGPoint(x: 200, y: 0),
            CGPoint(x: 200, y: 120), CGPoint(x: 0, y: 120),
        ]
        let lb = try XCTUnwrap(letterboxPiece(
            raster: raster, polygon: poly, imgsz: 200, padValue: 114, rotate: false
        ))
        // A box at scene (100, 60) is at crop (100, 60); canvas adds pad.
        let canvas = det(100 + lb.padX, 60 + lb.padY, w: 20, h: 40)
        let scene = try XCTUnwrap(mapDetFromCrop(
            canvas, homography: lb.homography, gain: lb.gain, padX: lb.padX, padY: lb.padY
        ))
        XCTAssertEqual(scene.cx, 100, accuracy: 1.5)
        XCTAssertEqual(scene.cy, 60, accuracy: 1.5)
        // mapDetFromCrop refits a min-area rect and may swap w/h so the
        // long side is `w` (same as its other callers).
        let sides = [scene.w, scene.h].sorted()
        XCTAssertEqual(sides[0], 20, accuracy: 1.5)
        XCTAssertEqual(sides[1], 40, accuracy: 1.5)
    }

    func testNeverUpsamplesSmallPiece() throws {
        let image = makeSolidCGImage(width: 400, height: 400)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let poly = [
            CGPoint(x: 10, y: 10), CGPoint(x: 50, y: 10),
            CGPoint(x: 50, y: 50), CGPoint(x: 10, y: 50),
        ]
        let lb = try XCTUnwrap(letterboxPiece(
            raster: raster, polygon: poly, imgsz: 1024, padValue: 114, rotate: false
        ))
        XCTAssertEqual(lb.scale, 1.0, accuracy: 1e-9)
        XCTAssertEqual(lb.gain, 1.0, accuracy: 1e-9)
        XCTAssertLessThan(lb.cropWidth, 1024)
    }

    // MARK: - Engine

    func testFitsNativeResolutionIsSingleLeafPass() throws {
        let image = makeSolidCGImage(width: 200, height: 160)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let lb = try XCTUnwrap(letterboxPiece(
            raster: raster,
            polygon: ZoomPiece.fullImage(width: 200, height: 160).polygon,
            imgsz: 1024, rotate: false
        ))
        var calls = 0
        let result = try jigsawZoomDetect(
            image: image,
            predict: { _ in
                calls += 1
                return self.inference([self.canvasDet(self.det(40, 80), lb), self.canvasDet(self.det(120, 80), lb)])
            },
            options: options,
            zoomOptions: JigsawZoomOptions(imgsz: 1024, rotatePieces: false)
        )
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(result.inferencePasses, 1)
        XCTAssertEqual(result.leafCount, 1)
        XCTAssertEqual(result.fallbackLeafCount, 0)
        XCTAssertFalse(result.usedRecursion)
        XCTAssertEqual(result.detections.count, 2)
    }

    func testUncuttableOversizedPieceFallsBackToLeaf() throws {
        // No detections → planCrops returns one full-image piece, which is
        // not a partition, so A-4 accepts the downsampled leaf.
        let image = makeSolidCGImage(width: 400, height: 400)
        var calls = 0
        let result = try jigsawZoomDetect(
            image: image,
            predict: { _ in
                calls += 1
                return self.inference([])
            },
            options: options,
            zoomOptions: JigsawZoomOptions(
                downsampleThreshold: 0.95, imgsz: 128, rotatePieces: false, maxDepth: 3
            )
        )
        XCTAssertEqual(calls, 1, "A-4 must not recurse when the cut is the same piece")
        XCTAssertEqual(result.fallbackLeafCount, 1)
        XCTAssertTrue(result.detections.isEmpty)
    }

    func testRecursesWhenPlanCropsSplitsADenseShelf() throws {
        let imgW = 400, imgH = 240
        let image = makeSolidCGImage(width: imgW, height: imgH)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let rootLB = try XCTUnwrap(letterboxPiece(
            raster: raster,
            polygon: ZoomPiece.fullImage(width: imgW, height: imgH).polygon,
            imgsz: 128, rotate: false
        ))
        // Four side-by-side spines in scene pixels, expressed in the root
        // canvas so the first predict maps back to those scene boxes.
        let sceneSpines = [
            det(50, 120, w: 40, h: 180),
            det(150, 120, w: 40, h: 180),
            det(250, 120, w: 40, h: 180),
            det(350, 120, w: 40, h: 180),
        ]
        let firstPass = sceneSpines.map { canvasDet($0, rootLB) }
        var calls = 0
        let result = try jigsawZoomDetect(
            image: image,
            predict: { img in
                calls += 1
                if calls == 1 {
                    return self.inference(firstPass)
                }
                return self.inference([self.det(Double(img.width) / 2, Double(img.height) / 2, w: 20, h: 40)])
            },
            options: options,
            zoomOptions: JigsawZoomOptions(
                downsampleThreshold: 0.95, imgsz: 128, rotatePieces: false, maxDepth: 4
            )
        )
        XCTAssertGreaterThan(calls, 1, "dense shelf larger than imgsz should cut")
        XCTAssertTrue(result.usedRecursion || result.plannedCropCount > 1 || result.leafCount > 1)
        XCTAssertGreaterThanOrEqual(result.detections.count, 1)
    }

    // MARK: - Leaf telemetry (A-5 / A-7 / A-8)

    func testLeafTelemetryMatchesCountsAndPassesRunInvariants() throws {
        let image = makeSolidCGImage(width: 200, height: 160)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let lb = try XCTUnwrap(letterboxPiece(
            raster: raster,
            polygon: ZoomPiece.fullImage(width: 200, height: 160).polygon,
            imgsz: 1024, rotate: false
        ))
        let result = try jigsawZoomDetect(
            image: image,
            predict: { _ in self.inference([self.canvasDet(self.det(40, 80), lb)]) },
            options: options,
            zoomOptions: JigsawZoomOptions(imgsz: 1024, rotatePieces: false)
        )
        XCTAssertEqual(result.leaves.count, result.leafCount)
        XCTAssertEqual(result.leaves.filter(\.isFallback).count, result.fallbackLeafCount)
        XCTAssertEqual(result.leaves.first?.depth, 0)
        XCTAssertEqual(result.leaves.first?.scale ?? 0, 1.0, accuracy: 1e-9)
        XCTAssertTrue(hardRulesOK(verifyZoomRun(result, downsampleThreshold: 0.95)))
    }

    func testFallbackLeafIsFlaggedAsFallbackNotAsThresholdReached() throws {
        // No detections -> no legal cut -> A-4 accepts a heavily downsampled
        // leaf, which the run invariants must tolerate but still flag.
        let image = makeSolidCGImage(width: 400, height: 400)
        let result = try jigsawZoomDetect(
            image: image,
            predict: { _ in self.inference([]) },
            options: options,
            zoomOptions: JigsawZoomOptions(downsampleThreshold: 0.95, imgsz: 128, rotatePieces: false, maxDepth: 3)
        )
        XCTAssertEqual(result.leaves.count, 1)
        XCTAssertTrue(result.leaves[0].isFallback)
        XCTAssertLessThan(result.leaves[0].scale, 0.95)
        let rules = verifyZoomRun(result, downsampleThreshold: 0.95)
        XCTAssertTrue(hardRulesOK(rules))
        XCTAssertEqual(rules.first { $0.rule == "R7_LEAF_SCALE_OK" }?.detail, "ok (fallback_leaves=1)")
    }

    // MARK: - Cutter selection (Stage 3)

    func testFreeFormCutterRecursesToNativeResolutionLeaves() throws {
        let imgW = 512, imgH = 512
        let image = makeSolidCGImage(width: imgW, height: imgH)
        let raster = try XCTUnwrap(SceneRaster(cgImage: image))
        let rootLB = try XCTUnwrap(letterboxPiece(
            raster: raster,
            polygon: ZoomPiece.fullImage(width: imgW, height: imgH).polygon,
            imgsz: 128, rotate: false
        ))
        // Two shelf rows of four spines, with gaps between the spines and
        // between the rows, so seams have somewhere to go on both axes.
        let sceneSpines = [Double(130), Double(380)].flatMap { rowY in
            [60.0, 190.0, 320.0, 450.0].map { det($0, rowY, w: 40, h: 180) }
        }
        let firstPass = sceneSpines.map { canvasDet($0, rootLB) }
        var calls = 0
        let result = try jigsawZoomDetect(
            image: image,
            predict: { img in
                calls += 1
                if calls == 1 { return self.inference(firstPass) }
                return self.inference([self.det(Double(img.width) / 2, Double(img.height) / 2, w: 20, h: 40)])
            },
            options: options,
            zoomOptions: JigsawZoomOptions(
                downsampleThreshold: 0.95, imgsz: 128, rotatePieces: false,
                maxDepth: 6, cutter: .v2
            )
        )
        // R-6: one cut descends several levels at once, so the root cut alone
        // yields a whole grid rather than the two halves a binary cutter
        // would give, and the recursion stays shallow. Here it cannot reach
        // native resolution in a single cut: the row bands themselves have to
        // be cut apart, which needs the finer detections of the next level.
        XCTAssertGreaterThanOrEqual(result.plannedCropCount, 4, "one cut should yield a grid, not a bisection")
        XCTAssertTrue(result.usedRecursion)
        XCTAssertGreaterThan(result.leafCount, 1)
        XCTAssertLessThanOrEqual(result.leaves.map(\.depth).max() ?? 0, 2)
        XCTAssertEqual(result.fallbackLeafCount, 0, "every leaf should reach the resolution threshold")
        XCTAssertTrue(hardRulesOK(verifyZoomRun(result, downsampleThreshold: 0.95)))
    }

    /// With nothing detected, v1 has no bands or blocks to plan from and
    /// surrenders the photo to A-4. v2 is resolution-driven — detections only
    /// place its seams — so it still zooms in, which is the whole point when
    /// the downsampled pass is what missed the spines.
    func testEmptyFirstPassStopsV1ButNotV2() throws {
        let image = makeSolidCGImage(width: 300, height: 300)
        func run(_ cutter: ZoomCutterKind) throws -> JigsawZoomResult {
            try jigsawZoomDetect(
                image: image,
                predict: { _ in self.inference([]) },
                options: options,
                zoomOptions: JigsawZoomOptions(imgsz: 128, rotatePieces: false, maxDepth: 3, cutter: cutter)
            )
        }
        let v1 = try run(.v1)
        XCTAssertEqual(v1.inferencePasses, 1)
        XCTAssertEqual(v1.fallbackLeafCount, 1)

        let v2 = try run(.v2)
        XCTAssertGreaterThan(v2.inferencePasses, 1)
        XCTAssertEqual(v2.fallbackLeafCount, 0, "an empty region has no OBB to avoid, so every seam is legal")
        XCTAssertTrue(hardRulesOK(verifyZoomRun(v2, downsampleThreshold: 0.95)))
    }
}
