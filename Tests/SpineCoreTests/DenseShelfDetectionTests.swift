import CoreGraphics
import XCTest

@testable import SpineCore

/// `denseShelfDetect` tests against a fake `predict` closure (no Core ML
/// model needed) -- covers the threshold gate, the rules-fail fallback, and
/// a full plan -> re-infer -> merge round trip with a hand-placed "new"
/// detection whose canvas-local coordinates are derived from the *same*
/// warp/pad math the function itself uses, so the assertion is a real
/// geometry round trip rather than a change-detector on internals.
final class DenseShelfDetectionTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, _ w: Double, _ h: Double, angleDeg: Double = 0, conf: Float = 0.9) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: conf)
    }

    private func inferenceResult(_ dets: [OBBDetection], alreadyNMSed: Bool = false) -> InferenceResult {
        InferenceResult(detections: dets, layout: .legacyChannelsFirst, alreadyNMSed: alreadyNMSed, inferenceMs: 0)
    }

    private let options = DetectionOptions(confidenceThreshold: 0.15, iouThreshold: 0.45, maxDetections: 500)

    // MARK: - Below threshold: single predict call, no jigsaw

    func testBelowDenseThresholdSkipsJigsawAndCallsPredictOnce() throws {
        let image = makeSolidCGImage(width: 300, height: 220)
        let dets = [det(60, 110, 40, 200), det(120, 110, 40, 200)]
        var callCount = 0
        let result = try denseShelfDetect(
            image: image,
            predict: { img in
                callCount += 1
                XCTAssertEqual(img.width, 300)
                return self.inferenceResult(dets)
            },
            options: options,
            denseOptions: DenseShelfDetectionOptions(denseThreshold: 10)
        )
        XCTAssertEqual(callCount, 1)
        XCTAssertFalse(result.usedJigsaw)
        XCTAssertEqual(result.firstPassCount, 2)
        XCTAssertEqual(result.detections.count, 2)
        XCTAssertEqual(result.newDetectionCount, 0)
    }

    // MARK: - Out-of-frame first-pass detections are dropped (matches CLI)

    func testFirstPassDropsDetectionsWithCenterOutsideImage() throws {
        let image = makeSolidCGImage(width: 300, height: 220)
        let dets = [det(60, 110, 40, 200), det(-10, 110, 40, 200)]
        let result = try denseShelfDetect(
            image: image, predict: { _ in self.inferenceResult(dets) },
            options: options, denseOptions: DenseShelfDetectionOptions(denseThreshold: 10)
        )
        XCTAssertEqual(result.firstPassCount, 1)
    }

    // MARK: - Above threshold, full plan -> re-infer -> merge round trip

    func testDenseThresholdEngagesJigsawAndMergesANewCropOnlyDetection() throws {
        let imgW = 300, imgH = 220
        let image = makeSolidCGImage(width: imgW, height: imgH)
        let raster = SceneRaster(cgImage: image)!
        // Same fixture as LayoutCropPlannerTests' "single shelf, single
        // block": four touching same-orientation spines forming one block,
        // with empty margin strips above/below (`planCrops` extends a band
        // to image edges when top/bottom margin > ~2px).
        let dets = [det(60, 110, 40, 200), det(120, 110, 40, 200), det(180, 110, 40, 200), det(240, 110, 40, 200)]
        let denseOptions = DenseShelfDetectionOptions(denseThreshold: 4)

        // Precompute the exact same plans `denseShelfDetect` will compute,
        // so we know the crop that owns our four members and can place a
        // fake "new" detection at a known scene point via that crop's own
        // warp + pad transform.
        let plans = planCrops(
            dets: dets, imgW: imgW, imgH: imgH, raster: raster,
            angleTolDeg: denseOptions.angleTolDeg, rowGapK: denseOptions.rowGapK, colGapK: denseOptions.colGapK,
            minBlockMembers: denseOptions.minBlockMembers, imgsz: denseOptions.imgsz, maxCropDimK: denseOptions.maxCropDimK
        )
        XCTAssertTrue(hardRulesOK(verifyPlan(dets: dets, plans: plans, imgW: imgW, imgH: imgH)))
        guard let memberPlanIndex = plans.firstIndex(where: { !$0.memberIndices.isEmpty }) else {
            return XCTFail("expected exactly one crop to own the four members")
        }
        let memberPlan = plans[memberPlanIndex]
        XCTAssertEqual(Set(memberPlan.memberIndices), Set(0..<dets.count))

        // A scene point clearly outside every existing spine's box (spines
        // span x in [40, 260], y in [10, 210]) but inside the member crop's
        // quad -- e.g. hugging the crop's own top-left corner.
        let newScenePoint = CGPoint(
            x: memberPlan.quad.map(\.x).min()! + 2,
            y: memberPlan.quad.map(\.y).min()! + 2
        )
        guard let warped = warpQuad(raster, quad: memberPlan.quad, maxSide: denseOptions.imgsz, padValue: denseOptions.padValue),
              let padded = padNoUpsize(warped.image, canvas: denseOptions.imgsz, padValue: denseOptions.padValue) else {
            return XCTFail("expected member crop to warp/pad successfully")
        }
        let warpedLocal = warped.homography.apply(newScenePoint)
        let canvasLocal = CGPoint(x: warpedLocal.x * padded.gain + padded.padX, y: warpedLocal.y * padded.gain + padded.padY)
        let fakeCropDet = det(Double(canvasLocal.x), Double(canvasLocal.y), 8, 8)

        var predictCallCount = 0
        let result = try denseShelfDetect(
            image: image,
            predict: { img in
                predictCallCount += 1
                if img.width == imgW, img.height == imgH {
                    return self.inferenceResult(dets) // first pass
                }
                // Every plan gets re-inferred; only the member-owning crop
                // reports a (fake) new detection, the rest find nothing.
                let isMemberCrop = predictCallCount == memberPlanIndex + 2 // +1 for first pass, +1 for 1-based plan order
                return self.inferenceResult(isMemberCrop ? [fakeCropDet] : [])
            },
            options: options,
            denseOptions: denseOptions
        )

        XCTAssertTrue(result.usedJigsaw)
        XCTAssertEqual(result.firstPassCount, 4)
        XCTAssertEqual(result.plannedCropCount, plans.count)
        XCTAssertEqual(predictCallCount, 1 + plans.count)
        XCTAssertEqual(result.newDetectionCount, 1, "the fabricated crop-only detection should survive merge as a distinct spine")
        XCTAssertEqual(result.detections.count, 5)

        guard let mapped = result.detections.first(where: { d in !dets.contains { $0.id == d.id } }) else {
            return XCTFail("expected exactly one net-new detection after merge")
        }
        XCTAssertEqual(mapped.cx, Double(newScenePoint.x), accuracy: 1.0)
        XCTAssertEqual(mapped.cy, Double(newScenePoint.y), accuracy: 1.0)
    }

    // MARK: - Dense mode with zero detections is a no-op beyond the first pass

    func testDenseThresholdOfZeroWithNoDetectionsStillReturnsFirstPassOnly() throws {
        // No bands -> `planCrops` returns one full-image empty plan, rules
        // trivially pass, and its own re-infer call finds nothing new.
        let image = makeSolidCGImage(width: 50, height: 50)
        var callCount = 0
        let result = try denseShelfDetect(
            image: image,
            predict: { _ in
                callCount += 1
                return self.inferenceResult([])
            },
            options: options,
            denseOptions: DenseShelfDetectionOptions(denseThreshold: 0)
        )
        XCTAssertEqual(result.firstPassCount, 0)
        XCTAssertEqual(result.detections.count, 0)
        XCTAssertEqual(result.newDetectionCount, 0)
    }
}
