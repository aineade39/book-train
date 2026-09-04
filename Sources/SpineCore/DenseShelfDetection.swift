import CoreGraphics
import Foundation

// Dense-shelf hardening path per docs/BOOK_ID_IOS_PIPELINE.md §Detection &
// isolation / §Delivery sequencing step 2: "For dense shelves only, support
// tiled first-pass detection and the layout-crops jigsaw planner (plan ->
// verify rules -> optional per-crop re-inference -> global rotated NMS)."
//
// This factors the CLI shape already proven out in
// `Sources/layout-crops/main.swift` (single-shot -> planCrops -> verifyPlan
// -> per-crop re-infer -> nmsRotated merge) into a reusable function so the
// app pipeline can opt into the same jigsaw re-inference for shelves dense
// enough that a single-shot pass is likely to miss or merge neighboring
// spines, without duplicating that logic or depending on the CLI target.
//
// `predict` is a plain closure rather than a concrete `SpineDetector` so
// this is unit-testable on Mac with a fake predictor (no Core ML model
// required) while the real pipeline / CLI can pass `detector.predict`.

public struct DenseShelfDetectionOptions {
    /// Single-shot detection count at/above which the jigsaw re-inference
    /// pass engages. Below this, `denseShelfDetect` is just a single-shot
    /// predict + NMS (the v1 "ship" path from the spec).
    public var denseThreshold: Int
    public var imgsz: Int
    public var angleTolDeg: Double
    public var rowGapK: Double
    public var colGapK: Double
    public var minBlockMembers: Int
    public var maxCropDimK: Double
    public var padValue: UInt8

    public init(
        denseThreshold: Int = 40,
        imgsz: Int = 1024,
        angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg,
        rowGapK: Double = 0.2,
        colGapK: Double = 1.0,
        minBlockMembers: Int = 2,
        maxCropDimK: Double = 1.5,
        padValue: UInt8 = 114
    ) {
        self.denseThreshold = denseThreshold
        self.imgsz = imgsz
        self.angleTolDeg = angleTolDeg
        self.rowGapK = rowGapK
        self.colGapK = colGapK
        self.minBlockMembers = minBlockMembers
        self.maxCropDimK = maxCropDimK
        self.padValue = padValue
    }

    public static let `default` = DenseShelfDetectionOptions()
}

public struct DenseShelfDetectionResult {
    /// Final detections: either the plain single-shot result, or the
    /// single-shot + per-crop re-inference merged by rotated NMS.
    public let detections: [OBBDetection]
    /// Whether the jigsaw re-inference pass actually ran (it is skipped
    /// below `denseThreshold`, when the scene raster can't be built, or
    /// when `verifyPlan`'s hard rules fail on the planned crops).
    public let usedJigsaw: Bool
    public let firstPassCount: Int
    public let plannedCropCount: Int
    /// Detections present in `detections` that were not in the first pass
    /// (i.e. only found by a crop re-inference) — the spines single-shot
    /// detection would otherwise have missed.
    public let newDetectionCount: Int
    /// Scene-space crop quads (TL, TR, BR, BL) from `planCrops`. Empty when
    /// the planner did not run. Present even if hard rules failed and the
    /// jigsaw re-inference pass was skipped.
    public let cropQuads: [[CGPoint]]

    public init(
        detections: [OBBDetection],
        usedJigsaw: Bool,
        firstPassCount: Int,
        plannedCropCount: Int,
        newDetectionCount: Int,
        cropQuads: [[CGPoint]] = []
    ) {
        self.detections = detections
        self.usedJigsaw = usedJigsaw
        self.firstPassCount = firstPassCount
        self.plannedCropCount = plannedCropCount
        self.newDetectionCount = newDetectionCount
        self.cropQuads = cropQuads
    }
}

/// Axis-aligned bounds of a crop-plan quad, for progress reporting.
private func boundingRect(of quad: [CGPoint]) -> CGRect {
    guard let firstPoint = quad.first else { return .zero }
    var minX = firstPoint.x, maxX = firstPoint.x
    var minY = firstPoint.y, maxY = firstPoint.y
    for p in quad.dropFirst() {
        minX = min(minX, p.x); maxX = max(maxX, p.x)
        minY = min(minY, p.y); maxY = max(maxY, p.y)
    }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
}

/// Incremental reporting from `denseShelfDetect` so a caller's UI can show
/// detect work as it happens (FUNC §5.2 "Finding books"). Additive: the
/// closure defaults to nil, so CLI callers are unchanged.
public enum DenseShelfDetectProgress {
    /// The full-frame pass finished: its NMS'd detections, plus the
    /// bounding rects (scene pixels) of any planned jigsaw areas about to
    /// be re-inferred — empty when the jigsaw pass is skipped, meaning
    /// detect is done.
    case firstPass(detections: [OBBDetection], plannedAreaRects: [CGRect])
    /// One planned jigsaw area finished re-inference (index into
    /// `plannedAreaRects`). Fires even when the area's warp/predict
    /// failed, so completed counts always reach the total.
    case areaCompleted(index: Int)
}

/// Single-shot detect -> (if dense) plan/verify/re-infer/merge. See
/// `Sources/layout-crops/main.swift` for the CLI this mirrors and
/// `docs/BOOK_ID_IOS_PIPELINE.md` for the policy this implements.
public func denseShelfDetect(
    image: CGImage,
    predict: (CGImage) throws -> InferenceResult,
    options: DetectionOptions,
    denseOptions: DenseShelfDetectionOptions = .default,
    progress: ((DenseShelfDetectProgress) -> Void)? = nil
) throws -> DenseShelfDetectionResult {
    let imgW = image.width
    let imgH = image.height

    let firstResult = try predict(image)
    var first = firstResult.alreadyNMSed
        ? Array(firstResult.detections.prefix(options.maxDetections))
        : nmsRotated(firstResult.detections, iouThreshold: options.iouThreshold, maxDetections: options.maxDetections)
    first = first.filter { $0.cx >= 0 && $0.cx <= Double(imgW) && $0.cy >= 0 && $0.cy <= Double(imgH) }

    guard first.count >= denseOptions.denseThreshold, let raster = SceneRaster(cgImage: image) else {
        progress?(.firstPass(detections: first, plannedAreaRects: []))
        return DenseShelfDetectionResult(
            detections: first, usedJigsaw: false, firstPassCount: first.count,
            plannedCropCount: 0, newDetectionCount: 0, cropQuads: []
        )
    }

    let plans = planCrops(
        dets: first, imgW: imgW, imgH: imgH, raster: raster,
        angleTolDeg: denseOptions.angleTolDeg, rowGapK: denseOptions.rowGapK, colGapK: denseOptions.colGapK,
        minBlockMembers: denseOptions.minBlockMembers, imgsz: denseOptions.imgsz, maxCropDimK: denseOptions.maxCropDimK
    )
    let ruleResults = verifyPlan(dets: first, plans: plans, imgW: imgW, imgH: imgH, angleTolDeg: denseOptions.angleTolDeg)
    guard hardRulesOK(ruleResults) else {
        progress?(.firstPass(detections: first, plannedAreaRects: []))
        return DenseShelfDetectionResult(
            detections: first, usedJigsaw: false, firstPassCount: first.count,
            plannedCropCount: plans.count, newDetectionCount: 0, cropQuads: plans.map(\.quad)
        )
    }

    progress?(.firstPass(detections: first, plannedAreaRects: plans.map { boundingRect(of: $0.quad) }))

    var cropDets: [OBBDetection] = []
    for (planIndex, plan) in plans.enumerated() {
        defer { progress?(.areaCompleted(index: planIndex)) }
        guard let warped = warpQuad(raster, quad: plan.quad, maxSide: denseOptions.imgsz, padValue: denseOptions.padValue),
              let padded = padNoUpsize(warped.image, canvas: denseOptions.imgsz, padValue: denseOptions.padValue),
              let raw = try? predict(padded.image) else { continue }
        for d in raw.detections {
            if let mapped = mapDetFromCrop(d, homography: warped.homography, gain: padded.gain, padX: padded.padX, padY: padded.padY),
               mapped.w > 1, mapped.h > 1 {
                cropDets.append(mapped)
            }
        }
    }

    let merged = nmsRotated(first + cropDets, iouThreshold: options.iouThreshold, maxDetections: options.maxDetections)
    let firstIds = Set(first.map(\.id))
    let newCount = merged.filter { !firstIds.contains($0.id) }.count
    return DenseShelfDetectionResult(
        detections: merged, usedJigsaw: true, firstPassCount: first.count,
        plannedCropCount: plans.count, newDetectionCount: newCount,
        cropQuads: plans.map(\.quad)
    )
}
