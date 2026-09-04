import CoreGraphics
import Foundation

// Stages 2-3 of the jigsaw-zoom plan (~/dev/book-id-design/docs/
// jigsaw-zoom-requirements.md R-1..R-7): recurse until a piece fits the
// model (R-1b), cut with the cutter named by `JigsawZoomOptions.cutter`
// (v1 = the `planCrops` quad adapter, v2 = the free-form staircase cutter
// in `PieceCutter.swift`), keep leaf detections only (R-5), fall back to
// the current downsampled detect when no legal cut exists (A-4).

public struct JigsawZoomOptions {
    /// Accept a piece as a leaf when its letterbox scale is at or above
    /// this (spec Q-2 default 0.95).
    public var downsampleThreshold: Double
    public var imgsz: Int
    public var padValue: UInt8
    public var rotatePieces: Bool
    public var maxDepth: Int
    public var angleTolDeg: Double
    public var rowGapK: Double
    public var colGapK: Double
    public var minBlockMembers: Int
    public var maxCropDimK: Double
    /// Which cutter partitions an oversized piece. Defaults to the frozen
    /// incumbent adapter so Stage 2's measured behavior is unchanged until a
    /// caller opts into v2.
    public var cutter: ZoomCutterKind
    public var freeForm: FreeFormCutterOptions

    public init(
        downsampleThreshold: Double = 0.95,
        imgsz: Int = 1024,
        padValue: UInt8 = 114,
        rotatePieces: Bool = false,
        maxDepth: Int = 8,
        angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg,
        rowGapK: Double = 0.2,
        colGapK: Double = 1.0,
        minBlockMembers: Int = 2,
        maxCropDimK: Double = 1.5,
        cutter: ZoomCutterKind = .v1,
        freeForm: FreeFormCutterOptions = .default
    ) {
        self.downsampleThreshold = downsampleThreshold
        self.imgsz = imgsz
        self.padValue = padValue
        self.rotatePieces = rotatePieces
        self.maxDepth = maxDepth
        self.angleTolDeg = angleTolDeg
        self.rowGapK = rowGapK
        self.colGapK = colGapK
        self.minBlockMembers = minBlockMembers
        self.maxCropDimK = maxCropDimK
        self.cutter = cutter
        self.freeForm = freeForm
    }

    public static let `default` = JigsawZoomOptions()

    /// The concrete cutter for `cutter`.
    func makeCutter() -> ZoomCutter {
        switch cutter {
        case .v1:
            return PlanCropsCutter(
                angleTolDeg: angleTolDeg, rowGapK: rowGapK, colGapK: colGapK,
                minBlockMembers: minBlockMembers, maxCropDimK: maxCropDimK
            )
        case .v2:
            return FreeFormCutter(options: freeForm)
        }
    }
}

/// One accepted leaf, for the A-5 / A-7 / A-8 telemetry: depth histogram,
/// "every leaf's letterbox scale is within the threshold", and how often the
/// A-4 fallback had to accept an under-scaled piece.
public struct ZoomLeafInfo {
    public let depth: Int
    /// Letterbox shrink this leaf was detected at (1.0 = native pixels).
    public let scale: Double
    /// True when the piece was accepted as a leaf because no legal cut
    /// existed (A-4), not because it reached the resolution threshold.
    public let isFallback: Bool
    public let detectionCount: Int

    public init(depth: Int, scale: Double, isFallback: Bool, detectionCount: Int) {
        self.depth = depth
        self.scale = scale
        self.isFallback = isFallback
        self.detectionCount = detectionCount
    }
}

public struct JigsawZoomResult {
    public let detections: [OBBDetection]
    public let firstPassCount: Int
    public let plannedCropCount: Int
    public let inferencePasses: Int
    public let leafCount: Int
    public let fallbackLeafCount: Int
    public let topLevelDropCount: Int
    public let dedupHitCount: Int
    public let cropQuads: [[CGPoint]]
    public let usedRecursion: Bool
    /// Per-leaf provenance (A-7 / A-8). `leaves.count == leafCount`.
    public let leaves: [ZoomLeafInfo]

    public init(
        detections: [OBBDetection],
        firstPassCount: Int,
        plannedCropCount: Int,
        inferencePasses: Int,
        leafCount: Int,
        fallbackLeafCount: Int,
        topLevelDropCount: Int,
        dedupHitCount: Int,
        cropQuads: [[CGPoint]],
        usedRecursion: Bool,
        leaves: [ZoomLeafInfo] = []
    ) {
        self.detections = detections
        self.firstPassCount = firstPassCount
        self.plannedCropCount = plannedCropCount
        self.inferencePasses = inferencePasses
        self.leafCount = leafCount
        self.fallbackLeafCount = fallbackLeafCount
        self.topLevelDropCount = topLevelDropCount
        self.dedupHitCount = dedupHitCount
        self.cropQuads = cropQuads
        self.usedRecursion = usedRecursion
        self.leaves = leaves
    }
}

private struct VisitState {
    var passes = 0
    var leafCount = 0
    var fallbackLeafCount = 0
    var firstLevelQuads: [[CGPoint]] = []
    var firstPass: [OBBDetection] = []
    var leaves: [ZoomLeafInfo] = []

    mutating func recordLeaf(depth: Int, scale: Double, isFallback: Bool, detectionCount: Int) {
        leafCount += 1
        if isFallback { fallbackLeafCount += 1 }
        leaves.append(ZoomLeafInfo(depth: depth, scale: scale, isFallback: isFallback, detectionCount: detectionCount))
    }
}

/// Recursive zoom detect. `predict` is a closure so tests can fake Core ML.
public func jigsawZoomDetect(
    image: CGImage,
    predict: (CGImage) throws -> InferenceResult,
    options: DetectionOptions,
    zoomOptions: JigsawZoomOptions = .default
) throws -> JigsawZoomResult {
    let imgW = image.width, imgH = image.height
    guard let raster = SceneRaster(cgImage: image) else {
        return JigsawZoomResult(
            detections: [], firstPassCount: 0, plannedCropCount: 0,
            inferencePasses: 0, leafCount: 0, fallbackLeafCount: 0,
            topLevelDropCount: 0, dedupHitCount: 0, cropQuads: [], usedRecursion: false
        )
    }

    var state = VisitState()
    let root = ZoomPiece.fullImage(width: imgW, height: imgH)
    let leafDets = try visit(
        piece: root, raster: raster, cutter: zoomOptions.makeCutter(),
        predict: predict, options: options, zoom: zoomOptions, state: &state
    )

    // Same in-frame filter as `denseShelfDetect` applies to its first pass.
    let inFrame = leafDets.filter { $0.cx >= 0 && $0.cx <= Double(imgW) && $0.cy >= 0 && $0.cy <= Double(imgH) }
    let beforeNMS = inFrame.count
    let merged = nmsRotated(inFrame, iouThreshold: options.iouThreshold, maxDetections: options.maxDetections)
    let dedupHits = max(0, beforeNMS - merged.count)

    let drop: Int
    if state.firstPass.isEmpty {
        drop = 0
    } else {
        let match = matchDetections(state.firstPass, merged, iouThreshold: 0.5)
        drop = match.onlyA.count
    }

    return JigsawZoomResult(
        detections: merged,
        firstPassCount: state.firstPass.count,
        plannedCropCount: state.firstLevelQuads.count,
        inferencePasses: state.passes,
        leafCount: state.leafCount,
        fallbackLeafCount: state.fallbackLeafCount,
        topLevelDropCount: drop,
        dedupHitCount: dedupHits,
        cropQuads: state.firstLevelQuads,
        usedRecursion: state.firstLevelQuads.count > 1 || state.leafCount > 1,
        leaves: state.leaves
    )
}

private func preparedDets(_ raw: InferenceResult, options: DetectionOptions) -> [OBBDetection] {
    raw.alreadyNMSed
        ? Array(raw.detections.prefix(options.maxDetections))
        : nmsRotated(raw.detections, iouThreshold: options.iouThreshold, maxDetections: options.maxDetections)
}

private func visit(
    piece: ZoomPiece,
    raster: SceneRaster,
    cutter: ZoomCutter,
    predict: (CGImage) throws -> InferenceResult,
    options: DetectionOptions,
    zoom: JigsawZoomOptions,
    state: inout VisitState
) throws -> [OBBDetection] {
    guard let letterbox = letterboxPiece(
        raster: raster, polygon: piece.polygon, imgsz: zoom.imgsz,
        padValue: zoom.padValue, rotate: zoom.rotatePieces
    ) else { return [] }

    state.passes += 1
    let raw = try predict(letterbox.image)
    let mapped = preparedDets(raw, options: options).compactMap { det -> OBBDetection? in
        guard let scene = mapDetFromCrop(
            det, homography: letterbox.homography,
            gain: letterbox.gain, padX: letterbox.padX, padY: letterbox.padY
        ), scene.w > 1, scene.h > 1 else { return nil }
        return scene
    }

    if piece.depth == 0 { state.firstPass = mapped }

    let hitResolution = letterbox.scale + 1e-9 >= zoom.downsampleThreshold
    if hitResolution || piece.depth >= zoom.maxDepth {
        state.recordLeaf(
            depth: piece.depth, scale: letterbox.scale,
            isFallback: !hitResolution, detectionCount: mapped.count
        )
        return mapped
    }

    let children = cutPiece(
        piece: piece, mapped: mapped, letterbox: letterbox,
        raster: raster, cutter: cutter, zoom: zoom
    )
    // A first cut often only shortens one axis (shelf bands still as
    // wide as the photo). Letterbox scale then stays flat until a later
    // split; treat a real area drop as progress.
    let parentArea = abs(polygonArea(piece.polygon))
    let progressing = children.contains { abs(polygonArea($0.polygon)) < parentArea * 0.85 }
    if children.count <= 1 || !progressing {
        state.recordLeaf(
            depth: piece.depth, scale: letterbox.scale,
            isFallback: true, detectionCount: mapped.count
        )
        return mapped
    }

    if piece.depth == 0 { state.firstLevelQuads = children.map(\.polygon) }

    var out: [OBBDetection] = []
    for child in children {
        out.append(contentsOf: try visit(
            piece: child, raster: raster, cutter: cutter,
            predict: predict, options: options, zoom: zoom, state: &state
        ))
    }
    return out
}

/// Runs the configured cutter in the piece's unpadded-crop coordinates and
/// maps the resulting pieces back to the scene. Both cutters plan against
/// real pixels (the *unmasked* crop, so edge energy sees the neighbors a
/// seam has to route between) even though inference sees the gray-filled
/// mask.
private func cutPiece(
    piece: ZoomPiece,
    mapped: [OBBDetection],
    letterbox: ZoomLetterbox,
    raster: SceneRaster,
    cutter: ZoomCutter,
    zoom: JigsawZoomOptions
) -> [ZoomPiece] {
    guard let cropImage = renderUnmaskedCrop(
        raster: raster, homography: letterbox.homography,
        width: letterbox.cropWidth, height: letterbox.cropHeight,
        padValue: zoom.padValue
    ), let cropRaster = SceneRaster(cgImage: cropImage),
        let cropEnergy = EdgeEnergy(raster: cropRaster) else { return [] }

    // Same in-frame filter as `denseShelfDetect`: an out-of-frame center
    // can never be owned by any crop, which fails R4 outright.
    let cropW = letterbox.cropWidth, cropH = letterbox.cropHeight
    let localDets = mapped
        .compactMap { sceneToCrop($0, homography: letterbox.homography) }
        .filter { $0.cx >= 0 && $0.cx <= Double(cropW) && $0.cy >= 0 && $0.cy <= Double(cropH) }

    let request = ZoomCutRequest(
        polygon: piece.polygon.map { letterbox.homography.apply($0) },
        dets: localDets,
        raster: cropRaster,
        energy: cropEnergy,
        width: cropW,
        height: cropH,
        imgsz: zoom.imgsz,
        scale: letterbox.scale,
        rotatePieces: zoom.rotatePieces,
        downsampleThreshold: zoom.downsampleThreshold,
        angleTolDeg: zoom.angleTolDeg,
        depth: piece.depth,
        debug: ProcessInfo.processInfo.environment["JIGSAW_ZOOM_DEBUG"] != nil
    )
    guard let cropPieces = cutter.cut(request) else { return [] }

    return cropPieces.compactMap { polygon -> ZoomPiece? in
        let scenePolygon = polygon.map { letterbox.homography.applyInverse($0) }
        let finite = scenePolygon.filter { $0.x.isFinite && $0.y.isFinite }
        guard finite.count >= 3 else { return nil }
        return ZoomPiece(polygon: scenePolygon, depth: piece.depth + 1)
    }
}

private func sceneToCrop(_ det: OBBDetection, homography: Homography) -> OBBDetection? {
    let cropCorners = det.corners.map { homography.apply($0) }
    guard let rect = minAreaRect(cropCorners) else { return nil }
    var rw = rect.width, rh = rect.height, angle = rect.angleRad
    guard rw >= 1, rh >= 1 else { return nil }
    if rw < rh {
        swap(&rw, &rh)
        angle += .pi / 2
    }
    return OBBDetection(cx: rect.cx, cy: rect.cy, w: rw, h: rh, angle: angle, conf: det.conf, id: det.id)
}
