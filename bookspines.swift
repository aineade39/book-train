import Foundation
import AppKit
import CoreML
import CoreImage
import Vision
import ImageIO
import UniformTypeIdentifiers

// Core ML OBB validation / inference harness for the spine detector.
//
// Runs a YOLO-OBB .mlpackage (exported with nms=False) on one image, with
// optional tiling for very dense shelves and optional Vision OCR on the
// final crops:
//
//   [tile grid] -> per tile: letterbox -> Core ML -> decode [1, 6, N] ->
//   confidence filter -> map tile-local dets to full-image pixels ->
//   single global rotated NMS -> (optional) perspective-crop + OCR
//
// Output: JSON on stdout + "<image>.spines.png" with oriented boxes drawn.
//
// Usage:
//   swift bookspines.swift <image> [--model path.mlpackage] [--conf 0.15]
//                          [--iou 0.45] [--max-det 500] [--tiles NxM] [--tile-overlap 0.25]
//                          [--ocr]
//
// Output layouts (auto-detected from tensor shape):
//   Legacy YOLO11  [1, 6, N]:   cx, cy, w, h, conf, angle   — needs rotated NMS
//   YOLO26 end2end [1, 300, 7]: cx, cy, w, h, conf, cls, angle — NMS already in-graph
// Coords are letterboxed model-input pixels; angle is radians (x-right / y-down).

// MARK: - CLI

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

var imagePath: String? = nil
// Default: promoted export under $BOOK_SPINES_DATA (or ~/ml/book-spines).
var modelPath: String = {
    let root = ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
        ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
    let production = "\(root)/models/production"
    // Promote script installs a symlink at SpineDetectorOBB.mlpackage. Until
    // then that name may still be a legacy real package — only follow it when
    // it is a symlink; otherwise keep the frozen aug baseline as default.
    let alias = "\(production)/SpineDetectorOBB.mlpackage"
    let aug = "\(production)/SpineDetectorOBB-aug.mlpackage"
    if let attrs = try? FileManager.default.attributesOfItem(atPath: alias),
       let type = attrs[.type] as? FileAttributeType,
       type == .typeSymbolicLink {
        return alias
    }
    return aug
}()
var confThreshold: Float = 0.15
var iouThreshold: Double = 0.45
var maxDetections = 500
var tilesArg: String? = nil          // nil = auto (1x1, or 2x2 for very large images)
var tileOverlap: CGFloat = 0.25
var enableOCR = false

var argIndex = 1
let args = CommandLine.arguments
while argIndex < args.count {
    let arg = args[argIndex]
    switch arg {
    case "--model":
        argIndex += 1; modelPath = args[argIndex]
    case "--conf":
        argIndex += 1; confThreshold = Float(args[argIndex]) ?? confThreshold
    case "--iou":
        argIndex += 1; iouThreshold = Double(args[argIndex]) ?? iouThreshold
    case "--max-det":
        argIndex += 1; maxDetections = Int(args[argIndex]) ?? maxDetections
    case "--tiles":
        argIndex += 1; tilesArg = args[argIndex]
    case "--tile-overlap":
        argIndex += 1; tileOverlap = CGFloat(Double(args[argIndex]) ?? Double(tileOverlap))
    case "--ocr":
        enableOCR = true
    default:
        if imagePath == nil { imagePath = arg } else {
            log("Unknown argument: \(arg)")
            exit(1)
        }
    }
    argIndex += 1
}

guard let imagePath else {
    log("""
        Usage: swift bookspines.swift <image> [--model path.mlpackage] [--conf 0.15] \
        [--iou 0.45] [--max-det 500] [--tiles NxM] [--tile-overlap 0.25] [--ocr]
        """)
    exit(1)
}

let imageURL = URL(fileURLWithPath: imagePath)
guard let nsImage = NSImage(contentsOf: imageURL),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    log("Error: could not load image at \(imageURL.path)")
    exit(1)
}
let imageW = CGFloat(cgImage.width)
let imageH = CGFloat(cgImage.height)

// MARK: - Model loading with a persistent compiled-model cache
//
// MLModel.compileModel(at:) recompiles into a fresh temp directory every call
// (~1-2s for this model). Cache the compiled .mlmodelc next to a
// modification-date check so repeat runs against the same .mlpackage skip
// recompilation entirely.

func cachedCompiledModelURL(for sourceURL: URL) throws -> URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("bookspines-model-cache", isDirectory: true)
    try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    let cachedName = sourceURL.lastPathComponent.replacingOccurrences(of: ".mlpackage", with: "") + ".mlmodelc"
    let cachedURL = cacheDir.appendingPathComponent(cachedName)

    let sourceModified = (try? sourceURL.resourceValues(forKeys: [.contentModificationDateKey]))?
        .contentModificationDate
    let cachedModified = (try? cachedURL.resourceValues(forKeys: [.contentModificationDateKey]))?
        .contentModificationDate

    if FileManager.default.fileExists(atPath: cachedURL.path),
       let sourceModified, let cachedModified, cachedModified >= sourceModified {
        return cachedURL
    }

    let compiled = try MLModel.compileModel(at: sourceURL)
    if FileManager.default.fileExists(atPath: cachedURL.path) {
        try FileManager.default.removeItem(at: cachedURL)
    }
    try FileManager.default.copyItem(at: compiled, to: cachedURL)
    return cachedURL
}

let modelURL = URL(fileURLWithPath: modelPath)
guard FileManager.default.fileExists(atPath: modelURL.path) else {
    log("Error: model not found at \(modelURL.path)")
    exit(1)
}

let model: MLModel
do {
    let compiledURL = try cachedCompiledModelURL(for: modelURL)
    let config = MLModelConfiguration()
    config.computeUnits = .all
    model = try MLModel(contentsOf: compiledURL, configuration: config)
} catch {
    log("Error: could not load model: \(error.localizedDescription)")
    exit(1)
}

guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
      let imageConstraint = model.modelDescription.inputDescriptionsByName[inputName]?.imageConstraint else {
    log("Error: model has no image input")
    exit(1)
}
let inputW = imageConstraint.pixelsWide
let inputH = imageConstraint.pixelsHigh
guard let outputName = model.modelDescription.outputDescriptionsByName.keys.first else {
    log("Error: model has no outputs")
    exit(1)
}

/// How the Core ML multiarray is laid out. Detected from shape (and refined on
/// the first inference if the constraint shape is symbolic).
enum OBBOutputLayout {
    case legacyChannelsFirst   // [1, 6, N]
    case end2endDetections     // [1, maxDet, 7]
}

func detectOutputLayout(shape: [NSNumber]) -> OBBOutputLayout? {
    guard shape.count == 3 else { return nil }
    let a = shape[1].intValue
    let b = shape[2].intValue
    if a == 6 { return .legacyChannelsFirst }
    if b == 7 { return .end2endDetections }
    if a == 7 { return .end2endDetections } // rare [1, 7, N] transpose
    return nil
}

final class OutputLayoutState {
    static var current: OBBOutputLayout = .end2endDetections
}

if let constraint = model.modelDescription.outputDescriptionsByName[outputName]?.multiArrayConstraint,
   let layout = detectOutputLayout(shape: constraint.shape) {
    OutputLayoutState.current = layout
} else if !modelPath.lowercased().contains("yolo26") {
    OutputLayoutState.current = .legacyChannelsFirst
}

log("Model input \(inputW)x\(inputH), output \"\(outputName)\", layout \(OutputLayoutState.current)")

// MARK: - Tile grid
//
// Detections use full-image, top-left-origin pixel coordinates throughout.

struct Tile {
    let rect: CGRect   // pixel space, top-left origin (matches CGImage.cropping)
}

func computeTiles(imageW: CGFloat, imageH: CGFloat, spec: String?, overlap: CGFloat) -> [Tile] {
    var cols = 1, rows = 1
    if let spec {
        let parts = spec.lowercased().split(separator: "x")
        if parts.count == 2, let c = Int(parts[0]), let r = Int(parts[1]), c > 0, r > 0 {
            cols = c; rows = r
        } else {
            log("Warning: could not parse --tiles \"\(spec)\", falling back to 1x1")
        }
    } else if max(imageW, imageH) > 3000 {
        cols = 2; rows = 2   // auto: dense full-bookcase shots benefit from tiling
    }

    guard cols > 1 || rows > 1 else {
        return [Tile(rect: CGRect(x: 0, y: 0, width: imageW, height: imageH))]
    }

    let tileW = imageW / (CGFloat(cols) - CGFloat(cols - 1) * overlap)
    let tileH = imageH / (CGFloat(rows) - CGFloat(rows - 1) * overlap)
    let stepX = tileW * (1 - overlap)
    let stepY = tileH * (1 - overlap)

    var tiles: [Tile] = []
    for row in 0..<rows {
        for col in 0..<cols {
            let x0 = min(max(0, CGFloat(col) * stepX), max(0, imageW - tileW))
            let y0 = min(max(0, CGFloat(row) * stepY), max(0, imageH - tileH))
            tiles.append(Tile(rect: CGRect(x: x0, y: y0, width: min(tileW, imageW), height: min(tileH, imageH))))
        }
    }
    return tiles
}

let tiles = computeTiles(imageW: imageW, imageH: imageH, spec: tilesArg, overlap: tileOverlap)
if tiles.count > 1 {
    log("Tiling into \(tiles.count) overlapping tiles (\(tilesArg ?? "auto"), overlap \(tileOverlap))")
}

// MARK: - Detection type + geometry (shared across tiles and final NMS)

struct OBBDetection {
    let cx, cy, w, h: CGFloat   // full-image pixels, top-left origin, y down
    let angle: CGFloat          // radians
    let confidence: Float

    // Corner order: front-right, back-right, back-left, front-left of the
    // rotated rect; a closed quad for drawing/JSON.
    var corners: [CGPoint] {
        let c = cos(angle), s = sin(angle)
        let v1 = CGPoint(x: c * w / 2, y: s * w / 2)
        let v2 = CGPoint(x: -s * h / 2, y: c * h / 2)
        return [
            CGPoint(x: cx + v1.x + v2.x, y: cy + v1.y + v2.y),
            CGPoint(x: cx + v1.x - v2.x, y: cy + v1.y - v2.y),
            CGPoint(x: cx - v1.x - v2.x, y: cy - v1.y - v2.y),
            CGPoint(x: cx - v1.x + v2.x, y: cy - v1.y + v2.y),
        ]
    }

    func offset(dx: CGFloat, dy: CGFloat) -> OBBDetection {
        OBBDetection(cx: cx + dx, cy: cy + dy, w: w, h: h, angle: angle, confidence: confidence)
    }
}

// MARK: - Per-tile inference: letterbox -> Core ML -> decode
//
// Supports both YOLO11 `[1,6,N]` (scan conf lane, then decode survivors) and
// YOLO26 end2end `[1,300,7]` (up to 300 already-NMS'd rows; filter by conf).

func runTileInference(tile cgTile: CGImage, tileOriginX: CGFloat, tileOriginY: CGFloat) -> ([OBBDetection], Double) {
    let tileW = CGFloat(cgTile.width)
    let tileH = CGFloat(cgTile.height)
    let gain = min(CGFloat(inputW) / tileW, CGFloat(inputH) / tileH)
    let scaledW = tileW * gain
    let scaledH = tileH * gain
    let padX = (CGFloat(inputW) - scaledW) / 2
    let padY = (CGFloat(inputH) - scaledH) / 2

    var pixelBuffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, inputW, inputH, kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferCGImageCompatibilityKey: true] as CFDictionary, &pixelBuffer)
    guard let pixelBuffer else {
        log("Error: could not allocate pixel buffer")
        return ([], 0)
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                           width: inputW, height: inputH,
                           bitsPerComponent: 8,
                           bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                           space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                               | CGBitmapInfo.byteOrder32Little.rawValue) {
        ctx.setFillColor(CGColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: inputW, height: inputH))
        // CGContext origin is bottom-left; padY works out the same because the
        // letterbox is vertically centered.
        ctx.interpolationQuality = .high
        ctx.draw(cgTile, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

    let started = Date()
    let prediction: MLFeatureProvider
    do {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
        prediction = try model.prediction(from: input)
    } catch {
        log("Error: inference failed: \(error.localizedDescription)")
        return ([], 0)
    }
    let inferenceMs = Date().timeIntervalSince(started) * 1000

    guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
        log("Error: no multiarray output \"\(outputName)\"")
        return ([], inferenceMs)
    }

    if let detected = detectOutputLayout(shape: output.shape) {
        OutputLayoutState.current = detected
    }

    let data = output.dataPointer.assumingMemoryBound(to: Float32.self)

    func mapToImage(cxRaw: Float, cyRaw: Float, wRaw: Float, hRaw: Float, angle: Float, conf: Float) -> OBBDetection? {
        let cx = (CGFloat(cxRaw) - padX) / gain + tileOriginX
        let cy = (CGFloat(cyRaw) - padY) / gain + tileOriginY
        let w = CGFloat(wRaw) / gain
        let h = CGFloat(hRaw) / gain
        guard w > 1, h > 1 else { return nil }
        return OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: CGFloat(angle), confidence: conf)
    }

    var tileDetections: [OBBDetection] = []

    switch OutputLayoutState.current {
    case .legacyChannelsFirst:
        // Shape [1, 6, N] — channels then anchors.
        let channels = output.shape[1].intValue
        let anchors = output.shape[2].intValue
        guard channels == 6 else {
            log("Error: legacy layout expected 6 channels, got \(channels)")
            return ([], inferenceMs)
        }
        let strideC = output.strides[1].intValue
        let strideN = output.strides[2].intValue
        let confBase = 4 * strideC
        var survivors: [Int] = []
        survivors.reserveCapacity(256)
        if strideN == 1 {
            let confLane = UnsafeBufferPointer(start: data + confBase, count: anchors)
            for i in 0..<anchors where confLane[i] >= confThreshold { survivors.append(i) }
        } else {
            for i in 0..<anchors where data[confBase + i * strideN] >= confThreshold { survivors.append(i) }
        }
        tileDetections.reserveCapacity(survivors.count)
        for i in survivors {
            let conf = data[confBase + i * strideN]
            let cxRaw = data[0 * strideC + i * strideN]
            let cyRaw = data[1 * strideC + i * strideN]
            let wRaw = data[2 * strideC + i * strideN]
            let hRaw = data[3 * strideC + i * strideN]
            let angle = data[5 * strideC + i * strideN]
            if let det = mapToImage(cxRaw: cxRaw, cyRaw: cyRaw, wRaw: wRaw, hRaw: hRaw, angle: angle, conf: conf) {
                tileDetections.append(det)
            }
        }

    case .end2endDetections:
        // Shape [1, maxDet, 7] — rows of [cx, cy, w, h, conf, cls, angle].
        // (Also handle a transposed [1, 7, maxDet] if strides imply it.)
        let dim1 = output.shape[1].intValue
        let dim2 = output.shape[2].intValue
        let stride1 = output.strides[1].intValue
        let stride2 = output.strides[2].intValue

        let rows: Int
        let featStride: Int
        let rowStride: Int
        if dim2 == 7 {
            rows = dim1
            rowStride = stride1
            featStride = stride2
        } else if dim1 == 7 {
            rows = dim2
            rowStride = stride2
            featStride = stride1
        } else {
            log("Error: end2end layout expected a 7-feature axis, got shape \(output.shape)")
            return ([], inferenceMs)
        }

        tileDetections.reserveCapacity(min(rows, 64))
        for i in 0..<rows {
            let base = i * rowStride
            let conf = data[base + 4 * featStride]
            guard conf >= confThreshold else { continue }
            let cxRaw = data[base + 0 * featStride]
            let cyRaw = data[base + 1 * featStride]
            let wRaw = data[base + 2 * featStride]
            let hRaw = data[base + 3 * featStride]
            let angle = data[base + 6 * featStride]
            if let det = mapToImage(cxRaw: cxRaw, cyRaw: cyRaw, wRaw: wRaw, hRaw: hRaw, angle: angle, conf: conf) {
                tileDetections.append(det)
            }
        }
    }

    return (tileDetections, inferenceMs)
}

// MARK: - Run all tiles

log("Running inference (\(tiles.count) tile\(tiles.count > 1 ? "s" : ""))...")
var allCandidates: [OBBDetection] = []
var totalInferenceMs: Double = 0
for tile in tiles {
    guard let cropped = cgImage.cropping(to: tile.rect) else {
        log("Warning: could not crop tile \(tile.rect)")
        continue
    }
    let (dets, ms) = runTileInference(tile: cropped, tileOriginX: tile.rect.minX, tileOriginY: tile.rect.minY)
    allCandidates.append(contentsOf: dets)
    totalInferenceMs += ms
}
log("\(allCandidates.count) raw candidates above conf \(confThreshold) across all tiles")

// MARK: - Global rotated NMS
//
// Convex polygon intersection (Sutherland-Hodgman) for exact rotated IoU.

func polygonArea(_ pts: [CGPoint]) -> CGFloat {
    guard pts.count >= 3 else { return 0 }
    var area: CGFloat = 0
    for i in 0..<pts.count {
        let p = pts[i], q = pts[(i + 1) % pts.count]
        area += p.x * q.y - q.x * p.y
    }
    return abs(area) / 2
}

func clip(_ subject: [CGPoint], edgeA: CGPoint, edgeB: CGPoint) -> [CGPoint] {
    // Keep points on the left side of edge a->b (for a CCW clip polygon).
    func side(_ p: CGPoint) -> CGFloat {
        (edgeB.x - edgeA.x) * (p.y - edgeA.y) - (edgeB.y - edgeA.y) * (p.x - edgeA.x)
    }
    func intersect(_ p: CGPoint, _ q: CGPoint) -> CGPoint {
        let a1 = edgeB.y - edgeA.y, b1 = edgeA.x - edgeB.x
        let c1 = a1 * edgeA.x + b1 * edgeA.y
        let a2 = q.y - p.y, b2 = p.x - q.x
        let c2 = a2 * p.x + b2 * p.y
        let det = a1 * b2 - a2 * b1
        guard abs(det) > 1e-9 else { return p }
        return CGPoint(x: (b2 * c1 - b1 * c2) / det, y: (a1 * c2 - a2 * c1) / det)
    }
    var result: [CGPoint] = []
    for i in 0..<subject.count {
        let current = subject[i]
        let previous = subject[(i + subject.count - 1) % subject.count]
        let currentInside = side(current) >= 0
        let previousInside = side(previous) >= 0
        if currentInside {
            if !previousInside { result.append(intersect(previous, current)) }
            result.append(current)
        } else if previousInside {
            result.append(intersect(previous, current))
        }
    }
    return result
}

func ensureCCW(_ pts: [CGPoint]) -> [CGPoint] {
    var signed: CGFloat = 0
    for i in 0..<pts.count {
        let p = pts[i], q = pts[(i + 1) % pts.count]
        signed += p.x * q.y - q.x * p.y
    }
    return signed < 0 ? pts.reversed() : pts
}

func rotatedIoU(_ a: OBBDetection, _ b: OBBDetection) -> CGFloat {
    // Cheap reject: centers further apart than the sum of half-diagonals.
    let reach = (hypot(a.w, a.h) + hypot(b.w, b.h)) / 2
    guard hypot(a.cx - b.cx, a.cy - b.cy) < reach else { return 0 }

    let quadA = ensureCCW(a.corners)
    let quadB = ensureCCW(b.corners)
    var inter = quadA
    for i in 0..<quadB.count {
        guard !inter.isEmpty else { break }
        inter = clip(inter, edgeA: quadB[i], edgeB: quadB[(i + 1) % quadB.count])
    }
    let interArea = polygonArea(inter)
    guard interArea > 0 else { return 0 }
    let union = polygonArea(quadA) + polygonArea(quadB) - interArea
    return union > 0 ? interArea / union : 0
}

allCandidates.sort { $0.confidence > $1.confidence }
var detections: [OBBDetection] = []
let skipNMS = (OutputLayoutState.current == .end2endDetections) && tiles.count == 1
if skipNMS {
    // YOLO26 end2end already suppressed duplicates in-graph.
    detections = Array(allCandidates.prefix(maxDetections))
    log("\(detections.count) spines (end2end, NMS skipped; inference \(String(format: "%.0f", totalInferenceMs)) ms)")
} else {
    for candidate in allCandidates {
        guard detections.count < maxDetections else { break }
        if !detections.contains(where: { rotatedIoU($0, candidate) > iouThreshold }) {
            detections.append(candidate)
        }
    }
    log("\(detections.count) spines after rotated NMS (inference \(String(format: "%.0f", totalInferenceMs)) ms total)")
}

// MARK: - Optional Vision OCR on each final spine crop
//
// Perspective-corrects each oriented box to an upright w x h crop (undoing
// the box's own rotation), then tries both reading orientations since spine
// text commonly runs along the long axis rather than horizontally.

struct SpineText {
    let text: String
    let confidence: Float
}

func uprightCrop(of detection: OBBDetection, from image: CGImage, imageH: CGFloat) -> CGImage? {
    let ciImage = CIImage(cgImage: image)
    // Pixel space is top-left/y-down; Core Image is bottom-left/y-up.
    let ciCenterX = detection.cx
    let ciCenterY = imageH - detection.cy

    var transform = CGAffineTransform(translationX: -ciCenterX, y: -ciCenterY)
    transform = transform.concatenating(CGAffineTransform(rotationAngle: detection.angle))
    transform = transform.concatenating(CGAffineTransform(translationX: detection.w / 2, y: detection.h / 2))

    let transformed = ciImage.transformed(by: transform)
    let cropRect = CGRect(x: 0, y: 0, width: detection.w, height: detection.h).integral
    guard cropRect.width > 0, cropRect.height > 0 else { return nil }
    let cropped = transformed.cropped(to: cropRect)

    let context = CIContext()
    return context.createCGImage(cropped, from: cropRect)
}

func recognizeText(in cgImage: CGImage, orientation: CGImagePropertyOrientation) -> [SpineText] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return []
    }
    return (request.results ?? []).compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        return SpineText(text: candidate.string, confidence: candidate.confidence)
    }
}

func ocrSpine(_ detection: OBBDetection) -> [SpineText] {
    guard let crop = uprightCrop(of: detection, from: cgImage, imageH: imageH) else { return [] }
    // Try reading the crop as-is, then rotated 90 degrees: spine titles run
    // either along the long axis (vertical spines) or horizontally (stacked).
    let up = recognizeText(in: crop, orientation: .up)
    let rotated = recognizeText(in: crop, orientation: .right)
    return up.count >= rotated.count ? up : rotated
}

var ocrResults: [[SpineText]] = []
if enableOCR {
    log("Running OCR on \(detections.count) spine crops...")
    for detection in detections {
        ocrResults.append(ocrSpine(detection))
    }
}

// MARK: - JSON output

struct TextJSON: Codable {
    let text: String
    let confidence: Float
}

struct SpineJSON: Codable {
    let confidence: Float
    let cxPx: Int
    let cyPx: Int
    let wPx: Int
    let hPx: Int
    let angleDeg: Double
    let cornersPx: [[Int]]
    let texts: [TextJSON]?
}

struct DocumentJSON: Codable {
    let image: [String: Int]
    let model: String
    let tiles: Int
    let count: Int
    let spines: [SpineJSON]
}

let spineJSONs = detections.enumerated().map { index, d in
    SpineJSON(
        confidence: d.confidence,
        cxPx: Int(d.cx.rounded()), cyPx: Int(d.cy.rounded()),
        wPx: Int(d.w.rounded()), hPx: Int(d.h.rounded()),
        angleDeg: Double(d.angle) * 180 / .pi,
        cornersPx: d.corners.map { [Int($0.x.rounded()), Int($0.y.rounded())] },
        texts: enableOCR ? ocrResults[index].map { TextJSON(text: $0.text, confidence: $0.confidence) } : nil
    )
}

let document = DocumentJSON(
    image: ["w": cgImage.width, "h": cgImage.height],
    model: modelURL.lastPathComponent,
    tiles: tiles.count,
    count: detections.count,
    spines: spineJSONs
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
if let jsonData = try? encoder.encode(document), let json = String(data: jsonData, encoding: .utf8) {
    print(json)
}

// MARK: - Annotated image

let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
if let ctx = CGContext(data: nil, width: cgImage.width, height: cgImage.height,
                       bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: imageW, height: imageH))

    // Detections use top-left-origin pixels; CGContext is bottom-left.
    func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: imageH - p.y) }

    ctx.setLineWidth(max(3, imageW / 800))
    for detection in detections {
        // Green for high confidence, fading to red for low.
        let t = CGFloat(min(max((detection.confidence - confThreshold) / (1 - confThreshold), 0), 1))
        ctx.setStrokeColor(CGColor(red: 1 - t, green: t, blue: 0.1, alpha: 0.95))
        let quad = detection.corners.map(flip)
        ctx.move(to: quad[0])
        for p in quad.dropFirst() { ctx.addLine(to: p) }
        ctx.closePath()
        ctx.strokePath()
    }

    if tiles.count > 1 {
        ctx.setStrokeColor(CGColor(red: 0.2, green: 0.6, blue: 1, alpha: 0.6))
        ctx.setLineWidth(max(2, imageW / 1500))
        for tile in tiles {
            let r = CGRect(x: tile.rect.minX, y: imageH - tile.rect.maxY, width: tile.rect.width, height: tile.rect.height)
            ctx.stroke(r)
        }
    }

    let outputURL = imageURL.deletingPathExtension().appendingPathExtension("spines.png")
    if let annotated = ctx.makeImage(),
       let destination = CGImageDestinationCreateWithURL(outputURL as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(destination, annotated, nil)
        CGImageDestinationFinalize(destination)
        log("Annotated image written to \(outputURL.path)")
    }
}
