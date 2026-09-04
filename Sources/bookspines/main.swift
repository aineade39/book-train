import AppKit
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import SpineCore
import UniformTypeIdentifiers
import Vision

// Core ML OBB validation / inference harness for the spine detector.
//
// Runs a YOLO-OBB .mlpackage (exported with nms=False) on one image, with
// optional tiling for very dense shelves and optional Vision OCR on the
// final crops:
//
//   [tile grid] -> per tile: SpineCore.SpineDetector.predict ->
//   single global rotated NMS -> (optional) perspective-crop + OCR
//
// Output: JSON on stdout + "<image>.obb.jpg" with blue OBB overlays (no labels).
//
// Usage:
//   swift run bookspines <image> [--model path.mlpackage] [--conf 0.15]
//                        [--iou 0.45] [--max-det 500] [--tiles NxM] [--tile-overlap 0.25]
//                        [--ocr]
//
// This is the SwiftPM-packaged successor to the root `bookspines.swift`
// script; output JSON schema and default model resolution are kept
// byte-for-byte compatible so existing callers/scripts don't need to change.

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - CLI

var imagePath: String?
var modelPath: String = defaultModelURL().path
var confThreshold: Float = 0.15
var iouThreshold: Double = 0.45
var maxDetections = 500
var tilesArg: String?          // nil = auto (1x1, or 2x2 for very large images)
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
        Usage: swift run bookspines <image> [--model path.mlpackage] [--conf 0.15] \
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

// MARK: - Model

let modelURL = URL(fileURLWithPath: modelPath)
let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: modelURL, computeUnits: .all)
} catch {
    log("Error: \(error)")
    exit(1)
}
log("Model input \(detector.inputWidth)x\(detector.inputHeight), output \"\(detector.outputName)\", layout \(detector.layout)")

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

// MARK: - Run all tiles

log("Running inference (\(tiles.count) tile\(tiles.count > 1 ? "s" : ""))...")
let detectionOptions = DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections)
var allCandidates: [OBBDetection] = []
var totalInferenceMs: Double = 0
var alreadyNMSed = false
for tile in tiles {
    guard let cropped = cgImage.cropping(to: tile.rect) else {
        log("Warning: could not crop tile \(tile.rect)")
        continue
    }
    do {
        let result = try detector.predict(cropped, originX: Double(tile.rect.minX), originY: Double(tile.rect.minY), options: detectionOptions)
        allCandidates.append(contentsOf: result.detections)
        totalInferenceMs += result.inferenceMs
        alreadyNMSed = result.alreadyNMSed
    } catch {
        log("Error: inference failed on tile \(tile.rect): \(error)")
    }
}
log("\(allCandidates.count) raw candidates above conf \(confThreshold) across all tiles")

// MARK: - Global rotated NMS
//
// A single-tile, already-NMS'd (YOLO26 end2end) pass can trust the model's
// own row order; anything else (legacy layout, or multiple tiles merging)
// needs a fresh rotated-NMS pass.

var detections: [OBBDetection] = []
let skipNMS = alreadyNMSed && tiles.count == 1
if skipNMS {
    detections = Array(allCandidates.prefix(maxDetections))
    log("\(detections.count) spines (end2end, NMS skipped; inference \(String(format: "%.0f", totalInferenceMs)) ms)")
} else {
    detections = nmsRotated(allCandidates, iouThreshold: iouThreshold, maxDetections: maxDetections)
    log("\(detections.count) spines after rotated NMS (inference \(String(format: "%.0f", totalInferenceMs)) ms total)")
}

// MARK: - Optional Vision OCR on each final spine crop
//
// Perspective-corrects each oriented box to an upright w x h crop (undoing
// the box's own rotation, via SpineCore.uprightWarp), then tries both
// reading orientations since spine text commonly runs along the long axis
// rather than horizontally.
//
// This CLI's `ocrSpine` is a placeholder only (fixed .up/.right pair, picked
// by observation *count*) — see docs/BOOK_ID_IOS_PIPELINE.md §OCR
// orientation design for the aspect-guided, confidence-summed target
// implementation in `SpinePerception`.

struct SpineText {
    let text: String
    let confidence: Float
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
    guard let crop = uprightWarp(of: detection, in: cgImage) else { return [] }
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
        confidence: d.conf,
        cxPx: Int(d.cx.rounded()), cyPx: Int(d.cy.rounded()),
        wPx: Int(d.w.rounded()), hPx: Int(d.h.rounded()),
        angleDeg: d.angle * 180 / .pi,
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

// MARK: - Annotated image (blue OBB polygons only — no confidence labels)

func ultralyticsLineWidth(imageW: CGFloat, imageH: CGFloat) -> CGFloat {
    max(CGFloat(round((imageW + imageH) / 2 * 0.003)), 2)
}

func drawUltralyticsOBBOverlay(on baseImage: CGImage, detections: [OBBDetection], tiles: [Tile]) -> CGImage? {
    let w = CGFloat(baseImage.width)
    let h = CGFloat(baseImage.height)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: baseImage.width, height: baseImage.height,
                               bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(baseImage, in: CGRect(x: 0, y: 0, width: w, height: h))

    func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: h - p.y) }

    let lw = ultralyticsLineWidth(imageW: w, imageH: h)
    ctx.setLineWidth(lw)
    ctx.setLineJoin(.round)
    // Ultralytics class-0 color #042aff in BGR -> sRGB (4, 42, 255).
    let boxColor = CGColor(red: 4 / 255, green: 42 / 255, blue: 1, alpha: 1)

    // result.plot() draws reversed so higher-index boxes land on top.
    for detection in detections.reversed() {
        let quad = detection.corners.map(flip)
        ctx.setStrokeColor(boxColor)
        ctx.move(to: quad[0])
        for p in quad.dropFirst() { ctx.addLine(to: p) }
        ctx.closePath()
        ctx.strokePath()
    }

    if tiles.count > 1 {
        ctx.setStrokeColor(CGColor(red: 0.2, green: 0.6, blue: 1, alpha: 0.6))
        ctx.setLineWidth(max(2, w / 1500))
        for tile in tiles {
            let r = CGRect(x: tile.rect.minX, y: h - tile.rect.maxY, width: tile.rect.width, height: tile.rect.height)
            ctx.stroke(r)
        }
    }

    return ctx.makeImage()
}

if let annotated = drawUltralyticsOBBOverlay(on: cgImage, detections: detections, tiles: tiles) {
    let obbURL = imageURL.deletingPathExtension().appendingPathExtension("obb.jpg")
    if let destination = CGImageDestinationCreateWithURL(obbURL as CFURL,
                                                           UTType.jpeg.identifier as CFString, 1, nil) {
        let options = [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary
        CGImageDestinationAddImage(destination, annotated, options)
        CGImageDestinationFinalize(destination)
        log("Annotated image written to \(obbURL.path)")
    }
}
