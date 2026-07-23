import AppKit
import CoreGraphics
import Foundation
import SpineCore
import SpinePerception

// Detect + isolate + read (no matching yet) macOS validation harness, per
// docs/BOOK_ID_IOS_PIPELINE.md's layered pipeline -- mirrors `bookspines`'
// shape but runs the real `SpinePerception` OCR orientation router (not
// `bookspines`' placeholder `ocrSpine`) plus the capture and OCR quality
// gates and the full-frame barcode fast path.
//
// Usage:
//   swift run spine-read <image> [--model path.mlpackage] [--conf 0.15] \
//       [--iou 0.45] [--max-det 500] [--json <path>]

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func nextArg(_ args: [String], _ i: inout Int) -> String {
    i += 1
    guard i < args.count else { fail("Missing value for \(args[i - 1])") }
    return args[i]
}

var imagePath: String?
var modelPath: String = defaultModelURL().path
var confThreshold: Float = 0.15
var iouThreshold: Double = 0.45
var maxDetections = 500
var jsonPath: String?

let cliArgs = CommandLine.arguments
var argIndex = 1
while argIndex < cliArgs.count {
    let arg = cliArgs[argIndex]
    switch arg {
    case "--model": modelPath = nextArg(cliArgs, &argIndex)
    case "--conf": confThreshold = Float(nextArg(cliArgs, &argIndex)) ?? confThreshold
    case "--iou": iouThreshold = Double(nextArg(cliArgs, &argIndex)) ?? iouThreshold
    case "--max-det": maxDetections = Int(nextArg(cliArgs, &argIndex)) ?? maxDetections
    case "--json": jsonPath = nextArg(cliArgs, &argIndex)
    default:
        if imagePath == nil { imagePath = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let imagePath else {
    fail("""
        Usage: swift run spine-read <image> [--model path.mlpackage] [--conf 0.15] \
        [--iou 0.45] [--max-det 500] [--json <path>]
        """)
}

let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
guard let nsImage = NSImage(contentsOf: imageURL),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fail("Could not read image: \(imageURL.path)")
}
log("Image: \(cgImage.width)x\(cgImage.height)")

// MARK: - Capture quality gate

let captureGate = CaptureQualityGate.default
let captureScore = captureGate.score(cgImage)
log("Capture quality: sharpness=\(String(format: "%.3f", captureScore.sharpness)) exposure=\(String(format: "%.3f", captureScore.exposure)) passes=\(captureGate.passes(cgImage))")

// MARK: - Detect

let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: URL(fileURLWithPath: modelPath), computeUnits: .all)
} catch {
    fail("Could not load model: \(error)")
}

let detectionOptions = DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections)
var detections: [OBBDetection]
do {
    let result = try detector.predict(cgImage, options: detectionOptions)
    detections = result.alreadyNMSed
        ? Array(result.detections.prefix(maxDetections))
        : nmsRotated(result.detections, iouThreshold: iouThreshold, maxDetections: maxDetections)
} catch {
    fail("Inference failed: \(error)")
}
log("Detected \(detections.count) spines")

// MARK: - Full-frame barcode fast path (never per-spine-crop -- see BarcodeReader.swift)

let barcodeReader = VisionBarcodeReader()
let barcodes = (try? barcodeReader.detectBarcodes(in: cgImage)) ?? []
if !barcodes.isEmpty {
    log("Barcodes: \(barcodes.map { "\($0.payload) (\($0.symbology), isbn=\($0.looksLikeISBN))" }.joined(separator: ", "))")
}

// MARK: - Isolate + read each spine

struct SpineReadJSON: Codable {
    let id: String
    let cx, cy, w, h, angleDeg: Double
    let detectionConfidence: Float
    let orientation: String
    let assembledText: String
    let qualityScore: Double
    let passedQualityGate: Bool
    let ranThirdPass: Bool
}

let router = OCROrientationRouter(recognizer: VisionTextRecognizer())
var reads: [SpineReadJSON] = []
for det in detections {
    guard let crop = uprightWarp(of: det, in: cgImage) else {
        log("  warning: could not warp detection \(det.id)")
        continue
    }
    do {
        let result = try router.recognize(crop: crop, detection: det)
        log("  [\(det.id.uuidString.prefix(8))] \"\(result.assembledText)\" (orientation=\(result.winningPass.orientation), quality=\(String(format: "%.2f", result.qualityScore)), passed=\(result.passedQualityGate), thirdPass=\(result.ranThirdPass))")
        reads.append(SpineReadJSON(
            id: det.id.uuidString, cx: det.cx, cy: det.cy, w: det.w, h: det.h,
            angleDeg: det.angle * 180 / .pi, detectionConfidence: det.conf,
            orientation: "\(result.winningPass.orientation)", assembledText: result.assembledText,
            qualityScore: result.qualityScore, passedQualityGate: result.passedQualityGate,
            ranThirdPass: result.ranThirdPass
        ))
    } catch {
        log("  warning: OCR failed for \(det.id): \(error)")
    }
}

// MARK: - JSON output

struct PayloadJSON: Codable {
    let image: String
    let captureSharpness: Double
    let captureExposure: Double
    let barcodes: [String]
    let spines: [SpineReadJSON]
}

let payload = PayloadJSON(
    image: imageURL.path, captureSharpness: captureScore.sharpness, captureExposure: captureScore.exposure,
    barcodes: barcodes.map(\.payload), spines: reads
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try! encoder.encode(payload)
if let jsonPath {
    try? data.write(to: URL(fileURLWithPath: jsonPath))
    log("Wrote \(jsonPath)")
} else {
    print(String(data: data, encoding: .utf8)!)
}
