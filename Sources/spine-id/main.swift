import AppKit
import CoreGraphics
import Foundation
import SpineCatalog
import SpineCore
import SpineMatching
import SpinePerception
import SpinePipeline
import SpineReasoning

// End-to-end detect -> isolate -> read -> normalize -> match -> confirm
// macOS validation harness for the full pipeline in
// docs/BOOK_ID_IOS_PIPELINE.md -- the one CLI `AGENTS.md`'s "scope
// anchoring" habit should treat as the primary deliverable when a request
// says "run/compare the pipeline" without naming a specific stage.
//
// Per the locked "Book ID OCR gains" plan §H ("spine-id: In scope --
// shared package APIs for ISBN, role retrieve/rerank, accept; CLI and app
// call the same code"), all of the actual detect/OCR/match orchestration
// lives in `SpinePipeline.SpineIdentificationEngine` -- this file is just
// argument parsing, image/catalog loading, and JSON serialization around
// it. The iOS/macOS app's `SpineIdentificationPipeline` builds the exact
// same engine and adapts its `SpinePipelineResult` to SwiftUI models
// instead of JSON.
//
// Usage:
//   swift run spine-id <image> --db <catalog.sqlite> [--model path.mlpackage] \
//       [--conf 0.15] [--iou 0.45] [--max-det 500] \
//       [--accept-threshold 90] [--margin 8] [--top-n 5] [--fm] [--json <path>]
//
// `--fm` opts into the docs/BOOK_ID_IOS_PIPELINE.md §Foundation Models
// enhancement: escalate a rate-limited subset of "hard case" (marginal
// OCR quality score) spines to the on-device Foundation Model, using its
// cleaned title+author as the match query instead of the raw OCR text.
// No-op (falls back to plain OCR matching, same as without the flag) when
// the OS/build doesn't have iOS/macOS 26+'s `FoundationModels` or the
// on-device model isn't available (see `SpineReasoningService.isAvailable`).

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
var dbPath: String?
var modelPath: String = defaultModelURL().path
var confThreshold: Float = 0.15
var iouThreshold: Double = 0.45
var maxDetections = 500
var acceptThreshold = 90.0
var marginThreshold = 8.0
var topN = 5
var jsonPath: String?
var useFM = false

let cliArgs = CommandLine.arguments
var argIndex = 1
while argIndex < cliArgs.count {
    let arg = cliArgs[argIndex]
    switch arg {
    case "--db": dbPath = nextArg(cliArgs, &argIndex)
    case "--model": modelPath = nextArg(cliArgs, &argIndex)
    case "--conf": confThreshold = Float(nextArg(cliArgs, &argIndex)) ?? confThreshold
    case "--iou": iouThreshold = Double(nextArg(cliArgs, &argIndex)) ?? iouThreshold
    case "--max-det": maxDetections = Int(nextArg(cliArgs, &argIndex)) ?? maxDetections
    case "--accept-threshold": acceptThreshold = Double(nextArg(cliArgs, &argIndex)) ?? acceptThreshold
    case "--margin": marginThreshold = Double(nextArg(cliArgs, &argIndex)) ?? marginThreshold
    case "--top-n": topN = Int(nextArg(cliArgs, &argIndex)) ?? topN
    case "--fm": useFM = true
    case "--json": jsonPath = nextArg(cliArgs, &argIndex)
    default:
        if imagePath == nil { imagePath = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let imagePath, let dbPath else {
    fail("""
        Usage: swift run spine-id <image> --db <catalog.sqlite> [--model path.mlpackage] \
        [--conf 0.15] [--iou 0.45] [--max-det 500] \
        [--accept-threshold 90] [--margin 8] [--top-n 5] [--fm] [--json <path>]
        """)
}

let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
guard let nsImage = NSImage(contentsOf: imageURL),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fail("Could not read image: \(imageURL.path)")
}

let catalog: BookCatalog
do {
    catalog = try BookCatalog(path: URL(fileURLWithPath: dbPath).standardizedFileURL.path)
} catch {
    fail("Could not open catalog: \(error)")
}

let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: URL(fileURLWithPath: modelPath), computeUnits: .all)
} catch {
    fail("Could not load model: \(error)")
}

let engine = SpineIdentificationEngine(
    detector: detector,
    detectionOptions: DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections),
    acceptPolicy: AcceptPolicy(acceptThreshold: acceptThreshold, marginThreshold: marginThreshold, topN: topN),
    useFM: useFM
)

if useFM, #unavailable(iOS 26.0, macOS 26.0) {
    log("--fm requested but this OS/build doesn't support FoundationModels; falling back to Vision-only")
}

let result: SpinePipelineResult
do {
    result = try await engine.run(on: cgImage, catalog: catalog)
} catch {
    fail("Pipeline run failed: \(error)")
}

if !result.isbnBarcodes.isEmpty {
    log("ISBN barcode fast path: \(result.isbnBarcodes.joined(separator: ", "))")
}
log("Detected \(result.spines.count) spine\(result.spines.count == 1 ? "" : "s")")

// MARK: - JSON output

struct SpineResultJSON: Codable {
    let id: String
    let cx: Double
    let cy: Double
    let w: Double
    let h: Double
    let angleDeg: Double
    let detectionConfidence: Float
    let assembledText: String
    let ocrQualityScore: Double
    let passedOCRQualityGate: Bool
    let decision: String
    let matchedTitle: String?
    let matchedAuthor: String?
    let matchedWorkKey: String?
    let score: Double?
    let margin: Double?
    let source: String
    let topCandidates: [String]
}

func toJSON(_ spine: SpinePipelineSpine) -> SpineResultJSON {
    let det = spine.detection
    let base = (
        id: spine.id.uuidString,
        cx: det.cx, cy: det.cy, w: det.w, h: det.h,
        angleDeg: det.angle * 180 / .pi,
        detectionConfidence: det.conf,
        assembledText: spine.assembledText,
        ocrQualityScore: spine.ocrQualityScore,
        margin: spine.matchMargin,
        source: spine.source.rawValue
    )
    switch spine.decision {
    case .didNotPassQualityGate:
        return SpineResultJSON(
            id: base.id, cx: base.cx, cy: base.cy, w: base.w, h: base.h,
            angleDeg: base.angleDeg, detectionConfidence: base.detectionConfidence,
            assembledText: base.assembledText, ocrQualityScore: base.ocrQualityScore,
            passedOCRQualityGate: false, decision: "no-match", matchedTitle: nil, matchedAuthor: nil,
            matchedWorkKey: nil, score: nil, margin: nil, source: base.source, topCandidates: []
        )
    case .autoAccepted(let title, let author, let workKey, let score):
        log("  [\(spine.id.uuidString.prefix(8))] auto-accept: \"\(spine.assembledText)\" -> \(title)")
        return SpineResultJSON(
            id: base.id, cx: base.cx, cy: base.cy, w: base.w, h: base.h,
            angleDeg: base.angleDeg, detectionConfidence: base.detectionConfidence,
            assembledText: base.assembledText, ocrQualityScore: base.ocrQualityScore,
            passedOCRQualityGate: true, decision: "auto-accept", matchedTitle: title, matchedAuthor: author,
            matchedWorkKey: workKey, score: score, margin: base.margin, source: base.source, topCandidates: [title]
        )
    case .needsConfirmation(let candidates):
        log("  [\(spine.id.uuidString.prefix(8))] ambiguous: \"\(spine.assembledText)\" -> \(candidates.map(\.candidate.title))")
        return SpineResultJSON(
            id: base.id, cx: base.cx, cy: base.cy, w: base.w, h: base.h,
            angleDeg: base.angleDeg, detectionConfidence: base.detectionConfidence,
            assembledText: base.assembledText, ocrQualityScore: base.ocrQualityScore,
            passedOCRQualityGate: true, decision: "ambiguous", matchedTitle: nil, matchedAuthor: nil,
            matchedWorkKey: nil, score: candidates.first?.score, margin: base.margin, source: base.source,
            topCandidates: candidates.map(\.candidate.title)
        )
    case .noMatch:
        return SpineResultJSON(
            id: base.id, cx: base.cx, cy: base.cy, w: base.w, h: base.h,
            angleDeg: base.angleDeg, detectionConfidence: base.detectionConfidence,
            assembledText: base.assembledText, ocrQualityScore: base.ocrQualityScore,
            passedOCRQualityGate: true, decision: "no-match", matchedTitle: nil, matchedAuthor: nil,
            matchedWorkKey: nil, score: nil, margin: base.margin, source: base.source, topCandidates: []
        )
    }
}

struct PayloadJSON: Codable {
    let image: String
    let captureSharpness: Double
    let captureExposure: Double
    let capturePassed: Bool
    let isbnBarcodes: [String]
    let spines: [SpineResultJSON]
}

let payload = PayloadJSON(
    image: imageURL.path,
    captureSharpness: result.captureSharpness,
    captureExposure: result.captureExposure,
    capturePassed: result.capturePassed,
    isbnBarcodes: result.isbnBarcodes,
    spines: result.spines.map(toJSON)
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
