import AppKit
import CoreGraphics
import Foundation
import SpineCatalog
import SpineCore
import SpineMatching
import SpinePerception
import SpineReasoning

// End-to-end detect -> isolate -> read -> normalize -> match -> confirm
// macOS validation harness for the full pipeline in
// docs/BOOK_ID_IOS_PIPELINE.md -- the one CLI `AGENTS.md`'s "scope
// anchoring" habit should treat as the primary deliverable when a request
// says "run/compare the pipeline" without naming a specific stage.
//
// Usage:
//   swift run spine-id <image> --db <catalog.sqlite> [--model path.mlpackage] \
//       [--conf 0.15] [--iou 0.45] [--max-det 500] \
//       [--accept-threshold 90] [--margin 8] [--top-n 5] [--fm] [--json <path>]
//
// `--fm` opts into the docs/BOOK_ID_IOS_PIPELINE.md §Foundation Models
// enhancement: escalate a rate-limited subset of "hard case" (marginal
// OCR quality score) spines to `SpineReasoningService`, using its cleaned
// title+author as the match query instead of the raw OCR text. No-op
// (falls back to plain OCR matching, same as without the flag) when the
// OS/build doesn't have iOS/macOS 26+'s `FoundationModels` or the
// on-device model isn't available -- see `SpineReasoningService.isAvailable`.

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

// MARK: - Full-frame barcode fast path -- bypasses fuzzy match entirely for ISBN hits.

let barcodeReader = VisionBarcodeReader()
let barcodes = (try? barcodeReader.detectBarcodes(in: cgImage))?.filter(\.looksLikeISBN) ?? []
if !barcodes.isEmpty {
    log("ISBN barcode fast path: \(barcodes.map(\.payload).joined(separator: ", "))")
}

// MARK: - Per-spine: read -> normalize -> match -> accept

struct SpineResultJSON: Codable {
    let id: String
    let assembledText: String
    let ocrQualityScore: Double
    let passedOCRQualityGate: Bool
    let decision: String
    let matchedTitle: String?
    let matchedAuthor: String?
    let score: Double?
    let margin: Double?
    let source: String
    let topCandidates: [String]
}

let router = OCROrientationRouter(recognizer: VisionTextRecognizer())
let policy = AcceptPolicy(acceptThreshold: acceptThreshold, marginThreshold: marginThreshold, topN: topN)

struct SpineOCRRecord {
    let detection: OBBDetection
    let assembledText: String
    let qualityScore: Double
    let passedQualityGate: Bool
}

var ocrRecords: [SpineOCRRecord] = []
for det in detections {
    guard let crop = uprightWarp(of: det, in: cgImage) else { continue }
    guard let ocr = try? router.recognize(crop: crop, detection: det) else { continue }
    ocrRecords.append(SpineOCRRecord(
        detection: det, assembledText: ocr.assembledText, qualityScore: ocr.qualityScore,
        passedQualityGate: ocr.passedQualityGate && !ocr.assembledText.isEmpty
    ))
}

// MARK: - Optional FM escalation (hard cases only, rate-limited)
//
// Runs *after* the OCR quality gate but *before* matching, per
// docs/BOOK_ID_IOS_PIPELINE.md's pipeline diagram
// ("QualityGate -> [optional FM @Generable parse] -> Normalize -> ...").
// A cleaned title+author (`SpineExtraction.matchQueryText`) replaces the
// raw OCR text as the match query for escalated spines only; everything
// else about matching is unchanged.

@available(iOS 26.0, macOS 26.0, *)
func runFMEscalation(records: [SpineOCRRecord], policy: FMEscalationPolicy) async -> [UUID: String] {
    let service = SpineReasoningService()
    guard service.isAvailable else {
        log("--fm requested but no on-device Foundation Model is available; falling back to Vision-only")
        return [:]
    }
    let passed = records.filter(\.passedQualityGate)
    let scores = passed.map { (id: $0.detection.id, qualityScore: $0.qualityScore) }
    let escalate = policy.selectForEscalation(scores: scores)
    guard !escalate.isEmpty else { return [:] }
    log("FM escalation: \(escalate.count) hard case(s) of \(passed.count) OCR-passed spine(s)")

    var overrides: [UUID: String] = [:]
    for record in passed where escalate.contains(record.detection.id) {
        guard let extraction = await service.extract(from: record.assembledText) else { continue }
        overrides[record.detection.id] = extraction.matchQueryText
        log("  [\(record.detection.id.uuidString.prefix(8))] fm: \"\(record.assembledText)\" -> \"\(extraction.matchQueryText)\"")
    }
    return overrides
}

var fmQueryOverrides: [UUID: String] = [:]
if useFM {
    if #available(iOS 26.0, macOS 26.0, *) {
        fmQueryOverrides = await runFMEscalation(records: ocrRecords, policy: .default)
    } else {
        log("--fm requested but this OS/build doesn't support FoundationModels; falling back to Vision-only")
    }
}

// MARK: - Per-spine: normalize -> match -> accept

var results: [SpineResultJSON] = []
for record in ocrRecords {
    let det = record.detection
    guard record.passedQualityGate else {
        results.append(SpineResultJSON(
            id: det.id.uuidString, assembledText: record.assembledText, ocrQualityScore: record.qualityScore,
            passedOCRQualityGate: false, decision: "no-match", matchedTitle: nil, matchedAuthor: nil,
            score: nil, margin: nil, source: "ocr", topCandidates: []
        ))
        continue
    }

    let fmOverride = fmQueryOverrides[det.id]
    let matchSource = fmOverride != nil ? "fm-assisted" : "ocr"
    let normalizedQuery = normalizeForSearch(fmOverride ?? record.assembledText)
    let candidates = (try? catalog.retrieveCandidates(forQuery: normalizedQuery)) ?? []
    let scored = candidates.map {
        ScoredCandidate(candidate: $0, score: tokenSetRatio(normalizedQuery, normalizeForSearch($0.searchableText)))
    }
    let decision = policy.decide(scored)

    let result: SpineResultJSON
    switch decision {
    case .autoAccept(let winner):
        result = SpineResultJSON(
            id: det.id.uuidString, assembledText: record.assembledText, ocrQualityScore: record.qualityScore,
            passedOCRQualityGate: true, decision: "auto-accept",
            matchedTitle: winner.candidate.title, matchedAuthor: winner.candidate.author,
            score: winner.score, margin: nil, source: matchSource, topCandidates: [winner.candidate.title]
        )
    case .ambiguous(let top):
        result = SpineResultJSON(
            id: det.id.uuidString, assembledText: record.assembledText, ocrQualityScore: record.qualityScore,
            passedOCRQualityGate: true, decision: "ambiguous", matchedTitle: nil, matchedAuthor: nil,
            score: top.first?.score, margin: nil, source: matchSource, topCandidates: top.map(\.candidate.title)
        )
    case .noMatch:
        result = SpineResultJSON(
            id: det.id.uuidString, assembledText: record.assembledText, ocrQualityScore: record.qualityScore,
            passedOCRQualityGate: true, decision: "no-match", matchedTitle: nil, matchedAuthor: nil,
            score: nil, margin: nil, source: matchSource, topCandidates: []
        )
    }
    log("  [\(det.id.uuidString.prefix(8))] \(result.decision): \"\(record.assembledText)\" -> \(result.matchedTitle ?? "-")")
    results.append(result)
}

// MARK: - JSON output

struct PayloadJSON: Codable {
    let image: String
    let isbnBarcodes: [String]
    let spines: [SpineResultJSON]
}

let payload = PayloadJSON(image: imageURL.path, isbnBarcodes: barcodes.map(\.payload), spines: results)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try! encoder.encode(payload)
if let jsonPath {
    try? data.write(to: URL(fileURLWithPath: jsonPath))
    log("Wrote \(jsonPath)")
} else {
    print(String(data: data, encoding: .utf8)!)
}
