import AppKit
import CoreGraphics
import Foundation
import SpineCore
import SpinePerception

// OCR quality-gate tuning harness per docs/BOOK_ID_IOS_PIPELINE.md
// §Delivery sequencing step 2 ("quality gate tuning on real rotation
// buckets") and `AGENTS.md`'s rotation-bucket eval culture
// (`tools/eval_rotation_sweep.py`, `MODELS.md`'s acceptance rule):
// aggregate pass rate can look fine while the gate is actually far too
// permissive/strict at extreme rotations, because rotated copies are a
// minority of any dataset. This buckets a directory of images by the
// `_rot<angle>` suffix `tools/build_spines_dataset.py` bakes in (plain
// filenames = "original") and reports the OCR quality gate's pass rate +
// mean score per bucket using the real detector + real Vision OCR + real
// `OCRQualityGate` -- no ground-truth OCR text needed: "does the gate
// pass this crop" is itself the signal being tuned.
//
// Usage:
//   swift run ocr-quality-sweep <images-dir> [--model path.mlpackage] \
//       [--angles 30,45,60,90] [--max-per-bucket 15] [--conf 0.15] \
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

var imagesDirPath: String?
var modelPath: String = defaultModelURL().path
var angles: [Int] = [30, 45, 60, 90]
var maxPerBucket = 15
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
    case "--angles": angles = nextArg(cliArgs, &argIndex).split(separator: ",").compactMap { Int($0) }
    case "--max-per-bucket": maxPerBucket = Int(nextArg(cliArgs, &argIndex)) ?? maxPerBucket
    case "--conf": confThreshold = Float(nextArg(cliArgs, &argIndex)) ?? confThreshold
    case "--iou": iouThreshold = Double(nextArg(cliArgs, &argIndex)) ?? iouThreshold
    case "--max-det": maxDetections = Int(nextArg(cliArgs, &argIndex)) ?? maxDetections
    case "--json": jsonPath = nextArg(cliArgs, &argIndex)
    default:
        if imagesDirPath == nil { imagesDirPath = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let imagesDirPath else {
    fail("""
        Usage: swift run ocr-quality-sweep <images-dir> [--model path.mlpackage] \
        [--angles 30,45,60,90] [--max-per-bucket 15] [--conf 0.15] [--iou 0.45] \
        [--max-det 500] [--json <path>]
        """)
}

// MARK: - Bucket the images directory by `_rot<angle>` filename suffix.

let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
let imagesDirURL = URL(fileURLWithPath: imagesDirPath).standardizedFileURL
guard let entries = try? FileManager.default.contentsOfDirectory(at: imagesDirURL, includingPropertiesForKeys: nil) else {
    fail("Could not list directory: \(imagesDirURL.path)")
}

let wantedAngles = Set(angles)
let bucketPattern = try! NSRegularExpression(pattern: "_rot(\\d+)$")

var buckets: [String: [URL]] = ["original": []]
for angle in angles { buckets["rot\(angle)"] = [] }

for url in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
    guard imageExtensions.contains(url.pathExtension.lowercased()) else { continue }
    let stem = url.deletingPathExtension().lastPathComponent
    let range = NSRange(stem.startIndex..., in: stem)
    if let match = bucketPattern.firstMatch(in: stem, range: range),
       let angleRange = Range(match.range(at: 1), in: stem), let angle = Int(stem[angleRange]),
       wantedAngles.contains(angle) {
        buckets["rot\(angle)", default: []].append(url)
    } else if bucketPattern.firstMatch(in: stem, range: range) == nil {
        buckets["original", default: []].append(url)
    } // rotation suffixes not requested are silently skipped, like the Python sweep.
}
buckets = buckets.filter { !$0.value.isEmpty }

// MARK: - Model + routers

let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: URL(fileURLWithPath: modelPath), computeUnits: .all)
} catch {
    fail("Could not load model: \(error)")
}
let detectionOptions = DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections)
let qualityGate = OCRQualityGate.default
let router = OCROrientationRouter(recognizer: VisionTextRecognizer(), qualityGate: qualityGate)

struct SpineReadingJSON: Codable {
    let qualityScore: Double
    let passedQualityGate: Bool
    let meanConfidence: Float
    let assembledTextLength: Int
}

struct BucketReportJSON: Codable {
    let bucket: String
    let images: Int
    let spinesRead: Int
    let passed: Int
    let passRate: Double
    let meanQualityScore: Double
    let meanConfidence: Double
    let readings: [SpineReadingJSON]
}

var bucketOrder = ["original"] + angles.map { "rot\($0)" }
bucketOrder = bucketOrder.filter { buckets[$0] != nil }

var reports: [BucketReportJSON] = []
for bucketName in bucketOrder {
    let urls = Array((buckets[bucketName] ?? []).prefix(maxPerBucket))
    guard !urls.isEmpty else { continue }
    log("Bucket \(bucketName): \(urls.count) image(s)")

    var readings: [SpineReadingJSON] = []
    for url in urls {
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            log("  warning: could not read \(url.lastPathComponent)")
            continue
        }
        let detections: [OBBDetection]
        do {
            let result = try detector.predict(cgImage, options: detectionOptions)
            detections = result.alreadyNMSed
                ? Array(result.detections.prefix(maxDetections))
                : nmsRotated(result.detections, iouThreshold: iouThreshold, maxDetections: maxDetections)
        } catch {
            log("  warning: inference failed for \(url.lastPathComponent): \(error)")
            continue
        }
        for det in detections {
            guard let crop = uprightWarp(of: det, in: cgImage),
                  let result = try? router.recognize(crop: crop, detection: det) else { continue }
            readings.append(SpineReadingJSON(
                qualityScore: result.qualityScore, passedQualityGate: result.passedQualityGate,
                meanConfidence: result.winningPass.meanConfidence, assembledTextLength: result.assembledText.count
            ))
        }
    }

    let passed = readings.filter(\.passedQualityGate).count
    let meanScore = readings.isEmpty ? 0 : readings.map(\.qualityScore).reduce(0, +) / Double(readings.count)
    let meanConf = readings.isEmpty ? 0 : Double(readings.map(\.meanConfidence).reduce(0, +)) / Double(readings.count)
    let passRate = readings.isEmpty ? 0 : Double(passed) / Double(readings.count)
    log("  spines read: \(readings.count)  passed: \(passed) (\(String(format: "%.1f%%", passRate * 100)))  mean score: \(String(format: "%.3f", meanScore))")

    reports.append(BucketReportJSON(
        bucket: bucketName, images: urls.count, spinesRead: readings.count, passed: passed,
        passRate: passRate, meanQualityScore: meanScore, meanConfidence: meanConf, readings: readings
    ))
}

// MARK: - Summary table + JSON output

log("")
log("bucket      images  spines  passed  pass%   meanScore  meanConf")
for r in reports {
    let namePadded = r.bucket.padding(toLength: 10, withPad: " ", startingAt: 0)
    let numbers = String(
        format: "%6d  %6d  %6d  %5.1f   %9.3f  %8.3f",
        r.images, r.spinesRead, r.passed, r.passRate * 100, r.meanQualityScore, r.meanConfidence
    )
    log("\(namePadded)  \(numbers)")
}

struct PayloadJSON: Codable {
    let imagesDir: String
    let model: String
    let acceptScoreThreshold: Double
    let buckets: [BucketReportJSON]
}
let payload = PayloadJSON(
    imagesDir: imagesDirURL.path, model: modelPath,
    acceptScoreThreshold: qualityGate.acceptScoreThreshold, buckets: reports
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
