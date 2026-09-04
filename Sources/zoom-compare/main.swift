import AppKit
import CoreGraphics
import Foundation
import SpineCore
import UniformTypeIdentifiers

// Stage 0 comparison harness for the jigsaw-zoom plan
// (~/dev/book-id-design/docs/jigsaw-zoom-requirements.md §A-8).
//
// Runs, per scene in the manifest:
//   1. single-shot   — one full-image predict + rotated NMS (the floor)
//   2. layout-crops  — denseShelfDetect with denseThreshold=0 (always
//                      jigsaw), the same code path as `layout-crops
//                      --infer-crops` and the SpinePipeline dense path
//   3. jigsaw-zoom   — recursive leaf-only detect (JigsawZoomEngine), once
//                      per `--cutter` selection: v1 is the `planCrops` quad
//                      adapter, v2 the free-form staircase cutter, `both`
//                      runs the Stage 3 cutter A/B in one pass
//
// Scoring: count-based recall proxy against hand-estimated per-scene spine
// counts (`gt_est` in the manifest; no trusted box ground truth exists),
// plus pairwise greedy rotated-IoU matching between pipelines. Outputs
// per-run JSON, a markdown summary table, and per-pipeline overlays with
// unique-to-this-pipeline detections highlighted.
//
// Outputs go under $BOOK_SPINES_DATA/eval/ (never committed).

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

// MARK: - CLI

var manifestPath = "tools/zoom_compare_scenes.json"
var modelPath = defaultModelURL().path
var imgsz = 1024
var confThreshold: Float = 0.15
var iouThreshold = 0.45
var maxDetections = 500
var matchIoU = 0.5
var outDir: String?
var writeOverlays = true
var onlyScene: String?
var zoomCutters: [ZoomCutterKind] = ZoomCutterKind.allCases

let cliArgs = CommandLine.arguments
var argIndex = 1
func nextArg(_ args: [String], _ i: inout Int) -> String {
    i += 1
    guard i < args.count else { fail("Missing value for \(args[i - 1])") }
    return args[i]
}
while argIndex < cliArgs.count {
    switch cliArgs[argIndex] {
    case "--manifest": manifestPath = nextArg(cliArgs, &argIndex)
    case "--model": modelPath = nextArg(cliArgs, &argIndex)
    case "--imgsz": imgsz = Int(nextArg(cliArgs, &argIndex)) ?? imgsz
    case "--conf": confThreshold = Float(nextArg(cliArgs, &argIndex)) ?? confThreshold
    case "--iou": iouThreshold = Double(nextArg(cliArgs, &argIndex)) ?? iouThreshold
    case "--match-iou": matchIoU = Double(nextArg(cliArgs, &argIndex)) ?? matchIoU
    case "--max-det": maxDetections = Int(nextArg(cliArgs, &argIndex)) ?? maxDetections
    case "--out": outDir = nextArg(cliArgs, &argIndex)
    case "--no-overlays": writeOverlays = false
    case "--only": onlyScene = nextArg(cliArgs, &argIndex)
    case "--cutter":
        let value = nextArg(cliArgs, &argIndex).lowercased()
        if value == "both" {
            zoomCutters = ZoomCutterKind.allCases
        } else if let kind = ZoomCutterKind(rawValue: value) {
            zoomCutters = [kind]
        } else {
            fail("--cutter expects v1, v2 or both (got \(value))")
        }
    default: fail("Unknown argument: \(cliArgs[argIndex])")
    }
    argIndex += 1
}

// MARK: - Manifest

struct SceneEntry: Codable {
    let image: String
    let gtEst: Int

    enum CodingKeys: String, CodingKey {
        case image
        case gtEst = "gt_est"
    }
}

struct Manifest: Codable {
    let scenesDir: String
    let scenes: [SceneEntry]

    enum CodingKeys: String, CodingKey {
        case scenesDir = "scenes_dir"
        case scenes
    }
}

let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
guard let manifestData = try? Data(contentsOf: manifestURL),
      let manifest = try? JSONDecoder().decode(Manifest.self, from: manifestData) else {
    fail("Could not read manifest: \(manifestURL.path)")
}
let scenesDir = URL(fileURLWithPath: NSString(string: manifest.scenesDir).expandingTildeInPath, isDirectory: true)

// MARK: - Output dir

func defaultOutDir() -> URL {
    let root = ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
        ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
    formatter.timeZone = TimeZone(identifier: "UTC")
    return URL(fileURLWithPath: root)
        .appendingPathComponent("eval", isDirectory: true)
        .appendingPathComponent("zoom-compare-\(formatter.string(from: Date()))", isDirectory: true)
}

let outURL = outDir.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? defaultOutDir()
try? FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

// MARK: - Model

let modelURL = URL(fileURLWithPath: modelPath)
print("Model: \(modelURL.path)")
let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: modelURL, computeUnits: .all)
} catch {
    fail("\(error)")
}
let detectionOptions = DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections)

// MARK: - Report shapes

/// The per-photo numbers A-8 asks every jigsaw-zoom run to report:
/// seam-dedup rate, top-level drop rate, recursion depth histogram, and the
/// A-7 run invariants.
struct ZoomTelemetryJSON: Codable {
    let cutter: String
    let leafCount: Int
    let fallbackLeafCount: Int
    let dedupHits: Int
    let topLevelDrops: Int
    let depthHistogram: [String: Int]
    let runRulesOk: Bool
    let runRules: [RuleResult]

    enum CodingKeys: String, CodingKey {
        case cutter
        case leafCount = "leaf_count"
        case fallbackLeafCount = "fallback_leaf_count"
        case dedupHits = "dedup_hits"
        case topLevelDrops = "top_level_drops"
        case depthHistogram = "depth_histogram"
        case runRulesOk = "run_rules_ok"
        case runRules = "run_rules"
    }
}

struct PipelineReportJSON: Codable {
    let name: String
    let score: SceneScore
    let inferencePasses: Int
    let wallMs: Double
    let usedJigsaw: Bool
    let plannedCrops: Int
    let newAfterMerge: Int
    let zoom: ZoomTelemetryJSON?
    let detections: [OBBDetectionJSON]

    enum CodingKeys: String, CodingKey {
        case name, score, zoom
        case inferencePasses = "inference_passes"
        case wallMs = "wall_ms"
        case usedJigsaw = "used_jigsaw"
        case plannedCrops = "planned_crops"
        case newAfterMerge = "new_after_merge"
        case detections
    }
}

struct PairwiseJSON: Codable {
    let a: String
    let b: String
    let matched: Int
    let onlyA: Int
    let onlyB: Int

    enum CodingKeys: String, CodingKey {
        case a, b, matched
        case onlyA = "only_a"
        case onlyB = "only_b"
    }
}

struct SceneReportJSON: Codable {
    let image: String
    let width: Int
    let height: Int
    let gtEst: Int
    let pipelines: [PipelineReportJSON]
    let pairwise: [PairwiseJSON]

    enum CodingKeys: String, CodingKey {
        case image, width, height
        case gtEst = "gt_est"
        case pipelines, pairwise
    }
}

struct RunReportJSON: Codable {
    let model: String
    let imgsz: Int
    let conf: Float
    let iou: Double
    let matchIoU: Double
    let cutters: [String]
    let scenes: [SceneReportJSON]

    enum CodingKeys: String, CodingKey {
        case model, imgsz, conf, iou, cutters
        case matchIoU = "match_iou"
        case scenes
    }
}

// MARK: - Overlay

func writePNG(_ image: CGImage, to url: URL) -> Bool {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
}

let cropEdgePalette: [CGColor] = [
    CGColor(red: 1.0, green: 220 / 255, blue: 0, alpha: 1),
    CGColor(red: 128 / 255, green: 1.0, blue: 0, alpha: 1),
    CGColor(red: 0, green: 180 / 255, blue: 1.0, alpha: 1),
    CGColor(red: 200 / 255, green: 0, blue: 1.0, alpha: 1),
    CGColor(red: 1.0, green: 80 / 255, blue: 80 / 255, alpha: 1),
    CGColor(red: 1.0, green: 160 / 255, blue: 0, alpha: 1),
    CGColor(red: 0, green: 1.0, blue: 200 / 255, alpha: 1),
    CGColor(red: 100 / 255, green: 100 / 255, blue: 1.0, alpha: 1),
]

func drawOverlay(
    scene: CGImage,
    dets: [OBBDetection],
    uniqueIndices: Set<Int>,
    cropQuads: [[CGPoint]],
    maxSide: Int = 2560
) -> CGImage? {
    let w = scene.width, h = scene.height
    let scale = min(1.0, Double(maxSide) / Double(max(w, h)))
    let outW = max(1, Int((Double(w) * scale).rounded()))
    let outH = max(1, Int((Double(h) * scale).rounded()))
    guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: nil, width: outW, height: outH, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.scaleBy(x: CGFloat(scale), y: CGFloat(scale))
    ctx.interpolationQuality = .high
    ctx.draw(scene, in: CGRect(x: 0, y: 0, width: w, height: h))
    func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: CGFloat(h) - p.y) }
    let lineWidth = max(2.0, ((Double(w) + Double(h)) / 2 * 0.002).rounded())

    let sharedColor = CGColor(red: 4 / 255, green: 42 / 255, blue: 1.0, alpha: 1)
    let uniqueColor = CGColor(red: 1.0, green: 0, blue: 1.0, alpha: 1)
    for (i, det) in dets.enumerated() {
        let isUnique = uniqueIndices.contains(i)
        ctx.setLineWidth(isUnique ? lineWidth + 1 : lineWidth)
        ctx.setStrokeColor(isUnique ? uniqueColor : sharedColor)
        let quad = det.corners.map(flip)
        ctx.move(to: quad[0])
        for p in quad.dropFirst() { ctx.addLine(to: p) }
        ctx.closePath()
        ctx.strokePath()
    }

    for (i, quad) in cropQuads.enumerated() {
        let pts = quad.filter { $0.x.isFinite && $0.y.isFinite }.map(flip)
        guard pts.count >= 3 else { continue }
        ctx.setLineWidth(lineWidth + 2)
        ctx.setStrokeColor(cropEdgePalette[i % cropEdgePalette.count])
        ctx.move(to: pts[0])
        for p in pts.dropFirst() { ctx.addLine(to: p) }
        ctx.closePath()
        ctx.strokePath()
    }
    return ctx.makeImage()
}

// MARK: - Run

var sceneReports: [SceneReportJSON] = []
var markdownRows: [String] = []

for entry in manifest.scenes {
    if let onlyScene, !entry.image.contains(onlyScene) { continue }
    let imageURL = scenesDir.appendingPathComponent(entry.image)
    guard let nsImage = NSImage(contentsOf: imageURL),
          let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("skip (unreadable): \(imageURL.path)")
        continue
    }
    let imgW = cgImage.width, imgH = cgImage.height
    print("\n=== \(entry.image) (\(imgW)x\(imgH), gt_est=\(entry.gtEst)) ===")

    let predict: (CGImage) throws -> InferenceResult = { try detector.predict($0, options: detectionOptions) }

    func runPipeline(name: String, denseThreshold: Int) throws -> (PipelineReportJSON, [OBBDetection], [[CGPoint]]) {
        let start = Date()
        let result = try denseShelfDetect(
            image: cgImage, predict: predict, options: detectionOptions,
            denseOptions: DenseShelfDetectionOptions(denseThreshold: denseThreshold, imgsz: imgsz)
        )
        let wallMs = Date().timeIntervalSince(start) * 1000
        let passes = 1 + (result.usedJigsaw ? result.plannedCropCount : 0)
        let report = PipelineReportJSON(
            name: name,
            score: sceneScore(dets: result.detections, gtEst: entry.gtEst),
            inferencePasses: passes,
            wallMs: (wallMs * 10).rounded() / 10,
            usedJigsaw: result.usedJigsaw,
            plannedCrops: result.plannedCropCount,
            newAfterMerge: result.newDetectionCount,
            zoom: nil,
            detections: result.detections.map(OBBDetectionJSON.init)
        )
        return (report, result.detections, result.cropQuads)
    }

    func runZoom(cutter: ZoomCutterKind) throws -> (PipelineReportJSON, [OBBDetection], [[CGPoint]]) {
        let zoomOptions = JigsawZoomOptions(imgsz: imgsz, cutter: cutter)
        let start = Date()
        let result = try jigsawZoomDetect(
            image: cgImage, predict: predict, options: detectionOptions,
            zoomOptions: zoomOptions
        )
        let wallMs = Date().timeIntervalSince(start) * 1000
        let runRules = verifyZoomRun(result, downsampleThreshold: zoomOptions.downsampleThreshold)
        var histogram: [String: Int] = [:]
        for leaf in result.leaves { histogram["d\(leaf.depth)", default: 0] += 1 }
        let report = PipelineReportJSON(
            name: "zoom-\(cutter.rawValue)",
            score: sceneScore(dets: result.detections, gtEst: entry.gtEst),
            inferencePasses: result.inferencePasses,
            wallMs: (wallMs * 10).rounded() / 10,
            usedJigsaw: result.usedRecursion,
            plannedCrops: result.plannedCropCount,
            newAfterMerge: result.topLevelDropCount,
            zoom: ZoomTelemetryJSON(
                cutter: cutter.rawValue,
                leafCount: result.leafCount,
                fallbackLeafCount: result.fallbackLeafCount,
                dedupHits: result.dedupHitCount,
                topLevelDrops: result.topLevelDropCount,
                depthHistogram: histogram,
                runRulesOk: hardRulesOK(runRules),
                runRules: runRules
            ),
            detections: result.detections.map(OBBDetectionJSON.init)
        )
        return (report, result.detections, result.cropQuads)
    }

    do {
        let (singleReport, singleDets, _) = try runPipeline(name: "single", denseThreshold: Int.max)
        let (layoutReport, layoutDets, _) = try runPipeline(name: "layout", denseThreshold: 0)
        var zoomRuns: [(kind: ZoomCutterKind, report: PipelineReportJSON, dets: [OBBDetection])] = []
        for kind in zoomCutters {
            let (report, dets, _) = try runZoom(cutter: kind)
            zoomRuns.append((kind, report, dets))
        }

        let matchSL = matchDetections(singleDets, layoutDets, iouThreshold: matchIoU)
        var pairwise = [
            PairwiseJSON(a: "single", b: "layout", matched: matchSL.pairs.count, onlyA: matchSL.onlyA.count, onlyB: matchSL.onlyB.count),
        ]
        // A-8's primary comparison: complete incumbent vs complete new, per cutter.
        for run in zoomRuns {
            let m = matchDetections(layoutDets, run.dets, iouThreshold: matchIoU)
            pairwise.append(PairwiseJSON(a: "layout", b: run.report.name, matched: m.pairs.count, onlyA: m.onlyA.count, onlyB: m.onlyB.count))
        }
        // Stage 3's cutter A/B, when both ran.
        if zoomRuns.count == 2 {
            let m = matchDetections(zoomRuns[0].dets, zoomRuns[1].dets, iouThreshold: matchIoU)
            pairwise.append(PairwiseJSON(a: zoomRuns[0].report.name, b: zoomRuns[1].report.name, matched: m.pairs.count, onlyA: m.onlyA.count, onlyB: m.onlyB.count))
        }

        if writeOverlays, let last = zoomRuns.last {
            let stem = (entry.image as NSString).deletingPathExtension
            func writeOverlay(_ name: String, dets: [OBBDetection], unique: Set<Int>, quads: [[CGPoint]]) {
                autoreleasepool {
                    guard let img = drawOverlay(scene: cgImage, dets: dets, uniqueIndices: unique, cropQuads: quads) else { return }
                    _ = writePNG(img, to: outURL.appendingPathComponent("\(stem).\(name).png"))
                }
            }
            // One full-scene bitmap per photo: a second 5k overlay in the
            // same process has been SIGSEGVing after the first write.
            writeOverlay("\(last.report.name)-n50", dets: last.dets.filter { $0.conf >= 0.50 }, unique: [], quads: [])
        }

        sceneReports.append(SceneReportJSON(
            image: entry.image, width: imgW, height: imgH, gtEst: entry.gtEst,
            pipelines: [singleReport, layoutReport] + zoomRuns.map(\.report), pairwise: pairwise
        ))

        func fmt(_ r: PipelineReportJSON) -> String {
            String(format: "%d/%d/%d rp15=%.2f", r.score.count15, r.score.count50, r.score.count70, r.score.recallProxy15)
        }
        print("  single: \(fmt(singleReport))  (\(String(format: "%.0f", singleReport.wallMs)) ms)")
        print("  layout: \(fmt(layoutReport))  +\(layoutReport.newAfterMerge) new, \(layoutReport.inferencePasses) passes (\(String(format: "%.0f", layoutReport.wallMs)) ms)")
        for run in zoomRuns {
            let t = run.report.zoom
            print("  \(run.report.name): \(fmt(run.report))  drops=\(run.report.newAfterMerge) dedup=\(t?.dedupHits ?? 0) "
                + "leaves=\(t?.leafCount ?? 0) (a4=\(t?.fallbackLeafCount ?? 0)) \(run.report.inferencePasses) passes "
                + "(\(String(format: "%.0f", run.report.wallMs)) ms) rules_ok=\(t?.runRulesOk ?? false)")
        }
        for p in pairwise.dropFirst() {
            print("  \(p.a)↔\(p.b)@\(matchIoU): matched=\(p.matched) only_\(p.a)=\(p.onlyA) only_\(p.b)=\(p.onlyB)")
        }

        var row = "| \(entry.image) | \(entry.gtEst) "
            + "| \(singleReport.score.count15) | \(singleReport.score.count50) "
            + "| \(layoutReport.score.count15) | \(layoutReport.score.count50) | \(layoutReport.inferencePasses) "
        for run in zoomRuns {
            row += "| \(run.report.score.count15) | \(run.report.score.count50) | \(run.report.inferencePasses) "
                + "| \(run.report.zoom?.fallbackLeafCount ?? 0) | \(run.report.zoom?.dedupHits ?? 0) "
        }
        if let lastPair = pairwise.dropFirst().first {
            row += "| \(lastPair.matched) | \(lastPair.onlyA) | \(lastPair.onlyB) "
        }
        markdownRows.append(row + "|")
    } catch {
        print("  ERROR: \(error)")
    }
}

// MARK: - Write reports

let runReport = RunReportJSON(
    model: modelURL.path, imgsz: imgsz, conf: confThreshold, iou: iouThreshold,
    matchIoU: matchIoU, cutters: zoomCutters.map(\.rawValue), scenes: sceneReports
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
do {
    let data = try encoder.encode(runReport)
    try data.write(to: outURL.appendingPathComponent("report.json"))
} catch {
    fail("Could not write report.json: \(error)")
}

var header = "| scene | gt_est | single n15 | single n50 | layout n15 | layout n50 | layout passes "
for kind in zoomCutters {
    let name = "zoom-\(kind.rawValue)"
    header += "| \(name) n15 | \(name) n50 | \(name) passes | \(name) a4 | \(name) dedup "
}
let firstZoomName = "zoom-\(zoomCutters[0].rawValue)"
header += "| layout∩\(firstZoomName) | only layout | only \(firstZoomName) |"
let separator = "|" + String(repeating: "---|", count: header.components(separatedBy: "|").count - 2)

var markdown = """
# zoom-compare (single / layout / jigsaw-zoom)

Model: `\(modelURL.lastPathComponent)`  imgsz=\(imgsz) conf=\(confThreshold) iou=\(iouThreshold) match_iou=\(matchIoU) cutters=\(zoomCutters.map(\.rawValue).joined(separator: ","))

`a4` = leaves accepted without reaching the resolution threshold (no legal cut);
`dedup` = leaf detections merged by the global rotated NMS, i.e. the seam-damage estimate (A-2).

\(header)
\(separator)

"""
markdown = markdown.trimmingCharacters(in: .newlines) + "\n" + markdownRows.joined(separator: "\n") + "\n"
try? markdown.write(to: outURL.appendingPathComponent("report.md"), atomically: true, encoding: .utf8)

print("\nWrote \(outURL.path)")
