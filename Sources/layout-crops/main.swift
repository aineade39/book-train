import AppKit
import CoreGraphics
import CoreText
import Foundation
import SpineCore
import UniformTypeIdentifiers

// Full-scene jigsaw crop planner CLI — Swift/Core ML successor to
// `tools/layout_crop_predict.py`, sharing its pipeline shape:
//
//   image -> Core ML first pass -> planCrops -> verifyPlan (hard rules) ->
//   optional: warp quad + pad -> per-crop Core ML -> global rotated NMS ->
//   JSON summary + optional scene overlay
//
// Exit code 0 if every hard rule passes, 2 otherwise (crop re-inference is
// skipped whenever a hard rule fails).

func log(_ message: String) {
    print(message)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

// MARK: - CLI flags (mirrors `tools/layout_crop_predict.py`'s argparse)

var imagePath: String?
var modelPath: String = defaultModelURL().path
var imgsz = 1024
var confThreshold: Float = 0.15
var iouThreshold: Double = 0.45
var maxDetections = 500
var angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg
var rowGapK = 0.2
var colGapK = 1.0
var minBlockMembers = 2
var maxCropDimK = 1.5
var padValue: UInt8 = 114
var planOnly = false
var inferCrops = false
var writeCropsDir: String?
var overlayPlan = false
var overlayCrops = false
var overlayDets = false
var outPath: String?
var jsonPath: String?

func nextArg(_ args: [String], _ i: inout Int) -> String {
    i += 1
    guard i < args.count else { fail("Missing value for \(args[i - 1])") }
    return args[i]
}

let cliArgs = CommandLine.arguments
var argIndex = 1
while argIndex < cliArgs.count {
    let arg = cliArgs[argIndex]
    switch arg {
    case "--model": modelPath = nextArg(cliArgs, &argIndex)
    case "--imgsz": imgsz = Int(nextArg(cliArgs, &argIndex)) ?? imgsz
    case "--conf": confThreshold = Float(nextArg(cliArgs, &argIndex)) ?? confThreshold
    case "--iou": iouThreshold = Double(nextArg(cliArgs, &argIndex)) ?? iouThreshold
    case "--max-det": maxDetections = Int(nextArg(cliArgs, &argIndex)) ?? maxDetections
    case "--angle-tol-deg": angleTolDeg = Double(nextArg(cliArgs, &argIndex)) ?? angleTolDeg
    case "--row-gap-k": rowGapK = Double(nextArg(cliArgs, &argIndex)) ?? rowGapK
    case "--col-gap-k": colGapK = Double(nextArg(cliArgs, &argIndex)) ?? colGapK
    case "--min-block-members": minBlockMembers = Int(nextArg(cliArgs, &argIndex)) ?? minBlockMembers
    case "--max-crop-dim-k": maxCropDimK = Double(nextArg(cliArgs, &argIndex)) ?? maxCropDimK
    case "--pad-value": padValue = UInt8(Int(nextArg(cliArgs, &argIndex)) ?? Int(padValue))
    case "--plan-only": planOnly = true
    case "--infer-crops": inferCrops = true
    case "--write-crops": writeCropsDir = nextArg(cliArgs, &argIndex)
    case "--overlay-plan": overlayPlan = true
    case "--overlay-crops": overlayCrops = true
    case "--overlay-dets": overlayDets = true
    case "--out": outPath = nextArg(cliArgs, &argIndex)
    case "--json": jsonPath = nextArg(cliArgs, &argIndex)
    default:
        if imagePath == nil { imagePath = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let imagePath else {
    fail("""
        Usage: swift run layout-crops <image> [--model path.mlpackage] [--imgsz 1024] [--conf 0.15] [--iou 0.45] [--max-det 500] \
        [--angle-tol-deg 25] [--row-gap-k 0.2] [--col-gap-k 1.0] [--min-block-members 2] [--max-crop-dim-k 1.5] [--pad-value 114] \
        [--plan-only] [--infer-crops] [--write-crops <dir>] [--overlay-plan] [--overlay-crops] [--overlay-dets] [--out <path>] [--json <path>]
        """)
}

// MARK: - Load image

let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
guard FileManager.default.fileExists(atPath: imageURL.path) else {
    fail("Image not found: \(imageURL.path)")
}
guard let nsImage = NSImage(contentsOf: imageURL),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fail("Could not read image: \(imageURL.path)")
}
let imgW = cgImage.width
let imgH = cgImage.height
log("Image: \(imgW)x\(imgH)")

guard let raster = SceneRaster(cgImage: cgImage) else {
    fail("Could not decode scene raster: \(imageURL.path)")
}

// MARK: - Model + first pass

let modelURL = URL(fileURLWithPath: modelPath)
log("Loading model: \(modelURL.path)")
let detector: SpineDetector
do {
    detector = try SpineDetector(modelURL: modelURL, computeUnits: .all)
} catch {
    fail("\(error)")
}

log("Running first-pass detection...")
let detectionOptions = DetectionOptions(confidenceThreshold: confThreshold, iouThreshold: iouThreshold, maxDetections: maxDetections)
var first: [OBBDetection]
do {
    let result = try detector.predict(cgImage, options: detectionOptions)
    first = result.alreadyNMSed
        ? Array(result.detections.prefix(maxDetections))
        : nmsRotated(result.detections, iouThreshold: iouThreshold, maxDetections: maxDetections)
} catch {
    fail("Inference failed: \(error)")
}
// Drop detections whose center lies outside the image (common OBB edge noise).
first = first.filter { $0.cx >= 0 && $0.cx <= Double(imgW) && $0.cy >= 0 && $0.cy <= Double(imgH) }
log("first-pass: \(first.count) spines (in-frame)")

// MARK: - Plan + verify

let plans = planCrops(
    dets: first, imgW: imgW, imgH: imgH, raster: raster,
    angleTolDeg: angleTolDeg, rowGapK: rowGapK, colGapK: colGapK,
    minBlockMembers: minBlockMembers, imgsz: imgsz, maxCropDimK: maxCropDimK
)
let ruleResults = verifyPlan(dets: first, plans: plans, imgW: imgW, imgH: imgH, angleTolDeg: angleTolDeg)
let rulesOk = hardRulesOK(ruleResults)
let emptyCells = plans.filter(\.memberIndices.isEmpty).count
log("planned crops: \(plans.count) (rules_ok=\(rulesOk); empty cells=\(emptyCells))")
for r in ruleResults {
    let flag = r.ok ? "OK" : (r.hard ? "FAIL" : "WARN")
    log("  [\(flag)] \(r.rule): \(r.detail)")
}
for p in plans {
    let rect = p.rect
    let angleStr = String(format: "%g", p.angleDeg)
    log("  \(p.name): ~\(Int((rect.x1 - rect.x0).rounded()))x\(Int((rect.y1 - rect.y0).rounded())) @(\(Int(rect.x0.rounded())),\(Int(rect.y0.rounded()))) angle\u{2248}\(angleStr)\u{00B0} members=\(p.memberIndices.count)")
}

// MARK: - Materialize crops (warp + pad [+ write] [+ re-infer])

let doInfer = inferCrops && !planOnly && !overlayPlan
let writeDir: URL? = overlayPlan ? nil : writeCropsDir.map { URL(fileURLWithPath: $0).standardizedFileURL }
if let writeDir {
    try? FileManager.default.createDirectory(at: writeDir, withIntermediateDirectories: true)
}
let needMaterialize = writeDir != nil || doInfer

struct CropMetaJSON: Codable {
    let name: String
    let shelfId: Int
    let blockId: Int
    let angleDeg: Double
    let quad: [[Double]]
    let rect: [Double]
    let members: Int
    let warpedSize: [Int]?
    let warpScale: Double?
    let gain: Double?
    let pad: [Double]?
    let path: String?
    let cropDetections: Int?

    enum CodingKeys: String, CodingKey {
        case name
        case shelfId = "shelf_id"
        case blockId = "block_id"
        case angleDeg = "angle_deg"
        case quad, rect, members
        case warpedSize = "warped_size"
        case warpScale = "warp_scale"
        case gain, pad, path
        case cropDetections = "crop_detections"
    }
}

func round2(_ v: Double) -> Double { (v * 100).rounded() / 100 }

var cropDets: [OBBDetection] = []
var cropMeta: [CropMetaJSON] = []

if needMaterialize {
    for plan in plans {
        guard let warped = warpQuad(raster, quad: plan.quad, maxSide: imgsz, padValue: padValue) else {
            log("  warning: could not warp \(plan.name)")
            continue
        }
        guard let padded = padNoUpsize(warped.image, canvas: imgsz, padValue: padValue) else {
            log("  warning: could not pad \(plan.name)")
            continue
        }

        var cropDetCount: Int?
        var path: String?

        if let writeDir {
            let outURL = writeDir.appendingPathComponent("\(plan.name).png")
            if writePNG(padded.image, to: outURL) {
                path = outURL.path
                log("  wrote \(outURL.lastPathComponent) (\(warped.outputWidth)x\(warped.outputHeight) -> \(imgsz))")
            }
        }

        if doInfer {
            do {
                let raw = try detector.predict(padded.image, options: detectionOptions)
                var mapped: [OBBDetection] = []
                for d in raw.detections {
                    if let md = mapDetFromCrop(d, homography: warped.homography, gain: padded.gain, padX: padded.padX, padY: padded.padY),
                       md.w > 1, md.h > 1 {
                        mapped.append(md)
                    }
                }
                cropDetCount = mapped.count
                cropDets.append(contentsOf: mapped)
            } catch {
                log("  warning: crop inference failed for \(plan.name): \(error)")
            }
        }

        let rect = plan.rect
        cropMeta.append(CropMetaJSON(
            name: plan.name, shelfId: plan.shelfId, blockId: plan.blockId, angleDeg: plan.angleDeg,
            quad: plan.quad.map { [round2(Double($0.x)), round2(Double($0.y))] },
            rect: [round2(rect.x0), round2(rect.y0), round2(rect.x1), round2(rect.y1)],
            members: plan.memberIndices.count,
            warpedSize: [warped.outputWidth, warped.outputHeight],
            warpScale: (warped.warpScale * 1e5).rounded() / 1e5,
            gain: (padded.gain * 1e5).rounded() / 1e5,
            pad: [round2(padded.padX), round2(padded.padY)],
            path: path,
            cropDetections: cropDetCount
        ))
    }
} else {
    for plan in plans {
        let rect = plan.rect
        cropMeta.append(CropMetaJSON(
            name: plan.name, shelfId: plan.shelfId, blockId: plan.blockId, angleDeg: plan.angleDeg,
            quad: plan.quad.map { [round2(Double($0.x)), round2(Double($0.y))] },
            rect: [round2(rect.x0), round2(rect.y0), round2(rect.x1), round2(rect.y1)],
            members: plan.memberIndices.count,
            warpedSize: nil, warpScale: nil, gain: nil, pad: nil, path: nil, cropDetections: nil
        ))
    }
}

// MARK: - Merge + partition net-new detections

var merged = first
var newDets: [OBBDetection] = []
if doInfer {
    merged = nmsRotated(first + cropDets, iouThreshold: iouThreshold, maxDetections: maxDetections)
    let mergedIds = Set(merged.map(\.id))
    let firstIds = Set(first.map(\.id))
    newDets = merged.filter { !firstIds.contains($0.id) && mergedIds.contains($0.id) }
    log("crop-pass raw: \(cropDets.count)  merged: \(merged.count)  new: \(newDets.count)  delta: \(newDets.count >= 0 ? "+" : "")\(newDets.count)")
}

// MARK: - Overlay

func writePNG(_ image: CGImage, to url: URL) -> Bool {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
}

func drawLabel(_ text: String, at point: CGPoint, color: CGColor, in ctx: CGContext) {
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 15, nil)
    let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: color]
    let attrStr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
    let line = CTLineCreateWithAttributedString(attrStr)
    ctx.textPosition = point
    CTLineDraw(line, ctx)
}

let overlayPalette: [CGColor] = [
    CGColor(red: 1.0, green: 220 / 255, blue: 0, alpha: 1),
    CGColor(red: 128 / 255, green: 1.0, blue: 0, alpha: 1),
    CGColor(red: 0, green: 180 / 255, blue: 1.0, alpha: 1),
    CGColor(red: 200 / 255, green: 0, blue: 1.0, alpha: 1),
    CGColor(red: 1.0, green: 80 / 255, blue: 80 / 255, alpha: 1),
    CGColor(red: 1.0, green: 160 / 255, blue: 0, alpha: 1),
    CGColor(red: 0, green: 1.0, blue: 200 / 255, alpha: 1),
    CGColor(red: 100 / 255, green: 100 / 255, blue: 1.0, alpha: 1),
]

func drawOverlay(scene: CGImage, plans: [CropPlan], dets: [OBBDetection]?, drawDets: Bool, newDetIds: Set<UUID>?) -> CGImage? {
    let w = scene.width, h = scene.height
    guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(scene, in: CGRect(x: 0, y: 0, width: w, height: h))
    func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: CGFloat(h) - p.y) }

    let lineWidth = max(2.0, ((Double(w) + Double(h)) / 2 * 0.002).rounded())

    if drawDets, let dets {
        let firstColor = CGColor(red: 4 / 255, green: 42 / 255, blue: 1.0, alpha: 1)
        let newColor = CGColor(red: 1.0, green: 0, blue: 1.0, alpha: 1)
        for det in dets {
            let isNew = newDetIds?.contains(det.id) ?? false
            ctx.setLineWidth(isNew ? lineWidth + 1 : lineWidth)
            ctx.setStrokeColor(isNew ? newColor : firstColor)
            let quad = det.corners.map(flip)
            ctx.move(to: quad[0])
            for p in quad.dropFirst() { ctx.addLine(to: p) }
            ctx.closePath()
            ctx.strokePath()
        }
    }

    for (i, plan) in plans.enumerated() {
        let color = overlayPalette[i % overlayPalette.count]
        ctx.setLineWidth(lineWidth + 1)
        ctx.setStrokeColor(color)
        let quad = plan.quad.map(flip)
        ctx.move(to: quad[0])
        for p in quad.dropFirst() { ctx.addLine(to: p) }
        ctx.closePath()
        ctx.strokePath()

        let label = plan.memberIndices.isEmpty ? "\(plan.name)*" : plan.name
        let tl = flip(plan.quad[0])
        drawLabel(label, at: CGPoint(x: tl.x + 4, y: tl.y - 22), color: color, in: ctx)
    }

    return ctx.makeImage()
}

let drawDets = overlayDets && !overlayPlan
let wantOverlay = overlayPlan || overlayCrops || overlayDets || outPath != nil
if wantOverlay {
    let suffix = overlayPlan ? "plan" : (overlayCrops ? "crops" : "layout")
    let outURL = outPath.map { URL(fileURLWithPath: $0) }
        ?? imageURL.deletingLastPathComponent().appendingPathComponent("\(imageURL.deletingPathExtension().lastPathComponent).\(suffix).png")
    let newDetIds = (doInfer && drawDets) ? Set(newDets.map(\.id)) : nil
    if let overlay = drawOverlay(scene: cgImage, plans: plans, dets: drawDets ? merged : nil, drawDets: drawDets, newDetIds: newDetIds) {
        if writePNG(overlay, to: outURL) {
            log("Wrote overlay \(outURL.path)")
        } else {
            log("warning: could not write overlay to \(outURL.path)")
        }
    }
}

// MARK: - JSON summary

struct ImageSizeJSON: Codable { let width: Int; let height: Int }

struct PayloadJSON: Codable {
    let image: String
    let imageSize: ImageSizeJSON
    let weights: String
    let imgsz: Int
    let conf: Float
    let overlap: Int
    let jigsaw: Bool
    let rulesOk: Bool
    let rules: [RuleResult]
    let firstPassCount: Int
    let firstPass: [OBBDetectionJSON]
    let plannedCrops: Int
    let emptyCells: Int
    let crops: [CropMetaJSON]
    let mergedCount: Int
    let newAfterMergeCount: Int
    let cropPassRaw: Int
    let detectionsMerged: [OBBDetectionJSON]?

    enum CodingKeys: String, CodingKey {
        case image
        case imageSize = "image_size"
        case weights, imgsz, conf, overlap, jigsaw
        case rulesOk = "rules_ok"
        case rules
        case firstPassCount = "first_pass_count"
        case firstPass = "first_pass"
        case plannedCrops = "planned_crops"
        case emptyCells = "empty_cells"
        case crops
        case mergedCount = "merged_count"
        case newAfterMergeCount = "new_after_merge_count"
        case cropPassRaw = "crop_pass_raw"
        case detectionsMerged = "detections_merged"
    }
}

let payload = PayloadJSON(
    image: imageURL.path,
    imageSize: ImageSizeJSON(width: imgW, height: imgH),
    weights: modelURL.path,
    imgsz: imgsz,
    conf: confThreshold,
    overlap: 0,
    jigsaw: true,
    rulesOk: rulesOk,
    rules: ruleResults,
    firstPassCount: first.count,
    firstPass: first.map(OBBDetectionJSON.init),
    plannedCrops: plans.count,
    emptyCells: emptyCells,
    crops: cropMeta,
    mergedCount: merged.count,
    newAfterMergeCount: newDets.count,
    cropPassRaw: doInfer ? cropDets.count : 0,
    detectionsMerged: doInfer ? merged.map(OBBDetectionJSON.init) : nil
)

func defaultJSONPath(image: URL) -> URL {
    let root = ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
        ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
    let evalDir = URL(fileURLWithPath: root).appendingPathComponent("eval", isDirectory: true)
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
    formatter.timeZone = TimeZone(identifier: "UTC")
    let ts = formatter.string(from: Date())
    let stem = image.deletingPathExtension().lastPathComponent
    return evalDir.appendingPathComponent("\(stem)_layout_crops_\(ts).json")
}

let resolvedJSONPath = jsonPath.map { URL(fileURLWithPath: $0) } ?? defaultJSONPath(image: imageURL)
try? FileManager.default.createDirectory(at: resolvedJSONPath.deletingLastPathComponent(), withIntermediateDirectories: true)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
do {
    let data = try encoder.encode(payload)
    try data.write(to: resolvedJSONPath)
    log("Wrote \(resolvedJSONPath.path)")
} catch {
    fail("Could not write JSON summary: \(error)")
}

exit(rulesOk ? 0 : 2)
