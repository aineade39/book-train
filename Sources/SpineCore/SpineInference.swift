import CoreGraphics
import CoreML
import Foundation

// Core ML loading, Ultralytics-compatible letterboxing, OBB decoding, and
// rotated NMS. Extracted from the inference path in `bookspines.swift` so
// both the `bookspines` and `layout-crops` executables share one decoder.
//
// This file never calls `exit`; every failure path throws `SpineCoreError`
// so callers (CLI `main.swift`s) decide how to report and exit.

public enum SpineCoreError: Error, CustomStringConvertible {
    case modelNotFound(String)
    case modelHasNoImageInput
    case modelHasNoOutputs
    case modelCompileFailed(String)
    case modelLoadFailed(String)
    case imageEncodeFailed
    case predictionFailed(String)
    case noMultiArrayOutput(String)
    case unsupportedOutputLayout(shape: [Int])
    case letterboxFailed

    public var description: String {
        switch self {
        case .modelNotFound(let path): return "model not found at \(path)"
        case .modelHasNoImageInput: return "model has no image input"
        case .modelHasNoOutputs: return "model has no outputs"
        case .modelCompileFailed(let msg): return "could not compile model: \(msg)"
        case .modelLoadFailed(let msg): return "could not load model: \(msg)"
        case .imageEncodeFailed: return "could not encode image for model input"
        case .predictionFailed(let msg): return "inference failed: \(msg)"
        case .noMultiArrayOutput(let name): return "no multiarray output \"\(name)\""
        case .unsupportedOutputLayout(let shape): return "unsupported output layout, shape \(shape)"
        case .letterboxFailed: return "could not letterbox image"
        }
    }
}

/// How the Core ML multiarray is laid out.
public enum OBBOutputLayout: Equatable, CustomStringConvertible {
    case legacyChannelsFirst   // [1, 6, N]
    case end2endDetections     // [1, maxDet, 7]

    public var description: String {
        switch self {
        case .legacyChannelsFirst: return "legacyChannelsFirst"
        case .end2endDetections: return "end2endDetections"
        }
    }

    /// Detects layout from a 3-D output shape (mirrors `bookspines.swift`'s
    /// `detectOutputLayout`). Returns `nil` if the shape can't be classified
    /// (e.g. still symbolic) — the caller should keep its previous guess.
    public static func detect(shape: [Int]) -> OBBOutputLayout? {
        guard shape.count == 3 else { return nil }
        let a = shape[1], b = shape[2]
        if a == 6 { return .legacyChannelsFirst }
        if b == 7 { return .end2endDetections }
        if a == 7 { return .end2endDetections } // rare [1, 7, N] transpose
        return nil
    }
}

public struct DetectionOptions {
    public var confidenceThreshold: Float
    public var iouThreshold: Double
    public var maxDetections: Int

    public init(confidenceThreshold: Float = 0.15, iouThreshold: Double = 0.45, maxDetections: Int = 500) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.maxDetections = maxDetections
    }
}

/// Raw decode result for one Core ML call (one tile / one crop / one image).
public struct InferenceResult {
    /// Confidence-filtered, image-pixel-mapped detections in model/row order
    /// (already deduped by the model itself when `alreadyNMSed` is true).
    public let detections: [OBBDetection]
    public let layout: OBBOutputLayout
    /// True when the model's own graph already performs NMS (YOLO26 end2end).
    /// A single-tile, single-crop caller may skip an extra NMS pass; a
    /// caller merging multiple tiles/crops must still run a fresh rotated
    /// NMS over the merged candidates regardless of this flag.
    public let alreadyNMSed: Bool
    public let inferenceMs: Double
}

// MARK: - Model cache

/// `MLModel.compileModel(at:)` recompiles into a fresh temp directory every
/// call. Cache the compiled `.mlmodelc` next to a modification-date check so
/// repeat runs against the same `.mlpackage` skip recompilation entirely.
/// Uses the same cache directory name as the original `bookspines.swift` so
/// existing warm caches on disk are reused.
public func cachedCompiledModelURL(for sourceURL: URL) throws -> URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("bookspines-model-cache", isDirectory: true)
    try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    let cachedName = sourceURL.lastPathComponent.replacingOccurrences(of: ".mlpackage", with: "") + ".mlmodelc"
    let cachedURL = cacheDir.appendingPathComponent(cachedName)

    let sourceModified = (try? sourceURL.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
    let cachedModified = (try? cachedURL.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate

    if FileManager.default.fileExists(atPath: cachedURL.path),
       let sourceModified, let cachedModified, cachedModified >= sourceModified {
        return cachedURL
    }

    let compiled: URL
    do {
        compiled = try MLModel.compileModel(at: sourceURL)
    } catch {
        throw SpineCoreError.modelCompileFailed(error.localizedDescription)
    }
    if FileManager.default.fileExists(atPath: cachedURL.path) {
        try FileManager.default.removeItem(at: cachedURL)
    }
    try FileManager.default.copyItem(at: compiled, to: cachedURL)
    return cachedURL
}

/// Default promoted export under `$BOOK_SPINES_DATA` (or `~/ml/book-spines`).
/// Mirrors `bookspines.swift`'s original default-model resolution: follow
/// the `SpineDetectorOBB.mlpackage` alias only once the promote script has
/// retargeted it (i.e. only when it's actually a symlink); otherwise fall
/// back to the frozen `SpineDetectorOBB-aug.mlpackage` baseline.
public func defaultModelURL(dataRoot: String? = nil) -> URL {
    let root = dataRoot
        ?? ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
        ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
    let production = "\(root)/models/production"
    let alias = "\(production)/SpineDetectorOBB.mlpackage"
    let aug = "\(production)/SpineDetectorOBB-aug.mlpackage"
    if let attrs = try? FileManager.default.attributesOfItem(atPath: alias),
       let type = attrs[.type] as? FileAttributeType,
       type == .typeSymbolicLink {
        return URL(fileURLWithPath: alias)
    }
    return URL(fileURLWithPath: aug)
}

// MARK: - Ultralytics letterbox

/// Matches `ultralytics.data.augment.LetterBox` (center, gray=114) and
/// `ultralytics.utils.ops.scale_boxes` (xywh=True, padding=True).
public struct LetterboxParams {
    public let gain: Double
    public let padX: Double
    public let padY: Double
    public let newUnpadW: Int
    public let newUnpadH: Int
}

public func ultralyticsLetterbox(imageW: Double, imageH: Double, modelW: Int, modelH: Int) -> LetterboxParams {
    let gain = min(Double(modelH) / imageH, Double(modelW) / imageW)
    let newUnpadW = Int((imageW * gain).rounded())
    let newUnpadH = Int((imageH * gain).rounded())
    let dw = Double(modelW) - Double(newUnpadW)
    let dh = Double(modelH) - Double(newUnpadH)
    let padX = (dw / 2 - 0.1).rounded()
    let padY = (dh / 2 - 0.1).rounded()
    return LetterboxParams(gain: gain, padX: padX, padY: padY, newUnpadW: newUnpadW, newUnpadH: newUnpadH)
}

private func mapModelBoxToImage(
    cxRaw: Float, cyRaw: Float, wRaw: Float, hRaw: Float, angle: Float, conf: Float,
    letterbox: LetterboxParams, originX: Double, originY: Double
) -> OBBDetection? {
    let cx = (Double(cxRaw) - letterbox.padX) / letterbox.gain + originX
    let cy = (Double(cyRaw) - letterbox.padY) / letterbox.gain + originY
    let w = Double(wRaw) / letterbox.gain
    let h = Double(hRaw) / letterbox.gain
    guard w > 1, h > 1 else { return nil }
    return OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: Double(angle), conf: conf)
}

private func letterboxedImage(from cgImage: CGImage, modelW: Int, modelH: Int) -> (CGImage, LetterboxParams)? {
    let imgW = Double(cgImage.width)
    let imgH = Double(cgImage.height)
    let letterbox = ultralyticsLetterbox(imageW: imgW, imageH: imgH, modelW: modelW, modelH: modelH)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let ctx = CGContext(data: nil, width: modelW, height: modelH,
                               bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
                               bitmapInfo: bitmapInfo) else { return nil }
    ctx.setFillColor(CGColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: modelW, height: modelH))
    guard let resizedCtx = CGContext(data: nil,
                                      width: letterbox.newUnpadW,
                                      height: letterbox.newUnpadH,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else { return nil }
    resizedCtx.interpolationQuality = .medium
    resizedCtx.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(letterbox.newUnpadW), height: CGFloat(letterbox.newUnpadH)))
    guard let resized = resizedCtx.makeImage() else { return nil }
    ctx.interpolationQuality = .medium
    ctx.draw(resized, in: CGRect(x: letterbox.padX, y: letterbox.padY,
                                  width: CGFloat(letterbox.newUnpadW), height: CGFloat(letterbox.newUnpadH)))
    guard let image = ctx.makeImage() else { return nil }
    return (image, letterbox)
}

// MARK: - Detector

/// Loads one Core ML OBB model and decodes detections from it. Holds no
/// tile/crop-specific state — safe to reuse across many `predict` calls.
public final class SpineDetector {
    public private(set) var model: MLModel
    public private(set) var inputName: String
    public private(set) var outputName: String
    public private(set) var imageConstraint: MLImageConstraint
    public private(set) var inputWidth: Int
    public private(set) var inputHeight: Int
    /// Best current guess at the model's output layout: from the model's
    /// declared output shape at load time, refined after every `predict`
    /// call that observes a concrete (non-symbolic) shape.
    public private(set) var layout: OBBOutputLayout

    /// Loads an uncompiled `.mlpackage`/`.mlmodel` source, compiling (and
    /// caching the compiled result) on first use — the path every macOS CLI
    /// (`bookspines`, `spine-read`, `spine-id`, ...) uses, since they always
    /// point at a source model on disk (e.g. `defaultModelURL()`).
    public convenience init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw SpineCoreError.modelNotFound(modelURL.path)
        }
        let compiledURL = try cachedCompiledModelURL(for: modelURL)
        try self.init(compiledModelURL: compiledURL, computeUnits: computeUnits)
    }

    /// Loads a model that is **already compiled** (a `.mlmodelc` bundle) --
    /// the shape an `.mlpackage` added as an Xcode resource ends up in
    /// inside an app bundle (Xcode's Core ML resource build phase compiles
    /// it in place and ships only the compiled output, never the raw
    /// source), so on-device app code can't route through
    /// `cachedCompiledModelURL`'s `MLModel.compileModel(at:)` step, which
    /// requires an *uncompiled* source. Use
    /// `Bundle.main.url(forResource:withExtension:"mlmodelc")` to locate it.
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        guard FileManager.default.fileExists(atPath: compiledModelURL.path) else {
            throw SpineCoreError.modelNotFound(compiledModelURL.path)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        do {
            model = try MLModel(contentsOf: compiledModelURL, configuration: config)
        } catch {
            throw SpineCoreError.modelLoadFailed(error.localizedDescription)
        }

        guard let inName = model.modelDescription.inputDescriptionsByName.keys.first,
              let constraint = model.modelDescription.inputDescriptionsByName[inName]?.imageConstraint else {
            throw SpineCoreError.modelHasNoImageInput
        }
        guard let outName = model.modelDescription.outputDescriptionsByName.keys.first else {
            throw SpineCoreError.modelHasNoOutputs
        }
        inputName = inName
        outputName = outName
        imageConstraint = constraint
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh

        if let outConstraint = model.modelDescription.outputDescriptionsByName[outName]?.multiArrayConstraint,
           let detected = OBBOutputLayout.detect(shape: outConstraint.shape.map(\.intValue)) {
            layout = detected
        } else {
            layout = .end2endDetections
        }
    }

    /// Runs inference on `image`, mapping decoded boxes into full-scene
    /// pixel coordinates via `(originX, originY)` (the top-left of `image`
    /// within the scene — 0,0 for a full-image pass, a tile/crop offset
    /// otherwise). Confidence-filters but does **not** run cross-call NMS;
    /// callers merging multiple tiles/crops must call `nmsRotated`
    /// themselves after combining candidates.
    public func predict(_ image: CGImage, originX: Double = 0, originY: Double = 0, options: DetectionOptions) throws -> InferenceResult {
        guard let (letterboxed, letterbox) = letterboxedImage(from: image, modelW: inputWidth, modelH: inputHeight) else {
            throw SpineCoreError.letterboxFailed
        }

        let started = Date()
        let prediction: MLFeatureProvider
        do {
            let feature = try MLFeatureValue(cgImage: letterboxed, constraint: imageConstraint, options: [:])
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: feature])
            prediction = try model.prediction(from: input)
        } catch {
            throw SpineCoreError.predictionFailed(error.localizedDescription)
        }
        let inferenceMs = Date().timeIntervalSince(started) * 1000

        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw SpineCoreError.noMultiArrayOutput(outputName)
        }

        let shape = output.shape.map(\.intValue)
        if let detected = OBBOutputLayout.detect(shape: shape) {
            layout = detected
        }
        let currentLayout = layout

        let detections = try SpineDetector.decodeDetections(
            output: output, shape: shape, layout: currentLayout,
            confidenceThreshold: options.confidenceThreshold,
            letterbox: letterbox, originX: originX, originY: originY
        )

        return InferenceResult(
            detections: detections,
            layout: currentLayout,
            alreadyNMSed: currentLayout == .end2endDetections,
            inferenceMs: inferenceMs
        )
    }

    /// Pure decode step (no model/self state) — a `static` function so it
    /// can be unit-tested directly against hand-built `MLMultiArray`
    /// fixtures (including non-contiguous strides) without loading a real
    /// Core ML model. Internal-only; exercised via `@testable import`.
    static func decodeDetections(
        output: MLMultiArray, shape: [Int], layout: OBBOutputLayout,
        confidenceThreshold: Float, letterbox: LetterboxParams, originX: Double, originY: Double
    ) throws -> [OBBDetection] {
        let data = output.dataPointer.assumingMemoryBound(to: Float32.self)
        var detections: [OBBDetection] = []

        switch layout {
        case .legacyChannelsFirst:
            // Shape [1, 6, N] — channels then anchors.
            let channels = shape[1]
            let anchors = shape[2]
            guard channels == 6 else { throw SpineCoreError.unsupportedOutputLayout(shape: shape) }
            let strideC = output.strides[1].intValue
            let strideN = output.strides[2].intValue
            let confBase = 4 * strideC
            var survivors: [Int] = []
            survivors.reserveCapacity(256)
            if strideN == 1 {
                let confLane = UnsafeBufferPointer(start: data + confBase, count: anchors)
                for i in 0..<anchors where confLane[i] > confidenceThreshold { survivors.append(i) }
            } else {
                for i in 0..<anchors where data[confBase + i * strideN] > confidenceThreshold { survivors.append(i) }
            }
            detections.reserveCapacity(survivors.count)
            for i in survivors {
                let conf = data[confBase + i * strideN]
                let cxRaw = data[0 * strideC + i * strideN]
                let cyRaw = data[1 * strideC + i * strideN]
                let wRaw = data[2 * strideC + i * strideN]
                let hRaw = data[3 * strideC + i * strideN]
                let angle = data[5 * strideC + i * strideN]
                if let det = mapModelBoxToImage(cxRaw: cxRaw, cyRaw: cyRaw, wRaw: wRaw, hRaw: hRaw, angle: angle, conf: conf,
                                                 letterbox: letterbox, originX: originX, originY: originY) {
                    detections.append(det)
                }
            }

        case .end2endDetections:
            // Shape [1, maxDet, 7] — rows of [cx, cy, w, h, conf, cls, angle].
            // (Also handles a transposed [1, 7, maxDet].)
            let dim1 = shape[1]
            let dim2 = shape[2]
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
                throw SpineCoreError.unsupportedOutputLayout(shape: shape)
            }

            detections.reserveCapacity(min(rows, 64))
            for i in 0..<rows {
                let base = i * rowStride
                let conf = data[base + 4 * featStride]
                guard conf > confidenceThreshold else { continue }
                let cxRaw = data[base + 0 * featStride]
                let cyRaw = data[base + 1 * featStride]
                let wRaw = data[base + 2 * featStride]
                let hRaw = data[base + 3 * featStride]
                let angle = data[base + 6 * featStride]
                if let det = mapModelBoxToImage(cxRaw: cxRaw, cyRaw: cyRaw, wRaw: wRaw, hRaw: hRaw, angle: angle, conf: conf,
                                                 letterbox: letterbox, originX: originX, originY: originY) {
                    detections.append(det)
                }
            }
        }

        return detections
    }
}

// MARK: - Rotated NMS

/// Greedy rotated NMS: sort by confidence, keep a candidate only if its IoU
/// with every already-kept detection is `<= iouThreshold`. Matches Python
/// `nms_rotated` (`tools/tiled_predict_obb.py`).
public func nmsRotated(_ candidates: [OBBDetection], iouThreshold: Double, maxDetections: Int) -> [OBBDetection] {
    var kept: [OBBDetection] = []
    for candidate in candidates.sorted(by: { $0.conf > $1.conf }) {
        guard kept.count < maxDetections else { break }
        if kept.allSatisfy({ rotatedIoU($0, candidate) <= iouThreshold }) {
            kept.append(candidate)
        }
    }
    return kept
}
