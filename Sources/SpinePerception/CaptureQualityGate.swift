import CoreGraphics
import Foundation

// Capture-quality pre-gate per docs/BOOK_ID_IOS_PIPELINE.md §Detection &
// isolation: "before detection, cheaply score the frame for focus / blur /
// exposure. Reject-and-prompt-retake on clearly bad frames rather than
// spending the full detect->OCR->match budget on unreadable input." This
// is the *first* quality gate; `OCRQualityGate` is the second.
//
// Runs entirely on a downsampled grayscale buffer (Core Graphics only, no
// Vision/Core Image dependency) so it is cheap, deterministic, and trivial
// to unit test on Mac with synthetic fixtures (a checkerboard vs. a solid
// gray image, etc.).

public struct CaptureQualityScore {
    /// Normalized `[0, 1)` sharpness proxy (variance of a Laplacian
    /// response over a downsampled grayscale image) — near `0` for a flat
    /// or heavily blurred frame, higher for one with crisp edges.
    public let sharpness: Double
    /// Normalized `[0, 1]` exposure score, peaking at mid-gray mean
    /// luminance and falling off toward black/white clipping.
    public let exposure: Double
}

public struct CaptureQualityGate {
    public var minSharpness: Double
    public var minExposure: Double
    /// Longest side (px) the frame is downsampled to before scoring —
    /// scoring is a cheap pre-gate, not a detail-preserving analysis.
    public var analysisMaxSide: Int

    public static let `default` = CaptureQualityGate()

    public init(minSharpness: Double = 0.12, minExposure: Double = 0.12, analysisMaxSide: Int = 192) {
        self.minSharpness = minSharpness
        self.minExposure = minExposure
        self.analysisMaxSide = analysisMaxSide
    }

    public func score(_ image: CGImage) -> CaptureQualityScore {
        guard let gray = downsampledGrayscale(image, maxSide: analysisMaxSide) else {
            return CaptureQualityScore(sharpness: 0, exposure: 0)
        }
        return CaptureQualityScore(
            sharpness: sharpnessScore(gray),
            exposure: exposureScore(gray)
        )
    }

    public func passes(_ image: CGImage) -> Bool {
        let s = score(image)
        return s.sharpness >= minSharpness && s.exposure >= minExposure
    }

    // MARK: - Grayscale downsample

    private struct GrayscaleBuffer {
        let pixels: [Double] // row-major, [0, 255]
        let width: Int
        let height: Int
    }

    private func downsampledGrayscale(_ image: CGImage, maxSide: Int) -> GrayscaleBuffer? {
        let longSide = max(image.width, image.height)
        let scale = longSide > maxSide ? Double(maxSide) / Double(longSide) : 1
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return nil }

        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height)
        var pixels = [Double](repeating: 0, count: width * height)
        for i in 0..<(width * height) { pixels[i] = Double(buffer[i]) }
        return GrayscaleBuffer(pixels: pixels, width: width, height: height)
    }

    // MARK: - Sharpness (variance of Laplacian)

    private func sharpnessScore(_ buffer: GrayscaleBuffer) -> Double {
        let w = buffer.width, h = buffer.height
        guard w >= 3, h >= 3 else { return 0 }
        let p = buffer.pixels

        var responses: [Double] = []
        responses.reserveCapacity((w - 2) * (h - 2))
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let center = p[y * w + x]
                let up = p[(y - 1) * w + x]
                let down = p[(y + 1) * w + x]
                let left = p[y * w + x - 1]
                let right = p[y * w + x + 1]
                responses.append(up + down + left + right - 4 * center)
            }
        }
        guard !responses.isEmpty else { return 0 }
        let mean = responses.reduce(0, +) / Double(responses.count)
        let variance = responses.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(responses.count)

        // Saturating map: 0 at variance 0, asymptotic toward 1. `k` is an
        // order-of-magnitude scale constant for 8-bit grayscale Laplacian
        // variance, not a precision-tuned threshold — real acceptance
        // tuning happens against production rotation-bucket photos per
        // AGENTS.md's evaluation guidance, not by editing this constant.
        let k = 40.0
        return variance / (variance + k)
    }

    // MARK: - Exposure

    private func exposureScore(_ buffer: GrayscaleBuffer) -> Double {
        guard !buffer.pixels.isEmpty else { return 0 }
        let mean = buffer.pixels.reduce(0, +) / Double(buffer.pixels.count)
        // Peaks at mid-gray (127.5), falls linearly to 0 at pure black/white.
        return 1 - abs(mean - 127.5) / 127.5
    }
}
