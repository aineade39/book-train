import CoreGraphics
import Foundation

/// Shared fixtures for `SpinePerceptionTests`. Mirrors
/// `Tests/SpineCoreTests/TestSupport.swift`'s construction (kept as a
/// separate copy since XCTest targets don't share sources) so tests here
/// don't need a real captured photo or a live Vision/model dependency.

private func imageFromTopDownRGBA(_ pixels: [UInt8], width: Int, height: Int) -> CGImage {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let provider = CGDataProvider(data: Data(pixels) as CFData)!
    return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: cs,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
    )!
}

/// A flat `width x height` RGB image, every pixel `(gray, gray, gray)`.
func makeSolidCGImage(width: Int, height: Int, gray: UInt8 = 114) -> CGImage {
    var buf = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        buf[i * 4] = gray; buf[i * 4 + 1] = gray; buf[i * 4 + 2] = gray
    }
    return imageFromTopDownRGBA(buf, width: width, height: height)
}

/// A `width x height` image split into `cols x rows` axis-aligned blocks
/// alternating black/white -- a stand-in for a "sharp, high-contrast
/// scene" fixture (as opposed to `makeSolidCGImage`'s flat/blurred stand-in).
func makeCheckerboardCGImage(width: Int, height: Int, cols: Int, rows: Int) -> CGImage {
    var buf = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
        let r = min(rows - 1, y * rows / height)
        for x in 0..<width {
            let c = min(cols - 1, x * cols / width)
            let black = (r + c) % 2 == 0
            let v: UInt8 = black ? 0 : 255
            let idx = (y * width + x) * 4
            buf[idx] = v; buf[idx + 1] = v; buf[idx + 2] = v
        }
    }
    return imageFromTopDownRGBA(buf, width: width, height: height)
}

/// A uniform mid-gray image with tiny random per-pixel noise -- a stand-in
/// for a well-exposed but out-of-focus frame (some high-frequency variance,
/// but far less than a real in-focus edge).
func makeNoisyGrayCGImage(width: Int, height: Int, gray: UInt8 = 128, noise: UInt8 = 4) -> CGImage {
    var generator = SplitMix64(seed: 42)
    var buf = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        let delta = Int(generator.next() % UInt64(noise * 2 + 1)) - Int(noise)
        let v = UInt8(clamping: Int(gray) + delta)
        buf[i * 4] = v; buf[i * 4 + 1] = v; buf[i * 4 + 2] = v
    }
    return imageFromTopDownRGBA(buf, width: width, height: height)
}

/// Minimal deterministic PRNG so noise fixtures are reproducible across runs.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
