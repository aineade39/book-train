import CoreGraphics
import Foundation
import XCTest

@testable import SpineCore

/// Shared fixtures/helpers for `SpineCoreTests`. Kept dependency-free (no
/// model, no dataset) so every test here runs offline and deterministically.

/// Builds a `CGImage` directly from a top-down (row 0 = top, y increasing
/// downward), row-major RGBA8 buffer via `CGDataProvider` -- the same
/// construction `warpPerspective` uses. Deliberately avoids
/// `CGContext.fill(rect:)`, whose `y` is in Quartz's bottom-left-origin,
/// y-up user space and therefore lands in the *opposite* end of the raw
/// buffer than a naive top-left-origin `y` would suggest.
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
/// Used wherever a test needs a `CGImage`/`SceneRaster` but doesn't care
/// about pixel content (e.g. homography/padding geometry tests, or a
/// zero-texture backdrop for deterministic pixel-seam search in planner
/// tests).
func makeSolidCGImage(width: Int, height: Int, gray: UInt8 = 114) -> CGImage {
    var buf = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        buf[i * 4] = gray; buf[i * 4 + 1] = gray; buf[i * 4 + 2] = gray
    }
    return imageFromTopDownRGBA(buf, width: width, height: height)
}

/// A `width x height` image split into `cols x rows` axis-aligned blocks
/// alternating black/white, top-left block (row 0, col 0, in top-down,
/// y-down pixel space) black — deterministic, easy to reason about
/// pixel-by-pixel, and (unlike a uniform fill) sensitive to any accidental
/// y-axis flip in a warp/crop path.
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

func XCTAssertPointsEqual(_ a: CGPoint, _ b: CGPoint, accuracy: Double = 1e-6, _ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(Double(a.x), Double(b.x), accuracy: accuracy, message, file: file, line: line)
    XCTAssertEqual(Double(a.y), Double(b.y), accuracy: accuracy, message, file: file, line: line)
}
