import CoreGraphics
import Foundation

/// Shared fixtures for `SpinePipelineTests`. Mirrors
/// `Tests/SpinePerceptionTests/TestSupport.swift`'s construction (kept as a
/// separate copy since XCTest targets don't share sources) so tests here
/// don't need a real captured photo.

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

/// A flat `width x height` RGB image, every pixel `(gray, gray, gray)` --
/// fails `CaptureQualityGate`'s sharpness check, standing in for "even if
/// capture gate would fail" fixtures (§F).
func makeSolidCGImage(width: Int, height: Int, gray: UInt8 = 114) -> CGImage {
    var buf = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        buf[i * 4] = gray; buf[i * 4 + 1] = gray; buf[i * 4 + 2] = gray
    }
    return imageFromTopDownRGBA(buf, width: width, height: height)
}

/// Linearly scales every pixel's RGB by `factor` (alpha untouched) --
/// drawing over an opaque black background at `factor` alpha is exactly
/// `result = factor * src + (1 - factor) * black = factor * src`. Used to
/// reproduce a real dim-room photo's exposure regime from a real,
/// content-bearing fixture (see
/// `SpineIdentificationEngineTests.testAdvisoryCaptureGateStillDetectsSpinesOnASharpButDarkFrame`)
/// without losing the spine-shaped content a flat synthetic fixture has
/// none of.
func darkenedCGImage(_ image: CGImage, factor: CGFloat) -> CGImage {
    let width = image.width, height = image.height
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.setAlpha(factor)
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
}
