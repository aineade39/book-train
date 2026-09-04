import CoreGraphics
import Foundation

// Stage 1 of the jigsaw-zoom plan (~/dev/book-id-design/docs/
// jigsaw-zoom-requirements.md R-3 / R-7 / A-6): a scene-space polygon
// piece, rigidly placed on the model canvas (optional rotation, gray-fill
// mask, never upsample). No perspective warp — that is the incumbent
// layout-crops path (`warpQuad`).

public struct ZoomPiece {
    /// Scene-pixel polygon (any winding; ≥ 3 points).
    public let polygon: [CGPoint]
    public let depth: Int

    public init(polygon: [CGPoint], depth: Int = 0) {
        self.polygon = polygon
        self.depth = depth
    }

    public static func fullImage(width: Int, height: Int) -> ZoomPiece {
        ZoomPiece(polygon: [
            CGPoint(x: 0, y: 0),
            CGPoint(x: width, y: 0),
            CGPoint(x: width, y: height),
            CGPoint(x: 0, y: height),
        ], depth: 0)
    }
}

/// Rigid letterbox of one piece onto an `imgsz` canvas.
public struct ZoomLetterbox {
    public let image: CGImage
    /// Scene → unpadded-crop (similarity). Combined with `gain`/`padX`/`padY`
    /// this is the inverse of `mapDetFromCrop`.
    public let homography: Homography
    public let gain: Double
    public let padX: Double
    public let padY: Double
    /// Shrink applied to fit the canvas; 1.0 means the piece already fits
    /// (R-7: never upsample). R-1b accepts the piece as a leaf when this
    /// is at or above the downsample threshold.
    public let scale: Double
    public let cropWidth: Int
    public let cropHeight: Int
}

/// Axis-aligned or min-area-rect placement of `polygon` onto a crop of
/// size `(cropWidth, cropHeight)`, without rendering.
public struct ZoomPlacement {
    public let homography: Homography
    public let cropWidth: Int
    public let cropHeight: Int

    /// `min(1, imgsz / max(cropWidth, cropHeight))` — the letterbox shrink.
    public func scale(imgsz: Int) -> Double {
        guard cropWidth > 0, cropHeight > 0, imgsz > 0 else { return 1 }
        return min(1.0, Double(imgsz) / Double(max(cropWidth, cropHeight)))
    }
}

/// Source quad + native crop size for `polygon`. `rotate` uses the
/// min-area rectangle (R-3); otherwise the axis-aligned bounding box.
func zoomSourceQuad(polygon: [CGPoint], rotate: Bool) -> (src: [CGPoint], cropW: Int, cropH: Int)? {
    let finite = polygon.filter { $0.x.isFinite && $0.y.isFinite }
    guard finite.count >= 3 else { return nil }

    if rotate, let mar = minAreaRect(finite), mar.width >= 1, mar.height >= 1 {
        return (minAreaRectCorners(mar), max(1, Int(mar.width.rounded())), max(1, Int(mar.height.rounded())))
    }
    let xs = finite.map { Double($0.x) }
    let ys = finite.map { Double($0.y) }
    let x0 = xs.min()!, y0 = ys.min()!, x1 = xs.max()!, y1 = ys.max()!
    guard x1 > x0, y1 > y0 else { return nil }
    let src = [
        CGPoint(x: x0, y: y0),
        CGPoint(x: x1, y: y0),
        CGPoint(x: x1, y: y1),
        CGPoint(x: x0, y: y1),
    ]
    return (src, max(1, Int((x1 - x0).rounded())), max(1, Int((y1 - y0).rounded())))
}

/// Placement for `polygon`. `rotate` uses the min-area rectangle (R-3);
/// otherwise the axis-aligned bounding box.
public func zoomPlacement(polygon: [CGPoint], rotate: Bool) -> ZoomPlacement? {
    guard let (src, cropW, cropH) = zoomSourceQuad(polygon: polygon, rotate: rotate) else { return nil }
    let dst = [
        CGPoint(x: 0, y: 0),
        CGPoint(x: cropW - 1, y: 0),
        CGPoint(x: cropW - 1, y: cropH - 1),
        CGPoint(x: 0, y: cropH - 1),
    ]
    guard let homography = Homography(from: src, to: dst) else { return nil }
    return ZoomPlacement(homography: homography, cropWidth: cropW, cropHeight: cropH)
}

/// Four corners of `mar` in TL, TR, BR, BL order in the rectangle's own
/// frame (y-down), matching `zoomPlacement`'s destination quad.
func minAreaRectCorners(_ mar: MinAreaRect) -> [CGPoint] {
    let c = cos(mar.angleRad), s = sin(mar.angleRad)
    let hx = mar.width / 2, hy = mar.height / 2
    let locals: [(Double, Double)] = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return locals.map { lx, ly in
        CGPoint(x: mar.cx + lx * c - ly * s, y: mar.cy + lx * s + ly * c)
    }
}

/// Renders `polygon` from `raster` onto an `imgsz` canvas: rigid placement,
/// pixels outside the polygon filled with `padValue` (R-3), never upsampled
/// (R-7).
///
/// The downscale (when the piece is larger than `imgsz`) is folded into the
/// render homography, so the mask/sample loop touches at most `imgsz²`
/// pixels instead of the piece's native pixel count — rendering a 4284x5712
/// root at native size and then shrinking it was the dominant cost of a
/// whole `jigsawZoomDetect` run.
public func letterboxPiece(
    raster: SceneRaster,
    polygon: [CGPoint],
    imgsz: Int,
    padValue: UInt8 = 114,
    rotate: Bool = true
) -> ZoomLetterbox? {
    guard let (src, cropW, cropH) = zoomSourceQuad(polygon: polygon, rotate: rotate) else { return nil }
    let scale = min(1.0, Double(imgsz) / Double(max(cropW, cropH)))
    let outW = max(1, Int((Double(cropW) * scale).rounded()))
    let outH = max(1, Int((Double(cropH) * scale).rounded()))
    let dst = [
        CGPoint(x: 0, y: 0),
        CGPoint(x: outW - 1, y: 0),
        CGPoint(x: outW - 1, y: outH - 1),
        CGPoint(x: 0, y: outH - 1),
    ]
    guard let homography = Homography(from: src, to: dst) else { return nil }
    guard let crop = renderMaskedCrop(
        raster: raster, homography: homography,
        width: outW, height: outH,
        polygon: polygon, padValue: padValue
    ) else { return nil }
    guard let padded = padNoUpsize(crop, canvas: imgsz, padValue: padValue) else { return nil }
    return ZoomLetterbox(
        image: padded.image,
        homography: homography,
        gain: padded.gain,
        padX: padded.padX,
        padY: padded.padY,
        scale: scale,
        cropWidth: outW,
        cropHeight: outH
    )
}

/// Unmasked crop (real pixels, including neighbors outside the piece).
/// Used as the `SceneRaster` for a local `planCrops` cut.
func renderUnmaskedCrop(
    raster: SceneRaster,
    homography: Homography,
    width: Int,
    height: Int,
    padValue: UInt8
) -> CGImage? {
    warpPerspective(raster, homography: homography, outputWidth: width, outputHeight: height, borderValue: padValue)
}

private func renderMaskedCrop(
    raster: SceneRaster,
    homography: Homography,
    width: Int,
    height: Int,
    polygon: [CGPoint],
    padValue: UInt8
) -> CGImage? {
    guard width > 0, height > 0 else { return nil }
    var out = [UInt8](repeating: padValue, count: width * height * 4)
    for i in 0..<(width * height) { out[i * 4 + 3] = 255 }

    // Scanline-fill the mask in *output* space instead of testing every
    // output pixel against the scene polygon: the render homography maps
    // lines to lines, so the transformed polygon is exact, and the cost
    // drops from O(width * height * vertices) to O(height * vertices) plus
    // one sample per interior pixel. Free-form staircase pieces (R-2) carry
    // tens of vertices, where the per-pixel containment test dominated the
    // whole run.
    let local = polygon.map { homography.apply($0) }
    guard let bounds = polygonBounds(local) else { return nil }
    var crossings: [Double] = []

    for oy in 0..<height {
        let y = Double(oy)
        // Mirrors the 1px boundary-touch tolerance the previous
        // `pointInPolygon` mask had, so edge rows stay inside the piece.
        guard y >= bounds.y0 - 1, y <= bounds.y1 + 1 else { continue }
        let scanY = min(max(y, bounds.y0 + 1e-6), bounds.y1 - 1e-6)

        crossings.removeAll(keepingCapacity: true)
        let n = local.count
        for i in 0..<n {
            let p = local[i], q = local[(i + 1) % n]
            let py = Double(p.y), qy = Double(q.y)
            guard (py > scanY) != (qy > scanY) else { continue }
            let t = (scanY - py) / (qy - py)
            crossings.append(Double(p.x) + t * (Double(q.x) - Double(p.x)))
        }
        guard crossings.count >= 2 else { continue }
        crossings.sort()

        var k = 0
        while k + 1 < crossings.count {
            let x0 = max(0, Int((crossings[k] - 1).rounded(.up)))
            let x1 = min(width - 1, Int((crossings[k + 1] + 1).rounded(.down)))
            if x0 <= x1 {
                for ox in x0...x1 {
                    let srcPt = homography.applyInverse(CGPoint(x: ox, y: oy))
                    let (r, g, b) = raster.sample(srcPt, borderValue: padValue)
                    let idx = (oy * width + ox) * 4
                    out[idx] = r; out[idx + 1] = g; out[idx + 2] = b
                }
            }
            k += 2
        }
    }

    guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
          let provider = CGDataProvider(data: Data(out) as CFData) else { return nil }
    return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: cs,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
    )
}
