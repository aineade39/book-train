import CoreGraphics
import Foundation
import simd

// Pixel-level operations for the layout-crop planner: edge-energy maps for
// whitespace-seam search, a hand-rolled projective homography (forward +
// inverse, both retained and independently testable — see
// `LayoutCropRules.swift`'s sibling tests), perspective warp via manual
// inverse-mapped bilinear sampling, downscale-only padding, and the inverse
// remap of a crop-local detection back into scene pixels.
//
// Deliberately does not use `CIFilter`/`CIPerspectiveCorrection`: that
// filter's output extent and corner-ordering contract is not part of its
// documented API, so relying on it would reintroduce exactly the kind of
// unverified coordinate assumption this port is trying to avoid. Every pixel
// read/write here goes through one homography we solve and can unit-test.

// MARK: - Edge energy (pixel-texture whitespace search)

/// Per-pixel `|Laplacian|` energy of a grayscale image, backed by a summed-
/// area table so arbitrary-rectangle mean queries are O(1). Mirrors Python
/// `compute_edge_energy` (`cv2.cvtColor` + `cv2.Laplacian(ksize=3)`).
public struct EdgeEnergy {
    public let width: Int
    public let height: Int
    private let integral: [Double] // (width+1) * (height+1), row-major

    public init?(cgImage: CGImage) {
        let w = cgImage.width, h = cgImage.height
        guard w > 0, h > 0 else { return nil }
        guard let gray = EdgeEnergy.grayscale(cgImage) else { return nil }
        let energy = EdgeEnergy.laplacianEnergy(gray, width: w, height: h)
        width = w
        height = h
        integral = EdgeEnergy.buildIntegral(energy, width: w, height: h)
    }

    /// Builds from an already-decoded `SceneRaster`, avoiding a second
    /// CGContext draw of a potentially large scene image.
    public init?(raster: SceneRaster) {
        let w = raster.width, h = raster.height
        guard w > 0, h > 0 else { return nil }
        var gray = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let (r, g, b) = raster.rgb(x: x, y: y)
                gray[y * w + x] = UInt8(min(255, max(0, (0.299 * Double(r) + 0.587 * Double(g) + 0.114 * Double(b)).rounded())))
            }
        }
        let energy = EdgeEnergy.laplacianEnergy(gray, width: w, height: h)
        width = w
        height = h
        integral = EdgeEnergy.buildIntegral(energy, width: w, height: h)
    }

    /// OpenCV's BGR2GRAY luma weights (Rec. 601), applied to an RGB raster.
    private static func grayscale(_ cgImage: CGImage) -> [UInt8]? {
        let w = cgImage.width, h = cgImage.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = rgba.withUnsafeMutableBytes({ ptr -> CGContext? in
            CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: cs, bitmapInfo: bitmapInfo)
        }) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        var gray = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let r = Double(rgba[i * 4]), g = Double(rgba[i * 4 + 1]), b = Double(rgba[i * 4 + 2])
            gray[i] = UInt8(min(255, max(0, (0.299 * r + 0.587 * g + 0.114 * b).rounded())))
        }
        return gray
    }

    /// 3x3 Laplacian (`[[0,1,0],[1,-4,1],[0,1,0]]`), reflect-101 border —
    /// matches `cv2.Laplacian`'s default `BORDER_REFLECT_101`.
    private static func laplacianEnergy(_ gray: [UInt8], width: Int, height: Int) -> [Double] {
        func reflect101(_ i: Int, _ n: Int) -> Int {
            if n == 1 { return 0 }
            var v = i
            while v < 0 || v >= n {
                if v < 0 { v = -v }
                if v >= n { v = 2 * (n - 1) - v }
            }
            return v
        }
        var out = [Double](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let center = Double(gray[y * width + x])
                let up = Double(gray[reflect101(y - 1, height) * width + x])
                let down = Double(gray[reflect101(y + 1, height) * width + x])
                let left = Double(gray[y * width + reflect101(x - 1, width)])
                let right = Double(gray[y * width + reflect101(x + 1, width)])
                out[y * width + x] = abs(up + down + left + right - 4 * center)
            }
        }
        return out
    }

    private static func buildIntegral(_ energy: [Double], width: Int, height: Int) -> [Double] {
        let stride = width + 1
        var integral = [Double](repeating: 0, count: stride * (height + 1))
        for y in 0..<height {
            var rowSum = 0.0
            for x in 0..<width {
                rowSum += energy[y * width + x]
                integral[(y + 1) * stride + (x + 1)] = rowSum + integral[y * stride + (x + 1)]
            }
        }
        return integral
    }

    /// Sum of energy over `[x0, x1) x [y0, y1)`, clamped to image bounds.
    public func rectSum(x0: Int, y0: Int, x1: Int, y1: Int) -> Double {
        let cx0 = max(0, min(x0, width)), cx1 = max(0, min(x1, width))
        let cy0 = max(0, min(y0, height)), cy1 = max(0, min(y1, height))
        guard cx1 > cx0, cy1 > cy0 else { return 0 }
        let stride = width + 1
        return integral[cy1 * stride + cx1] - integral[cy0 * stride + cx1]
            - integral[cy1 * stride + cx0] + integral[cy0 * stride + cx0]
    }
}

public enum ProfileAxis {
    case x
    case y
}

/// Mean energy along `axis` over `[lo, hi)`, averaged across the
/// perpendicular band `[fixedLo, fixedHi)`. Out-of-image positions read as 0
/// — matches Python `_axis_profile`.
public func axisProfile(_ energy: EdgeEnergy, axis: ProfileAxis, lo: Int, hi: Int, fixedLo: Int, fixedHi: Int) -> [Double] {
    let length = max(0, hi - lo)
    var out = [Double](repeating: 0, count: length)
    guard length > 0 else { return out }
    let imgW = energy.width, imgH = energy.height
    let f0: Int, f1: Int, p0: Int, p1: Int
    if axis == .x {
        f0 = max(0, min(fixedLo, imgH)); f1 = max(0, min(fixedHi, imgH))
        p0 = max(0, min(lo, imgW)); p1 = max(0, min(hi, imgW))
    } else {
        f0 = max(0, min(fixedLo, imgW)); f1 = max(0, min(fixedHi, imgW))
        p0 = max(0, min(lo, imgH)); p1 = max(0, min(hi, imgH))
    }
    guard f1 > f0, p1 > p0 else { return out }
    let bandCount = Double(f1 - f0)
    let off = p0 - lo
    for p in p0..<p1 {
        let sum = axis == .x
            ? energy.rectSum(x0: p, y0: f0, x1: p + 1, y1: f1)
            : energy.rectSum(x0: f0, y0: p, x1: f1, y1: p + 1)
        let idx = off + (p - p0)
        if idx >= 0, idx < length { out[idx] = sum / bandCount }
    }
    return out
}

private func boxcarSmooth(_ profile: [Double], k kRaw: Int) -> [Double] {
    guard kRaw > 1, !profile.isEmpty else { return profile }
    let k = min(kRaw, profile.count)
    let half = k / 2
    var out = [Double](repeating: 0, count: profile.count)
    for i in 0..<profile.count {
        var sum = 0.0
        for j in 0..<k {
            let idx = i - half + j
            if idx >= 0, idx < profile.count { sum += profile[idx] }
        }
        out[i] = sum / Double(k)
    }
    return out
}

/// Position (x if `axis == .x` else y) of minimum texture energy within
/// `[searchLo, searchHi]`, averaged over the perpendicular
/// `[fixedLo, fixedHi)` band, boxcar-smoothed, and restricted to stay
/// `marginPx` from the search-range ends and outside every `forbidden`
/// sub-range. Returns `nil` if every position is excluded. Matches Python
/// `find_pixel_seam`, including its first-minimum tie-break.
public func findPixelSeam(
    _ energy: EdgeEnergy,
    axis: ProfileAxis,
    fixedLo: Double, fixedHi: Double,
    searchLo: Double, searchHi: Double,
    forbidden: [(Double, Double)],
    smoothPx: Int,
    marginPx: Double
) -> Double? {
    let loI = Int(searchLo.rounded(.down))
    let hiI = Int(searchHi.rounded(.up))
    guard hiI - loI >= 3 else { return nil }
    var prof = axisProfile(energy, axis: axis, lo: loI, hi: hiI, fixedLo: Int(fixedLo.rounded()), fixedHi: Int(fixedHi.rounded()))
    let k = max(1, smoothPx | 1)
    if k > 1 { prof = boxcarSmooth(prof, k: k) }

    var bestIdx: Int? = nil
    var bestValue = Double.infinity
    for i in 0..<prof.count {
        let position = Double(loI + i)
        guard position >= searchLo + marginPx, position <= searchHi - marginPx else { continue }
        var excluded = false
        for (fLo, fHi) in forbidden where position >= fLo && position <= fHi {
            excluded = true
            break
        }
        guard !excluded else { continue }
        if prof[i] < bestValue {
            bestValue = prof[i]
            bestIdx = i
        }
    }
    guard let bestIdx else { return nil }
    return Double(loI + bestIdx)
}

// MARK: - Homography

/// A planar projective transform (3x3, homogeneous), solved from four point
/// correspondences via direct linear transform. `forward` maps
/// `src -> dst`; `inverse` maps `dst -> src`. Both are retained (not just
/// derived on demand) so a caller can assert on either direction directly.
public struct Homography {
    public let forward: simd_double3x3
    public let inverse: simd_double3x3

    /// Solves for the homography mapping each `src[i]` to `dst[i]`.
    /// Returns `nil` if the four correspondences are degenerate (no unique
    /// solution, e.g. three or more collinear points).
    public init?(from src: [CGPoint], to dst: [CGPoint]) {
        guard src.count == 4, dst.count == 4 else { return nil }
        guard let h = Homography.solve(src: src, dst: dst) else { return nil }
        let det = h.determinant
        guard det.isFinite, abs(det) > 1e-12 else { return nil }
        forward = h
        inverse = h.inverse
    }

    public func apply(_ p: CGPoint) -> CGPoint {
        let v = forward * simd_double3(Double(p.x), Double(p.y), 1)
        return CGPoint(x: v.x / v.z, y: v.y / v.z)
    }

    public func applyInverse(_ p: CGPoint) -> CGPoint {
        let v = inverse * simd_double3(Double(p.x), Double(p.y), 1)
        return CGPoint(x: v.x / v.z, y: v.y / v.z)
    }

    /// Builds a `simd_double3x3` whose *rows* are `r0, r1, r2` (simd's
    /// memberwise initializer treats its three `SIMD3` arguments as
    /// columns, so this transposes that column-built matrix).
    private static func matrix(rows r0: SIMD3<Double>, _ r1: SIMD3<Double>, _ r2: SIMD3<Double>) -> simd_double3x3 {
        simd_double3x3(r0, r1, r2).transpose
    }

    private static func solve(src: [CGPoint], dst: [CGPoint]) -> simd_double3x3? {
        var a = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
        var b = [Double](repeating: 0, count: 8)
        for i in 0..<4 {
            let x = Double(src[i].x), y = Double(src[i].y)
            let X = Double(dst[i].x), Y = Double(dst[i].y)
            a[2 * i] = [x, y, 1, 0, 0, 0, -x * X, -y * X]
            b[2 * i] = X
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * Y, -y * Y]
            b[2 * i + 1] = Y
        }
        guard let h = gaussJordanSolve(a: a, b: b) else { return nil }
        return matrix(
            rows: SIMD3(h[0], h[1], h[2]),
            SIMD3(h[3], h[4], h[5]),
            SIMD3(h[6], h[7], 1)
        )
    }
}

/// Gauss-Jordan elimination with partial pivoting for a small dense system.
/// Returns `nil` if the system is (near-)singular.
private func gaussJordanSolve(a: [[Double]], b: [Double]) -> [Double]? {
    let n = a.count
    var m = a
    var rhs = b
    for col in 0..<n {
        var pivotRow = col
        var maxVal = abs(m[col][col])
        for r in (col + 1)..<n where abs(m[r][col]) > maxVal {
            maxVal = abs(m[r][col])
            pivotRow = r
        }
        guard maxVal > 1e-12 else { return nil }
        if pivotRow != col {
            m.swapAt(col, pivotRow)
            rhs.swapAt(col, pivotRow)
        }
        let pivot = m[col][col]
        for r in 0..<n where r != col {
            let factor = m[r][col] / pivot
            guard factor != 0 else { continue }
            for c in col..<n { m[r][c] -= factor * m[col][c] }
            rhs[r] -= factor * rhs[col]
        }
    }
    var x = [Double](repeating: 0, count: n)
    for i in 0..<n { x[i] = rhs[i] / m[i][i] }
    return x
}

// MARK: - Scene raster + perspective warp (manual inverse-mapped sampling)

/// A decoded RGB raster of one `CGImage`, kept resident for repeated
/// bilinear sampling while warping many crop quads out of the same scene.
public struct SceneRaster {
    public let width: Int
    public let height: Int
    private let pixels: [UInt8] // RGBA8, width * height * 4

    public init?(cgImage: CGImage) {
        let w = cgImage.width, h = cgImage.height
        guard w > 0, h > 0 else { return nil }
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = buf.withUnsafeMutableBytes({ ptr -> CGContext? in
            CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: cs, bitmapInfo: bitmapInfo)
        }) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        width = w
        height = h
        pixels = buf
    }

    /// Raw RGB at integer pixel `(x, y)` (unchecked bounds — callers must
    /// stay within `0..<width` / `0..<height`).
    func rgb(x: Int, y: Int) -> (UInt8, UInt8, UInt8) {
        let idx = (y * width + x) * 4
        return (pixels[idx], pixels[idx + 1], pixels[idx + 2])
    }

    /// Bilinear sample at continuous coordinate `p` (pixel-index space,
    /// top-left origin, matching `cv2.warpPerspective`'s convention of using
    /// raw pixel indices with no half-pixel offset). Out-of-bounds reads
    /// return `(borderValue, borderValue, borderValue)`.
    func sample(_ p: CGPoint, borderValue: UInt8) -> (UInt8, UInt8, UInt8) {
        let x = Double(p.x), y = Double(p.y)
        guard x > -1, y > -1, x < Double(width), y < Double(height) else {
            return (borderValue, borderValue, borderValue)
        }
        let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
        let x1 = x0 + 1, y1 = y0 + 1
        let fx = x - Double(x0), fy = y - Double(y0)

        func px(_ xi: Int, _ yi: Int) -> (Double, Double, Double) {
            guard xi >= 0, yi >= 0, xi < width, yi < height else {
                return (Double(borderValue), Double(borderValue), Double(borderValue))
            }
            let idx = (yi * width + xi) * 4
            return (Double(pixels[idx]), Double(pixels[idx + 1]), Double(pixels[idx + 2]))
        }
        func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }
        let c00 = px(x0, y0), c10 = px(x1, y0), c01 = px(x0, y1), c11 = px(x1, y1)
        let r = lerp(lerp(c00.0, c10.0, fx), lerp(c01.0, c11.0, fx), fy)
        let g = lerp(lerp(c00.1, c10.1, fx), lerp(c01.1, c11.1, fx), fy)
        let b = lerp(lerp(c00.2, c10.2, fx), lerp(c01.2, c11.2, fx), fy)
        return (UInt8(r.rounded()), UInt8(g.rounded()), UInt8(b.rounded()))
    }
}

/// Renders a `outputWidth x outputHeight` image by, for every destination
/// pixel, mapping back through `homography.applyInverse` into `raster` and
/// bilinear-sampling — the same inverse-mapping convention as
/// `cv2.warpPerspective(src, M, size)` (forward `M`, sampled via `M^-1`).
public func warpPerspective(_ raster: SceneRaster, homography: Homography, outputWidth: Int, outputHeight: Int, borderValue: UInt8 = 114) -> CGImage? {
    guard outputWidth > 0, outputHeight > 0 else { return nil }
    var out = [UInt8](repeating: borderValue, count: outputWidth * outputHeight * 4)
    for oy in 0..<outputHeight {
        for ox in 0..<outputWidth {
            let srcPt = homography.applyInverse(CGPoint(x: ox, y: oy))
            let (r, g, b) = raster.sample(srcPt, borderValue: borderValue)
            let idx = (oy * outputWidth + ox) * 4
            out[idx] = r; out[idx + 1] = g; out[idx + 2] = b; out[idx + 3] = 255
        }
    }
    guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
          let provider = CGDataProvider(data: Data(out) as CFData) else { return nil }
    return CGImage(
        width: outputWidth, height: outputHeight, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: outputWidth * 4, space: cs,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
    )
}

// MARK: - Warp quad + pad-no-upsize

public struct WarpResult {
    public let image: CGImage
    public let homography: Homography
    public let warpScale: Double
    public let outputWidth: Int
    public let outputHeight: Int
}

/// `(w, h)` of the axis-aligned crop that best fits quad `[tl, tr, br, bl]`
/// without distortion — matches Python `quad_output_size`.
public func quadOutputSize(_ quad: [CGPoint]) -> (Int, Int) {
    let tl = quad[0], tr = quad[1], br = quad[2], bl = quad[3]
    let w = max(dist(tl, tr), dist(bl, br))
    let h = max(dist(tl, bl), dist(tr, br))
    return (max(1, Int(w.rounded())), max(1, Int(h.rounded())))
}

/// Perspective-warps scene quad `quad` into its own upright crop (never
/// upsized past `maxSide` on its long side) — matches Python `warp_quad`.
public func warpQuad(_ raster: SceneRaster, quad: [CGPoint], maxSide: Int?, padValue: UInt8 = 114) -> WarpResult? {
    let (nativeW, nativeH) = quadOutputSize(quad)
    var scale = 1.0
    if let maxSide, max(nativeW, nativeH) > maxSide {
        scale = Double(maxSide) / Double(max(nativeW, nativeH))
    }
    let outW = max(1, Int((Double(nativeW) * scale).rounded()))
    let outH = max(1, Int((Double(nativeH) * scale).rounded()))
    let dst = [CGPoint(x: 0, y: 0), CGPoint(x: outW - 1, y: 0), CGPoint(x: outW - 1, y: outH - 1), CGPoint(x: 0, y: outH - 1)]
    guard let homography = Homography(from: quad, to: dst) else { return nil }
    guard let image = warpPerspective(raster, homography: homography, outputWidth: outW, outputHeight: outH, borderValue: padValue) else { return nil }
    return WarpResult(image: image, homography: homography, warpScale: scale, outputWidth: outW, outputHeight: outH)
}

public struct PadResult {
    public let image: CGImage
    public let gain: Double
    public let padX: Double
    public let padY: Double
}

/// Centers `crop` in a `canvas x canvas` square, downscaling (never
/// upscaling) to fit — matches Python `pad_no_upsize`.
public func padNoUpsize(_ crop: CGImage, canvas: Int, padValue: UInt8 = 114) -> PadResult? {
    let w = crop.width, h = crop.height
    guard w > 0, h > 0, canvas > 0 else { return nil }
    let scale = min(1.0, Double(canvas) / Double(max(w, h)))
    let newW = max(1, Int((Double(w) * scale).rounded()))
    let newH = max(1, Int((Double(h) * scale).rounded()))
    let padX = (Double(canvas) - Double(newW)) / 2.0
    let padY = (Double(canvas) - Double(newH)) / 2.0
    let x0 = Int(padX.rounded())
    let y0 = Int(padY.rounded())

    guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let ctx = CGContext(data: nil, width: canvas, height: canvas, bitsPerComponent: 8, bytesPerRow: 0, space: cs, bitmapInfo: bitmapInfo) else {
        return nil
    }
    let gray = Double(padValue) / 255.0
    ctx.setFillColor(CGColor(red: gray, green: gray, blue: gray, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: canvas, height: canvas))
    ctx.interpolationQuality = .high
    ctx.draw(crop, in: CGRect(x: x0, y: y0, width: newW, height: newH))
    guard let outImage = ctx.makeImage() else { return nil }
    return PadResult(image: outImage, gain: scale, padX: Double(x0), padY: Double(y0))
}

// MARK: - Minimum-area enclosing rectangle (rotating calipers)

public struct MinAreaRect {
    public let cx: Double
    public let cy: Double
    public let width: Double
    public let height: Double
    /// Radians. Reconstructing `OBBDetection(cx, cy, width, height, angleRad)`
    /// and reading its `.corners` reproduces this rectangle exactly.
    public let angleRad: Double
}

/// Minimum-area rectangle enclosing `points`, via rotating calipers over
/// their convex hull. Used in place of `cv2.minAreaRect` (PCA-based fits are
/// not exact for a general point set and were explicitly rejected for this
/// port).
public func minAreaRect(_ points: [CGPoint]) -> MinAreaRect? {
    let hull = convexHull(points)
    guard !hull.isEmpty else { return nil }
    if hull.count == 1 {
        return MinAreaRect(cx: Double(hull[0].x), cy: Double(hull[0].y), width: 0, height: 0, angleRad: 0)
    }
    if hull.count == 2 {
        let p = hull[0], q = hull[1]
        let angle = atan2(Double(q.y - p.y), Double(q.x - p.x))
        return MinAreaRect(cx: Double(p.x + q.x) / 2, cy: Double(p.y + q.y) / 2, width: dist(p, q), height: 0, angleRad: angle)
    }

    var best: MinAreaRect?
    var bestArea = Double.infinity
    let n = hull.count
    for i in 0..<n {
        let p1 = hull[i], p2 = hull[(i + 1) % n]
        let edgeAngle = atan2(Double(p2.y - p1.y), Double(p2.x - p1.x))
        let c = cos(edgeAngle), s = sin(edgeAngle)
        var minX = Double.infinity, maxX = -Double.infinity
        var minY = Double.infinity, maxY = -Double.infinity
        for pt in hull {
            let x = Double(pt.x), y = Double(pt.y)
            let rx = x * c + y * s
            let ry = -x * s + y * c
            minX = min(minX, rx); maxX = max(maxX, rx)
            minY = min(minY, ry); maxY = max(maxY, ry)
        }
        let width = maxX - minX
        let height = maxY - minY
        let area = width * height
        if area < bestArea {
            bestArea = area
            let centerRX = (minX + maxX) / 2
            let centerRY = (minY + maxY) / 2
            let cx = centerRX * c - centerRY * s
            let cy = centerRX * s + centerRY * c
            best = MinAreaRect(cx: cx, cy: cy, width: width, height: height, angleRad: edgeAngle)
        }
    }
    return best
}

/// Maps one detection from a crop's padded-canvas pixel space back into
/// scene pixels: undoes `padNoUpsize` (via `gain`, `padX`, `padY`), then
/// `homography.applyInverse` (warped-crop -> scene) on its four corners,
/// then re-fits the minimum-area enclosing rectangle. Matches Python
/// `map_det_from_crop`.
public func mapDetFromCrop(_ det: OBBDetection, homography: Homography, gain: Double, padX: Double, padY: Double) -> OBBDetection? {
    guard gain > 1e-9 else { return nil }
    let cxC = (det.cx - padX) / gain
    let cyC = (det.cy - padY) / gain
    let wC = det.w / gain
    let hC = det.h / gain

    let c = cos(det.angle), s = sin(det.angle)
    let v1x = c * wC / 2, v1y = s * wC / 2
    let v2x = -s * hC / 2, v2y = c * hC / 2
    let cornersWarped = [
        CGPoint(x: cxC + v1x + v2x, y: cyC + v1y + v2y),
        CGPoint(x: cxC + v1x - v2x, y: cyC + v1y - v2y),
        CGPoint(x: cxC - v1x - v2x, y: cyC - v1y - v2y),
        CGPoint(x: cxC - v1x + v2x, y: cyC - v1y + v2y),
    ]
    let sceneCorners = cornersWarped.map { homography.applyInverse($0) }
    guard let rect = minAreaRect(sceneCorners) else { return nil }
    var rw = rect.width, rh = rect.height
    var angleRad = rect.angleRad
    guard rw >= 1, rh >= 1 else { return nil }
    if rw < rh {
        swap(&rw, &rh)
        angleRad += .pi / 2
    }
    return OBBDetection(cx: rect.cx, cy: rect.cy, w: rw, h: rh, angle: angleRad, conf: det.conf)
}
