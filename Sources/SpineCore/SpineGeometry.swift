import CoreGraphics
import Foundation

// Shared geometry primitives for OBB detections and crop-plan polygons.
//
// Coordinate convention (matches tools/tiled_predict_obb.py and
// tools/layout_crop_predict.py): every `CGPoint`/`CGFloat` value that flows
// through this file is in *full-scene image pixels*, origin top-left, y
// increasing downward. The same primitives are reused for crop-local pixel
// coordinates inside `LayoutCropImage.swift` (post-unwarp, pre-scene-remap);
// callers are responsible for keeping the two spaces separate — there is no
// runtime tag, so never mix a scene-space point with a crop-space point in
// the same call. Core Image's bottom-left/y-up convention is never used
// here; it is only ever touched, and immediately converted away from, at the
// narrow rendering boundary in `LayoutCropImage.swift`.

/// Central home for numeric tolerances shared across rules/planner/geometry,
/// so intentional deviations from the Python thresholds are easy to spot.
public enum LayoutConstants {
    /// R5: how far (px) a crop-quad corner may sit outside the image and
    /// still be considered "finite" (matches Python `pad = 3.0`).
    public static let quadFinitePad: Double = 3.0
    /// R2: pairwise crop-quad intersection area (px^2) above which two crops
    /// are considered overlapping (matches Python `area > 4.0`).
    public static let overlapAreaEps: Double = 4.0
    /// R1: fractional tolerance between summed crop area and image area.
    public static let coverAreaTolFraction: Double = 0.02
    /// R1: default coarse sample grid resolution (48x48, matches Python).
    public static let defaultGridSamples = 48
    /// Default max angle difference (deg) to stay in the same orientation
    /// block / purity check (matches Python `angle_tol_deg`).
    public static let defaultAngleTolDeg: Double = 25.0
    /// `_point_in_poly` boundary-touch tolerance (matches Python `eps=1.0`).
    public static let pointOnSegmentEpsContainment: Double = 1.0
    /// `_point_strictly_inside` boundary-touch tolerance (Python `eps=1.5`).
    public static let pointOnSegmentEpsStrict: Double = 1.5
    /// SAT separating-axis / full-separation epsilon (Python `eps=1e-3`).
    public static let satSeparationEps: Double = 1e-3
    /// `_seam_usable` / `quad_from_lines` in-image margin (Python literal 2).
    public static let seamUsableImageMargin: Double = 2.0
    /// Minimum row-gap floor in px (Python `max(6.0, row_gap_k * mss)`).
    public static let minRowGapFloor: Double = 6.0
    /// Minimum column-gap floor in px (Python `max(8.0, col_gap_k * mss)`).
    public static let minColGapFloor: Double = 8.0
}

// MARK: - Angle helpers

/// Normalize an angle (radians) into `[0, pi)`, matching Python's `a % math.pi`.
public func normAngle(_ a: Double) -> Double {
    var r = a.truncatingRemainder(dividingBy: .pi)
    if r < 0 { r += .pi }
    return r
}

/// Smallest difference (radians, in `[0, pi/2]`) between two mod-pi angles.
public func angleDiff(_ a: Double, _ b: Double) -> Double {
    let d = abs(normAngle(a) - normAngle(b))
    return min(d, Double.pi - d)
}

/// Circular mean of mod-pi angles (doubles the angle, averages on the unit
/// circle, halves back) — matches Python `_circular_mean` / `circular_mean`.
public func circularMean(_ angles: [Double]) -> Double {
    guard !angles.isEmpty else { return 0.0 }
    let s = angles.reduce(0.0) { $0 + sin(2 * normAngle($1)) }
    let c = angles.reduce(0.0) { $0 + cos(2 * normAngle($1)) }
    return normAngle(0.5 * atan2(s, c))
}

// MARK: - OBB detection

/// One oriented bounding box detection in full-scene pixel coordinates.
public struct OBBDetection: Sendable {
    public var cx: Double
    public var cy: Double
    public var w: Double
    public var h: Double
    public var angle: Double   // radians
    public var conf: Float
    /// Stable identity independent of field values, so callers can classify
    /// "first pass vs. net-new after merge" without relying on Swift value
    /// identity (which Python's `id(obj)` approach implicitly assumed).
    public let id: UUID

    public init(cx: Double, cy: Double, w: Double, h: Double, angle: Double, conf: Float, id: UUID = UUID()) {
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle
        self.conf = conf
        self.id = id
    }

    /// Four corners (front-right, back-right, back-left, front-left of the
    /// rotated rect), forming a closed quad. Matches Python `Det.corners()`.
    public var corners: [CGPoint] {
        let c = cos(angle), s = sin(angle)
        let v1x = c * w / 2, v1y = s * w / 2
        let v2x = -s * h / 2, v2y = c * h / 2
        return [
            CGPoint(x: cx + v1x + v2x, y: cy + v1y + v2y),
            CGPoint(x: cx + v1x - v2x, y: cy + v1y - v2y),
            CGPoint(x: cx - v1x - v2x, y: cy - v1y - v2y),
            CGPoint(x: cx - v1x + v2x, y: cy - v1y + v2y),
        ]
    }

    public func offset(dx: Double, dy: Double) -> OBBDetection {
        OBBDetection(cx: cx + dx, cy: cy + dy, w: w, h: h, angle: angle, conf: conf, id: id)
    }

    /// Absolute image-space direction (mod pi) of the OBB's *long* axis.
    /// The model doesn't guarantee `w` is the long side, so this compares
    /// `w` vs `h` before reading off `angle` — mirrors Python `long_axis_angle`.
    public func longAxisAngle() -> Double {
        let base = w >= h ? angle : angle + .pi / 2
        return normAngle(base)
    }

    /// x-span of `corners` (min, max).
    public var xSpan: (Double, Double) {
        let xs = corners.map { Double($0.x) }
        return (xs.min()!, xs.max()!)
    }

    /// y-span of `corners` (min, max).
    public var ySpan: (Double, Double) {
        let ys = corners.map { Double($0.y) }
        return (ys.min()!, ys.max()!)
    }
}

/// JSON-serializable snapshot matching Python `dataclasses.asdict(Det)`.
public struct OBBDetectionJSON: Codable {
    public let cx, cy, w, h, angle: Double
    public let conf: Float

    public init(_ d: OBBDetection) {
        cx = d.cx; cy = d.cy; w = d.w; h = d.h; angle = d.angle; conf = d.conf
    }
}

public func dist(_ a: CGPoint, _ b: CGPoint) -> Double {
    Foundation.hypot(Double(a.x - b.x), Double(a.y - b.y))
}

// MARK: - Lines (ax + by + c = 0)

/// A line in implicit form `ax + by + c = 0`. Mirrors the `(a, b, c)` tuples
/// used throughout `tools/layout_crop_predict.py`.
public struct Line: Equatable {
    public var a: Double
    public var b: Double
    public var c: Double

    public init(a: Double, b: Double, c: Double) {
        self.a = a
        self.b = b
        self.c = c
    }

    public static func vertical(atX x: Double) -> Line { Line(a: 1, b: 0, c: -x) }
    public static func horizontal(atY y: Double) -> Line { Line(a: 0, b: 1, c: -y) }

    public static func through(_ p: CGPoint, _ q: CGPoint) -> Line {
        let a = Double(p.y - q.y)
        let b = Double(q.x - p.x)
        let c = -(a * Double(p.x) + b * Double(p.y))
        return Line(a: a, b: b, c: c)
    }

    /// Signed distance-like value (not normalized) — matches Python `_signed`.
    public func signed(_ p: CGPoint) -> Double {
        a * Double(p.x) + b * Double(p.y) + c
    }

    /// y at a given x, or 0 if the line is (near-)vertical — matches
    /// Python `y_on_line` exactly, including its degenerate fallback.
    public func yValue(atX x: Double) -> Double {
        guard abs(b) >= 1e-9 else { return 0.0 }
        return -(a * x + c) / b
    }

    /// Intersection point, or `nil` if the lines are parallel — matches
    /// Python `intersect_lines`.
    public func intersection(with other: Line) -> CGPoint? {
        let det = a * other.b - other.a * b
        guard abs(det) >= 1e-9 else { return nil }
        let x = (other.b * (-c) - b * (-other.c)) / det
        let y = (a * (-other.c) - other.a * (-c)) / det
        return CGPoint(x: x, y: y)
    }
}

/// Least-squares line through `points`, preferring a near-horizontal
/// (`y = m x + b`) fit — matches Python `fit_line`. Falls back to a
/// horizontal line at the mean y if the points are (numerically) vertical,
/// which `numpy.linalg.lstsq` would instead resolve via its pseudo-inverse;
/// this is an intentional, documented deviation since that path is not
/// exercised by any well-formed shelf/column scene.
public func fitLine(_ points: [CGPoint]) -> Line {
    guard points.count >= 2 else {
        let y = points.first.map { Double($0.y) } ?? 0.0
        return .horizontal(atY: y)
    }
    let n = Double(points.count)
    let xs = points.map { Double($0.x) }
    let ys = points.map { Double($0.y) }
    let sumX = xs.reduce(0, +)
    let sumY = ys.reduce(0, +)
    let sumXY = zip(xs, ys).reduce(0.0) { $0 + $1.0 * $1.1 }
    let sumXX = xs.reduce(0.0) { $0 + $1 * $1 }
    let denom = n * sumXX - sumX * sumX
    guard abs(denom) >= 1e-9 else {
        return .horizontal(atY: sumY / n)
    }
    let m = (n * sumXY - sumX * sumY) / denom
    let b = (sumY - m * sumX) / n
    return Line(a: -m, b: 1.0, c: -b)
}

// MARK: - Polygon primitives

/// Signed-area magnitude of a (possibly non-convex, simple) polygon.
public func polygonArea(_ pts: [CGPoint]) -> Double {
    guard pts.count >= 3 else { return 0.0 }
    var area = 0.0
    for i in 0..<pts.count {
        let p = pts[i], q = pts[(i + 1) % pts.count]
        area += Double(p.x * q.y - q.x * p.y)
    }
    return abs(area) / 2.0
}

/// Reorders `pts` to counter-clockwise (by signed area sign) if needed.
public func ensureCCW(_ pts: [CGPoint]) -> [CGPoint] {
    var signedArea = 0.0
    for i in 0..<pts.count {
        let p = pts[i], q = pts[(i + 1) % pts.count]
        signedArea += Double(p.x * q.y - q.x * p.y)
    }
    return signedArea < 0 ? pts.reversed() : pts
}

/// Sutherland–Hodgman clip of `subject` against the half-plane left of
/// directed edge `edgeA -> edgeB` (for a CCW clip polygon).
public func clipPolygon(_ subject: [CGPoint], edgeA: CGPoint, edgeB: CGPoint) -> [CGPoint] {
    guard !subject.isEmpty else { return [] }
    func side(_ p: CGPoint) -> Double {
        Double((edgeB.x - edgeA.x) * (p.y - edgeA.y) - (edgeB.y - edgeA.y) * (p.x - edgeA.x))
    }
    func intersect(_ p: CGPoint, _ q: CGPoint) -> CGPoint {
        let a1 = Double(edgeB.y - edgeA.y), b1 = Double(edgeA.x - edgeB.x)
        let c1 = a1 * Double(edgeA.x) + b1 * Double(edgeA.y)
        let a2 = Double(q.y - p.y), b2 = Double(p.x - q.x)
        let c2 = a2 * Double(p.x) + b2 * Double(p.y)
        let det = a1 * b2 - a2 * b1
        guard abs(det) >= 1e-12 else { return p }
        return CGPoint(x: (b2 * c1 - b1 * c2) / det, y: (a1 * c2 - a2 * c1) / det)
    }
    var out: [CGPoint] = []
    for i in 0..<subject.count {
        let current = subject[i]
        let previous = subject[(i + subject.count - 1) % subject.count]
        let curIn = side(current) >= 0
        let prevIn = side(previous) >= 0
        if curIn {
            if !prevIn { out.append(intersect(previous, current)) }
            out.append(current)
        } else if prevIn {
            out.append(intersect(previous, current))
        }
    }
    return out
}

/// Intersection area of two polygons via repeated half-plane clipping. Exact
/// only while the *clip* polygon `b` is convex; use
/// `simplePolyIntersectionArea` when either side may be a non-convex piece.
public func polyIntersectionArea(_ a: [CGPoint], _ b: [CGPoint]) -> Double {
    var inter = a
    for i in 0..<b.count {
        if inter.isEmpty { break }
        inter = clipPolygon(inter, edgeA: b[i], edgeB: b[(i + 1) % b.count])
    }
    return polygonArea(inter)
}

/// True if `(x, y)` lies on segment `(x1,y1)-(x2,y2)` within `eps` (also
/// used as the boundary-touch tolerance for containment checks below).
public func pointOnSegment(_ x: Double, _ y: Double, _ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double, eps: Double = 1.0) -> Bool {
    let cross = abs((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1))
    if cross > eps * max(1.0, Foundation.hypot(x2 - x1, y2 - y1)) { return false }
    let dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -eps { return false }
    if dot > (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + eps { return false }
    return true
}

/// Point-in-polygon (ray casting), with boundary touches counted as inside.
public func pointInPolygon(_ x: Double, _ y: Double, _ poly: [CGPoint]) -> Bool {
    let n = poly.count
    guard n >= 3 else { return false }
    var inside = false
    var j = n - 1
    for i in 0..<n {
        let xi = Double(poly[i].x), yi = Double(poly[i].y)
        let xj = Double(poly[j].x), yj = Double(poly[j].y)
        if pointOnSegment(x, y, xi, yi, xj, yj, eps: LayoutConstants.pointOnSegmentEpsContainment) {
            return true
        }
        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi)
        if intersect { inside.toggle() }
        j = i
    }
    return inside
}

/// Point strictly inside `poly` (inside AND not touching any edge within a
/// slightly wider tolerance than `pointInPolygon`'s own boundary check).
public func pointStrictlyInside(_ x: Double, _ y: Double, _ poly: [CGPoint]) -> Bool {
    guard pointInPolygon(x, y, poly) else { return false }
    let n = poly.count
    for i in 0..<n {
        let p1 = poly[i], p2 = poly[(i + 1) % n]
        if pointOnSegment(x, y, Double(p1.x), Double(p1.y), Double(p2.x), Double(p2.y), eps: LayoutConstants.pointOnSegmentEpsStrict) {
            return false
        }
    }
    return true
}

/// Andrew's monotone-chain convex hull, CCW, deduplicated, no repeated last point.
public func convexHull(_ points: [CGPoint]) -> [CGPoint] {
    let sortedPts = points.sorted { ($0.x, $0.y) < ($1.x, $1.y) }
    var dedup: [CGPoint] = []
    for p in sortedPts where dedup.last != p {
        dedup.append(p)
    }
    guard dedup.count > 2 else { return dedup }

    func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    var lower: [CGPoint] = []
    for p in dedup {
        while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
            lower.removeLast()
        }
        lower.append(p)
    }
    var upper: [CGPoint] = []
    for p in dedup.reversed() {
        while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
            upper.removeLast()
        }
        upper.append(p)
    }
    return Array(lower.dropLast()) + Array(upper.dropLast())
}

// MARK: - Separating lines (SAT)

/// True iff every point in `cornersA` is strictly on one side of `line` and
/// every point in `cornersB` is strictly on the other — matches Python
/// `_fully_separates`.
public func fullySeparates(_ line: Line, _ cornersA: [CGPoint], _ cornersB: [CGPoint], eps: Double = LayoutConstants.satSeparationEps) -> Bool {
    let sa = cornersA.map { line.signed($0) }
    let sb = cornersB.map { line.signed($0) }
    guard let maxA = sa.max(), let minA = sa.min(), let maxB = sb.max(), let minB = sb.min() else { return false }
    return (maxA < -eps && minB > eps) || (minA > eps && maxB < -eps)
}

/// A seam line must cross both a row band's top and bottom boundary lines
/// at finite, in-image points to be usable as a column boundary — matches
/// Python `_seam_usable`.
public func seamUsable(_ line: Line, topLine: Line, bottomLine: Line, imgW: Int, imgH: Int) -> Bool {
    guard let topPt = line.intersection(with: topLine),
          let botPt = line.intersection(with: bottomLine) else { return false }
    for p in [topPt, botPt] {
        guard p.x.isFinite, p.y.isFinite else { return false }
        let margin = LayoutConstants.seamUsableImageMargin
        if Double(p.x) < -margin || Double(p.y) < -margin
            || Double(p.x) > Double(imgW) + margin || Double(p.y) > Double(imgH) + margin {
            return false
        }
    }
    return true
}

public enum SeamPreference {
    case vertical
    case horizontal
}

/// Exact linear-separability test via the separating axis theorem — matches
/// Python `_sat_separating_line`. Returns `nil` if no hull-edge normal both
/// separates the two point sets and yields a seam usable given
/// `topLine`/`bottomLine`.
public func satSeparatingLine(
    cornersA: [CGPoint],
    cornersB: [CGPoint],
    topLine: Line,
    bottomLine: Line,
    imgW: Int,
    imgH: Int,
    prefer: SeamPreference = .vertical
) -> Line? {
    let hullA = convexHull(cornersA)
    let hullB = convexHull(cornersB)
    guard hullA.count >= 2, hullB.count >= 2 else { return nil }

    var axes: [(Double, Double)] = []
    for hull in [hullA, hullB] {
        let n = hull.count
        for i in 0..<n {
            let p1 = hull[i], p2 = hull[(i + 1) % n]
            let ex = Double(p2.x - p1.x), ey = Double(p2.y - p1.y)
            let length = Foundation.hypot(ex, ey)
            if length > 1e-9 {
                axes.append((-ey / length, ex / length))
            }
        }
    }

    var bestLine: Line? = nil
    var bestScore = Double.infinity
    for (nx, ny) in axes {
        let projA = cornersA.map { nx * Double($0.x) + ny * Double($0.y) }
        let projB = cornersB.map { nx * Double($0.x) + ny * Double($0.y) }
        guard let maxA = projA.max(), let minA = projA.min(),
              let maxB = projB.max(), let minB = projB.min() else { continue }
        let t: Double
        if maxA < minB {
            t = 0.5 * (maxA + minB)
        } else if maxB < minA {
            t = 0.5 * (maxB + minA)
        } else {
            continue
        }
        let line = Line(a: nx, b: ny, c: -t)
        guard seamUsable(line, topLine: topLine, bottomLine: bottomLine, imgW: imgW, imgH: imgH) else { continue }
        let score = prefer == .vertical ? abs(ny) : abs(nx)
        if score < bestScore {
            bestScore = score
            bestLine = line
        }
    }
    return bestLine
}

// MARK: - Rotated IoU

/// Rotated (OBB) intersection-over-union via convex clipping — matches
/// Python `rotated_iou` / the existing implementation in `bookspines.swift`.
public func rotatedIoU(_ a: OBBDetection, _ b: OBBDetection) -> Double {
    let reach = (Foundation.hypot(a.w, a.h) + Foundation.hypot(b.w, b.h)) / 2
    guard Foundation.hypot(a.cx - b.cx, a.cy - b.cy) < reach else { return 0 }

    let quadA = ensureCCW(a.corners)
    let quadB = ensureCCW(b.corners)
    var inter = quadA
    for i in 0..<quadB.count {
        if inter.isEmpty { break }
        inter = clipPolygon(inter, edgeA: quadB[i], edgeB: quadB[(i + 1) % quadB.count])
    }
    let interArea = polygonArea(inter)
    guard interArea > 0 else { return 0 }
    let union = polygonArea(quadA) + polygonArea(quadB) - interArea
    return union > 0 ? interArea / union : 0
}
