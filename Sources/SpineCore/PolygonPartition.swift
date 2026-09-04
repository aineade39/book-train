import CoreGraphics
import Foundation

// Stage 3 of the jigsaw-zoom plan (~/dev/book-id-design/docs/
// jigsaw-zoom-requirements.md R-2: "Pieces may be any shape. There is no
// quad or straight-seam requirement."): the polygon primitives the
// free-form cutter needs and the incumbent quad planner never did.
//
// Two gaps in `SpineGeometry.swift` motivate this file:
//   1. `polyIntersectionArea` clips half-plane by half-plane, so it is only
//      exact when the clip polygon is convex. A staircase piece is not.
//   2. Nothing there splits a polygon with a polyline.
//
// A "monotone seam" is a staircase polyline that advances monotonically
// along one axis while its position varies along the other. It generalizes
// the incumbent planner's straight seams (a straight cut is the degenerate
// case) and can weave between interlocked OBBs that no straight line can
// separate — the A-4 pressure point of the quad cutter.

// MARK: - Bounds

/// Axis-aligned bounds of a polygon, or `nil` if fewer than three of its
/// points are finite.
public func polygonBounds(_ pts: [CGPoint]) -> (x0: Double, y0: Double, x1: Double, y1: Double)? {
    let finite = pts.filter { $0.x.isFinite && $0.y.isFinite }
    guard finite.count >= 3 else { return nil }
    let xs = finite.map { Double($0.x) }
    let ys = finite.map { Double($0.y) }
    return (xs.min()!, ys.min()!, xs.max()!, ys.max()!)
}

/// Extent of `pts` along `axis`.
public func polygonExtent(_ pts: [CGPoint], axis: ProfileAxis) -> Double {
    guard let b = polygonBounds(pts) else { return 0 }
    return axis == .x ? b.x1 - b.x0 : b.y1 - b.y0
}

/// The dimension that decides whether a piece still needs cutting (R-1b):
/// the longer side of the rectangle the engine will letterbox it into —
/// its minimum-area rect when pieces may be rotated (R-3), otherwise its
/// axis-aligned bounding box.
public func letterboxSourceDim(_ polygon: [CGPoint], rotate: Bool) -> Double {
    if rotate, let mar = minAreaRect(polygon) {
        return max(mar.width, mar.height)
    }
    guard let b = polygonBounds(polygon) else { return 0 }
    return max(b.x1 - b.x0, b.y1 - b.y0)
}

// MARK: - Rings

/// Drops non-finite and consecutive-duplicate points, and un-closes an
/// explicitly closed ring, so `count` is the true vertex count.
func dedupeRing(_ pts: [CGPoint], eps: Double = 1e-6) -> [CGPoint] {
    var out: [CGPoint] = []
    for p in pts {
        guard p.x.isFinite, p.y.isFinite else { continue }
        if let last = out.last, dist(last, p) <= eps { continue }
        out.append(p)
    }
    while out.count >= 2, dist(out[0], out[out.count - 1]) <= eps {
        out.removeLast()
    }
    return out
}

/// True when `pts` is convex (either winding) — the precondition for
/// `polyIntersectionArea`'s half-plane clipping to be exact.
public func isConvexPolygon(_ pts: [CGPoint]) -> Bool {
    let ring = dedupeRing(pts)
    guard ring.count >= 3 else { return false }
    let n = ring.count
    var sign = 0
    for i in 0..<n {
        let a = ring[i], b = ring[(i + 1) % n], c = ring[(i + 2) % n]
        let cross = Double((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x))
        if abs(cross) < 1e-9 { continue }
        let s = cross > 0 ? 1 : -1
        if sign == 0 {
            sign = s
        } else if s != sign {
            return false
        }
    }
    return true
}

// MARK: - Triangulation

private func crossZ(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> Double {
    Double((a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x))
}

private func pointInTriangle(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint, _ c: CGPoint, includeBoundary: Bool) -> Bool {
    let d1 = crossZ(a, b, p), d2 = crossZ(b, c, p), d3 = crossZ(c, a, p)
    let eps = 1e-9
    if includeBoundary {
        return (d1 >= -eps && d2 >= -eps && d3 >= -eps) || (d1 <= eps && d2 <= eps && d3 <= eps)
    }
    return (d1 > eps && d2 > eps && d3 > eps) || (d1 < -eps && d2 < -eps && d3 < -eps)
}

/// Ear-clipping triangulation of a simple (hole-free) polygon. Returns an
/// empty array when the ring is degenerate or no ear can be found, which
/// callers must treat as "triangulation unavailable" rather than "zero
/// area" — `simplePolyIntersectionArea` checks the reconstructed area for
/// exactly that reason.
public func triangulatePolygon(_ pts: [CGPoint]) -> [[CGPoint]] {
    var ring = ensureCCW(dedupeRing(pts))
    guard ring.count >= 3 else { return [] }
    var triangles: [[CGPoint]] = []
    var iterations = 0
    let iterationCap = 4 * ring.count * ring.count + 16

    while ring.count > 3 {
        iterations += 1
        guard iterations <= iterationCap else { return [] }
        let n = ring.count
        var clipped = false
        // Two passes. The first rejects any ear with another vertex merely
        // *touching* it, because clipping such an ear leaves a degenerate
        // remainder that no later pass can triangulate — exactly what a
        // staircase piece produces, where every second vertex sits on the
        // diagonal an over-eager first ear would span. The second pass
        // relaxes to strictly-inside so a polygon whose only ears touch a
        // collinear neighbour still makes progress.
        for includeBoundary in [true, false] {
            for i in 0..<n {
                let prevIdx = (i + n - 1) % n
                let nextIdx = (i + 1) % n
                let prev = ring[prevIdx], cur = ring[i], next = ring[nextIdx]
                guard crossZ(prev, cur, next) > 1e-9 else { continue }
                var isEar = true
                for j in 0..<n where j != i && j != prevIdx && j != nextIdx {
                    if pointInTriangle(ring[j], prev, cur, next, includeBoundary: includeBoundary) {
                        isEar = false
                        break
                    }
                }
                guard isEar else { continue }
                triangles.append([prev, cur, next])
                ring.remove(at: i)
                clipped = true
                break
            }
            if clipped { break }
        }
        guard clipped else { return [] }
    }
    triangles.append(ring)
    return triangles
}

/// Intersection area of two *simple* polygons, convex or not.
///
/// Half-plane clipping is exact as long as the *clip* polygon is convex (a
/// non-convex subject only picks up zero-width bridges, which contribute no
/// area), so a convex partner on either side is enough. When both are
/// non-convex, both get triangulated — triangles are convex, so the clip is
/// exact again — and the pairwise areas summed. If either triangulation
/// fails to reproduce its polygon's area, this falls back to the convex
/// approximation rather than under-reporting an overlap to the rules.
public func simplePolyIntersectionArea(_ a: [CGPoint], _ b: [CGPoint]) -> Double {
    guard let ba = polygonBounds(a), let bb = polygonBounds(b) else { return 0 }
    if ba.x1 <= bb.x0 || bb.x1 <= ba.x0 || ba.y1 <= bb.y0 || bb.y1 <= ba.y0 { return 0 }

    let ccwA = ensureCCW(dedupeRing(a))
    let ccwB = ensureCCW(dedupeRing(b))
    guard ccwA.count >= 3, ccwB.count >= 3 else { return 0 }
    if isConvexPolygon(ccwB) { return polyIntersectionArea(ccwA, ccwB) }
    if isConvexPolygon(ccwA) { return polyIntersectionArea(ccwB, ccwA) }

    let triA = triangulatePolygon(ccwA)
    let triB = triangulatePolygon(ccwB)
    let areaA = polygonArea(ccwA), areaB = polygonArea(ccwB)
    let sumA = triA.reduce(0.0) { $0 + polygonArea($1) }
    let sumB = triB.reduce(0.0) { $0 + polygonArea($1) }
    guard !triA.isEmpty, !triB.isEmpty,
          abs(sumA - areaA) <= max(1.0, 0.01 * areaA),
          abs(sumB - areaB) <= max(1.0, 0.01 * areaB)
    else {
        return polyIntersectionArea(ccwA, ccwB)
    }

    var total = 0.0
    for t1 in triA {
        guard let b1 = polygonBounds(t1) else { continue }
        for t2 in triB {
            guard let b2 = polygonBounds(t2) else { continue }
            if b1.x1 <= b2.x0 || b2.x1 <= b1.x0 || b1.y1 <= b2.y0 || b2.y1 <= b1.y0 { continue }
            total += polyIntersectionArea(ensureCCW(t1), ensureCCW(t2))
        }
    }
    return total
}

// MARK: - Monotone seams

/// A staircase cut. `splitAxis == .x` means a vertical-ish seam whose
/// points advance in y and whose x varies (it separates left from right);
/// `.y` is the transpose.
public struct MonotoneSeam {
    /// Ordered along the advance axis. The first and last points lie
    /// outside the polygon being cut, so the seam crosses it completely.
    public let points: [CGPoint]
    public let splitAxis: ProfileAxis

    public init(points: [CGPoint], splitAxis: ProfileAxis) {
        self.points = points
        self.splitAxis = splitAxis
    }

    /// Advance-axis coordinate of `p` (the coordinate the seam is monotone in).
    func advance(_ p: CGPoint) -> Double {
        splitAxis == .x ? Double(p.y) : Double(p.x)
    }

    /// Split-axis coordinate of `p` (the coordinate the seam varies in).
    func position(_ p: CGPoint) -> Double {
        splitAxis == .x ? Double(p.x) : Double(p.y)
    }
}

/// Segment-segment intersection. Returns the parameters along each segment
/// and the point, or `nil` for parallel/non-crossing segments. Endpoint
/// touches count (`t`/`u` in `[0, 1]`).
func segmentIntersection(
    _ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, _ p4: CGPoint
) -> (t: Double, u: Double, point: CGPoint)? {
    let x1 = Double(p1.x), y1 = Double(p1.y)
    let x2 = Double(p2.x), y2 = Double(p2.y)
    let x3 = Double(p3.x), y3 = Double(p3.y)
    let x4 = Double(p4.x), y4 = Double(p4.y)
    let rx = x2 - x1, ry = y2 - y1
    let sx = x4 - x3, sy = y4 - y3
    let denom = rx * sy - ry * sx
    guard abs(denom) > 1e-12 else { return nil }
    let t = ((x3 - x1) * sy - (y3 - y1) * sx) / denom
    let u = ((x3 - x1) * ry - (y3 - y1) * rx) / denom
    guard t >= -1e-9, t <= 1 + 1e-9, u >= -1e-9, u <= 1 + 1e-9 else { return nil }
    return (t, u, CGPoint(x: x1 + t * rx, y: y1 + t * ry))
}

private struct SeamCrossing {
    let edgeIndex: Int
    let edgeT: Double
    /// Position along the seam polyline: segment index plus its own `u`.
    let seamPos: Double
    let point: CGPoint
}

/// Cuts `polygon` in two along `seam`.
///
/// Requires the seam to cross the polygon boundary exactly twice — the
/// case where "the two sides of the seam" is unambiguous. Returns `nil`
/// otherwise, and also when the two halves fail to conserve the parent's
/// area, so a caller can fall back (a different axis, or the A-4 leaf)
/// instead of emitting a plan that would break the strict partition.
public func splitPolygonByMonotoneSeam(
    _ polygon: [CGPoint],
    seam: MonotoneSeam
) -> (a: [CGPoint], b: [CGPoint])? {
    let ring = dedupeRing(polygon)
    guard ring.count >= 3, seam.points.count >= 2 else { return nil }

    var crossings: [SeamCrossing] = []
    for e in 0..<ring.count {
        let p1 = ring[e], p2 = ring[(e + 1) % ring.count]
        for s in 0..<(seam.points.count - 1) {
            guard let hit = segmentIntersection(p1, p2, seam.points[s], seam.points[s + 1]) else { continue }
            let crossing = SeamCrossing(
                edgeIndex: e, edgeT: hit.t,
                seamPos: Double(s) + hit.u, point: hit.point
            )
            // A seam vertex landing exactly on a polygon edge, or a
            // polygon vertex on the seam, yields the same point twice.
            if crossings.contains(where: { dist($0.point, crossing.point) <= 1e-6 }) { continue }
            crossings.append(crossing)
        }
    }
    guard crossings.count == 2 else { return nil }

    let ordered = crossings.sorted { ($0.edgeIndex, $0.edgeT) < ($1.edgeIndex, $1.edgeT) }
    let first = ordered[0], second = ordered[1]

    /// Ring vertices `v[i+1] ... v[j]` — those lying between a crossing on
    /// edge `i` and a crossing on edge `j`, walking the boundary forward.
    /// With `i == j`, `wrapWholeRing` picks between the empty arc (the
    /// short way, no vertex between the two crossings) and the full ring.
    func vertices(fromEdge i: Int, toEdge j: Int, wrapWholeRing: Bool) -> [CGPoint] {
        if i == j, !wrapWholeRing { return [] }
        var out: [CGPoint] = []
        var idx = (i + 1) % ring.count
        while out.count <= ring.count {
            out.append(ring[idx])
            if idx == j { break }
            idx = (idx + 1) % ring.count
        }
        return out
    }

    let arcForward = vertices(fromEdge: first.edgeIndex, toEdge: second.edgeIndex, wrapWholeRing: false)
    let arcBackward = vertices(fromEdge: second.edgeIndex, toEdge: first.edgeIndex, wrapWholeRing: true)

    // Seam vertices strictly between the crossings, ordered first -> second.
    let lo = min(first.seamPos, second.seamPos)
    let hi = max(first.seamPos, second.seamPos)
    var interior: [CGPoint] = []
    for s in 0..<seam.points.count {
        let pos = Double(s)
        if pos > lo + 1e-9, pos < hi - 1e-9 { interior.append(seam.points[s]) }
    }
    if first.seamPos > second.seamPos { interior.reverse() }

    let pieceA = dedupeRing([first.point] + arcForward + [second.point] + interior.reversed())
    let pieceB = dedupeRing([second.point] + arcBackward + [first.point] + interior)
    guard pieceA.count >= 3, pieceB.count >= 3 else { return nil }

    let parentArea = polygonArea(ring)
    let areaA = polygonArea(pieceA), areaB = polygonArea(pieceB)
    guard areaA > 1, areaB > 1 else { return nil }
    guard abs(areaA + areaB - parentArea) <= max(2.0, 0.005 * parentArea) else { return nil }
    return (pieceA, pieceB)
}
