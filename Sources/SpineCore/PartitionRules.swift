import CoreGraphics
import Foundation

// Stage 3 of the jigsaw-zoom plan: the incumbent quad rules
// (`LayoutCropRules.swift`) generalized to arbitrary polygons, per
// ~/dev/book-id-design/docs/jigsaw-zoom-requirements.md A-7 — "the prior
// rules file (cover, no overlap, no OBB cut, every OBB owned, finite
// geometry) generalizes directly from quads to arbitrary polygons and is
// worth keeping as the verification oracle for any cutter implementation,
// plus two new checks: recursion terminates and every leaf's letterbox
// scale is within the threshold."
//
// Three things change versus `verifyPlan`:
//   * pieces are polygons with any vertex count, so the no-OBB-cut edge
//     walk is over `polygon.count` edges rather than exactly four;
//   * overlap area goes through `simplePolyIntersectionArea`, which stays
//     exact for the non-convex staircase pieces the free-form cutter emits;
//   * coverage is checked against an explicit `region` polygon rather than
//     the whole image, because a recursive cutter partitions a *piece*.
//
// `verifyPlan` delegates here with `region` set to the image rectangle, so
// quads remain a special case and the Python-parity contract holds.

/// One member of a partition: a polygon plus the indices of the detections
/// it owns. The polygon generalization of `CropPlan`.
public struct PartitionPiece {
    public let name: String
    public let polygon: [CGPoint]
    public let memberIndices: [Int]

    public init(name: String, polygon: [CGPoint], memberIndices: [Int] = []) {
        self.name = name
        self.polygon = polygon
        self.memberIndices = memberIndices
    }
}

/// The smallest-area piece whose polygon contains `(x, y)`, or `nil` if
/// none do — the polygon form of `owningCrop`.
public func owningPiece(x: Double, y: Double, pieces: [PartitionPiece]) -> PartitionPiece? {
    let hits = pieces.filter { pointInPolygon(x, y, $0.polygon) }
    return hits.min { polygonArea($0.polygon) < polygonArea($1.polygon) }
}

/// `%.0f`-style rounding for detail strings (renders non-finite values as
/// "nan"/"inf" instead of trapping, matching `LayoutCropRules`).
private func fmt0(_ x: Double) -> String { String(format: "%.0f", x) }

/// Rectangle polygon (TL, TR, BR, BL) for an image of `imgW x imgH`.
public func imageRegion(imgW: Int, imgH: Int) -> [CGPoint] {
    [
        CGPoint(x: 0, y: 0),
        CGPoint(x: imgW, y: 0),
        CGPoint(x: imgW, y: imgH),
        CGPoint(x: 0, y: imgH),
    ]
}

/// Hard/soft rule results for a partition of `region` into `pieces`. Rule
/// names, hard flags and pass semantics match `verifyPlan` exactly.
public func verifyPartition(
    dets: [OBBDetection],
    pieces: [PartitionPiece],
    region: [CGPoint],
    imgW: Int,
    imgH: Int,
    grid: Int = LayoutConstants.defaultGridSamples,
    angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg
) -> [RuleResult] {
    var results: [RuleResult] = []
    guard !pieces.isEmpty else {
        return [RuleResult(rule: "R5_QUAD_FINITE", hard: true, ok: false, detail: "no crops")]
    }

    // R5: every corner finite and within a small pad of the image.
    let pad = LayoutConstants.quadFinitePad
    var badFinite: [String] = []
    for p in pieces {
        for pt in p.polygon {
            if !(Double(pt.x).isFinite && Double(pt.y).isFinite) {
                badFinite.append(p.name)
                break
            }
            if Double(pt.x) < -pad || Double(pt.y) < -pad || Double(pt.x) > Double(imgW) + pad || Double(pt.y) > Double(imgH) + pad {
                badFinite.append(String(format: "%@@(%.0f,%.0f)", p.name, Double(pt.x), Double(pt.y)))
                break
            }
        }
    }
    results.append(RuleResult(
        rule: "R5_QUAD_FINITE", hard: true, ok: badFinite.isEmpty,
        detail: badFinite.isEmpty ? "ok" : "bad corners: \(Array(badFinite.prefix(8)))"
    ))

    // R2: pairwise overlap.
    var overlaps: [String] = []
    let polys = pieces.map { ensureCCW($0.polygon) }
    for i in 0..<polys.count {
        for j in (i + 1)..<polys.count {
            let area = simplePolyIntersectionArea(polys[i], polys[j])
            if area > LayoutConstants.overlapAreaEps {
                overlaps.append("\(pieces[i].name)/\(pieces[j].name):\(fmt0(area))")
            }
        }
    }
    results.append(RuleResult(
        rule: "R2_NO_OVERLAP", hard: true, ok: overlaps.isEmpty,
        detail: overlaps.isEmpty ? "ok" : "overlaps \(Array(overlaps.prefix(6)))"
    ))

    // R1: coverage — total area within tolerance of the region's, and every
    // grid sample that falls inside the region covered by some piece.
    let areaSum = polys.reduce(0.0) { $0 + polygonArea($1) }
    let expected = polygonArea(region)
    let areaOk = expected > 0 && abs(areaSum - expected) <= LayoutConstants.coverAreaTolFraction * expected
    var miss = 0
    var multi = 0
    if let bounds = polygonBounds(region), grid > 0 {
        let stepX = (bounds.x1 - bounds.x0) / Double(grid)
        let stepY = (bounds.y1 - bounds.y0) / Double(grid)
        for gy in 0..<grid {
            for gx in 0..<grid {
                let x = bounds.x0 + (Double(gx) + 0.5) * stepX
                let y = bounds.y0 + (Double(gy) + 0.5) * stepY
                guard pointInPolygon(x, y, region) else { continue }
                let hits = pieces.reduce(0) { $0 + (pointInPolygon(x, y, $1.polygon) ? 1 : 0) }
                if hits == 0 { miss += 1 } else if hits > 2 { multi += 1 }
            }
        }
    }
    results.append(RuleResult(
        rule: "R1_COVER_IMAGE", hard: true, ok: areaOk && miss == 0,
        detail: "area=\(fmt0(areaSum))/\(fmt0(expected)) miss=\(miss) multi=\(multi)"
    ))

    // R3 + R4: OBB ownership and no-cut.
    var orphan: [Int] = []
    var cutViolations: [Int] = []
    var split: [String] = []
    for (di, det) in dets.enumerated() {
        guard let centerOwner = owningPiece(x: det.cx, y: det.cy, pieces: pieces) else {
            orphan.append(di)
            continue
        }
        var bad: Set<String> = []
        let obbCorners = det.corners
        for corner in obbCorners {
            for p in pieces where p.name != centerOwner.name {
                if pointStrictlyInside(Double(corner.x), Double(corner.y), p.polygon) {
                    bad.insert(p.name)
                }
            }
        }
        // Piece edge through the OBB: any non-owner edge whose midpoint is
        // strictly inside it.
        for p in pieces where p.name != centerOwner.name {
            let q = p.polygon
            guard q.count >= 3 else { continue }
            for i in 0..<q.count {
                let a = q[i], b = q[(i + 1) % q.count]
                let mx = Double(a.x + b.x) / 2, my = Double(a.y + b.y) / 2
                if pointStrictlyInside(mx, my, obbCorners) {
                    bad.insert(p.name)
                }
            }
        }
        if !bad.isEmpty {
            cutViolations.append(di)
            split.append("det\(di):\(centerOwner.name)->\(bad.sorted())")
        }
    }
    results.append(RuleResult(
        rule: "R4_EVERY_OBB_OWNED", hard: true, ok: orphan.isEmpty,
        detail: orphan.isEmpty ? "ok" : "orphans n=\(orphan.count) ids=\(Array(orphan.prefix(12)))"
    ))
    results.append(RuleResult(
        rule: "R3_NO_OBB_CUT", hard: true, ok: cutViolations.isEmpty,
        detail: cutViolations.isEmpty ? "ok" : "split=\(Array(split.prefix(8)))"
    ))

    // S2: empty pieces are allowed (informational) — missed-spine recovery.
    results.append(RuleResult(
        rule: "S2_EMPTY_CELLS_OK", hard: false, ok: true,
        detail: "empty_cells=\(pieces.filter { $0.memberIndices.isEmpty }.count)"
    ))

    // S3: orientation purity within a piece.
    var purityBad: [String] = []
    for p in pieces where p.memberIndices.count >= 2 {
        let angles = p.memberIndices.map { dets[$0].longAxisAngle() }
        let meanA = circularMean(angles)
        let spread = angles.map { angleDiff($0, meanA) }.max() ?? 0
        if spread > (angleTolDeg * 2) * .pi / 180 {
            purityBad.append("\(p.name):spread=\(String(format: "%.1f", spread * 180 / .pi))deg")
        }
    }
    results.append(RuleResult(
        rule: "S3_BLOCK_ORIENTATION_PURITY", hard: false, ok: purityBad.isEmpty,
        detail: purityBad.isEmpty ? "ok" : "impure blocks: \(Array(purityBad.prefix(8)))"
    ))

    return results
}

// MARK: - Run-level invariants (A-7)

/// The two run-level checks A-7 adds on top of the per-cut rules:
/// recursion terminated with at least one leaf, and every leaf either
/// reached the resolution threshold (R-1b) or is an explicitly logged A-4
/// fallback. Soft by design — an A-4 leaf is legal, but a rising count is
/// the pressure signal A-4 asks to log.
public func verifyZoomRun(_ result: JigsawZoomResult, downsampleThreshold: Double) -> [RuleResult] {
    var results: [RuleResult] = []

    results.append(RuleResult(
        rule: "R6_TERMINATED", hard: true, ok: !result.leaves.isEmpty,
        detail: result.leaves.isEmpty
            ? "no leaf reached"
            : "leaves=\(result.leaves.count) passes=\(result.inferencePasses) maxDepth=\(result.leaves.map(\.depth).max() ?? 0)"
    ))

    let underscaled = result.leaves.filter { !$0.isFallback && $0.scale + 1e-9 < downsampleThreshold }
    results.append(RuleResult(
        rule: "R7_LEAF_SCALE_OK", hard: true, ok: underscaled.isEmpty,
        detail: underscaled.isEmpty
            ? "ok (fallback_leaves=\(result.fallbackLeafCount))"
            : "under-scaled non-fallback leaves n=\(underscaled.count) worst=\(String(format: "%.3f", underscaled.map(\.scale).min() ?? 0))"
    ))

    let hist = Dictionary(grouping: result.leaves, by: \.depth)
        .sorted { $0.key < $1.key }
        .map { "d\($0.key)=\($0.value.count)" }
        .joined(separator: " ")
    results.append(RuleResult(
        rule: "S4_DEPTH_HISTOGRAM", hard: false, ok: true,
        detail: hist.isEmpty ? "no leaves" : hist
    ))

    return results
}
