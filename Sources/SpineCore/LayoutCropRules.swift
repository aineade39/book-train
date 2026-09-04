import CoreGraphics
import Foundation

// Verification rules for layout crop plans against first-pass OBB
// detections. Direct port of `tools/layout_crop_rules.py`.
//
// Hard rules
// ----------
// R1 COVER_IMAGE       Crop polygons tile the image: area ~= W*H; every
//                       coarse grid sample is covered.
// R2 NO_OVERLAP         Pairwise interior intersection area of crop
//                       polygons is ~= 0.
// R3 NO_OBB_CUT         No OBB has a corner *strictly inside* a crop other
//                       than the crop that owns the OBB center. Shared
//                       boundary touches are allowed.
// R4 EVERY_OBB_OWNED    Every OBB center lies in at least one crop polygon.
// R5 QUAD_FINITE        All crop corners are finite and inside the image
//                       (small pad).
//
// Soft
// ----
// S2 EMPTY_CELLS_OK     Empty crops (0 members) are allowed for
//                       missed-spine recovery.
// S3 BLOCK_ORIENTATION_PURITY
//                       Warn if a block's member angles spread beyond
//                       `2 * angleTolDeg` from their circular mean.

/// A planned crop: a scene-space quad (TL, TR, BR, BL) plus the indices of
/// first-pass detections it owns. Matches Python `CropPlan`.
public struct CropPlan {
    public let shelfId: Int
    public let blockId: Int
    public let angleDeg: Double
    public let quad: [CGPoint]
    public let memberIndices: [Int]

    public init(shelfId: Int, blockId: Int, angleDeg: Double, quad: [CGPoint], memberIndices: [Int] = []) {
        self.shelfId = shelfId
        self.blockId = blockId
        self.angleDeg = angleDeg
        self.quad = quad
        self.memberIndices = memberIndices
    }

    public var name: String { "shelf\(shelfId)_block\(blockId)" }

    /// (x0, y0, x1, y1) axis-aligned bounds of `quad`.
    public var rect: (x0: Double, y0: Double, x1: Double, y1: Double) {
        let xs = quad.map { Double($0.x) }
        let ys = quad.map { Double($0.y) }
        return (xs.min()!, ys.min()!, xs.max()!, ys.max()!)
    }
}

public struct RuleResult: Codable {
    public let rule: String
    public let hard: Bool
    public let ok: Bool
    public let detail: String

    public init(rule: String, hard: Bool, ok: Bool, detail: String) {
        self.rule = rule
        self.hard = hard
        self.ok = ok
        self.detail = detail
    }
}

/// `%.0f`-style rounding for detail strings, matching Python's `:.0f`
/// formatting (which renders non-finite values as "nan"/"inf" instead of
/// trapping) — `Int(_:.rounded())` would crash on a NaN/Inf quad corner.
private func fmt0(_ x: Double) -> String { String(format: "%.0f", x) }

/// The smallest-area crop whose quad contains `(x, y)`, or `nil` if none do
/// — matches Python `owning_crop`.
public func owningCrop(x: Double, y: Double, plans: [CropPlan]) -> CropPlan? {
    let hits = plans.filter { pointInPolygon(x, y, $0.quad) }
    return hits.min { polygonArea($0.quad) < polygonArea($1.quad) }
}

public func verifyPlan(
    dets: [OBBDetection],
    plans: [CropPlan],
    imgW: Int,
    imgH: Int,
    grid: Int = LayoutConstants.defaultGridSamples,
    angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg
) -> [RuleResult] {
    // Quads are the special case of the polygon rules in
    // `PartitionRules.swift`, with the whole image as the region: identical
    // rule names, hard flags, detail formats and pass semantics (the
    // Python-parity contract), just not restricted to four corners.
    verifyPartition(
        dets: dets,
        pieces: plans.map { PartitionPiece(name: $0.name, polygon: $0.quad, memberIndices: $0.memberIndices) },
        region: imageRegion(imgW: imgW, imgH: imgH),
        imgW: imgW, imgH: imgH, grid: grid, angleTolDeg: angleTolDeg
    )
}

/// The pre-delegation quad implementation, kept as the reference the
/// polygon generalization is checked against in tests.
func verifyPlanQuadReference(
    dets: [OBBDetection],
    plans: [CropPlan],
    imgW: Int,
    imgH: Int,
    grid: Int = LayoutConstants.defaultGridSamples,
    angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg
) -> [RuleResult] {
    var results: [RuleResult] = []
    guard !plans.isEmpty else {
        return [RuleResult(rule: "R5_QUAD_FINITE", hard: true, ok: false, detail: "no crops")]
    }

    // R5: every quad corner finite and within a small pad of the image.
    let pad = LayoutConstants.quadFinitePad
    var badFinite: [String] = []
    for p in plans {
        for pt in p.quad {
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
    let polys = plans.map { ensureCCW($0.quad) }
    for i in 0..<polys.count {
        for j in (i + 1)..<polys.count {
            let area = polyIntersectionArea(polys[i], polys[j])
            if area > LayoutConstants.overlapAreaEps {
                overlaps.append("\(plans[i].name)/\(plans[j].name):\(fmt0(area))")
            }
        }
    }
    results.append(RuleResult(
        rule: "R2_NO_OVERLAP", hard: true, ok: overlaps.isEmpty,
        detail: overlaps.isEmpty ? "ok" : "overlaps \(Array(overlaps.prefix(6)))"
    ))

    // R1: coverage — total area within tolerance, and every grid sample hit.
    let areaSum = polys.reduce(0.0) { $0 + polygonArea($1) }
    let expected = Double(imgW * imgH)
    let areaOk = abs(areaSum - expected) <= LayoutConstants.coverAreaTolFraction * expected
    var miss = 0
    var multi = 0
    let stepX = Double(imgW) / Double(grid)
    let stepY = Double(imgH) / Double(grid)
    for gy in 0..<grid {
        for gx in 0..<grid {
            let x = (Double(gx) + 0.5) * stepX
            let y = (Double(gy) + 0.5) * stepY
            let hits = plans.reduce(0) { $0 + (pointInPolygon(x, y, $1.quad) ? 1 : 0) }
            if hits == 0 { miss += 1 } else if hits > 2 { multi += 1 }
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
        guard let centerOwner = owningCrop(x: det.cx, y: det.cy, plans: plans) else {
            orphan.append(di)
            continue
        }
        var bad: Set<String> = []
        let obbCorners = det.corners
        for cx_ in obbCorners {
            for p in plans where p.name != centerOwner.name {
                if pointStrictlyInside(Double(cx_.x), Double(cx_.y), p.quad) {
                    bad.insert(p.name)
                }
            }
        }
        // Crop edge through OBB: any crop edge (except owner) with midpoint inside OBB.
        for p in plans where p.name != centerOwner.name {
            let q = p.quad
            for i in 0..<4 {
                let a = q[i], b = q[(i + 1) % 4]
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

    // S2: empty cells are allowed (informational).
    results.append(RuleResult(
        rule: "S2_EMPTY_CELLS_OK", hard: false, ok: true,
        detail: "empty_cells=\(plans.filter { $0.memberIndices.isEmpty }.count)"
    ))

    // S3: orientation purity within a block.
    var purityBad: [String] = []
    for p in plans where p.memberIndices.count >= 2 {
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

public func hardRulesOK(_ results: [RuleResult]) -> Bool {
    results.filter(\.hard).allSatisfy(\.ok)
}
