import CoreGraphics
import Foundation

// Stage 3 of the jigsaw-zoom plan (~/dev/book-id-design/docs/
// jigsaw-zoom-requirements.md R-2 / R-6 / A-1 / A-2 / A-9): the cutter
// behind a protocol, so the engine can run either the incumbent quad
// planner (v1, `planCrops`) or the free-form staircase cutter (v2).
//
// What v2 changes versus v1:
//   * Seams are monotone staircases, not straight lines, so a cut stays
//     legal in the interlocked-OBB scenes where a straight seam has to give
//     up and surrender the whole piece to the A-4 fallback.
//   * There is no shelf-band / column-block structure (spec §4 explicitly
//     does not require one) and no empty gutter cells; a cut is a recursive
//     binary split of the piece polygon itself.
//   * The split target is resolution, not a `maxCropDimK` budget: one cut
//     descends until every piece would letterbox at or above the downsample
//     threshold (R-6, "ideally a single cut yields pieces that are already
//     at native resolution"), so the engine spends inference passes on
//     leaves rather than on intermediate levels (A-9).
//   * Cuts partition the *piece polygon*, not its bounding rectangle, so a
//     re-cut of a non-rectangular piece cannot leak into a sibling's area.

/// Which cutter the zoom engine partitions an oversized piece with.
public enum ZoomCutterKind: String, Codable, CaseIterable, Sendable {
    /// Incumbent `planCrops` quad planner (shelf bands + column blocks).
    case v1
    /// Free-form staircase-seam cutter.
    case v2
}

/// One cut request, entirely in the piece's local crop-pixel space.
public struct ZoomCutRequest {
    /// The piece being cut, in crop pixels.
    public let polygon: [CGPoint]
    /// This level's detections, in crop pixels. Cut placement only (R-1a).
    public let dets: [OBBDetection]
    public let raster: SceneRaster
    public let energy: EdgeEnergy
    public let width: Int
    public let height: Int
    public let imgsz: Int
    /// Scene -> crop shrink already applied to this space.
    public let scale: Double
    /// Whether the engine letterboxes by min-area rect (R-3 rotation) or by
    /// axis-aligned box — decides which dimension has to reach `targetDim`.
    public let rotatePieces: Bool
    public let downsampleThreshold: Double
    public let angleTolDeg: Double
    public let depth: Int
    public let debug: Bool

    public init(
        polygon: [CGPoint], dets: [OBBDetection], raster: SceneRaster, energy: EdgeEnergy,
        width: Int, height: Int, imgsz: Int, scale: Double, rotatePieces: Bool,
        downsampleThreshold: Double, angleTolDeg: Double, depth: Int, debug: Bool
    ) {
        self.polygon = polygon
        self.dets = dets
        self.raster = raster
        self.energy = energy
        self.width = width
        self.height = height
        self.imgsz = imgsz
        self.scale = scale
        self.rotatePieces = rotatePieces
        self.downsampleThreshold = downsampleThreshold
        self.angleTolDeg = angleTolDeg
        self.depth = depth
        self.debug = debug
    }

    /// Crop-pixel dimension at or below which a child piece letterboxes at
    /// the downsample threshold, i.e. becomes a leaf (R-1b). A child's scene
    /// dimension is `cropDim / scale`, and it is a leaf when
    /// `imgsz / sceneDim >= threshold`.
    public var targetDim: Double {
        Double(imgsz) * scale / max(1e-6, downsampleThreshold)
    }
}

/// Partitions one oversized piece. `nil` means "no legal cut" and sends the
/// engine to its A-4 fallback (accept the piece as a downsampled leaf).
public protocol ZoomCutter {
    func cut(_ request: ZoomCutRequest) -> [[CGPoint]]?
}

private func debugLog(_ message: String) {
    FileHandle.standardError.write(Data("[zoom-debug] \(message)\n".utf8))
}

// MARK: - v1: the incumbent quad planner as a cutter

/// Adapter over `planCrops` + `verifyPlan`. Frozen incumbent behavior: it
/// partitions the whole crop rectangle (not the piece polygon) and keeps the
/// `maxCropDimK` size budget, which is what the Stage 2 numbers were
/// measured with.
public struct PlanCropsCutter: ZoomCutter {
    public var angleTolDeg: Double
    public var rowGapK: Double
    public var colGapK: Double
    public var minBlockMembers: Int
    public var maxCropDimK: Double

    public init(
        angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg,
        rowGapK: Double = 0.2,
        colGapK: Double = 1.0,
        minBlockMembers: Int = 2,
        maxCropDimK: Double = 1.5
    ) {
        self.angleTolDeg = angleTolDeg
        self.rowGapK = rowGapK
        self.colGapK = colGapK
        self.minBlockMembers = minBlockMembers
        self.maxCropDimK = maxCropDimK
    }

    public func cut(_ request: ZoomCutRequest) -> [[CGPoint]]? {
        func planAndVerify(bandsOnly: Bool) -> [CropPlan]? {
            let plans = planCrops(
                dets: request.dets,
                imgW: request.width,
                imgH: request.height,
                raster: request.raster,
                angleTolDeg: angleTolDeg,
                rowGapK: rowGapK,
                colGapK: colGapK,
                minBlockMembers: minBlockMembers,
                imgsz: request.imgsz,
                // Planning runs in the letterboxed crop's scaled pixels; the
                // oversized-crop budget is a native-pixel concept, so shrink it
                // by the same scale or nothing ever looks oversized and
                // resolution-driven splits stop firing (R-6/A-1).
                maxCropDimK: maxCropDimK * request.scale,
                bandsOnly: bandsOnly
            )
            let rules = verifyPlan(
                dets: request.dets, plans: plans,
                imgW: request.width, imgH: request.height,
                angleTolDeg: angleTolDeg
            )
            if request.debug {
                for r in rules where r.hard && !r.ok {
                    debugLog("v1 depth=\(request.depth) bandsOnly=\(bandsOnly) plans=\(plans.count) FAIL \(r.rule): \(r.detail)")
                }
            }
            return hardRulesOK(rules) ? plans : nil
        }

        // The planner's column-seam resolution can emit an overlapping /
        // OBB-cutting plan on hard scenes. Rather than surrendering the whole
        // piece to A-4, retry with full-width shelf-band strips (horizontal
        // seams only), which cannot overlap and are far more likely to pass
        // hard rules; the recursion splits columns at the next level where
        // resolution is higher.
        guard let plans = planAndVerify(bandsOnly: false) ?? planAndVerify(bandsOnly: true) else { return nil }
        return plans.map(\.quad)
    }
}

// MARK: - v2: free-form staircase cutter

public struct FreeFormCutterOptions {
    /// Staircase resolution: how many steps a seam may take along its
    /// advance axis. More steps weave tighter but cost DP time and add
    /// polygon vertices.
    public var maxSeamRows: Int
    /// Uniform fallback positions across the split axis, for stretches with
    /// no detections to derive a seam location from.
    public var uniformCandidates: Int
    /// How far outside an OBB's side face to place its derived candidate
    /// (crop px). Small on purpose: the gap between touching spines is a
    /// couple of pixels, and a candidate grid coarser than that never finds
    /// the one legal seam in a dense shelf.
    public var sideFaceOffset: Double
    /// Candidate steps a seam may move per row — its maximum slope.
    public var maxCandidateJump: Int
    /// Penalty per median-candidate-step of jitter (keeps seams as straight
    /// as the obstacles allow).
    public var smoothWeight: Double
    /// Pull toward the equal-area split position (A-1's soft objective).
    public var balanceWeight: Double
    /// Extra OBB dilation before seam search, in median-short-side units.
    /// Zero by default: the blocked mask's cell already keeps a seam about a
    /// pixel clear, and anything larger closes the real gaps between
    /// adjacent spines and forces A-4 on every dense shelf.
    public var obbPadK: Double
    /// A-9 fragmentation ceiling: pieces one cut may produce.
    public var maxPieces: Int
    /// Never split a side shorter than this (crop px).
    public var minPieceDim: Double
    /// Blocked-mask cell size (crop px).
    public var maskCell: Double

    public init(
        maxSeamRows: Int = 40,
        uniformCandidates: Int = 24,
        sideFaceOffset: Double = 1.5,
        maxCandidateJump: Int = 4,
        smoothWeight: Double = 0.6,
        balanceWeight: Double = 1.2,
        obbPadK: Double = 0,
        maxPieces: Int = 64,
        minPieceDim: Double = 24,
        maskCell: Double = 1.0
    ) {
        self.maxSeamRows = maxSeamRows
        self.uniformCandidates = uniformCandidates
        self.sideFaceOffset = sideFaceOffset
        self.maxCandidateJump = maxCandidateJump
        self.smoothWeight = smoothWeight
        self.balanceWeight = balanceWeight
        self.obbPadK = obbPadK
        self.maxPieces = maxPieces
        self.minPieceDim = minPieceDim
        self.maskCell = maskCell
    }

    public static let `default` = FreeFormCutterOptions()
}

/// Rasterized "a seam may not pass here" map of the dilated OBBs, so the
/// seam DP's feasibility test is a constant-time lookup instead of a scan
/// over every detection.
struct BlockedMask {
    private let cell: Double
    private let cols: Int
    private let rows: Int
    private var bits: [Bool]

    init(obstacles: [[CGPoint]], width: Int, height: Int, cell: Double) {
        let c = max(1.0, cell)
        self.cell = c
        cols = max(1, Int((Double(width) / c).rounded(.up)) + 1)
        rows = max(1, Int((Double(height) / c).rounded(.up)) + 1)
        bits = [Bool](repeating: false, count: cols * rows)

        for poly in obstacles {
            guard let b = polygonBounds(poly) else { continue }
            let gx0 = max(0, Int(b.x0 / c) - 1), gx1 = min(cols - 1, Int(b.x1 / c) + 1)
            let gy0 = max(0, Int(b.y0 / c) - 1), gy1 = min(rows - 1, Int(b.y1 / c) + 1)
            guard gx0 <= gx1, gy0 <= gy1 else { continue }
            for gy in gy0...gy1 {
                for gx in gx0...gx1 {
                    let x = (Double(gx) + 0.5) * c
                    let y = (Double(gy) + 0.5) * c
                    if pointInPolygon(x, y, poly) { bits[gy * cols + gx] = true }
                }
            }
            // A spine thinner than a cell can slip between cell centers, so
            // also mark every cell its outline walks through.
            for i in 0..<poly.count {
                let p = poly[i], q = poly[(i + 1) % poly.count]
                let steps = max(1, Int((dist(p, q) / (c / 2)).rounded(.up)))
                for s in 0...steps {
                    let t = Double(s) / Double(steps)
                    let x = Double(p.x) + (Double(q.x) - Double(p.x)) * t
                    let y = Double(p.y) + (Double(q.y) - Double(p.y)) * t
                    mark(x, y)
                }
            }
        }
    }

    private mutating func mark(_ x: Double, _ y: Double) {
        let gx = Int(x / cell), gy = Int(y / cell)
        guard gx >= 0, gx < cols, gy >= 0, gy < rows else { return }
        bits[gy * cols + gx] = true
    }

    func blocked(_ x: Double, _ y: Double) -> Bool {
        let gx = Int(x / cell), gy = Int(y / cell)
        guard gx >= 0, gx < cols, gy >= 0, gy < rows else { return false }
        return bits[gy * cols + gx]
    }

    /// True if any point along `p -> q` is blocked.
    func segmentBlocked(_ p: CGPoint, _ q: CGPoint) -> Bool {
        let steps = max(1, Int((dist(p, q) / (cell / 2)).rounded(.up)))
        for s in 0...steps {
            let t = Double(s) / Double(steps)
            let x = Double(p.x) + (Double(q.x) - Double(p.x)) * t
            let y = Double(p.y) + (Double(q.y) - Double(p.y)) * t
            if blocked(x, y) { return true }
        }
        return false
    }
}

/// OBB corners grown by `pad` on every side.
func dilatedCorners(_ det: OBBDetection, pad: Double) -> [CGPoint] {
    OBBDetection(
        cx: det.cx, cy: det.cy,
        w: det.w + 2 * pad, h: det.h + 2 * pad,
        angle: det.angle, conf: det.conf, id: det.id
    ).corners
}

public struct FreeFormCutter: ZoomCutter {
    public var options: FreeFormCutterOptions

    public init(options: FreeFormCutterOptions = .default) {
        self.options = options
    }

    public func cut(_ request: ZoomCutRequest) -> [[CGPoint]]? {
        let ring = dedupeRing(request.polygon)
        guard ring.count >= 3 else { return nil }

        let pad = options.obbPadK * medianShortSide(request.dets)
        let mask = BlockedMask(
            obstacles: request.dets.map { dilatedCorners($0, pad: pad) },
            width: request.width, height: request.height, cell: options.maskCell
        )
        let reference = referenceEnergy(request)

        var splitBudget = max(0, options.maxPieces - 1)
        let pieces = splitToTarget(
            ring, request: request, mask: mask, reference: reference, splitBudget: &splitBudget
        )
        guard pieces.count > 1 else {
            if request.debug { debugLog("v2 depth=\(request.depth) no legal cut (A-4)") }
            return nil
        }

        // A cut is only worth an inference pass per piece if it actually
        // raises resolution. When the dominant axis is blocked — say a spine
        // spanning the full height of an already-narrow column — splitting
        // the other axis leaves every piece letterboxed at the parent's
        // scale, so the honest answer is A-4, not N passes for no gain.
        let parentDim = letterboxSourceDim(ring, rotate: request.rotatePieces)
        let worstChildDim = pieces.map { letterboxSourceDim($0, rotate: request.rotatePieces) }.max() ?? parentDim
        guard worstChildDim < parentDim - 0.5 else {
            if request.debug {
                debugLog("v2 depth=\(request.depth) cut gains no resolution (\(Int(parentDim))px -> \(Int(worstChildDim))px), taking A-4")
            }
            return nil
        }

        // Verified against the piece itself, not the crop rectangle: a
        // non-rectangular piece's children must stay inside it or the global
        // partition breaks (R-2).
        let named = pieces.enumerated().map { index, polygon in
            PartitionPiece(
                name: "piece\(index)",
                polygon: polygon,
                memberIndices: request.dets.indices.filter {
                    pointInPolygon(request.dets[$0].cx, request.dets[$0].cy, polygon)
                }
            )
        }
        let rules = verifyPartition(
            dets: request.dets, pieces: named, region: ring,
            imgW: request.width, imgH: request.height, angleTolDeg: request.angleTolDeg
        )
        guard hardRulesOK(rules) else {
            if request.debug {
                for r in rules where r.hard && !r.ok {
                    debugLog("v2 depth=\(request.depth) pieces=\(pieces.count) FAIL \(r.rule): \(r.detail)")
                }
            }
            return nil
        }
        if request.debug {
            debugLog("v2 depth=\(request.depth) pieces=\(pieces.count) target=\(String(format: "%.0f", request.targetDim))px")
        }
        return pieces
    }

    /// Mean edge energy of the crop, used to keep seam costs scale-free.
    private func referenceEnergy(_ request: ZoomCutRequest) -> Double {
        let area = Double(request.width * request.height)
        guard area > 0 else { return 1 }
        let sum = request.energy.rectSum(x0: 0, y0: 0, x1: request.width, y1: request.height)
        return max(1e-6, sum / area)
    }

    /// Splits until every piece would letterbox at the resolution threshold
    /// (R-6), always working on the piece that is currently worst so a
    /// `maxPieces` cap spends its budget evenly instead of over-splitting one
    /// branch and leaving another at the parent's scale.
    private func splitToTarget(
        _ polygon: [CGPoint],
        request: ZoomCutRequest,
        mask: BlockedMask,
        reference: Double,
        splitBudget: inout Int
    ) -> [[CGPoint]] {
        var pieces = [polygon]
        var unsplittable = Set<Int>()

        func dim(_ index: Int) -> Double {
            letterboxSourceDim(pieces[index], rotate: request.rotatePieces)
        }

        while splitBudget > 0 {
            let candidates = pieces.indices.filter { !unsplittable.contains($0) && dim($0) > request.targetDim }
            guard let worst = candidates.max(by: { dim($0) < dim($1) }) else { break }
            guard let halves = split(
                pieces[worst], request: request, mask: mask, reference: reference
            ) else {
                // A-4 pressure point: interlocked OBBs leave no legal seam.
                unsplittable.insert(worst)
                continue
            }
            pieces[worst] = halves.a
            pieces.append(halves.b)
            splitBudget -= 1
        }
        return pieces
    }

    /// One binary cut of `polygon`, or `nil` when no legal seam exists.
    private func split(
        _ polygon: [CGPoint],
        request: ZoomCutRequest,
        mask: BlockedMask,
        reference: Double
    ) -> (a: [CGPoint], b: [CGPoint])? {
        guard let bounds = polygonBounds(polygon) else { return nil }
        let extentX = bounds.x1 - bounds.x0, extentY = bounds.y1 - bounds.y0
        // Only the longest side is worth cutting: the letterbox scale is set
        // by `max(width, height)`, so shortening the other axis costs a piece
        // and buys no resolution (A-1/A-9). The near-tie allowance lets a
        // square piece fall back to the other axis when the first is blocked.
        var axes: [ProfileAxis] = extentX >= extentY ? [.x] : [.y]
        if min(extentX, extentY) >= 0.95 * max(extentX, extentY) {
            axes.append(axes[0] == .x ? .y : .x)
        }
        let parentArea = polygonArea(polygon)

        for axis in axes {
            let extent = axis == .x ? extentX : extentY
            guard extent >= 2 * options.minPieceDim else { continue }
            let lo = axis == .x ? bounds.x0 : bounds.y0
            // A-9: fewest pieces that still reach native resolution. Splitting
            // at the halfway point would round every cut up to a power of two;
            // aiming at `floor(k/2)/k` of the extent keeps the count at k.
            let needed = max(2, Int((extent / max(1.0, request.targetDim)).rounded(.up)))
            let ideal = lo + extent * Double(needed / 2) / Double(needed)

            guard let seam = bestSeam(
                polygon: polygon, bounds: bounds, splitAxis: axis, ideal: ideal,
                request: request, mask: mask, reference: reference
            ), let halves = splitPolygonByMonotoneSeam(polygon, seam: seam) else { continue }

            // A sliver split makes no resolution progress on the big half.
            guard polygonArea(halves.a) < parentArea * 0.95,
                  polygonArea(halves.b) < parentArea * 0.95 else { continue }
            return halves
        }
        return nil
    }

    /// Where a seam may cross the split axis: just outside every detection's
    /// side faces, plus a coarse uniform grid for stretches with no
    /// detections. This is the "arrangement of OBB side faces plus low-energy
    /// seams" of A-2 — and the only way to hit the one- or two-pixel gap
    /// between touching spines on a dense shelf, which a uniform grid coarse
    /// enough to be affordable always steps over.
    private func candidatePositions(
        request: ZoomCutRequest,
        splitAxis: ProfileAxis,
        candLo: Double,
        candHi: Double
    ) -> [Double] {
        var positions: [Double] = []
        let uniform = max(3, min(options.uniformCandidates, Int((candHi - candLo) / 8)))
        for i in 0..<uniform {
            positions.append(candLo + (candHi - candLo) * Double(i) / Double(max(1, uniform - 1)))
        }
        for det in request.dets {
            let span = splitAxis == .x ? det.xSpan : det.ySpan
            positions.append(span.0 - options.sideFaceOffset)
            positions.append(span.1 + options.sideFaceOffset)
        }

        var out: [Double] = []
        for p in positions.filter({ $0 >= candLo && $0 <= candHi }).sorted() {
            if let last = out.last, p - last <= 1.0 { continue }
            out.append(p)
        }
        return out
    }

    /// Lowest-cost monotone staircase across `polygon`, or `nil` when every
    /// candidate path would cut an OBB. Cost per step is edge energy (A-2:
    /// route seams through low-texture regions), plus a jitter penalty, plus
    /// a pull toward the equal-area position (A-1).
    private func bestSeam(
        polygon: [CGPoint],
        bounds: (x0: Double, y0: Double, x1: Double, y1: Double),
        splitAxis: ProfileAxis,
        ideal: Double,
        request: ZoomCutRequest,
        mask: BlockedMask,
        reference: Double
    ) -> MonotoneSeam? {
        let advanceLo = splitAxis == .x ? bounds.y0 : bounds.x0
        let advanceHi = splitAxis == .x ? bounds.y1 : bounds.x1
        let splitLo = splitAxis == .x ? bounds.x0 : bounds.y0
        let splitHi = splitAxis == .x ? bounds.x1 : bounds.y1
        let advanceExtent = advanceHi - advanceLo
        let splitExtent = splitHi - splitLo
        guard advanceExtent > 4, splitExtent > 4 else { return nil }

        let rowCount = max(3, min(options.maxSeamRows, Int((advanceExtent / 12).rounded(.up)) + 1))
        let margin = max(2.0, min(options.minPieceDim, 0.25 * splitExtent))
        let candLo = splitLo + margin, candHi = splitHi - margin
        guard candHi > candLo else { return nil }
        let candidates = candidatePositions(
            request: request, splitAxis: splitAxis, candLo: candLo, candHi: candHi
        )
        let candidateCount = candidates.count
        guard candidateCount >= 2 else { return nil }
        let medianStep = max(
            1.0,
            (candidates.last! - candidates.first!) / Double(candidateCount - 1)
        )

        func position(_ j: Int) -> Double { candidates[j] }
        func advance(_ i: Int) -> Double {
            advanceLo + advanceExtent * Double(i) / Double(rowCount - 1)
        }
        func point(_ i: Int, _ j: Int) -> CGPoint {
            splitAxis == .x
                ? CGPoint(x: position(j), y: advance(i))
                : CGPoint(x: advance(i), y: position(j))
        }

        let halfSpan = max(1.0, splitExtent / 2)
        let energyBox = 3
        let boxArea = Double((2 * energyBox + 1) * (2 * energyBox + 1))
        func nodeCost(_ i: Int, _ j: Int) -> Double {
            let p = point(i, j)
            if mask.blocked(Double(p.x), Double(p.y)) { return .infinity }
            let x = Int(Double(p.x).rounded()), y = Int(Double(p.y).rounded())
            let sum = request.energy.rectSum(
                x0: x - energyBox, y0: y - energyBox,
                x1: x + energyBox + 1, y1: y + energyBox + 1
            )
            let texture = sum / boxArea / reference
            let balance = abs(position(j) - ideal) / halfSpan
            return texture + options.balanceWeight * balance
        }

        var cost = [[Double]](repeating: [Double](repeating: .infinity, count: candidateCount), count: rowCount)
        var back = [[Int]](repeating: [Int](repeating: -1, count: candidateCount), count: rowCount)
        for j in 0..<candidateCount { cost[0][j] = nodeCost(0, j) }
        for i in 1..<rowCount {
            for j in 0..<candidateCount {
                let node = nodeCost(i, j)
                guard node.isFinite else { continue }
                let from = max(0, j - options.maxCandidateJump)
                let to = min(candidateCount - 1, j + options.maxCandidateJump)
                var bestCost = Double.infinity
                var bestPrev = -1
                for k in from...to {
                    let prior = cost[i - 1][k]
                    guard prior.isFinite else { continue }
                    let step = prior + options.smoothWeight * abs(position(j) - position(k)) / medianStep
                    guard step < bestCost else { continue }
                    if mask.segmentBlocked(point(i - 1, k), point(i, j)) { continue }
                    bestCost = step
                    bestPrev = k
                }
                guard bestPrev >= 0 else { continue }
                cost[i][j] = bestCost + node
                back[i][j] = bestPrev
            }
        }

        let last = rowCount - 1
        var bestEnd = -1
        var bestEndCost = Double.infinity
        for j in 0..<candidateCount where cost[last][j] < bestEndCost {
            bestEndCost = cost[last][j]
            bestEnd = j
        }
        guard bestEnd >= 0, bestEndCost.isFinite else { return nil }

        var path: [CGPoint] = []
        var j = bestEnd
        for i in stride(from: last, through: 0, by: -1) {
            path.append(point(i, j))
            if i > 0 {
                j = back[i][j]
                guard j >= 0 else { return nil }
            }
        }
        path.reverse()

        // Extend past the piece so the seam crosses its boundary rather than
        // ending on it.
        let overshoot = max(4.0, 0.05 * advanceExtent)
        let head = splitAxis == .x
            ? CGPoint(x: path[0].x, y: advanceLo - overshoot)
            : CGPoint(x: advanceLo - overshoot, y: path[0].y)
        let tail = splitAxis == .x
            ? CGPoint(x: path[path.count - 1].x, y: advanceHi + overshoot)
            : CGPoint(x: advanceHi + overshoot, y: path[path.count - 1].y)
        return MonotoneSeam(points: [head] + path + [tail], splitAxis: splitAxis)
    }
}
