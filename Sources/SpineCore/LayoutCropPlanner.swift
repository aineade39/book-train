import CoreGraphics
import Foundation

// Full-scene jigsaw crop planner: shelf bands + orientation-aware column
// blocks, each its own crop by default; oversized groups are cut further
// using pixel-texture-guided whitespace search rather than trusting OBB
// gaps alone. Direct port of `tools/layout_crop_predict.py`'s planning
// logic (`plan_crops` and everything it calls).

// MARK: - OBB edge/orientation helpers specific to seam-finding

func distFromVertical(_ theta: Double) -> Double {
    let d = abs(normAngle(theta) - .pi / 2)
    return min(d, .pi - d)
}

func allEdges(_ det: OBBDetection) -> [(CGPoint, CGPoint)] {
    let pts = det.corners
    return (0..<4).map { (pts[$0], pts[($0 + 1) % 4]) }
}

/// The pair of OBB edges that face left/right — the faces relevant for
/// column seams. Picks whichever edge-length class is closer to vertical in
/// image space, robust regardless of which of w/h the model calls the long
/// side. Matches Python `side_edges`.
func sideEdges(_ det: OBBDetection) -> [(CGPoint, CGPoint)] {
    let pts = det.corners
    let hEdges = [(pts[0], pts[1]), (pts[2], pts[3])]
    let wEdges = [(pts[1], pts[2]), (pts[3], pts[0])]
    return distFromVertical(det.angle + .pi / 2) <= distFromVertical(det.angle) ? hEdges : wEdges
}

func edgeMidX(_ edge: (CGPoint, CGPoint)) -> Double {
    Double(edge.0.x + edge.1.x) / 2
}

/// Most extreme side-face edge in x — normally the OBB's true left/right
/// face, with a defensive fallback to the most extreme raw corner edge.
/// Matches Python `side_edge_line`.
func sideEdgeLine(_ det: OBBDetection, preferMinX: Bool) -> Line {
    let scored = allEdges(det).sorted { edgeMidX($0) < edgeMidX($1) }
    let extreme = preferMinX ? scored.first! : scored.last!
    let faceScored = sideEdges(det).sorted { edgeMidX($0) < edgeMidX($1) }
    let cand = preferMinX ? faceScored.first! : faceScored.last!
    let shortDim = max(1.0, min(det.w, det.h))
    if abs(edgeMidX(cand) - edgeMidX(extreme)) <= 0.3 * shortDim {
        return .through(cand.0, cand.1)
    }
    return .through(extreme.0, extreme.1)
}

func allCorners(_ dets: [OBBDetection], _ indices: [Int]) -> [CGPoint] {
    indices.flatMap { dets[$0].corners }
}

// MARK: - Shelf / block clustering

/// Median true short-side (spine thickness) across `dets` — matches Python
/// `median_short_side`. Cannot just use `h`: this model doesn't guarantee
/// `h` is the short side.
func medianShortSide(_ dets: [OBBDetection]) -> Double {
    guard !dets.isEmpty else { return 10.0 }
    let vals = dets.map { min($0.w, $0.h) }.sorted()
    let n = vals.count
    return n % 2 == 1 ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0
}

/// Strict `(min, max)` y-extent of a band's members.
func bandExtent(_ dets: [OBBDetection], _ indices: [Int]) -> (Double, Double) {
    let tops = indices.map { dets[$0].ySpan.0 }
    let bots = indices.map { dets[$0].ySpan.1 }
    return (tops.min()!, bots.max()!)
}

/// Row (shelf) bands via interval-union on OBB y-extents — matches Python
/// `build_shelf_bands`.
func buildShelfBands(_ dets: [OBBDetection], rowGapK: Double) -> [[Int]] {
    guard !dets.isEmpty else { return [] }
    let minRowGap = max(LayoutConstants.minRowGapFloor, rowGapK * medianShortSide(dets))

    let order = (0..<dets.count).sorted { dets[$0].ySpan.0 < dets[$1].ySpan.0 }
    var bands: [[Int]] = [[order[0]]]
    var curBot = dets[order[0]].ySpan.1
    for idx in order.dropFirst() {
        let (top, bot) = dets[idx].ySpan
        if top <= curBot + minRowGap {
            bands[bands.count - 1].append(idx)
            curBot = max(curBot, bot)
        } else {
            bands.append([idx])
            curBot = bot
        }
    }

    // Absorb tiny bands (likely stray detections) into the closer neighbor.
    let minMembers = 2
    var changed = true
    while changed, bands.count > 1 {
        changed = false
        for i in 0..<bands.count {
            let band = bands[i]
            if band.count >= minMembers { continue }
            let neighbors = [i - 1, i + 1].filter { $0 >= 0 && $0 < bands.count }
            guard !neighbors.isEmpty else { continue }
            let cy = band.map { dets[$0].cy }.reduce(0, +) / Double(band.count)
            let bestJ = neighbors.min { a, b in
                let ma = bands[a].map { dets[$0].cy }.reduce(0, +) / Double(bands[a].count)
                let mb = bands[b].map { dets[$0].cy }.reduce(0, +) / Double(bands[b].count)
                return abs(ma - cy) < abs(mb - cy)
            }!
            bands[bestJ] = (bands[bestJ] + band).sorted { dets[$0].cy < dets[$1].cy }
            bands.remove(at: i)
            changed = true
            break
        }
    }
    for i in 0..<bands.count {
        bands[i].sort { dets[$0].cy < dets[$1].cy }
    }
    return bands
}

/// Column blocks within one shelf band via 1-D chain-merge on members
/// sorted by `cx`: consecutive detections stay together only if both their
/// x-extents are within `colGapPx` of touching AND their orientation is
/// within `angleTolDeg`. Matches Python `build_column_blocks`.
func buildColumnBlocks(_ dets: [OBBDetection], members: [Int], angleTolDeg: Double, colGapPx: Double, minBlockMembers: Int) -> [[Int]] {
    guard !members.isEmpty else { return [] }
    let angleTolRad = angleTolDeg * .pi / 180
    let order = members.sorted { dets[$0].cx < dets[$1].cx }
    var blocks: [[Int]] = [[order[0]]]
    for idx in order.dropFirst() {
        let prevIdx = blocks[blocks.count - 1].last!
        let gap = dets[idx].xSpan.0 - dets[prevIdx].xSpan.1
        let sameOrientation = angleDiff(dets[prevIdx].longAxisAngle(), dets[idx].longAxisAngle()) <= angleTolRad
        if gap <= colGapPx, sameOrientation {
            blocks[blocks.count - 1].append(idx)
        } else {
            blocks.append([idx])
        }
    }

    var changed = true
    while changed, blocks.count > 1 {
        changed = false
        for i in 0..<blocks.count {
            let block = blocks[i]
            if block.count >= minBlockMembers { continue }
            let neighbors = [i - 1, i + 1].filter { $0 >= 0 && $0 < blocks.count }
            guard !neighbors.isEmpty else { continue }
            let blockAngle = circularMean(block.map { dets[$0].longAxisAngle() })
            let blockCx = block.map { dets[$0].cx }.reduce(0, +) / Double(block.count)
            func neighborKey(_ j: Int) -> (Int, Double) {
                let nAngle = circularMean(blocks[j].map { dets[$0].longAxisAngle() })
                let same = angleDiff(blockAngle, nAngle) <= angleTolRad ? 0 : 1
                let nCx = blocks[j].map { dets[$0].cx }.reduce(0, +) / Double(blocks[j].count)
                return (same, abs(nCx - blockCx))
            }
            let bestJ = neighbors.min { neighborKey($0) < neighborKey($1) }!
            blocks[bestJ] = (blocks[bestJ] + block).sorted { dets[$0].cx < dets[$1].cx }
            blocks.remove(at: i)
            changed = true
            break
        }
    }
    return blocks
}

// MARK: - Oversized-group recursive splitting

/// Rough guess at a (possibly tilted) row boundary: samples local whitespace
/// minima in narrow x-slices, fits a line through them, and bipartitions
/// members by side. Seed only — refined by an exact SAT separator. Matches
/// Python `_rough_tilt_partition`.
func roughTiltPartition(_ dets: [OBBDetection], idxs: [Int], energy: EdgeEnergy, lo: Double, hi: Double, xLo: Double, xHi: Double, nSamples: Int = 24) -> ([Int], [Int])? {
    let width = xHi - xLo
    guard width > 1.0, idxs.count >= 4 else { return nil }
    let step = width / Double(nSamples)
    let sliceW = max(step, 40.0)
    let margin = max(4.0, 0.08 * (hi - lo))
    let mss = medianShortSide(idxs.map { dets[$0] })
    let smoothPx = max(5, Int((0.5 * mss).rounded()))

    var points: [CGPoint] = []
    for k in 0..<nSamples {
        let cx0 = xLo + Double(k) * step
        let cx1 = cx0 + sliceW
        if let yLocal = findPixelSeam(energy, axis: .y, fixedLo: cx0, fixedHi: cx1, searchLo: lo, searchHi: hi, forbidden: [], smoothPx: smoothPx, marginPx: margin) {
            points.append(CGPoint(x: (cx0 + cx1) / 2.0, y: yLocal))
        }
    }
    guard points.count >= 2 else { return nil }
    let line = fitLine(points)
    let top = idxs.filter { line.signed(CGPoint(x: dets[$0].cx, y: dets[$0].cy)) < 0 }
    let bot = idxs.filter { line.signed(CGPoint(x: dets[$0].cx, y: dets[$0].cy)) >= 0 }
    guard !top.isEmpty, !bot.isEmpty else { return nil }
    return (top, bot)
}

/// Fallback for a row group with genuinely zero OBB-extent gap anywhere
/// (camera tilt): rough-partitions, then finds an exact (possibly tilted)
/// SAT separator. Matches Python `_tilted_bipartition_seam`.
func tiltedBipartitionSeam(_ dets: [OBBDetection], idxs: [Int], energy: EdgeEnergy, lo: Double, hi: Double, pLo: Double, pHi: Double, imgW: Int, imgH: Int) -> (Line, [Int], [Int])? {
    guard let (top, bot) = roughTiltPartition(dets, idxs: idxs, energy: energy, lo: lo, hi: hi, xLo: pLo, xHi: pHi) else { return nil }
    let cornersTop = allCorners(dets, top)
    let cornersBot = allCorners(dets, bot)
    guard let line = satSeparatingLine(
        cornersA: cornersTop, cornersB: cornersBot,
        topLine: .vertical(atX: 0), bottomLine: .vertical(atX: Double(imgW)),
        imgW: imgW, imgH: imgH, prefer: .horizontal
    ), fullySeparates(line, cornersTop, cornersBot) else { return nil }
    return (line, top, bot)
}

/// Recursively splits `group` whenever its extent along `axis` exceeds
/// `maxDim`: "crop each shelf group on its own, further cropped if long."
/// The split point comes from `findPixelSeam` with every member's own span
/// marked forbidden, so a size-forced cut can never land on a detected OBB;
/// for `axis == .y` only, falls back to `tiltedBipartitionSeam` if no
/// axis-aligned seam is safe. Matches Python `split_oversized_group`.
/// Returns `(groups, seams)` where `seams[k]` is the exact boundary between
/// `groups[k]` and `groups[k+1]`.
func splitOversizedGroup(_ dets: [OBBDetection], group: [Int], energy: EdgeEnergy, axis: ProfileAxis, maxDim: Double, minSplitMembers: Int, imgW: Int, imgH: Int) -> ([[Int]], [Line]) {
    func ownSpan(_ i: Int) -> (Double, Double) { axis == .y ? dets[i].ySpan : dets[i].xSpan }
    func perpSpan(_ i: Int) -> (Double, Double) { axis == .y ? dets[i].xSpan : dets[i].ySpan }
    func extent(_ idxs: [Int]) -> (Double, Double) {
        let spans = idxs.map(ownSpan)
        return (spans.map(\.0).min()!, spans.map(\.1).max()!)
    }
    func perpExtent(_ idxs: [Int]) -> (Double, Double) {
        let spans = idxs.map(perpSpan)
        return (spans.map(\.0).min()!, spans.map(\.1).max()!)
    }

    func rec(_ idxs: [Int]) -> ([[Int]], [Line]) {
        guard idxs.count >= 2 * minSplitMembers else { return ([idxs], []) }
        let (lo, hi) = extent(idxs)
        guard hi - lo > maxDim else { return ([idxs], []) }

        let (pLo, pHi) = perpExtent(idxs)
        let mss = medianShortSide(idxs.map { dets[$0] })
        let pad = 0.15 * mss
        let forbidden: [(Double, Double)] = idxs.map { i in
            let s = ownSpan(i)
            return (s.0 - pad, s.1 + pad)
        }
        let margin = max(4.0, 0.5 * mss)
        let splitPos = findPixelSeam(
            energy, axis: axis, fixedLo: pLo, fixedHi: pHi, searchLo: lo, searchHi: hi,
            forbidden: forbidden, smoothPx: max(3, Int((0.15 * mss).rounded())), marginPx: margin
        )

        var a: [Int]?
        var b: [Int]?
        var line: Line?
        if let splitPos {
            let candA = idxs.filter { ownSpan($0).1 <= splitPos }
            let candB = idxs.filter { ownSpan($0).0 >= splitPos }
            let candLine: Line = axis == .y ? .horizontal(atY: splitPos) : .vertical(atX: splitPos)
            if !candA.isEmpty, !candB.isEmpty, candA.count + candB.count == idxs.count,
               fullySeparates(candLine, allCorners(dets, candA), allCorners(dets, candB)) {
                a = candA; b = candB; line = candLine
            }
        }
        if line == nil, axis == .y,
           let (tiltedLine, tTop, tBot) = tiltedBipartitionSeam(dets, idxs: idxs, energy: energy, lo: lo, hi: hi, pLo: pLo, pHi: pHi, imgW: imgW, imgH: imgH) {
            line = tiltedLine; a = tTop; b = tBot
        }
        guard let finalLine = line, let aIdx = a, let bIdx = b else { return ([idxs], []) }

        let keyFn: (Int) -> Double = axis == .y ? { dets[$0].cy } : { dets[$0].cx }
        let (aGroups, aSeams) = rec(aIdx.sorted { keyFn($0) < keyFn($1) })
        let (bGroups, bSeams) = rec(bIdx.sorted { keyFn($0) < keyFn($1) })
        return (aGroups + bGroups, aSeams + [finalLine] + bSeams)
    }

    return rec(group)
}

/// Applies `splitOversizedGroup` to every group, preserving order along
/// `axis` (discards internal seam lines). Matches Python
/// `split_oversized_group_list`.
func splitOversizedGroupList(_ dets: [OBBDetection], groups: [[Int]], energy: EdgeEnergy, axis: ProfileAxis, maxDim: Double, minSplitMembers: Int, imgW: Int, imgH: Int) -> [[Int]] {
    let keyFn: (Int) -> Double = axis == .y ? { dets[$0].cy } : { dets[$0].cx }
    var out: [[Int]] = []
    for group in groups {
        let (subGroups, _) = splitOversizedGroup(dets, group: group, energy: energy, axis: axis, maxDim: maxDim, minSplitMembers: minSplitMembers, imgW: imgW, imgH: imgH)
        out.append(contentsOf: subGroups)
    }
    out.sort { a, b in
        (a.map(keyFn).reduce(0, +) / Double(a.count)) < (b.map(keyFn).reduce(0, +) / Double(b.count))
    }
    return out
}

// MARK: - Full-scene jigsaw with OBB-edge seams

/// Finds a straight seam with every OBB corner of `leftBlock` strictly on
/// one side and every corner of `rightBlock` on the other. Tries, in
/// priority order: the pixel-texture minimum within any real whitespace
/// gap, the gap midpoint, the facing side-face of the two nearest end
/// books, then an exact SAT separator. Returns `nil` if the two blocks are
/// geometrically interleaved and cannot be split by any straight line.
/// Matches Python `find_separating_line`.
func findSeparatingLine(_ dets: [OBBDetection], leftBlock: [Int], rightBlock: [Int], colGapPx: Double, topLine: Line, bottomLine: Line, imgW: Int, imgH: Int, energy: EdgeEnergy?) -> Line? {
    let leftEnd = dets[leftBlock.max { dets[$0].cx < dets[$1].cx }!]
    let rightEnd = dets[rightBlock.min { dets[$0].cx < dets[$1].cx }!]
    let gapLo = leftEnd.xSpan.1
    let gapHi = rightEnd.xSpan.0
    let leftCorners = allCorners(dets, leftBlock)
    let rightCorners = allCorners(dets, rightBlock)

    var candidates: [Line] = []
    if gapHi > gapLo {
        if let energy {
            let yLo = min(topLine.yValue(atX: gapLo), topLine.yValue(atX: gapHi))
            let yHi = max(bottomLine.yValue(atX: gapLo), bottomLine.yValue(atX: gapHi))
            if let pxX = findPixelSeam(
                energy, axis: .x, fixedLo: yLo, fixedHi: yHi, searchLo: gapLo, searchHi: gapHi,
                forbidden: [], smoothPx: max(3, Int((0.2 * max(colGapPx, gapHi - gapLo)).rounded())),
                marginPx: max(2.0, 0.15 * (gapHi - gapLo))
            ) {
                candidates.append(.vertical(atX: pxX))
            }
        }
        candidates.append(.vertical(atX: 0.5 * (gapLo + gapHi)))
    }
    candidates.append(sideEdgeLine(leftEnd, preferMinX: false))
    candidates.append(sideEdgeLine(rightEnd, preferMinX: true))

    for cand in candidates
    where seamUsable(cand, topLine: topLine, bottomLine: bottomLine, imgW: imgW, imgH: imgH)
        && fullySeparates(cand, leftCorners, rightCorners) {
        return cand
    }

    if let satLine = satSeparatingLine(cornersA: leftCorners, cornersB: rightCorners, topLine: topLine, bottomLine: bottomLine, imgW: imgW, imgH: imgH),
       fullySeparates(satLine, leftCorners, rightCorners) {
        return satLine
    }
    return nil
}

/// Builds vertical seams between consecutive column blocks, merging any
/// adjacent pair for which `findSeparatingLine` fails so the result never
/// cuts a book. Matches Python `resolve_column_seams`.
func resolveColumnSeams(_ dets: [OBBDetection], blocks: [[Int]], colGapPx: Double, topLine: Line, bottomLine: Line, imgW: Int, imgH: Int, energy: EdgeEnergy?) -> ([[Int]], [Line]) {
    func sep(_ a: [Int], _ b: [Int]) -> Line? {
        findSeparatingLine(dets, leftBlock: a, rightBlock: b, colGapPx: colGapPx, topLine: topLine, bottomLine: bottomLine, imgW: imgW, imgH: imgH, energy: energy)
    }
    var blks = blocks
    var changed = true
    while changed, blks.count > 1 {
        changed = false
        for i in 0..<(blks.count - 1) {
            if sep(blks[i], blks[i + 1]) == nil {
                blks[i] = (blks[i] + blks[i + 1]).sorted { dets[$0].cx < dets[$1].cx }
                blks.remove(at: i + 1)
                changed = true
                break
            }
        }
    }
    var vLines: [Line] = [.vertical(atX: 0)]
    for bi in 0..<(blks.count - 1) {
        guard let line = sep(blks[bi], blks[bi + 1]) else {
            fatalError("resolveColumnSeams: adjacent blocks must be separable after the merge loop above")
        }
        vLines.append(line)
    }
    vLines.append(.vertical(atX: Double(imgW)))
    return (blks, vLines)
}

/// Intersects `top`/`bottom` with `left`/`right` to form a scene-space quad
/// (TL, TR, BR, BL), rejecting non-finite or off-image geometry. Matches
/// Python `quad_from_lines`.
func quadFromLines(top: Line, bottom: Line, left: Line, right: Line, imgW: Int, imgH: Int) -> [CGPoint]? {
    guard let tl = top.intersection(with: left),
          let tr = top.intersection(with: right),
          let br = bottom.intersection(with: right),
          let bl = bottom.intersection(with: left) else { return nil }
    let pts = [tl, tr, br, bl]
    let margin = LayoutConstants.seamUsableImageMargin
    for p in pts {
        guard p.x.isFinite, p.y.isFinite else { return nil }
        if Double(p.x) < -margin || Double(p.y) < -margin || Double(p.x) > Double(imgW) + margin || Double(p.y) > Double(imgH) + margin {
            return nil
        }
    }
    let clamped = pts.map { CGPoint(x: min(max($0.x, 0), CGFloat(imgW)), y: min(max($0.y, 0), CGFloat(imgH))) }
    if dist(clamped[0], clamped[1]) < 2 || dist(clamped[0], clamped[3]) < 2 { return nil }
    return clamped
}

// MARK: - planCrops

/// Partitions the scene into quads bounded by shelf-band whitespace cuts
/// and orientation-aware column seams. Matches Python `plan_crops`.
public func planCrops(
    dets: [OBBDetection],
    imgW: Int,
    imgH: Int,
    raster: SceneRaster,
    angleTolDeg: Double = LayoutConstants.defaultAngleTolDeg,
    rowGapK: Double = 0.2,
    colGapK: Double = 1.0,
    minBlockMembers: Int = 2,
    imgsz: Int = 1024,
    maxCropDimK: Double = 1.5
) -> [CropPlan] {
    guard let energy = EdgeEnergy(raster: raster) else { return [] }
    let maxDim = maxCropDimK * Double(imgsz)

    let originalBands = buildShelfBands(dets, rowGapK: rowGapK)
    let colGapPx = max(LayoutConstants.minColGapFloor, colGapK * medianShortSide(dets))

    var rowSpans: [(Line, Line, [Int])] = []
    if originalBands.isEmpty {
        rowSpans.append((.horizontal(atY: 0), .horizontal(atY: Double(imgH)), []))
    } else {
        let extents = originalBands.map { bandExtent(dets, $0) }
        let expansions = originalBands.map {
            splitOversizedGroup(dets, group: $0, energy: energy, axis: .y, maxDim: maxDim, minSplitMembers: minBlockMembers, imgW: imgW, imgH: imgH)
        }
        var prev = Line.horizontal(atY: 0)
        let firstTop = extents[0].0
        if firstTop > 2.0 {
            let cut = Line.horizontal(atY: firstTop)
            rowSpans.append((prev, cut, []))
            prev = cut
        }
        for i in 0..<originalBands.count {
            let (subBands, internalSeams) = expansions[i]
            let endCut: Line
            if i + 1 < originalBands.count {
                let mid = 0.5 * (extents[i].1 + extents[i + 1].0)
                endCut = .horizontal(atY: min(max(mid, 1.0), Double(imgH) - 1.0))
            } else {
                let cutY = extents[i].1 + 1.0
                endCut = .horizontal(atY: min(max(cutY, 1.0), Double(imgH)))
            }
            let boundaries = [prev] + internalSeams + [endCut]
            for j in 0..<subBands.count {
                rowSpans.append((boundaries[j], boundaries[j + 1], subBands[j]))
            }
            prev = endCut
        }
        if prev.yValue(atX: Double(imgW) / 2) < Double(imgH) - 0.5 {
            rowSpans.append((prev, .horizontal(atY: Double(imgH)), []))
        }
    }

    var plans: [CropPlan] = []
    for (shelfId, (topLine, bottomLine, members)) in rowSpans.enumerated() {
        if members.isEmpty {
            if let quad = quadFromLines(top: topLine, bottom: bottomLine, left: .vertical(atX: 0), right: .vertical(atX: Double(imgW)), imgW: imgW, imgH: imgH) {
                plans.append(CropPlan(shelfId: shelfId, blockId: 0, angleDeg: 0.0, quad: quad, memberIndices: []))
            }
            continue
        }

        var blocks = buildColumnBlocks(dets, members: members, angleTolDeg: angleTolDeg, colGapPx: colGapPx, minBlockMembers: minBlockMembers)
        blocks = splitOversizedGroupList(dets, groups: blocks, energy: energy, axis: .x, maxDim: maxDim, minSplitMembers: minBlockMembers, imgW: imgW, imgH: imgH)
        let (resolvedBlocks, vLines) = resolveColumnSeams(dets, blocks: blocks, colGapPx: colGapPx, topLine: topLine, bottomLine: bottomLine, imgW: imgW, imgH: imgH, energy: energy)

        for (blockId, block) in resolvedBlocks.enumerated() {
            guard let quad = quadFromLines(top: topLine, bottom: bottomLine, left: vLines[blockId], right: vLines[blockId + 1], imgW: imgW, imgH: imgH) else { continue }
            let angleDeg = circularMean(block.map { dets[$0].longAxisAngle() }) * 180 / .pi
            plans.append(CropPlan(shelfId: shelfId, blockId: blockId, angleDeg: (angleDeg * 100).rounded() / 100, quad: quad, memberIndices: block))
        }
    }
    return plans
}
