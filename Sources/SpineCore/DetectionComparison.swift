import Foundation

// Stage 0 of the jigsaw-zoom plan (~/dev/book-id-design/docs/
// jigsaw-zoom-requirements.md §A-8): pipeline-vs-pipeline comparison
// primitives for the `zoom-compare` harness. No ground-truth boxes exist
// for the scenes corpus (YOLO dataset labels are not trusted), so scoring
// is (a) count-based recall proxy against hand-estimated per-scene spine
// counts and (b) pairwise greedy rotated-IoU matching between two
// pipelines' detections.

// MARK: - Pairwise matching

/// One-to-one greedy assignment between two detection sets, highest IoU
/// first. `onlyA` / `onlyB` are the indices left unmatched — the
/// detections unique to each pipeline at this IoU threshold.
public struct DetectionMatchResult {
    public let pairs: [(a: Int, b: Int, iou: Double)]
    public let onlyA: [Int]
    public let onlyB: [Int]

    public init(pairs: [(a: Int, b: Int, iou: Double)], onlyA: [Int], onlyB: [Int]) {
        self.pairs = pairs
        self.onlyA = onlyA
        self.onlyB = onlyB
    }
}

/// Greedy rotated-IoU matching between `a` and `b`: all cross pairs with
/// IoU >= `iouThreshold`, taken best-first, each detection used at most
/// once. Greedy (not Hungarian) is enough here: spine detections rarely
/// have ambiguous multi-way overlaps at IoU >= 0.5, and the harness only
/// consumes matched/unmatched counts.
public func matchDetections(
    _ a: [OBBDetection], _ b: [OBBDetection], iouThreshold: Double = 0.5
) -> DetectionMatchResult {
    var candidates: [(iou: Double, a: Int, b: Int)] = []
    for (i, da) in a.enumerated() {
        for (j, db) in b.enumerated() {
            let iou = rotatedIoU(da, db)
            if iou >= iouThreshold {
                candidates.append((iou: iou, a: i, b: j))
            }
        }
    }
    candidates.sort { $0.iou > $1.iou }

    var usedA = Set<Int>()
    var usedB = Set<Int>()
    var pairs: [(a: Int, b: Int, iou: Double)] = []
    for c in candidates where !usedA.contains(c.a) && !usedB.contains(c.b) {
        usedA.insert(c.a)
        usedB.insert(c.b)
        pairs.append((a: c.a, b: c.b, iou: c.iou))
    }
    return DetectionMatchResult(
        pairs: pairs,
        onlyA: (0..<a.count).filter { !usedA.contains($0) },
        onlyB: (0..<b.count).filter { !usedB.contains($0) }
    )
}

// MARK: - Count-based recall proxy

/// Count-based score against a hand-estimated scene spine count — same
/// shape as `eval/scenes-review-20260718/recall_proxy.json` so new runs
/// stay comparable with that review. `recallProxy* = min(1, count / gtEst)`;
/// there are no reference boxes, so this is a recall *proxy*, not recall.
public struct SceneScore: Codable {
    public let count15: Int
    public let count50: Int
    public let count70: Int
    public let gtEst: Int
    public let recallProxy15: Double
    public let recallProxy50: Double

    enum CodingKeys: String, CodingKey {
        case count15 = "n15"
        case count50 = "n50"
        case count70 = "n70"
        case gtEst = "gt_est"
        case recallProxy15 = "recall_proxy_15"
        case recallProxy50 = "recall_proxy_50"
    }
}

public func sceneScore(dets: [OBBDetection], gtEst: Int) -> SceneScore {
    let count15 = dets.filter { $0.conf >= 0.15 }.count
    let count50 = dets.filter { $0.conf >= 0.50 }.count
    let count70 = dets.filter { $0.conf >= 0.70 }.count
    func proxy(_ n: Int) -> Double {
        gtEst > 0 ? min(1.0, Double(n) / Double(gtEst)) : 0.0
    }
    return SceneScore(
        count15: count15, count50: count50, count70: count70,
        gtEst: gtEst, recallProxy15: proxy(count15), recallProxy50: proxy(count50)
    )
}
