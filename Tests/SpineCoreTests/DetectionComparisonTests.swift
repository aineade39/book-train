import XCTest

@testable import SpineCore

final class DetectionComparisonTests: XCTestCase {
    private func det(_ cx: Double, _ cy: Double, w: Double = 40, h: Double = 200, angleDeg: Double = 0, conf: Float = 0.9) -> OBBDetection {
        OBBDetection(cx: cx, cy: cy, w: w, h: h, angle: angleDeg * .pi / 180, conf: conf)
    }

    // MARK: - matchDetections

    func testIdenticalSetsFullyMatch() {
        let a = [det(100, 100), det(300, 100), det(500, 100)]
        let result = matchDetections(a, a)
        XCTAssertEqual(result.pairs.count, 3)
        XCTAssertTrue(result.onlyA.isEmpty)
        XCTAssertTrue(result.onlyB.isEmpty)
        for p in result.pairs {
            XCTAssertEqual(p.a, p.b, "identical sets should match index-to-index")
            XCTAssertEqual(p.iou, 1.0, accuracy: 1e-9)
        }
    }

    func testDisjointSetsMatchNothing() {
        let a = [det(100, 100)]
        let b = [det(900, 100)]
        let result = matchDetections(a, b)
        XCTAssertTrue(result.pairs.isEmpty)
        XCTAssertEqual(result.onlyA, [0])
        XCTAssertEqual(result.onlyB, [0])
    }

    func testGreedyPrefersHigherIoUAndAssignsOneToOne() {
        // b0 overlaps a0 exactly; b1 is a shifted copy that still overlaps
        // a0 above threshold but must not steal it from b0.
        let a = [det(100, 100)]
        let b = [det(100, 100), det(110, 100)]
        let result = matchDetections(a, b, iouThreshold: 0.3)
        XCTAssertEqual(result.pairs.count, 1)
        XCTAssertEqual(result.pairs[0].b, 0, "exact copy should win over the shifted one")
        XCTAssertEqual(result.onlyB, [1])
        XCTAssertTrue(result.onlyA.isEmpty)
    }

    func testThresholdExcludesWeakOverlap() {
        // Half-width shift: IoU well below 0.5 but above 0.2.
        let a = [det(100, 100, w: 40)]
        let b = [det(130, 100, w: 40)]
        XCTAssertTrue(matchDetections(a, b, iouThreshold: 0.5).pairs.isEmpty)
        XCTAssertEqual(matchDetections(a, b, iouThreshold: 0.1).pairs.count, 1)
    }

    // MARK: - sceneScore

    func testSceneScoreCountsPerConfidenceTier() {
        let dets = [
            det(0, 0, conf: 0.16), det(0, 0, conf: 0.55),
            det(0, 0, conf: 0.75), det(0, 0, conf: 0.10),
        ]
        let score = sceneScore(dets: dets, gtEst: 6)
        XCTAssertEqual(score.count15, 3)
        XCTAssertEqual(score.count50, 2)
        XCTAssertEqual(score.count70, 1)
        XCTAssertEqual(score.recallProxy15, 0.5, accuracy: 1e-9)
        XCTAssertEqual(score.recallProxy50, 2.0 / 6.0, accuracy: 1e-9)
    }

    func testSceneScoreRecallProxyIsCappedAtOne() {
        let dets = (0..<10).map { _ in det(0, 0, conf: 0.9) }
        let score = sceneScore(dets: dets, gtEst: 5)
        XCTAssertEqual(score.recallProxy15, 1.0)
        XCTAssertEqual(score.recallProxy50, 1.0)
    }

    func testSceneScoreZeroGtEstYieldsZeroProxy() {
        let score = sceneScore(dets: [det(0, 0, conf: 0.9)], gtEst: 0)
        XCTAssertEqual(score.recallProxy15, 0.0)
        XCTAssertEqual(score.recallProxy50, 0.0)
    }
}
