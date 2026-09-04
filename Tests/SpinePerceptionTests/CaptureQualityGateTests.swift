import XCTest

@testable import SpinePerception

final class CaptureQualityGateTests: XCTestCase {
    let gate = CaptureQualityGate.default

    // MARK: - Sharpness

    func testCheckerboardScoresSharperThanSolidGray() {
        let sharp = makeCheckerboardCGImage(width: 200, height: 200, cols: 20, rows: 20)
        let flat = makeSolidCGImage(width: 200, height: 200, gray: 128)
        XCTAssertGreaterThan(gate.score(sharp).sharpness, gate.score(flat).sharpness)
    }

    func testCheckerboardScoresSharperThanNoisyGray() {
        let sharp = makeCheckerboardCGImage(width: 200, height: 200, cols: 20, rows: 20)
        let noisyBlur = makeNoisyGrayCGImage(width: 200, height: 200, gray: 128, noise: 3)
        XCTAssertGreaterThan(gate.score(sharp).sharpness, gate.score(noisyBlur).sharpness)
    }

    func testSolidGrayFailsSharpnessGate() {
        let flat = makeSolidCGImage(width: 200, height: 200, gray: 128)
        XCTAssertFalse(gate.passes(flat))
    }

    func testCheckerboardPassesGate() {
        let sharp = makeCheckerboardCGImage(width: 200, height: 200, cols: 20, rows: 20)
        XCTAssertTrue(gate.passes(sharp))
    }

    // MARK: - Exposure

    func testMidGrayExposureScoresHigherThanNearBlack() {
        let midGray = makeSolidCGImage(width: 100, height: 100, gray: 128)
        let nearBlack = makeSolidCGImage(width: 100, height: 100, gray: 5)
        XCTAssertGreaterThan(gate.score(midGray).exposure, gate.score(nearBlack).exposure)
    }

    func testMidGrayExposureScoresHigherThanNearWhite() {
        let midGray = makeSolidCGImage(width: 100, height: 100, gray: 128)
        let nearWhite = makeSolidCGImage(width: 100, height: 100, gray: 250)
        XCTAssertGreaterThan(gate.score(midGray).exposure, gate.score(nearWhite).exposure)
    }

    func testNearBlackFailsExposureGate() {
        let nearBlack = makeSolidCGImage(width: 100, height: 100, gray: 2)
        XCTAssertFalse(gate.passes(nearBlack))
    }

    func testCheckerboardExposureIsWellBalanced() {
        // Half black / half white pixels averages to mid-gray.
        let checkerboard = makeCheckerboardCGImage(width: 200, height: 200, cols: 20, rows: 20)
        XCTAssertGreaterThan(gate.score(checkerboard).exposure, 0.8)
    }
}
