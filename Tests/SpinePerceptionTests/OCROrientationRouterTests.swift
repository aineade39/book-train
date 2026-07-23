import CoreGraphics
import ImageIO
import XCTest

@testable import SpineCore
@testable import SpinePerception

final class OCROrientationRouterTests: XCTestCase {

    func testOrientationOrderIsAspectGuided() {
        let tall = OCROrientationRouter.orientationOrder(width: 40, height: 120)
        XCTAssertEqual(tall.primary, .right)
        XCTAssertEqual(tall.secondary, .up)

        let wide = OCROrientationRouter.orientationOrder(width: 200, height: 40)
        XCTAssertEqual(wide.primary, .up)
        XCTAssertEqual(wide.secondary, .right)
    }

    /// The winning pass is chosen by confidence *sum*, not observation
    /// count -- the exact anti-pattern the placeholder `ocrSpine` used.
    /// Rig a pass with many low-confidence junk observations against a
    /// pass with one clean, high-confidence observation.
    func testWinnerIsChosenByConfidenceSumNotObservationCount() throws {
        let det = OBBDetection(cx: 100, cy: 100, w: 40, h: 120, angle: 0, conf: 0.9)
        let (primary, secondary) = OCROrientationRouter.orientationOrder(width: det.w, height: det.h)

        let manyJunk = (0..<10).map { fullCropObservation(text: "x\($0)", confidence: 0.05) }
        let oneClean = [fullCropObservation(text: "Dune", confidence: 0.95)]

        let recognizer = FakeTextRecognizer(byOrientation: [primary: manyJunk, secondary: oneClean])
        let router = OCROrientationRouter(recognizer: recognizer)
        let result = try router.recognize(crop: makeSolidCGImage(width: 40, height: 120), detection: det)

        XCTAssertEqual(result.winningPass.orientation, secondary)
        XCTAssertEqual(result.assembledText, "Dune")
        XCTAssertFalse(result.ranThirdPass)
    }

    func testThirdPassIsSkippedWhenFirstTwoPassTheQualityGate() throws {
        let det = OBBDetection(cx: 100, cy: 100, w: 40, h: 120, angle: 0, conf: 0.9)
        let (primary, secondary) = OCROrientationRouter.orientationOrder(width: det.w, height: det.h)

        // A sentinel for `.down` that should never be consulted.
        let recognizer = FakeTextRecognizer(byOrientation: [
            primary: [fullCropObservation(text: "Dune Messiah", confidence: 0.9)],
            secondary: [],
            .down: [fullCropObservation(text: "SHOULD NOT BE USED", confidence: 0.99)],
        ])
        let router = OCROrientationRouter(recognizer: recognizer)
        let result = try router.recognize(crop: makeSolidCGImage(width: 40, height: 120), detection: det)

        XCTAssertFalse(result.ranThirdPass)
        XCTAssertEqual(result.assembledText, "Dune Messiah")
    }

    func testThirdPassRunsOnlyWhenBothFirstPassesFailQualityGate() throws {
        let det = OBBDetection(cx: 100, cy: 100, w: 40, h: 120, angle: 0, conf: 0.9)
        let (primary, secondary) = OCROrientationRouter.orientationOrder(width: det.w, height: det.h)

        let recognizer = FakeTextRecognizer(byOrientation: [
            primary: [fullCropObservation(text: "x", confidence: 0.02)],
            secondary: [],
            .down: [fullCropObservation(text: "Dune Messiah", confidence: 0.95)],
        ])
        let router = OCROrientationRouter(recognizer: recognizer)
        let result = try router.recognize(crop: makeSolidCGImage(width: 40, height: 120), detection: det)

        XCTAssertTrue(result.ranThirdPass)
        XCTAssertEqual(result.winningPass.orientation, .down)
        XCTAssertEqual(result.assembledText, "Dune Messiah")
        XCTAssertTrue(result.passedQualityGate)
    }

    func testAllPassesEmptyReturnsUngatedEmptyResult() throws {
        let det = OBBDetection(cx: 100, cy: 100, w: 40, h: 120, angle: 0, conf: 0.9)
        let recognizer = FakeTextRecognizer(byOrientation: [:])
        let router = OCROrientationRouter(recognizer: recognizer)
        let result = try router.recognize(crop: makeSolidCGImage(width: 40, height: 120), detection: det)

        XCTAssertTrue(result.ranThirdPass)
        XCTAssertFalse(result.passedQualityGate)
        XCTAssertEqual(result.assembledText, "")
    }
}
