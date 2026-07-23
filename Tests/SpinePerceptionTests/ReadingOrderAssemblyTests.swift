import CoreGraphics
import XCTest

@testable import SpineCore
@testable import SpinePerception

final class ReadingOrderAssemblyTests: XCTestCase {

    func testEmptyObservationsProducesEmptyString() {
        let det = OBBDetection(cx: 0, cy: 0, w: 40, h: 100, angle: 0, conf: 0.9)
        XCTAssertEqual(assembleReadingOrder(observations: [], orientation: .up, detection: det), "")
    }

    func testSingleObservationIsReturnedAsIs() {
        let det = OBBDetection(cx: 100, cy: 200, w: 40, h: 100, angle: 0, conf: 0.9)
        let text = assembleReadingOrder(
            observations: [fullCropObservation(text: "Solo", confidence: 0.9)],
            orientation: .up, detection: det
        )
        XCTAssertEqual(text, "Solo")
    }

    /// Tall spine (h > w, angle 0): the long axis is vertical. A title band
    /// near the top of the crop and an author band near the bottom must
    /// assemble title-then-author, matching how a spine is actually laid
    /// out (title above author, read top to bottom).
    func testTallSpineOrdersTopToBottom() {
        let det = OBBDetection(cx: 100, cy: 200, w: 40, h: 100, angle: 0, conf: 0.9)
        let title = bandObservation(text: "TITLE", confidence: 0.9, yBottom: 0.6, yTop: 1.0)
        let author = bandObservation(text: "AUTHOR", confidence: 0.9, yBottom: 0.0, yTop: 0.4)
        // Deliberately passed in reverse (raw Vision result order should
        // not matter -- only position along the long axis should).
        let text = assembleReadingOrder(observations: [author, title], orientation: .up, detection: det)
        XCTAssertEqual(text, "TITLE AUTHOR")
    }

    func testWideSpineOrdersLeftToRight() {
        let det = OBBDetection(cx: 200, cy: 50, w: 200, h: 40, angle: 0, conf: 0.9)
        let left = columnObservation(text: "LEFT", confidence: 0.9, xLeft: 0.0, xRight: 0.4)
        let right = columnObservation(text: "RIGHT", confidence: 0.9, xLeft: 0.6, xRight: 1.0)
        let text = assembleReadingOrder(observations: [right, left], orientation: .up, detection: det)
        XCTAssertEqual(text, "LEFT RIGHT")
    }

    /// `longAxisAngle()` is only defined mod pi (a direction *line*, not a
    /// signed vector — see `SpineGeometry.longAxisAngle`), so which
    /// geometric end projects "first" can flip sign as the detection's
    /// angle sweeps through a full rotation. What must stay true
    /// regardless: whichever slot (top-of-crop band vs. bottom-of-crop
    /// band) projects first is a property of the *geometry*, not of which
    /// text happens to occupy that slot — swapping the two texts between
    /// the same two fixed slots must exactly reverse the assembled order.
    func testSwappingWhichTextOccupiesEachSlotReversesAssembledOrder() {
        for angleDeg in [0.0, 30.0, 90.0, 160.0] {
            let det = OBBDetection(cx: 300, cy: 300, w: 40, h: 120, angle: angleDeg * .pi / 180, conf: 0.9)
            let topSlot = { (text: String) in bandObservation(text: text, confidence: 0.9, yBottom: 0.6, yTop: 1.0) }
            let bottomSlot = { (text: String) in bandObservation(text: text, confidence: 0.9, yBottom: 0.0, yTop: 0.4) }

            let orderA = assembleReadingOrder(observations: [topSlot("X"), bottomSlot("Y")], orientation: .up, detection: det)
            let orderB = assembleReadingOrder(observations: [topSlot("Y"), bottomSlot("X")], orientation: .up, detection: det)

            let reversedA = orderA.split(separator: " ").reversed().joined(separator: " ")
            XCTAssertEqual(reversedA, orderB, "swapping slot occupants should reverse the order at angle \(angleDeg)deg")
        }
    }

    func testOrderIsIndependentOfInputOrderAndConfidence() {
        let det = OBBDetection(cx: 100, cy: 200, w: 40, h: 100, angle: 0, conf: 0.9)
        let title = bandObservation(text: "TITLE", confidence: 0.2, yBottom: 0.6, yTop: 1.0)
        let author = bandObservation(text: "AUTHOR", confidence: 0.99, yBottom: 0.0, yTop: 0.4)
        // Even though "AUTHOR" has far higher confidence, position (not
        // confidence) determines assembly order.
        let text = assembleReadingOrder(observations: [author, title], orientation: .up, detection: det)
        XCTAssertEqual(text, "TITLE AUTHOR")
    }
}

private func columnObservation(text: String, confidence: Float, xLeft: Double, xRight: Double) -> RecognizedTextObservation {
    RecognizedTextObservation(
        text: text, confidence: confidence,
        topLeft: CGPoint(x: xLeft, y: 1), topRight: CGPoint(x: xRight, y: 1),
        bottomRight: CGPoint(x: xRight, y: 0), bottomLeft: CGPoint(x: xLeft, y: 0)
    )
}
