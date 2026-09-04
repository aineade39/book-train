import XCTest

@testable import SpineMatching

final class OCRConfusionTests: XCTestCase {
    func testExpandsWithinLengthAndConfidenceGate() {
        // "d0nut" (5 chars) with a confusable "0" -> "donut".
        let variants = OCRConfusion.expand(token: "d0nut", confidence: 0.5)
        XCTAssertTrue(variants.contains("donut"), "expected 0->o expansion; got \(variants)")
    }

    func testNoExpansionAboveConfidenceCeiling() {
        let variants = OCRConfusion.expand(token: "d0nut", confidence: 0.95)
        XCTAssertTrue(variants.isEmpty)
    }

    func testNoExpansionBelowMinLength() {
        // 3 chars, below the 4-char floor.
        let variants = OCRConfusion.expand(token: "d0g", confidence: 0.5)
        XCTAssertTrue(variants.isEmpty)
    }

    func testNoExpansionAboveMaxLength() {
        let longToken = String(repeating: "a", count: 19)
        XCTAssertTrue(OCRConfusion.expand(token: longToken, confidence: 0.5).isEmpty)
    }

    func testCapsAtTwoExpansionsPerToken() {
        // Contains several confusable characters/substrings.
        let variants = OCRConfusion.expand(token: "b00k5", confidence: 0.4)
        XCTAssertLessThanOrEqual(variants.count, OCRConfusion.maxExpansionsPerToken)
    }

    func testMultiCharacterRuleExpandsRnToM() {
        let variants = OCRConfusion.expand(token: "corner", confidence: 0.3)
        XCTAssertTrue(variants.contains("comer"), "expected rn->m expansion; got \(variants)")
    }

    func testNeverReturnsTheOriginalTokenAsAVariant() {
        let variants = OCRConfusion.expand(token: "regular", confidence: 0.3)
        XCTAssertFalse(variants.contains("regular"))
    }
}
