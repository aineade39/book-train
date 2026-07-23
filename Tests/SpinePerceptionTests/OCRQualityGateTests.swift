import XCTest

@testable import SpinePerception

final class OCRQualityGateTests: XCTestCase {
    let gate = OCRQualityGate.default

    func testHighConfidencePlausibleTextPasses() {
        let inputs = OCRQualityGateInputs(
            meanConfidence: 0.9, orientationAgreement: 0.6,
            assembledText: "Project Hail Mary", detectionConfidence: 0.85
        )
        XCTAssertTrue(gate.passes(inputs))
    }

    func testEmptyTextFails() {
        let inputs = OCRQualityGateInputs(meanConfidence: 0.9, orientationAgreement: 0.9, assembledText: "")
        XCTAssertFalse(gate.passes(inputs))
    }

    func testLowConfidenceGarbageFails() {
        let inputs = OCRQualityGateInputs(meanConfidence: 0.05, orientationAgreement: 0.0, assembledText: "x")
        XCTAssertFalse(gate.passes(inputs))
    }

    func testPunctuationOnlyTextScoresLowerThanWordLikeText() {
        let punctuationOnly = OCRQualityGateInputs(meanConfidence: 0.8, orientationAgreement: 0.5, assembledText: "!!!///###")
        let wordLike = OCRQualityGateInputs(meanConfidence: 0.8, orientationAgreement: 0.5, assembledText: "Dune Messiah")
        XCTAssertLessThan(gate.score(punctuationOnly), gate.score(wordLike))
    }

    func testDetectionConfidenceIsFoldedIntoScoreWhenProvided() {
        let withoutDetection = OCRQualityGateInputs(meanConfidence: 0.5, orientationAgreement: 0.5, assembledText: "Dune")
        let withLowDetection = OCRQualityGateInputs(
            meanConfidence: 0.5, orientationAgreement: 0.5, assembledText: "Dune", detectionConfidence: 0.05
        )
        XCTAssertGreaterThan(gate.score(withoutDetection), gate.score(withLowDetection))
    }

    func testScoreIsMonotonicInConfidence() {
        let low = OCRQualityGateInputs(meanConfidence: 0.1, orientationAgreement: 0.5, assembledText: "Dune Messiah")
        let high = OCRQualityGateInputs(meanConfidence: 0.95, orientationAgreement: 0.5, assembledText: "Dune Messiah")
        XCTAssertLessThan(gate.score(low), gate.score(high))
    }
}
