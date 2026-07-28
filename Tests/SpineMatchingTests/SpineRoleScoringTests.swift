import CoreGraphics
import XCTest

@testable import SpineMatching

final class SpineRoleScoringTests: XCTestCase {
    private let cropWidth = 40.0
    private let cropHeight = 300.0

    private func line(text: String, confidence: Float, top: Double, height: Double) -> SpineTextLine {
        SpineTextLine(
            text: text, confidence: confidence,
            topLeft: CGPoint(x: 2, y: top), topRight: CGPoint(x: 38, y: top),
            bottomRight: CGPoint(x: 38, y: top + height), bottomLeft: CGPoint(x: 2, y: top + height),
            cropWidth: cropWidth, cropHeight: cropHeight
        )
    }

    func testRelativeHeightMatchesRectangleHeightFraction() {
        // A clean axis-aligned 30px-tall line in a 300px-tall crop -> 0.1,
        // not 0.2 (regression for the "2A/w double-counts area" bug).
        let l = line(text: "TITLE LINE", confidence: 0.9, top: 0, height: 30)
        XCTAssertEqual(SpineRoleScoring.relativeHeight(of: l), 0.1, accuracy: 0.001)
    }

    func testSpinePositionIsMeanYFractionOfCropHeight() {
        let l = line(text: "TITLE", confidence: 0.9, top: 0, height: 30)
        XCTAssertEqual(SpineRoleScoring.spinePosition(of: l), 15.0 / 300.0, accuracy: 0.001)
    }

    func testTallModerateHeightNearTopScoresHighestAsTitle() {
        // ~10% of crop height, near the top -- squarely in the title band.
        let l = line(text: "PROJECT HAIL MARY", confidence: 0.9, top: 0, height: 30)
        let scores = SpineRoleScoring.roleScores(for: l)
        XCTAssertGreaterThan(scores.title, scores.author)
        XCTAssertGreaterThan(scores.title, scores.other)
    }

    func testShortLineInAuthorBandWithPersonNameBoostScoresHighestAsAuthor() {
        // ~4% of crop height, positioned mid-spine, reads like "By Author Name".
        let l = line(text: "By Andy Weir", confidence: 0.9, top: 99, height: 12)
        let scores = SpineRoleScoring.roleScores(for: l)
        XCTAssertGreaterThan(scores.author, scores.title)
        XCTAssertGreaterThan(scores.author, scores.other)
    }

    func testISBNLikeLineGetsOtherExclusionBoost() {
        let plain = line(text: "9780593135204", confidence: 0.9, top: 270, height: 8)
        let scores = SpineRoleScoring.roleScores(for: plain)
        XCTAssertGreaterThan(scores.other, scores.title)
        XCTAssertGreaterThan(scores.other, scores.author)
    }

    func testRoleScoresAlwaysSumToOne() {
        for (top, height, text) in [(0.0, 30.0, "TITLE"), (99.0, 12.0, "By Someone"), (270.0, 8.0, "PUBLISHER CO")] {
            let l = line(text: text, confidence: 0.8, top: top, height: height)
            let scores = SpineRoleScoring.roleScores(for: l)
            XCTAssertEqual(scores.title + scores.author + scores.other, 1.0, accuracy: 0.0001)
        }
    }
}
