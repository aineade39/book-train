import XCTest

@testable import SpineMatching

final class NormalizationTests: XCTestCase {
    func testLowercasesAndFoldsDiacritics() {
        XCTAssertEqual(normalizeForSearch("Café DU MONDE"), "cafe du monde")
    }

    func testCollapsesWhitespace() {
        XCTAssertEqual(normalizeForSearch("The   Great\nGatsby"), "the great gatsby")
    }

    func testStripsDecorativePunctuationButKeepsMeaningfulMarks() {
        XCTAssertEqual(normalizeForSearch("O'Brien: \"The Things They Carried\""), "o'brien the things they carried")
        XCTAssertEqual(normalizeForSearch("Jean-Paul Sartre"), "jean-paul sartre")
        XCTAssertEqual(normalizeForSearch("AT&T Vol. 2"), "at&t vol. 2")
    }

    func testDoesNotDeleteAllWhitespaceOrPunctuation() {
        // Anti-pattern guard: normalization must not collapse a
        // multi-word title into one run-on token.
        let normalized = normalizeForSearch("Gone Girl")
        XCTAssertTrue(normalized.contains(" "), "expected whitespace preserved, got \(normalized)")
        XCTAssertEqual(searchTokens(normalized).count, 2)
    }

    func testEmptyStringNormalizesToEmpty() {
        XCTAssertEqual(normalizeForSearch(""), "")
    }
}
