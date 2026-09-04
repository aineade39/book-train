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
    }

    func testAmpersandFoldsToAnd() {
        // 2j-5c regression (see tools/catalog/ol_common.py's Python port for
        // the full rationale): GR/OL disagree on "&" vs "and" in the same
        // title in at least one observed pair.
        XCTAssertEqual(
            normalizeForSearch("The Serpent & the Wings of Night"),
            normalizeForSearch("The Serpent and the Wings of Night")
        )
        XCTAssertEqual(normalizeForSearch("AT&T"), "atandt")
    }

    func testPeriodIsDroppedLikeSlash() {
        // 2j-5b regression: GR/OL disagree on "." vs "/" as a date-title
        // separator ("11.22.63" vs "11/22/63").
        XCTAssertEqual(normalizeForSearch("11.22.63"), normalizeForSearch("11/22/63"))
        XCTAssertEqual(normalizeForSearch("J.R.R. Tolkien"), "jrr tolkien")
    }

    func testInvisibleFormatCharactersAreDropped() {
        // 2j-5a regression: a real Goodreads title, "The \u{200B}Crown of
        // Gilded Bones", has a stray zero-width space right after the real
        // space before "Crown" -- invisible on screen but a distinct
        // scalar, so it silently broke the OL title match.
        XCTAssertEqual(
            normalizeForSearch("The \u{200B}Crown of Gilded Bones"),
            normalizeForSearch("The Crown of Gilded Bones")
        )
        XCTAssertEqual(normalizeForSearch("Dune\u{200C}Messiah"), "dunemessiah")
        XCTAssertEqual(normalizeForSearch("Dune\u{200D}Messiah"), "dunemessiah")
        XCTAssertEqual(normalizeForSearch("\u{FEFF}Dune Messiah"), "dune messiah")
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

    func testCurlyApostropheFoldsToStraight() {
        // Stage 2i regression (see tools/catalog/ol_common.py's Python port
        // for the full rationale): curly and straight apostrophes must
        // normalize identically, or the same word silently produces two
        // different search keys depending on which glyph the source used.
        XCTAssertEqual(normalizeForSearch("Assassin\u{2019}s Blade"), normalizeForSearch("Assassin's Blade"))
        XCTAssertEqual(normalizeForSearch("Assassin\u{2019}s Blade"), "assassin's blade")
    }
}
