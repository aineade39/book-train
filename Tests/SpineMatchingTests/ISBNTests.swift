import XCTest

@testable import SpineMatching

final class ISBNTests: XCTestCase {
    func testValidISBN13PassesThrough() {
        // Dune, Frank Herbert
        let isbn = ISBN13(rawPayload: "9780441013593")
        XCTAssertEqual(isbn?.value, "9780441013593")
    }

    func testSeparatorsAndSpacesAreStripped() {
        let isbn = ISBN13(rawPayload: "978-0-441-01359-3")
        XCTAssertEqual(isbn?.value, "9780441013593")
    }

    func testInvalidISBN13ChecksumIsRejected() {
        XCTAssertNil(ISBN13(rawPayload: "9780441013590"))
    }

    func testValidISBN10IsConvertedToISBN13() {
        // "0441013597" is a valid ISBN-10 for Dune -> canonical 13.
        let isbn = ISBN13(rawPayload: "0441013597")
        XCTAssertEqual(isbn?.value, "9780441013593")
    }

    func testISBN10WithLowercaseXCheckDigitIsAccepted() {
        // "080442957X" is a known-valid ISBN-10 with an X check digit.
        let isbn = ISBN13(rawPayload: "080442957x")
        XCTAssertNotNil(isbn)
        XCTAssertEqual(isbn?.value.count, 13)
    }

    func testInvalidISBN10ChecksumIsRejected() {
        XCTAssertNil(ISBN13(rawPayload: "0441013598"))
    }

    func testGarbagePayloadIsRejected() {
        XCTAssertNil(ISBN13(rawPayload: "not an isbn"))
        XCTAssertNil(ISBN13(rawPayload: "12345"))
        XCTAssertNil(ISBN13(rawPayload: ""))
    }

    func testTooManyOrTooFewDigitsIsRejected() {
        XCTAssertNil(ISBN13(rawPayload: "97804410135931"))
        XCTAssertNil(ISBN13(rawPayload: "978044101359"))
    }
}
