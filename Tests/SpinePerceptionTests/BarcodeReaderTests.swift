import CoreGraphics
import XCTest

@testable import SpinePerception

private struct FakeBarcodeReader: BarcodeReader {
    let results: [DetectedBarcode]
    func detectBarcodes(in image: CGImage) throws -> [DetectedBarcode] { results }
}

final class BarcodeReaderTests: XCTestCase {

    // MARK: - ISBN-shaped payload detection

    func testEAN13WithBookland978PrefixLooksLikeISBN() {
        XCTAssertTrue(DetectedBarcode(payload: "9780441013593", symbology: "EAN13").looksLikeISBN)
    }

    func testEAN13WithBookland979PrefixLooksLikeISBN() {
        XCTAssertTrue(DetectedBarcode(payload: "9791234567896", symbology: "EAN13").looksLikeISBN)
    }

    func testISBN10LooksLikeISBN() {
        XCTAssertTrue(DetectedBarcode(payload: "0441013597", symbology: "Code128").looksLikeISBN)
    }

    func testNonBooklandEAN13DoesNotLookLikeISBN() {
        // A generic grocery EAN-13 (arbitrary, doesn't start 978/979).
        XCTAssertFalse(DetectedBarcode(payload: "0012345678905", symbology: "EAN13").looksLikeISBN)
    }

    func testArbitraryQRPayloadDoesNotLookLikeISBN() {
        XCTAssertFalse(DetectedBarcode(payload: "https://example.com", symbology: "QR").looksLikeISBN)
    }

    func testISBNWithHyphensStillRecognized() {
        // Vision/real-world payloads are typically bare digits, but be
        // lenient about incidental formatting characters.
        XCTAssertTrue(DetectedBarcode(payload: "978-0-441-01359-3", symbology: "EAN13").looksLikeISBN)
    }

    // MARK: - Protocol injection (no live Vision dependency)

    func testFakeReaderRoundTrips() throws {
        let fake = FakeBarcodeReader(results: [DetectedBarcode(payload: "9780441013593", symbology: "EAN13")])
        let results = try fake.detectBarcodes(in: makeSolidCGImage(width: 10, height: 10))
        XCTAssertEqual(results.count, 1)
        XCTAssertTrue(results[0].looksLikeISBN)
    }
}
