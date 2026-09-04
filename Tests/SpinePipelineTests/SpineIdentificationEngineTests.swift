import AppKit
import CoreGraphics
import XCTest

@testable import SpineCatalog
@testable import SpineCore
@testable import SpinePerception
@testable import SpinePipeline

/// Covers the locked "Book ID OCR gains" plan §D/§F barcode-first capture
/// short circuit, end to end through `SpineIdentificationEngine.run` --
/// the same entry point both `spine-id` and the app's
/// `SpineIdentificationPipeline` call (§H "CLI and app call the same
/// code"). Uses a real `SpineDetector` (skipping if this machine has no
/// local production Core ML model, mirroring
/// `SpineDetectorCompiledModelInitTests`).
///
/// The capture-gate scenarios below use flat/solid synthetic fixtures,
/// which fail `CaptureQualityGate` *and* contain no spine-shaped content
/// for the real detector to find -- so `result.spines.isEmpty` in those
/// tests is a property of the fixture's content (there's nothing there),
/// not of the gate blocking detection. `CaptureQualityGate` is advisory
/// only (`capturePassed` is surfaced for the UI/telemetry, never used to
/// skip detect/OCR/match) -- see
/// `testAdvisoryCaptureGateStillDetectsSpinesOnASharpButDarkFrame` below
/// for the regression test proving detection still runs on a real,
/// content-bearing frame that fails the gate.
final class SpineIdentificationEngineTests: XCTestCase {
    private struct FakeBarcodeReader: BarcodeReader {
        let results: [DetectedBarcode]
        func detectBarcodes(in image: CGImage) throws -> [DetectedBarcode] { results }
    }

    /// A real, valid ISBN-13 (Dune's) -- checksum-valid so `ISBN13(rawPayload:)`
    /// accepts it.
    private let duneISBN13 = "9780441013593"

    private func makeEngine(barcodeReader: BarcodeReader) throws -> SpineIdentificationEngine {
        let modelURL = defaultModelURL()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let detector = try SpineDetector(modelURL: modelURL)
        return SpineIdentificationEngine(detector: detector, barcodeReader: barcodeReader, useFM: false)
    }

    func testUniqueISBNShortCircuitsToAutoAcceptedEvenWhenCaptureGateWouldFail() async throws {
        let catalog = try BookCatalog.inMemory()
        try catalog.insert(title: "Dune", author: "Frank Herbert", workKey: "/works/OL1W")
        try catalog.insertISBNs([(isbn13: duneISBN13, workKey: "/works/OL1W")])

        let barcodeReader = FakeBarcodeReader(results: [DetectedBarcode(payload: duneISBN13, symbology: "EAN13")])
        let engine = try makeEngine(barcodeReader: barcodeReader)

        // Flat gray frame fails `CaptureQualityGate`'s sharpness check --
        // the short circuit must still fire "even if capture gate would
        // fail" (§F).
        let blurryFrame = makeSolidCGImage(width: 200, height: 300)
        let result = try await engine.run(on: blurryFrame, catalog: catalog)

        XCTAssertTrue(result.capturePassed, "the short circuit reports capture as passed -- the gate was never consulted")
        XCTAssertEqual(result.isbnBarcodes, [duneISBN13])
        XCTAssertEqual(result.spines.count, 1)
        guard case .autoAccepted(let title, let author, _, _) = result.spines[0].decision else {
            return XCTFail("expected .autoAccepted, got \(result.spines[0].decision)")
        }
        XCTAssertEqual(title, "Dune")
        XCTAssertEqual(author, "Frank Herbert")
        XCTAssertEqual(result.spines[0].source, .barcode)
    }

    func testMultiWorkISBNDoesNotShortCircuitAndFallsThroughToNormalGate() async throws {
        let catalog = try BookCatalog.inMemory()
        try catalog.insert(title: "Book A", author: "Author A", workKey: "/works/OLA")
        try catalog.insert(title: "Book B", author: "Author B", workKey: "/works/OLB")
        try catalog.insertISBNs([
            (isbn13: duneISBN13, workKey: "/works/OLA"),
            (isbn13: duneISBN13, workKey: "/works/OLB"),
        ])

        let barcodeReader = FakeBarcodeReader(results: [DetectedBarcode(payload: duneISBN13, symbology: "EAN13")])
        let engine = try makeEngine(barcodeReader: barcodeReader)

        let blurryFrame = makeSolidCGImage(width: 200, height: 300)
        let result = try await engine.run(on: blurryFrame, catalog: catalog)

        // A multi-work ISBN isn't a "unique work" hit, so the normal
        // (advisory) capture gate applies -- the flat/blurry fixture still
        // fails it -- but detect now always runs regardless; this flat,
        // content-free fixture just has no spine-shaped content for the
        // real detector to find, which is a different thing from the gate
        // blocking detection (see class doc comment).
        XCTAssertFalse(result.capturePassed)
        XCTAssertEqual(result.isbnBarcodes, [duneISBN13])
        XCTAssertTrue(result.spines.isEmpty, "a flat fixture has no spine-shaped content, so there's nothing to associate the barcode with")
    }

    func testInvalidChecksumBarcodeIsDiscardedSilently() async throws {
        let catalog = try BookCatalog.inMemory()
        try catalog.insert(title: "Dune", author: "Frank Herbert", workKey: "/works/OL1W")
        try catalog.insertISBNs([(isbn13: duneISBN13, workKey: "/works/OL1W")])

        // Same shape as a real ISBN-13 (13 digits, 978 prefix) but with a
        // corrupted check digit.
        let badISBN = "9780441013599"
        let barcodeReader = FakeBarcodeReader(results: [DetectedBarcode(payload: badISBN, symbology: "EAN13")])
        let engine = try makeEngine(barcodeReader: barcodeReader)

        let blurryFrame = makeSolidCGImage(width: 200, height: 300)
        let result = try await engine.run(on: blurryFrame, catalog: catalog)

        XCTAssertFalse(result.capturePassed)
        XCTAssertTrue(result.isbnBarcodes.isEmpty, "an invalid-checksum barcode must be discarded with no banner (§D)")
    }

    func testNoBarcodesFallsThroughToNormalGateUnaffected() async throws {
        let catalog = try BookCatalog.inMemory()
        let engine = try makeEngine(barcodeReader: FakeBarcodeReader(results: []))

        let blurryFrame = makeSolidCGImage(width: 200, height: 300)
        let result = try await engine.run(on: blurryFrame, catalog: catalog)

        XCTAssertFalse(result.capturePassed)
        XCTAssertTrue(result.isbnBarcodes.isEmpty)
        // Empty because the flat fixture has no spine-shaped content, not
        // because the (advisory-only) gate blocked detection.
        XCTAssertTrue(result.spines.isEmpty)
    }

    /// Regression for a real production incident (see the "Book ID OCR
    /// gains" capture-gate post-mortem): two genuine bookshelf photos
    /// scored sharpness > 0.98 but exposure ~0.06-0.09 (a dim-but-legible
    /// room, well under `CaptureQualityGate.default.minExposure == 0.12`)
    /// and were silently reduced to zero spines by a hard veto in `run()`,
    /// even though the identical frames produced dozens of correctly
    /// matched spines once detect/OCR/match actually ran. Reproduces that
    /// exact regime -- comfortably sharp, comfortably under the exposure
    /// floor, **not** a blurry/blank frame -- by darkening a real,
    /// content-bearing fixture photo, and asserts detection still runs.
    func testAdvisoryCaptureGateStillDetectsSpinesOnASharpButDarkFrame() async throws {
        let modelURL = defaultModelURL()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path),
              let nsImage = NSImage(contentsOf: imageURL),
              let brightImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }

        let gate = CaptureQualityGate.default
        XCTAssertTrue(gate.passes(brightImage), "sanity: the un-darkened fixture must pass the gate on its own")

        // factor 0.10 empirically reproduces bedroom1's real incident
        // numbers almost exactly (exposure ~0.089) while leaving sharpness
        // comfortably above the floor -- the same "sharp, just dark"
        // combination, not a blur/glare failure.
        let darkImage = darkenedCGImage(brightImage, factor: 0.10)
        let darkScore = gate.score(darkImage)
        XCTAssertLessThan(darkScore.exposure, gate.minExposure, "darkened fixture should fail the exposure floor, matching the real incident")
        XCTAssertGreaterThanOrEqual(
            darkScore.sharpness, gate.minSharpness,
            "darkening must not also fail sharpness -- this must reproduce the real incident's regime (sharp, just dark), not a blurry frame"
        )
        XCTAssertFalse(gate.passes(darkImage))

        let detector = try SpineDetector(modelURL: modelURL)
        let engine = SpineIdentificationEngine(detector: detector, useFM: false)
        let result = try await engine.run(on: darkImage, catalog: nil)

        XCTAssertFalse(result.capturePassed, "advisory gate still reports the frame as marginal -- that reporting itself isn't the bug")
        XCTAssertFalse(
            result.spines.isEmpty,
            "the advisory gate must not block detection: a sharp-but-dark frame with real spine content must still produce spines"
        )
    }
}
