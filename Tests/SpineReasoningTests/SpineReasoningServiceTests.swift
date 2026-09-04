import FoundationModels
import XCTest

@testable import SpineReasoning

/// Opt-in integration test against the *real* on-device Foundation Model
/// -- `XCTSkip`s itself (matching the pattern in
/// `Tests/SpineCoreTests/ParityIntegrationTests.swift` etc.) when this
/// Mac/OS/build doesn't actually have the model available, since that's
/// exactly the condition `docs/BOOK_ID_IOS_PIPELINE.md`'s "Vision-only
/// fallback" is meant to handle gracefully rather than something this
/// suite should hard-fail on.
final class SpineReasoningServiceTests: XCTestCase {
    func testIsAvailableMatchesSystemLanguageModelAvailability() throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("FoundationModels requires iOS/macOS 26+")
        }
        let service = SpineReasoningService()
        XCTAssertEqual(service.isAvailable, SystemLanguageModel.default.availability == .available)
    }

    func testExtractReturnsNilWhenModelIsUnavailable() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("FoundationModels requires iOS/macOS 26+")
        }
        guard SystemLanguageModel.default.availability != .available else {
            throw XCTSkip("model is available on this machine -- see testExtractCleansUpNoisyOCRTextWhenModelIsAvailable")
        }
        let service = SpineReasoningService()
        let result = await service.extract(from: "some noisy ocr text")
        XCTAssertNil(result)
    }

    func testExtractCleansUpNoisyOCRTextWhenModelIsAvailable() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("FoundationModels requires iOS/macOS 26+")
        }
        guard SystemLanguageModel.default.availability == .available else {
            throw XCTSkip("no on-device Foundation Model available on this Mac/OS/build")
        }
        let service = SpineReasoningService()
        // Deliberately noisy: merged words, stray punctuation, a rotated-OCR-style
        // run-on -- the kind of text the quality gate would call "marginal".
        let noisy = "THE-GRE4T GATSBY f.scott.fitzgerald"
        guard let extraction = await service.extract(from: noisy) else {
            XCTFail("expected an extraction when the model reports .available")
            return
        }
        XCTAssertFalse(extraction.title.isEmpty)
        XCTAssertTrue(
            extraction.title.localizedCaseInsensitiveContains("gatsby"),
            "expected FM to recover \"Gatsby\" from the noisy OCR text, got: \(extraction.title)"
        )
    }

    /// Regression test for a real hang observed in this repo: running
    /// several FM sessions back-to-back across processes (multiple
    /// `swift test` xctest bundles + a CLI subprocess all using
    /// `SpineReasoningService` around the same time) left one
    /// `LanguageModelSession.respond` call running far past what should
    /// be a multi-second budget -- `extract`'s `timeout` parameter exists
    /// specifically so a caller is never stuck waiting on that, even if
    /// the underlying call itself doesn't return promptly.
    func testExtractNeverWaitsMeaningfullyLongerThanItsTimeout() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("FoundationModels requires iOS/macOS 26+")
        }
        guard SystemLanguageModel.default.availability == .available else {
            throw XCTSkip("no on-device Foundation Model available on this Mac/OS/build")
        }
        let service = SpineReasoningService()
        let start = Date()
        // Real model, real call -- but a timeout far shorter than any
        // observed successful response (~1-2s), so this exercises the
        // actual race, not just the unavailable-model early-return path.
        _ = await service.extract(from: "some ocr text", timeout: .milliseconds(1))
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 5, "extract(timeout:) should bound wall-clock time close to the requested timeout, took \(elapsed)s")
    }
}
