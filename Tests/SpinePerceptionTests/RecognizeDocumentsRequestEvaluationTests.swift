import AppKit
import CoreGraphics
import Vision
import XCTest

import SpineCore
import SpinePerception

/// Optional evaluation per docs/BOOK_ID_IOS_PIPELINE.md §Delivery
/// sequencing step 4: "`RecognizeDocumentsRequest` evaluation for
/// multi-line / vertical reading order where platform support allows."
///
/// Runs the real Vision `RecognizeDocumentsRequest` (macOS/iOS 26+)
/// against real upright-warped spine crops from the repo's fixture photo,
/// alongside the existing `OCROrientationRouter`, and prints which one
/// recovers usable text more often -- evidence for whether production
/// spine reads should route through it instead of/in addition to
/// `VNRecognizeTextRequest`, rather than assuming either way.
///
/// `RecognizeDocumentsRequest` is built for laid-out documents (title +
/// paragraphs + tables + lists), not a single run of (often
/// vertical/rotated) text on a spine, and unlike `OCROrientationRouter`
/// it has no built-in "try a few orientations and pick the best" pass --
/// callers must already know the orientation. `XCTSkip`s itself (same
/// pattern as `Tests/SpineCoreTests/SpineIdCLIIntegrationTests.swift`)
/// when its prerequisites (OS version, production model, fixture photo)
/// are missing.
final class RecognizeDocumentsRequestEvaluationTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testRecognizeDocumentsRequestAgainstOCROrientationRouterOnRealSpineCrops() async throws {
        guard #available(macOS 26.0, iOS 26.0, *) else {
            throw XCTSkip("RecognizeDocumentsRequest requires macOS/iOS 26+")
        }
        let modelURL = Self.defaultModelURLForTests()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }
        guard let nsImage = NSImage(contentsOf: imageURL),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw XCTSkip("could not decode fixture photo")
        }

        let detector = try SpineDetector(modelURL: modelURL)
        let options = DetectionOptions(confidenceThreshold: 0.15, iouThreshold: 0.45, maxDetections: 500)
        let inferenceResult = try detector.predict(cgImage, options: options)
        let detections = inferenceResult.alreadyNMSed
            ? inferenceResult.detections
            : nmsRotated(inferenceResult.detections, iouThreshold: options.iouThreshold, maxDetections: options.maxDetections)
        XCTAssertGreaterThan(detections.count, 10, "expected a dense shelf photo")

        let router = OCROrientationRouter(recognizer: VisionTextRecognizer())
        var routerRecoveredCount = 0
        var documentsRecoveredCount = 0
        var sampleCount = 0

        // Sample every 4th detection rather than all ~90 --
        // RecognizeDocumentsRequest's per-call latency is much higher than
        // VNRecognizeTextRequest's (it's a full document-layout analysis
        // pass), and this test only needs a representative comparison,
        // not exhaustive coverage.
        let sampled = detections.enumerated().filter { $0.offset % 4 == 0 }.map(\.element)
        for det in sampled {
            guard let crop = uprightWarp(of: det, in: cgImage) else { continue }
            sampleCount += 1

            let routerText = (try? router.recognize(crop: crop, detection: det))?.assembledText ?? ""
            if !routerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                routerRecoveredCount += 1
            }

            let request = RecognizeDocumentsRequest()
            // Match the router's primary aspect-guided orientation guess
            // (see OCROrientationRouter.swift): tall crops are read
            // bottom-to-top on most spines, so rotate 90 degrees before
            // asking Vision to read left-to-right.
            let orientation: CGImagePropertyOrientation = crop.height >= crop.width ? .right : .up
            let observations = (try? await request.perform(on: crop, orientation: orientation)) ?? []
            let documentsText = observations.first?.document.text.transcript ?? ""
            if !documentsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                documentsRecoveredCount += 1
            }
        }

        XCTAssertGreaterThan(sampleCount, 0)
        // Deliberately not a hard pass/fail on the *comparison* itself --
        // this task is framed as "optionally evaluate", not "must beat"
        // (unlike the rotation-sweep acceptance rule in AGENTS.md, which
        // *is* a hard gate for detector changes) -- so the finding is
        // surfaced for a human to read rather than asserted on.
        print(
            "RecognizeDocumentsRequest evaluation: \(sampleCount) crops sampled, "
                + "OCROrientationRouter recovered text on \(routerRecoveredCount), "
                + "RecognizeDocumentsRequest recovered text on \(documentsRecoveredCount)."
        )
    }

    private static func defaultModelURLForTests() -> URL {
        let root = ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
            ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
        return URL(fileURLWithPath: "\(root)/models/production/SpineDetectorOBB.mlpackage")
    }
}
