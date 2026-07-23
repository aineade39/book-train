import Foundation
import XCTest

/// Opt-in integration/golden test for the `spine-read` and `spine-id`
/// end-to-end macOS CLIs (Sources/spine-read, Sources/spine-id) against the
/// local production Core ML model + the repo's real fixture shelf photo --
/// the same photo/model gating `testLayoutCropsCLIPlanOnlyAgainstRealModelAndPhoto`
/// above uses. Deliberately avoids asserting exact OCR text for most
/// spines (Vision's output can shift slightly across OS/framework
/// versions, same reasoning as that test); only asserts structural shape
/// plus an end-to-end match against a tiny fixture catalog seeded with a
/// couple of the photo's most legible spine titles.
final class SpineIdCLIIntegrationTests: XCTestCase {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    // Seeded from a real `spine-read` run against bookcase.books.spines.png:
    // both titles OCR very cleanly (quality >= 0.97) on that photo.
    private static let fixtureCSV = """
    title,author,isbn
    Moon Utah,Moon Travel Guides,9781640492345
    Day Hiking Mount Rainier,Dan A. Nelson,9781594850852
    """

    func testSpineReadJSONShapeAgainstRealModelAndPhoto() throws {
        guard let spineReadURL = Self.executableURL(named: "spine-read") else {
            throw XCTSkip("could not locate the built spine-read executable next to the test bundle")
        }
        let modelURL = defaultModelURLForTests()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("spine-read-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let jsonURL = tmpDir.appendingPathComponent("spine-read.json")

        let result = try Self.run(spineReadURL, args: [imageURL.path, "--json", jsonURL.path])
        XCTAssertEqual(result.exitCode, 0, "spine-read should exit 0; stderr: \(result.stderr)")
        let payload = try JSONDecoder().decode(SpineReadPayloadJSON.self, from: Data(contentsOf: jsonURL))

        XCTAssertGreaterThan(payload.spines.count, 20, "expected many spine detections on a dense shelf photo")
        let passCount = payload.spines.filter(\.passedQualityGate).count
        XCTAssertGreaterThan(passCount, payload.spines.count / 2, "expected most spines to pass the OCR quality gate on a clear photo")
        for spine in payload.spines where spine.passedQualityGate {
            XCTAssertFalse(spine.assembledText.isEmpty, "a passing spine should have non-empty assembled text")
            XCTAssertGreaterThan(spine.qualityScore, 0)
        }
    }

    func testSpineIdEndToEndAutoAcceptsCleanOCRAgainstFixtureCatalog() throws {
        guard let spineIdURL = Self.executableURL(named: "spine-id") else {
            throw XCTSkip("could not locate the built spine-id executable next to the test bundle")
        }
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }
        let modelURL = defaultModelURLForTests()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("spine-id-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let csvURL = tmpDir.appendingPathComponent("catalog.csv")
        try Self.fixtureCSV.write(to: csvURL, atomically: true, encoding: .utf8)
        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [csvURL.path, "--db", dbURL.path])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build should exit 0; stderr: \(buildResult.stderr)")

        let jsonURL = tmpDir.appendingPathComponent("spine-id.json")
        let result = try Self.run(spineIdURL, args: [imageURL.path, "--db", dbURL.path, "--json", jsonURL.path])
        XCTAssertEqual(result.exitCode, 0, "spine-id should exit 0; stderr: \(result.stderr)")
        let payload = try JSONDecoder().decode(SpineIdPayloadJSON.self, from: Data(contentsOf: jsonURL))

        let autoAccepted = payload.spines.filter { $0.decision == "auto-accept" }
        let matchedTitles = Set(autoAccepted.compactMap(\.matchedTitle))
        XCTAssertTrue(
            matchedTitles.contains("Moon Utah") || matchedTitles.contains("Day Hiking Mount Rainier"),
            "expected at least one clean-OCR spine to auto-accept against the fixture catalog; got decisions: \(payload.spines.map(\.decision))"
        )
    }

    /// `--fm` opt-in smoke test: exercises the real on-device Foundation
    /// Model when available (see `SpineReasoningService`), and otherwise
    /// asserts the flag is a harmless no-op -- either way the two clean
    /// titles from `fixtureCSV` must still auto-accept, matching
    /// `testSpineIdEndToEndAutoAcceptsCleanOCRAgainstFixtureCatalog`'s
    /// baseline, since `--fm` must never change the outcome for spines it
    /// doesn't escalate.
    func testSpineIdWithFMFlagStillAutoAcceptsCleanOCRAndOnlyUsesValidSources() throws {
        guard let spineIdURL = Self.executableURL(named: "spine-id") else {
            throw XCTSkip("could not locate the built spine-id executable next to the test bundle")
        }
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }
        let modelURL = defaultModelURLForTests()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("spine-id-fm-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let csvURL = tmpDir.appendingPathComponent("catalog.csv")
        try Self.fixtureCSV.write(to: csvURL, atomically: true, encoding: .utf8)
        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [csvURL.path, "--db", dbURL.path])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build should exit 0; stderr: \(buildResult.stderr)")

        let jsonURL = tmpDir.appendingPathComponent("spine-id.json")
        let result = try Self.run(spineIdURL, args: [imageURL.path, "--db", dbURL.path, "--fm", "--json", jsonURL.path])
        XCTAssertEqual(result.exitCode, 0, "spine-id --fm should exit 0; stderr: \(result.stderr)")
        let payload = try JSONDecoder().decode(SpineIdPayloadJSON.self, from: Data(contentsOf: jsonURL))

        for spine in payload.spines {
            XCTAssertTrue(
                spine.source == "ocr" || spine.source == "fm-assisted",
                "unexpected source \"\(spine.source)\" for spine \(spine.matchedTitle ?? "-")"
            )
        }

        let autoAccepted = payload.spines.filter { $0.decision == "auto-accept" }
        let matchedTitles = Set(autoAccepted.compactMap(\.matchedTitle))
        XCTAssertTrue(
            matchedTitles.contains("Moon Utah") || matchedTitles.contains("Day Hiking Mount Rainier"),
            "--fm must not regress a clean-OCR auto-accept; got decisions: \(payload.spines.map(\.decision))"
        )
    }

    // MARK: - Helpers (mirrors Tests/SpineCoreTests/ParityIntegrationTests.swift)

    private static func executableURL(named name: String) -> URL? {
        let candidate = Bundle(for: SpineIdCLIIntegrationTests.self).bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    private struct RunResult { let exitCode: Int32; let stdout: String; let stderr: String }

    private static func run(_ executable: URL, args: [String]) throws -> RunResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = args
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return RunResult(
            exitCode: process.terminationStatus,
            stdout: String(data: outData, encoding: .utf8) ?? "",
            stderr: String(data: errData, encoding: .utf8) ?? ""
        )
    }
}

/// Standalone copy of `SpineCore.defaultModelURL()`'s resolution logic --
/// this test target doesn't depend on `SpineCore`, and the two CLIs under
/// test already exercise the real implementation.
private func defaultModelURLForTests() -> URL {
    let root = ProcessInfo.processInfo.environment["BOOK_SPINES_DATA"]
        ?? NSString(string: "~/ml/book-spines").expandingTildeInPath
    let alias = "\(root)/models/production/SpineDetectorOBB.mlpackage"
    return URL(fileURLWithPath: alias)
}

// MARK: - JSON payload shapes (mirror Sources/spine-read/main.swift and Sources/spine-id/main.swift)

private struct SpineReadSpineJSON: Decodable {
    let assembledText: String
    let qualityScore: Double
    let passedQualityGate: Bool
}

private struct SpineReadPayloadJSON: Decodable {
    let spines: [SpineReadSpineJSON]
}

private struct SpineIdSpineJSON: Decodable {
    let decision: String
    let matchedTitle: String?
    let source: String
}

private struct SpineIdPayloadJSON: Decodable {
    let spines: [SpineIdSpineJSON]
}
