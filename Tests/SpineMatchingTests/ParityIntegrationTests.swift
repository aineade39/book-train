import Foundation
import XCTest

@testable import SpineMatching

/// Opt-in integration test -- diffs `Sources/SpineMatching/FuzzyMatch.swift`'s
/// `ratio`/`tokenSortRatio`/`tokenSetRatio` against RapidFuzz, and
/// `Normalization.swift`'s `normalizeForSearch` against an independent
/// Python re-implementation, on a fixed set of fixture strings shared with
/// `tools/spine_matching_parity_fixture.py`. `XCTSkip`s itself when its
/// prerequisite (a Python venv with `rapidfuzz`) isn't present, so `swift
/// test` stays fully offline and dependency-free by default. Run explicitly
/// with `swift test --filter SpineMatchingTests.ParityIntegrationTests`.
final class ParityIntegrationTests: XCTestCase {

    private var repoRoot: URL {
        // Tests/SpineMatchingTests/ParityIntegrationTests.swift -> repo root.
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testFuzzyScorersAndNormalizationMatchPythonReferenceOnSharedFixture() throws {
        let pythonURL = repoRoot.appendingPathComponent(".venv/bin/python3")
        let scriptURL = repoRoot.appendingPathComponent("tools/spine_matching_parity_fixture.py")
        guard FileManager.default.fileExists(atPath: pythonURL.path) else {
            throw XCTSkip("no .venv/bin/python3 -- run tools/build_spines_dataset.py's venv setup first")
        }
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw XCTSkip("missing tools/spine_matching_parity_fixture.py")
        }

        let process = Process()
        process.executableURL = pythonURL
        process.arguments = [scriptURL.path]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let msg = String(data: errData, encoding: .utf8) ?? "<no stderr>"
            throw XCTSkip("Python fixture script failed (missing rapidfuzz in .venv? `pip install rapidfuzz`): \(msg)")
        }

        let payload = try JSONDecoder().decode(PyParityPayload.self, from: outData)

        for pair in payload.fuzzy {
            XCTAssertEqual(ratio(pair.a, pair.b), pair.ratio, accuracy: 0.01, "ratio diverged from rapidfuzz.fuzz.ratio(\"\(pair.a)\", \"\(pair.b)\")")
            XCTAssertEqual(tokenSortRatio(pair.a, pair.b), pair.tokenSortRatio, accuracy: 0.01, "tokenSortRatio diverged from rapidfuzz.fuzz.token_sort_ratio(\"\(pair.a)\", \"\(pair.b)\")")
            XCTAssertEqual(tokenSetRatio(pair.a, pair.b), pair.tokenSetRatio, accuracy: 0.01, "tokenSetRatio diverged from rapidfuzz.fuzz.token_set_ratio(\"\(pair.a)\", \"\(pair.b)\")")
        }

        for pair in payload.normalization {
            XCTAssertEqual(normalizeForSearch(pair.input), pair.normalized, "normalizeForSearch diverged from Python reference for input \"\(pair.input)\"")
        }
    }
}

// MARK: - JSON payload shape (mirrors tools/spine_matching_parity_fixture.py)

private struct PyParityFuzzyPair: Decodable {
    let a: String
    let b: String
    let ratio: Double
    let tokenSortRatio: Double
    let tokenSetRatio: Double

    enum CodingKeys: String, CodingKey {
        case a, b, ratio
        case tokenSortRatio = "token_sort_ratio"
        case tokenSetRatio = "token_set_ratio"
    }
}

private struct PyParityNormalizationPair: Decodable {
    let input: String
    let normalized: String
}

private struct PyParityPayload: Decodable {
    let fuzzy: [PyParityFuzzyPair]
    let normalization: [PyParityNormalizationPair]
}
