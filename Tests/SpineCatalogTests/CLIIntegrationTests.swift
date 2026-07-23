import Foundation
import XCTest

/// Opt-in integration/golden test for the `catalog-build` and `book-match`
/// macOS CLIs (Sources/catalog-build, Sources/book-match) -- exercises the
/// real compiled executables end to end (CSV parsing, GRDB/FTS5 catalog
/// build, retrieval + fuzzy rerank + accept policy), the same way
/// `Tests/SpineCoreTests/ParityIntegrationTests.swift` exercises
/// `layout-crops`. `XCTSkip`s itself when it can't locate the built
/// executables next to the test bundle (e.g. `swift test` run before `swift
/// build`), so `swift test` stays offline and dependency-free by default.
final class CLIIntegrationTests: XCTestCase {

    // MARK: - Fixture CSV (quoted fields + embedded comma, to exercise the
    // hand-rolled RFC4180-ish parser in Sources/catalog-build/main.swift).

    private static let fixtureCSV = """
    title,author,isbn,workKey
    Dune,Frank Herbert,9780441013593,
    Dune Messiah,Frank Herbert,9780441172696,
    "Project Hail Mary","Andy Weir",9780593135204,
    "Gone Girl, A Novel",Gillian Flynn,9780307588371,gone-girl
    Gone Girl,Gillian Flynn,9780307588388,gone-girl
    """

    func testCatalogBuildThenBookMatchEndToEnd() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }
        guard let bookMatchURL = Self.executableURL(named: "book-match") else {
            throw XCTSkip("could not locate the built book-match executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-cli-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let csvURL = tmpDir.appendingPathComponent("catalog.csv")
        try Self.fixtureCSV.write(to: csvURL, atomically: true, encoding: .utf8)
        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")

        // --- catalog-build ---
        let buildResult = try Self.run(catalogBuildURL, args: [csvURL.path, "--db", dbURL.path])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build should exit 0; stderr: \(buildResult.stderr)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dbURL.path), "expected a catalog.sqlite to be written")
        XCTAssertTrue(buildResult.stderr.contains("inserted 5 rows"), "expected all 5 fixture rows to be inserted; stderr: \(buildResult.stderr)")

        // --- book-match: clean, unambiguous query -> auto-accept ---
        let cleanResult = try Self.run(bookMatchURL, args: ["Project Hail Mary Andy Weir", "--db", dbURL.path, "--json"])
        XCTAssertEqual(cleanResult.exitCode, 0, "book-match should exit 0; stderr: \(cleanResult.stderr)")
        let cleanPayload = try JSONDecoder().decode(BookMatchResultJSON.self, from: Data(cleanResult.stdout.utf8))
        XCTAssertEqual(cleanPayload.decision, "auto-accept")
        XCTAssertEqual(cleanPayload.winner?.title, "Project Hail Mary")
        XCTAssertEqual(cleanPayload.winner?.author, "Andy Weir")

        // --- book-match: quoted-CSV-embedded-comma field round-trips intact ---
        let quotedResult = try Self.run(bookMatchURL, args: ["Gone Girl A Novel Gillian Flynn", "--db", dbURL.path, "--json"])
        XCTAssertEqual(quotedResult.exitCode, 0, "book-match should exit 0; stderr: \(quotedResult.stderr)")
        let quotedPayload = try JSONDecoder().decode(BookMatchResultJSON.self, from: Data(quotedResult.stdout.utf8))
        XCTAssertEqual(quotedPayload.decision, "auto-accept")
        XCTAssertEqual(quotedPayload.winner?.title, "Gone Girl, A Novel", "the embedded comma inside the quoted CSV field should survive parsing intact")

        // Both editions of "Gone Girl" should share the CSV's explicit workKey.
        XCTAssertTrue(quotedPayload.topCandidates.contains { $0.workKey == "gone-girl" })

        // --- book-match: nonsense query -> no-match, not an error ---
        let noMatchResult = try Self.run(bookMatchURL, args: ["Completely Unrelated Nonexistent Title Zzyzx", "--db", dbURL.path, "--json"])
        XCTAssertEqual(noMatchResult.exitCode, 0)
        let noMatchPayload = try JSONDecoder().decode(BookMatchResultJSON.self, from: Data(noMatchResult.stdout.utf8))
        XCTAssertEqual(noMatchPayload.decision, "no-match")
        XCTAssertNil(noMatchPayload.winner)
    }

    // MARK: - Helpers (mirrors Tests/SpineCoreTests/ParityIntegrationTests.swift)

    private static func executableURL(named name: String) -> URL? {
        let candidate = Bundle(for: CLIIntegrationTests.self).bundleURL
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

// MARK: - JSON payload shape (mirrors Sources/book-match/main.swift)

private struct BookMatchCandidateJSON: Decodable {
    let id: Int64
    let title: String
    let author: String
    let isbn: String?
    let workKey: String
    let score: Double
}

private struct BookMatchResultJSON: Decodable {
    let query: String
    let retrievedCount: Int
    let decision: String
    let winner: BookMatchCandidateJSON?
    let topCandidates: [BookMatchCandidateJSON]
}
