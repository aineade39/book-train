import Foundation
import GRDB
import SpineCatalog
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

    func testOLIntermediateBuildThenBookMatch() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }
        guard let bookMatchURL = Self.executableURL(named: "book-match") else {
            throw XCTSkip("could not locate the built book-match executable next to the test bundle")
        }

        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixture = repoRoot.appendingPathComponent("Tests/fixtures/ol-mini")
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("ol-mini fixture missing")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-ol-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let intermediate = tmpDir.appendingPathComponent("intermediate")
        try FileManager.default.createDirectory(at: intermediate, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            repoRoot.appendingPathComponent("tools/catalog/process_ol.py").path,
            "--raw-dir", fixture.path,
            "--editions", fixture.appendingPathComponent("editions.jsonl").path,
            "--works", fixture.appendingPathComponent("works.jsonl").path,
            "--authors", fixture.appendingPathComponent("authors.jsonl").path,
            "--out-dir", intermediate.path,
            "--min-editions", "1",
        ]
        process.currentDirectoryURL = repoRoot
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)

        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [
            "--intermediate", intermediate.path,
            "--output", dbURL.path,
            "--languages", "eng",
            "--min-editions", "1",
        ])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build OL mode stderr: \(buildResult.stderr)")

        let matchResult = try Self.run(bookMatchURL, args: ["project hail mary andy weir", "--db", dbURL.path, "--json"])
        XCTAssertEqual(matchResult.exitCode, 0)
        let payload = try JSONDecoder().decode(BookMatchResultJSON.self, from: Data(matchResult.stdout.utf8))
        XCTAssertEqual(payload.decision, "auto-accept")
        XCTAssertEqual(payload.winner?.title, "Project Hail Mary")

        // --- §D book_isbns: OL isbns.jsonl.gz -> unique-ISBN lookup ---
        let catalog = try BookCatalog(path: dbURL.path)
        let isbnMatches = try catalog.lookupISBN("9780593135204")
        XCTAssertEqual(isbnMatches.count, 1, "9780593135204 should map to exactly one work")
        XCTAssertEqual(isbnMatches.first?.title, "Project Hail Mary")
        XCTAssertTrue(
            try catalog.lookupISBN("0000000000000").isEmpty,
            "an ISBN absent from the fixture should return no matches, not an error"
        )

        // --- §C customWords: authors-only lexicon from the shipped rows ---
        let customWords = try catalog.customWords()
        XCTAssertTrue(customWords.contains("Weir"), "author name words should be in the customWords lexicon")
        XCTAssertTrue(customWords.contains("Herbert"))
        XCTAssertFalse(
            customWords.contains { $0.lowercased() == "project" || $0.lowercased() == "hail" || $0.lowercased() == "mary" },
            "customWords is authors-only -- title words must never appear"
        )
    }

    // MARK: - Build-time match-field dedup (Part B) + D1-D4 build pipeline

    /// Two intermediate works sharing a normalized (title, author) match
    /// field via punctuation (not casing) but different `workKey`s -- the
    /// OL data-quality pattern build-time dedup collapses. `works.jsonl.gz`
    /// is already popularity-ordered by `process_ol.py`'s
    /// `export_intermediate`, so listing the better-ranked one first
    /// mirrors a real intermediate.
    func testBuildFromIntermediateDedupesSharedMatchFieldKeepingBestRanked() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-dedup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let intermediate = tmpDir.appendingPathComponent("intermediate")
        try FileManager.default.createDirectory(at: intermediate, withIntermediateDirectories: true)
        try Self.writeGzippedJSONL(
            [
                #"{"workKey": "/works/OLB", "title": "Dune!", "author": "Frank Herbert", "isbn13": null, "editionCount": 2, "popularityRank": 1, "languages": ["eng"]}"#,
                #"{"workKey": "/works/OLA", "title": "Dune", "author": "Frank Herbert", "isbn13": null, "editionCount": 1, "popularityRank": 2, "languages": ["eng"]}"#,
            ],
            to: intermediate.appendingPathComponent("works.jsonl.gz")
        )

        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [
            "--intermediate", intermediate.path, "--output", dbURL.path, "--min-editions", "1",
        ])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build OL mode stderr: \(buildResult.stderr)")

        let catalog = try BookCatalog(path: dbURL.path)
        XCTAssertEqual(try catalog.countBooks(), 1, "the two colliding works should collapse to one shipped row")
        let rows = try catalog.dbQueue.read { db in try Row.fetchAll(db, sql: "SELECT workKey FROM books") }
        XCTAssertEqual(
            rows.first?["workKey"] as String?, "/works/OLB",
            "the earlier-in-stream (lower/better popularityRank) row should win"
        )
    }

    /// `maxWorks` must count *unique, post-dedup* shipped works, not raw
    /// scanned rows -- a duplicate dropped by the match-field-dedup guard
    /// must not consume a slot that a later, distinct work should get.
    func testBuildFromIntermediateMaxWorksCountsPostDedupUniqueWorks() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-maxworks-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let intermediate = tmpDir.appendingPathComponent("intermediate")
        try FileManager.default.createDirectory(at: intermediate, withIntermediateDirectories: true)
        try Self.writeGzippedJSONL(
            [
                #"{"workKey": "/works/OLB", "title": "Dune!", "author": "Frank Herbert", "isbn13": null, "editionCount": 2, "popularityRank": 1, "languages": ["eng"]}"#,
                #"{"workKey": "/works/OLA", "title": "Dune", "author": "Frank Herbert", "isbn13": null, "editionCount": 1, "popularityRank": 2, "languages": ["eng"]}"#,
                #"{"workKey": "/works/OLC", "title": "Dune Messiah", "author": "Frank Herbert", "isbn13": null, "editionCount": 1, "popularityRank": 3, "languages": ["eng"]}"#,
            ],
            to: intermediate.appendingPathComponent("works.jsonl.gz")
        )

        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [
            "--intermediate", intermediate.path, "--output", dbURL.path, "--min-editions", "1", "--max-works", "2",
        ])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build OL mode stderr: \(buildResult.stderr)")

        let catalog = try BookCatalog(path: dbURL.path)
        let workKeys = Set(try catalog.dbQueue.read { db in try String.fetchAll(db, sql: "SELECT workKey FROM books") })
        XCTAssertEqual(
            workKeys, ["/works/OLB", "/works/OLC"],
            "--max-works 2 should ship the 2 unique post-dedup works (OLB's dup OLA doesn't count against the cap), not stop after 2 raw rows"
        )
    }

    /// D3: `buildFromIntermediate` relaxes durability pragmas for the
    /// bulk-insert phase, then restores them before the build is
    /// considered complete -- verify the *restored* state is actually in
    /// effect post-build (not just that the build completed), and that
    /// the output DB round-trips correctly (readable, correct row count).
    func testBuildFromIntermediateRestoresDurabilityPragmasAfterBulkLoad() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-pragmas-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let intermediate = tmpDir.appendingPathComponent("intermediate")
        try FileManager.default.createDirectory(at: intermediate, withIntermediateDirectories: true)
        try Self.writeGzippedJSONL(
            [#"{"workKey": "/works/OLA", "title": "Dune", "author": "Frank Herbert", "isbn13": null, "editionCount": 1, "popularityRank": 1, "languages": ["eng"]}"#],
            to: intermediate.appendingPathComponent("works.jsonl.gz")
        )

        let dbURL = tmpDir.appendingPathComponent("catalog.sqlite")
        let buildResult = try Self.run(catalogBuildURL, args: [
            "--intermediate", intermediate.path, "--output", dbURL.path, "--min-editions", "1",
        ])
        XCTAssertEqual(buildResult.exitCode, 0, "catalog-build OL mode stderr: \(buildResult.stderr)")

        let catalog = try BookCatalog(path: dbURL.path)
        XCTAssertEqual(try catalog.countBooks(), 1, "output DB should round-trip correctly (readable, correct row count)")

        let (journalMode, synchronous) = try catalog.dbQueue.read { db in
            (
                try String.fetchOne(db, sql: "PRAGMA journal_mode") ?? "",
                try Int.fetchOne(db, sql: "PRAGMA synchronous") ?? -1
            )
        }
        XCTAssertEqual(journalMode.lowercased(), "delete", "durable journal_mode should be restored, not left at the bulk-load MEMORY setting")
        XCTAssertEqual(synchronous, 2, "durable synchronous=FULL (2) should be restored, not left at the bulk-load OFF (0) setting")
    }

    /// D4: the deferred-index/FTS-rebuild path (`buildFromIntermediate`)
    /// must produce a schema identical to the normal always-indexed
    /// migrator path (CSV mode, Mode A) -- same secondary indexes, same
    /// FTS sync triggers, same FTS query results on equivalent content.
    func testBuildFromIntermediateSchemaMatchesNormalMigratorPath() throws {
        guard let catalogBuildURL = Self.executableURL(named: "catalog-build") else {
            throw XCTSkip("could not locate the built catalog-build executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-schema-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Normal migrator path: CSV mode never touches D3/D4 at all.
        let csvURL = tmpDir.appendingPathComponent("catalog.csv")
        try "title,author,isbn\nProject Hail Mary,Andy Weir,9780593135204\n".write(to: csvURL, atomically: true, encoding: .utf8)
        let normalDBURL = tmpDir.appendingPathComponent("normal.sqlite")
        let normalResult = try Self.run(catalogBuildURL, args: [csvURL.path, "--db", normalDBURL.path])
        XCTAssertEqual(normalResult.exitCode, 0, "csv-mode build stderr: \(normalResult.stderr)")

        // Deferred-index/FTS-rebuild path: buildFromIntermediate.
        let intermediate = tmpDir.appendingPathComponent("intermediate")
        try FileManager.default.createDirectory(at: intermediate, withIntermediateDirectories: true)
        try Self.writeGzippedJSONL(
            [#"{"workKey": "/works/OL3W", "title": "Project Hail Mary", "author": "Andy Weir", "isbn13": "9780593135204", "editionCount": 1, "popularityRank": 1, "languages": ["eng"]}"#],
            to: intermediate.appendingPathComponent("works.jsonl.gz")
        )
        let deferredDBURL = tmpDir.appendingPathComponent("deferred.sqlite")
        let deferredResult = try Self.run(catalogBuildURL, args: [
            "--intermediate", intermediate.path, "--output", deferredDBURL.path, "--min-editions", "1",
        ])
        XCTAssertEqual(deferredResult.exitCode, 0, "intermediate-mode build stderr: \(deferredResult.stderr)")

        func schemaObjects(_ dbURL: URL, type: String) throws -> Set<String> {
            let catalog = try BookCatalog(path: dbURL.path)
            return Set(try catalog.dbQueue.read { db in
                try String.fetchAll(db, sql: "SELECT name FROM sqlite_master WHERE type = ? AND tbl_name = 'books'", arguments: [type])
            })
        }

        XCTAssertEqual(
            try schemaObjects(normalDBURL, type: "index"), try schemaObjects(deferredDBURL, type: "index"),
            "the deferred-index path must produce the same secondary indexes as the normal migrator path"
        )
        XCTAssertEqual(
            try schemaObjects(normalDBURL, type: "trigger"), try schemaObjects(deferredDBURL, type: "trigger"),
            "the deferred-index path must recreate the same FTS sync triggers as the normal migrator path"
        )

        // Same FTS query results on equivalent content.
        func ftsTitles(_ dbURL: URL) throws -> [String] {
            let catalog = try BookCatalog(path: dbURL.path)
            return try catalog.dbQueue.read { db in
                try String.fetchAll(
                    db,
                    sql: """
                        SELECT books.title FROM books_fts
                        JOIN books ON books.id = books_fts.rowid
                        WHERE books_fts MATCH '"andy"' ORDER BY rank
                        """
                )
            }
        }
        XCTAssertEqual(try ftsTitles(normalDBURL), ["Project Hail Mary"])
        XCTAssertEqual(try ftsTitles(deferredDBURL), ["Project Hail Mary"], "the FTS rebuild after deferred bulk-insert must be queryable identically to the normal path")
    }

    // MARK: - Helpers (mirrors Tests/SpineCoreTests/ParityIntegrationTests.swift)

    /// Gzip-compresses `lines` (newline-joined) to `url` via `/usr/bin/gzip`
    /// -- `CatalogOLBuild.streamGzippedJSONL` shells out to `gunzip -c`, so
    /// intermediate fixtures built inline for a test need to be real gzip
    /// data, not just a `.gz`-named plain-text file.
    private static func writeGzippedJSONL(_ lines: [String], to url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gzip")
        process.arguments = ["-c"]
        let inPipe = Pipe()
        let outPipe = Pipe()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        try process.run()
        inPipe.fileHandleForWriting.write(Data((lines.joined(separator: "\n") + "\n").utf8))
        try inPipe.fileHandleForWriting.close()
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        try outData.write(to: url)
    }

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
