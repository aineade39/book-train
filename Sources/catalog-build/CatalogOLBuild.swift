import Foundation
import GRDB
import SpineCatalog
import SpineMatching

struct CatalogProfileFilters {
    var maxWorks: Int?
    var languages: [String]
    var minEditions: Int
}

struct IntermediateWork: Decodable {
    let workKey: String
    let title: String
    let author: String
    let isbn13: String?
    let editionCount: Int
    let popularityRank: Int
    let languages: [String]
}

struct IntermediateISBN: Decodable {
    let isbn13: String
    let workKey: String
}

enum CatalogOLBuild {
    /// Above this many shipped works, `VACUUM`'s ~2x-disk-space rebuild
    /// (SQLite copies the whole DB into a temp file before replacing the
    /// original) isn't worth the risk for a from-scratch bulk-insert build
    /// -- `ios_en` (250k) and `dev_smoke` (10k) still get vacuumed; `full`
    /// (~39M) does not. See docs/BOOK_CATALOG.md / the "Build full.sqlite"
    /// plan's disk-space notes.
    private static let vacuumRowThreshold = 1_000_000

    /// Bounds peak memory for OL intermediate streaming to O(batch size)
    /// instead of O(corpus size) -- each full batch becomes its own
    /// `bulkInsert`/`insertISBNs` transaction. See the "Build full.sqlite"
    /// plan's OOM postmortem: the previous version decoded the *entire*
    /// ~39M-work intermediate into one array before any filtering ran,
    /// which is what got the build `SIGKILL`ed.
    private static let streamBatchSize = 25_000

    private static let progressInterval = 500_000

    static func buildFromIntermediate(
        intermediate: URL,
        output: URL,
        filters: CatalogProfileFilters,
        log: (String) -> Void
    ) throws {
        let worksFile = intermediate.appendingPathComponent("works.jsonl.gz")
        guard FileManager.default.fileExists(atPath: worksFile.path) else {
            throw NSError(domain: "catalog-build", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(worksFile.path)"])
        }

        if FileManager.default.fileExists(atPath: output.path) {
            try FileManager.default.removeItem(at: output)
        }
        let catalog = try BookCatalog(path: output.path)

        // Pass 1: stream works.jsonl.gz -> filter -> batch-insert. Never
        // holds more than `streamBatchSize` decoded works at once. Also
        // accumulates the two lightweight pieces of state later phases
        // need without holding onto full `IntermediateWork`/`WorkInsert`
        // rows: which workKeys shipped (for scoping the ISBN pass below)
        // and the customWords lexicon's per-word aggregates (via
        // `CustomWordsBuilder.Accumulator`, keyed by word, not by author).
        let langSet = Set(filters.languages.map { $0.lowercased() })
        var shippedWorkKeys: Set<String> = []
        var customWordsAcc = CustomWordsBuilder.Accumulator()
        var batch: [BookCatalog.WorkInsert] = []
        batch.reserveCapacity(streamBatchSize)
        var inserted = 0
        var scanned = 0

        func flushWorks() throws {
            guard !batch.isEmpty else { return }
            inserted += try catalog.bulkInsert(batch)
            batch.removeAll(keepingCapacity: true)
        }

        try streamGzippedJSONL(from: worksFile) { (work: IntermediateWork) in
            scanned += 1
            if scanned % progressInterval == 0 {
                log("catalog-build: scanned \(scanned) works, shipped \(inserted + batch.count)...")
            }
            guard work.editionCount >= filters.minEditions else { return }
            if !langSet.isEmpty {
                let workLangs = Set(work.languages.map { $0.lowercased() })
                if workLangs.isDisjoint(with: langSet) { return }
            }
            // Matches the old "filter everything, then Array(prefix(max))"
            // semantics exactly: works.jsonl.gz is popularity-ordered, so
            // stopping at `max` *shipped* works (in file order) keeps the
            // same N works the old array-then-prefix approach kept.
            let shippedSoFar = inserted + batch.count
            if let max = filters.maxWorks, shippedSoFar >= max { return }

            let row = BookCatalog.WorkInsert(
                workKey: work.workKey, title: work.title, author: work.author, isbn: work.isbn13,
                popularityRank: work.popularityRank, editionCount: work.editionCount
            )
            shippedWorkKeys.insert(row.workKey)
            customWordsAcc.add(author: row.author, popularityRank: row.popularityRank ?? shippedSoFar)
            batch.append(row)
            if batch.count >= streamBatchSize {
                try flushWorks()
            }
        }
        try flushWorks()

        // Pass 2: stream isbns.jsonl.gz -> scope to shippedWorkKeys ->
        // batch-insert. Same unbounded-array problem as works.jsonl.gz
        // (33.8M rows for `full`), fixed the same way.
        let isbnsFile = intermediate.appendingPathComponent("isbns.jsonl.gz")
        var isbnCount = 0
        if FileManager.default.fileExists(atPath: isbnsFile.path) {
            var isbnBatch: [(isbn13: String, workKey: String)] = []
            isbnBatch.reserveCapacity(streamBatchSize)
            func flushISBNs() throws {
                guard !isbnBatch.isEmpty else { return }
                try catalog.insertISBNs(isbnBatch)
                isbnCount += isbnBatch.count
                isbnBatch.removeAll(keepingCapacity: true)
            }
            try streamGzippedJSONL(from: isbnsFile) { (row: IntermediateISBN) in
                guard shippedWorkKeys.contains(row.workKey) else { return }
                // Re-validated/normalized through the single shared
                // `SpineMatching.ISBN13` path rather than trusted verbatim
                // from the Python intermediate (§H: shared match/ISBN APIs).
                guard let isbn13 = ISBN13(rawPayload: row.isbn13) else { return }
                isbnBatch.append((isbn13: isbn13.value, workKey: row.workKey))
                if isbnBatch.count >= streamBatchSize {
                    try flushISBNs()
                }
            }
            try flushISBNs()
        } else {
            log("catalog-build: no isbns.jsonl.gz found at \(isbnsFile.path); book_isbns will be empty")
        }

        try finalizeCatalog(
            catalog, worksInserted: inserted, isbnsInserted: isbnCount,
            customWords: customWordsAcc.finish(), output: output, log: log
        )
    }

    static func buildFromSubset(
        source: URL,
        output: URL,
        filters: CatalogProfileFilters,
        log: (String) -> Void
    ) throws {
        // NOTE: this still fetches the entire source catalog's `books` (and
        // later `book_isbns`) tables into memory in one array -- fine while
        // every caller's source is `ios_en`/`dev_smoke`-scale, but it has
        // the same unbounded-materialization shape that got
        // `buildFromIntermediate` OOM-killed if a future caller ever
        // passes a `full.sqlite`-scale (~39M row) `--subset-from` source.
        // Not fixed here since nothing calls it that way today -- stream
        // via `sourceCatalog.dbQueue.read` + a GRDB cursor instead of
        // `fetchAll` if that changes.
        let sourceCatalog = try BookCatalog(path: source.path)
        let records: [BookRecord] = try sourceCatalog.dbQueue.read { db in
            try BookRecord
                .order(Column("popularityRank").asc)
                .fetchAll(db)
        }
        var inserts: [BookCatalog.WorkInsert] = []
        for record in records {
            if record.editionCount ?? 0 < filters.minEditions { continue }
            inserts.append(BookCatalog.WorkInsert(
                workKey: record.workKey, title: record.title, author: record.author, isbn: record.isbn,
                popularityRank: record.popularityRank, editionCount: record.editionCount
            ))
        }
        if let max = filters.maxWorks { inserts = Array(inserts.prefix(max)) }

        // Subset builds carry `book_isbns` forward from the source
        // catalog, scoped to the workKeys kept after filtering.
        let shippedWorkKeys = Set(inserts.map(\.workKey))
        let isbnRows: [(isbn13: String, workKey: String)] = try sourceCatalog.dbQueue.read { db in
            let rows = try Row.fetchAll(db, sql: "SELECT isbn13, workKey FROM book_isbns")
            return rows.compactMap { row -> (isbn13: String, workKey: String)? in
                let workKey: String = row["workKey"]
                guard shippedWorkKeys.contains(workKey) else { return nil }
                guard let isbn13 = ISBN13(rawPayload: row["isbn13"]) else { return nil }
                return (isbn13: isbn13.value, workKey: workKey)
            }
        }

        try materializeInserts(inserts, isbns: isbnRows, output: output, log: log)
    }

    private static func materializeInserts(
        _ rows: [BookCatalog.WorkInsert],
        isbns: [(isbn13: String, workKey: String)] = [],
        output: URL,
        log: (String) -> Void
    ) throws {
        if FileManager.default.fileExists(atPath: output.path) {
            try FileManager.default.removeItem(at: output)
        }
        let catalog = try BookCatalog(path: output.path)
        let inserted = try catalog.bulkInsert(rows)

        try catalog.insertISBNs(isbns)

        // customWords lexicon (§C): authors-only, derived from the same
        // popularity-ordered rows actually shipped in this build.
        var customWordsAcc = CustomWordsBuilder.Accumulator()
        for (index, row) in rows.enumerated() {
            customWordsAcc.add(author: row.author, popularityRank: row.popularityRank ?? index)
        }

        try finalizeCatalog(
            catalog, worksInserted: inserted, isbnsInserted: isbns.count,
            customWords: customWordsAcc.finish(), output: output, log: log
        )
    }

    /// Shared tail end of every build path: install the customWords
    /// lexicon, optionally `VACUUM` (see `vacuumRowThreshold`), and log the
    /// final row/byte counts.
    private static func finalizeCatalog(
        _ catalog: BookCatalog,
        worksInserted: Int,
        isbnsInserted: Int,
        customWords: [String],
        output: URL,
        log: (String) -> Void
    ) throws {
        try catalog.insertCustomWords(customWords)

        try catalog.dbQueue.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA journal_mode=DELETE")
        }
        if worksInserted <= vacuumRowThreshold {
            try catalog.dbQueue.vacuum()
        } else {
            log(
                "catalog-build: skipping VACUUM for \(worksInserted) works (over the "
                    + "\(vacuumRowThreshold)-row threshold) -- a from-scratch bulk-insert build has little "
                    + "fragmentation to reclaim, and VACUUM's ~2x-disk-space rebuild isn't worth the risk at this scale"
            )
        }
        let bytes = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int64) ?? 0
        log(
            "catalog-build: inserted \(worksInserted) works, \(isbnsInserted) isbns, "
                + "\(customWords.count) customWords, bytes=\(bytes), db=\(output.path)"
        )
    }

    /// Streams a gzipped JSONL file line-by-line via a `gunzip` subprocess
    /// pipe, decoding and handing off one `T` at a time -- never buffers
    /// the decompressed text or a `[T]` array of every row. Replaces the
    /// old `readGzippedJSONL`, which called `readDataToEndOfFile()`
    /// (buffering the *entire* decompressed file, ~12-15GB for `full`'s
    /// `works.jsonl.gz`) and decoded every line into one array before
    /// returning -- the root cause of the OOM `SIGKILL` this streaming
    /// rewrite fixes.
    ///
    /// Reads in 1MB chunks off the pipe and only ever holds one chunk plus
    /// at most one partial trailing line in memory (`leftover`), regardless
    /// of total file size.
    private static func streamGzippedJSONL<T: Decodable>(
        from url: URL,
        onLine: (T) throws -> Void
    ) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
        process.arguments = ["-c", url.path]
        let outPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = Pipe()
        try process.run()
        defer { if process.isRunning { process.terminate() } }

        let handle = outPipe.fileHandleForReading
        let decoder = JSONDecoder()
        let newline: UInt8 = 0x0A
        var leftover = Data()
        let chunkSize = 1 << 20

        while true {
            let chunk = handle.readData(ofLength: chunkSize)
            if chunk.isEmpty { break }
            leftover.append(chunk)
            while let newlineIndex = leftover.firstIndex(of: newline) {
                let lineData = leftover[leftover.startIndex..<newlineIndex]
                if !lineData.isEmpty {
                    try onLine(try decoder.decode(T.self, from: lineData))
                }
                leftover.removeSubrange(leftover.startIndex...newlineIndex)
            }
        }
        if !leftover.isEmpty {
            try onLine(try decoder.decode(T.self, from: leftover))
        }

        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "catalog-build", code: 2, userInfo: [NSLocalizedDescriptionKey: "gunzip failed for \(url.path)"])
        }
    }
}
