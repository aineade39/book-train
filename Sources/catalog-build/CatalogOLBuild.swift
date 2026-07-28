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
        let works = try readIntermediateWorks(from: worksFile)
        let rows = applyFilters(works, filters: filters)

        // book_isbns (§D): populated from the OL `isbns.jsonl.gz`
        // many-to-many mapping, scoped down to the workKeys actually
        // shipped in this catalog build (maxWorks/language/minEditions
        // filters can shrink `rows` well below the full intermediate).
        var isbnRows: [(isbn13: String, workKey: String)] = []
        let isbnsFile = intermediate.appendingPathComponent("isbns.jsonl.gz")
        if FileManager.default.fileExists(atPath: isbnsFile.path) {
            let shippedWorkKeys = Set(rows.map(\.workKey))
            let allISBNs = try readIntermediateISBNs(from: isbnsFile)
            // Re-validated/normalized through the single shared
            // `SpineMatching.ISBN13` path rather than trusted verbatim
            // from the Python intermediate (§H: shared match/ISBN APIs).
            isbnRows = allISBNs.compactMap { row in
                guard shippedWorkKeys.contains(row.workKey) else { return nil }
                guard let isbn13 = ISBN13(rawPayload: row.isbn13) else { return nil }
                return (isbn13: isbn13.value, workKey: row.workKey)
            }
        } else {
            log("catalog-build: no isbns.jsonl.gz found at \(isbnsFile.path); book_isbns will be empty")
        }

        try materializeInserts(rows, isbns: isbnRows, output: output, log: log)
    }

    static func buildFromSubset(
        source: URL,
        output: URL,
        filters: CatalogProfileFilters,
        log: (String) -> Void
    ) throws {
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

    private static func applyFilters(_ works: [IntermediateWork], filters: CatalogProfileFilters) -> [BookCatalog.WorkInsert] {
        let langSet = Set(filters.languages.map { $0.lowercased() })
        var out: [BookCatalog.WorkInsert] = []
        for work in works {
            if work.editionCount < filters.minEditions { continue }
            if !langSet.isEmpty {
                let workLangs = Set(work.languages.map { $0.lowercased() })
                if workLangs.isDisjoint(with: langSet) { continue }
            }
            out.append(BookCatalog.WorkInsert(
                workKey: work.workKey, title: work.title, author: work.author, isbn: work.isbn13,
                popularityRank: work.popularityRank, editionCount: work.editionCount
            ))
        }
        if let max = filters.maxWorks { out = Array(out.prefix(max)) }
        return out
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
        let authorEntries = rows.enumerated().map { index, row in
            CustomWordsBuilder.AuthorEntry(author: row.author, popularityRank: row.popularityRank ?? index)
        }
        let customWords = CustomWordsBuilder.build(from: authorEntries)
        try catalog.insertCustomWords(customWords)

        try catalog.dbQueue.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA journal_mode=DELETE")
        }
        try catalog.dbQueue.vacuum()
        let bytes = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int64) ?? 0
        log(
            "catalog-build: inserted \(inserted) works, \(isbns.count) isbns, "
                + "\(customWords.count) customWords, bytes=\(bytes), db=\(output.path)"
        )
    }

    private static func readIntermediateWorks(from url: URL) throws -> [IntermediateWork] {
        try readGzippedJSONL(from: url)
    }

    private static func readIntermediateISBNs(from url: URL) throws -> [IntermediateISBN] {
        try readGzippedJSONL(from: url)
    }

    private static func readGzippedJSONL<T: Decodable>(from url: URL) throws -> [T] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
        process.arguments = ["-c", url.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "catalog-build", code: 2, userInfo: [NSLocalizedDescriptionKey: "gunzip failed for \(url.path)"])
        }
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        var rows: [T] = []
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let row = try decoder.decode(T.self, from: Data(line.utf8))
            rows.append(row)
        }
        return rows
    }
}
