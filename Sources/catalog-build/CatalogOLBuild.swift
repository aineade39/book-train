import Foundation
import GRDB
import SpineCatalog

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
        try materialize(rows: rows, output: output, log: log)
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
        try materializeInserts(inserts, output: output, log: log)
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

    private static func materialize(rows: [BookCatalog.WorkInsert], output: URL, log: (String) -> Void) throws {
        try materializeInserts(rows, output: output, log: log)
    }

    private static func materializeInserts(_ rows: [BookCatalog.WorkInsert], output: URL, log: (String) -> Void) throws {
        if FileManager.default.fileExists(atPath: output.path) {
            try FileManager.default.removeItem(at: output)
        }
        let catalog = try BookCatalog(path: output.path)
        let inserted = try catalog.bulkInsert(rows)
        try catalog.dbQueue.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA journal_mode=DELETE")
        }
        try catalog.dbQueue.vacuum()
        let bytes = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int64) ?? 0
        log("catalog-build: inserted \(inserted) works, bytes=\(bytes), db=\(output.path)")
    }

    private static func readIntermediateWorks(from url: URL) throws -> [IntermediateWork] {
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
        var works: [IntermediateWork] = []
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let row = try decoder.decode(IntermediateWork.self, from: Data(line.utf8))
            works.append(row)
        }
        return works
    }
}
