import Foundation
import SpineCatalog

// Builds `SpineCatalog.BookCatalog` SQLite files:
//
// Mode A — CSV (small lists, tests):
//   swift run catalog-build <catalog.csv> --db <catalog.sqlite>
//
// Mode B — OL intermediate (bulk):
//   swift run catalog-build --intermediate <dir> --output <catalog.sqlite> \
//       [--max-works N] [--languages eng ...] [--min-editions N]
//
// Mode C — subset from existing full DB:
//   swift run catalog-build --subset-from <full.sqlite> --output <catalog.sqlite> \
//       [--max-works N] [--min-editions N]
//
// See docs/BOOK_CATALOG.md.

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func nextArg(_ args: [String], _ i: inout Int) -> String {
    i += 1
    guard i < args.count else { fail("Missing value for \(args[i - 1])") }
    return args[i]
}

struct CLIOptions {
    var csvPath: String?
    var dbPath: String?
    var intermediatePath: String?
    var outputPath: String?
    var subsetFromPath: String?
    var maxWorks: Int?
    var languages: [String] = []
    var minEditions = 1
}

func parseArgs(_ cliArgs: [String]) -> CLIOptions {
    var opts = CLIOptions()
    var i = 1
    while i < cliArgs.count {
        let arg = cliArgs[i]
        switch arg {
        case "--db": opts.dbPath = nextArg(cliArgs, &i)
        case "--intermediate": opts.intermediatePath = nextArg(cliArgs, &i)
        case "--output": opts.outputPath = nextArg(cliArgs, &i)
        case "--subset-from": opts.subsetFromPath = nextArg(cliArgs, &i)
        case "--max-works": opts.maxWorks = Int(nextArg(cliArgs, &i))
        case "--languages":
            i += 1
            while i < cliArgs.count, !cliArgs[i].hasPrefix("--") {
                opts.languages.append(cliArgs[i])
                i += 1
            }
            continue
        case "--min-editions": opts.minEditions = Int(nextArg(cliArgs, &i)) ?? opts.minEditions
        default:
            if opts.csvPath == nil, !arg.hasPrefix("--") { opts.csvPath = arg }
            else { fail("Unknown argument: \(arg)") }
        }
        i += 1
    }
    return opts
}

let opts = parseArgs(CommandLine.arguments)
let filters = CatalogProfileFilters(maxWorks: opts.maxWorks, languages: opts.languages, minEditions: opts.minEditions)

if opts.subsetFromPath != nil || opts.intermediatePath != nil {
    guard let output = opts.outputPath else {
        fail("OL mode requires --output <catalog.sqlite>")
    }
    let outputURL = URL(fileURLWithPath: output).standardizedFileURL
    do {
        if let subset = opts.subsetFromPath {
            try CatalogOLBuild.buildFromSubset(
                source: URL(fileURLWithPath: subset).standardizedFileURL,
                output: outputURL,
                filters: filters,
                log: log
            )
        } else if let intermediate = opts.intermediatePath {
            try CatalogOLBuild.buildFromIntermediate(
                intermediate: URL(fileURLWithPath: intermediate).standardizedFileURL,
                output: outputURL,
                filters: filters,
                log: log
            )
        }
    } catch {
        fail("\(error)")
    }
} else {
    guard let csvPath = opts.csvPath, let dbPath = opts.dbPath else {
        fail("""
            Usage:
              swift run catalog-build <catalog.csv> --db <catalog.sqlite>
              swift run catalog-build --intermediate <dir> --output <catalog.sqlite> [filters]
              swift run catalog-build --subset-from <full.sqlite> --output <catalog.sqlite> [filters]
            """)
    }
    runCSVBuild(csvPath: csvPath, dbPath: dbPath)
}

func runCSVBuild(csvPath: String, dbPath: String) {
    let csvURL = URL(fileURLWithPath: csvPath).standardizedFileURL
    guard let contents = try? String(contentsOf: csvURL, encoding: .utf8) else {
        fail("Could not read CSV: \(csvURL.path)")
    }

    let rows = parseCSV(contents)
    guard let header = rows.first else { fail("Empty CSV: \(csvURL.path)") }

    let lowerHeader = header.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
    guard let titleCol = lowerHeader.firstIndex(of: "title"), let authorCol = lowerHeader.firstIndex(of: "author") else {
        fail("CSV header must include 'title' and 'author' columns; got \(header)")
    }
    let isbnCol = lowerHeader.firstIndex(of: "isbn")
    let workKeyCol = lowerHeader.firstIndex(of: "workkey")

    let dbURL = URL(fileURLWithPath: dbPath).standardizedFileURL
    let catalog: BookCatalog
    do {
        catalog = try BookCatalog(path: dbURL.path)
    } catch {
        fail("Could not open/create catalog at \(dbURL.path): \(error)")
    }

    var inserted = 0
    var skipped = 0
    for row in rows.dropFirst() {
        guard row.count > max(titleCol, authorCol) else { skipped += 1; continue }
        let title = row[titleCol].trimmingCharacters(in: .whitespaces)
        let author = row[authorCol].trimmingCharacters(in: .whitespaces)
        guard !title.isEmpty, !author.isEmpty else { skipped += 1; continue }
        let isbn = isbnCol.flatMap { row.count > $0 ? row[$0].trimmingCharacters(in: .whitespaces) : nil }
        let workKey = workKeyCol.flatMap { row.count > $0 ? row[$0].trimmingCharacters(in: .whitespaces) : nil }

        do {
            try catalog.insert(
                title: title, author: author,
                isbn: (isbn?.isEmpty ?? true) ? nil : isbn,
                workKey: (workKey?.isEmpty ?? true) ? nil : workKey
            )
            inserted += 1
        } catch {
            log("warning: could not insert row \(row): \(error)")
            skipped += 1
        }
    }

    log("catalog-build: inserted \(inserted) rows, skipped \(skipped), db=\(dbURL.path)")
}

// MARK: - Minimal CSV parsing (RFC4180-ish)

func parseCSV(_ text: String) -> [[String]] {
    var rows: [[String]] = []
    var currentRow: [String] = []
    var field = ""
    var inQuotes = false
    var iterator = text.makeIterator()
    var pending: Character?

    func endField() { currentRow.append(field); field = "" }
    func endRow() { endField(); rows.append(currentRow); currentRow = [] }

    while let c = pending ?? iterator.next() {
        pending = nil
        if inQuotes {
            if c == "\"" {
                if let next = iterator.next() {
                    if next == "\"" { field.append("\"") } else { inQuotes = false; pending = next }
                } else {
                    inQuotes = false
                }
            } else {
                field.append(c)
            }
            continue
        }
        switch c {
        case "\"": inQuotes = true
        case ",": endField()
        case "\n": endRow()
        case "\r": continue
        default: field.append(c)
        }
    }
    if !field.isEmpty || !currentRow.isEmpty { endRow() }
    return rows.filter { !($0.count == 1 && $0[0].isEmpty) }
}
