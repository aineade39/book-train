import Foundation
import SpineCatalog

// Builds (or appends to) a `SpineCatalog.BookCatalog` SQLite file from a CSV
// of `title,author,isbn` rows -- the macOS-side counterpart to whatever
// seeds the on-device catalog per docs/BOOK_ID_IOS_PIPELINE.md §Data model.
//
// Usage:
//   swift run catalog-build <catalog.csv> --db <catalog.sqlite> [--work-key-column workKey]
//
// CSV must have a header row; recognized columns (case-insensitive):
// `title` (required), `author` (required), `isbn` (optional),
// `workKey` (optional -- defaults to normalized title|author, grouping
// editions per §Data model).

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

var csvPath: String?
var dbPath: String?

let cliArgs = CommandLine.arguments
var argIndex = 1
while argIndex < cliArgs.count {
    let arg = cliArgs[argIndex]
    switch arg {
    case "--db": dbPath = nextArg(cliArgs, &argIndex)
    default:
        if csvPath == nil { csvPath = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let csvPath, let dbPath else {
    fail("Usage: swift run catalog-build <catalog.csv> --db <catalog.sqlite>")
}

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

// MARK: - Minimal CSV parsing (RFC4180-ish: quoted fields, "" escapes a
// literal quote, commas/newlines allowed inside quotes). Deliberately
// dependency-free -- catalog CSVs are small (a personal book list, not a
// bulk import), so a hand-rolled parser is fine here.

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
