import Foundation
import SpineCatalog
import SpineMatching

// Stage 1 (FTS5 retrieval) + Stage 2 (fuzzy rerank + accept policy) macOS
// validation harness -- runs the exact matcher `spine-id` uses, without
// needing a photo/model, per docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching.
//
// Usage:
//   swift run book-match "<query>" --db <catalog.sqlite> \
//       [--limit 50] [--accept-threshold 90] [--margin 8] [--top-n 5] [--json]

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func nextArg(_ args: [String], _ i: inout Int) -> String {
    i += 1
    guard i < args.count else { fail("Missing value for \(args[i - 1])") }
    return args[i]
}

var query: String?
var dbPath: String?
var limit = 50
var acceptThreshold = 90.0
var marginThreshold = 8.0
var topN = 5
var wantJSON = false

let cliArgs = CommandLine.arguments
var argIndex = 1
while argIndex < cliArgs.count {
    let arg = cliArgs[argIndex]
    switch arg {
    case "--db": dbPath = nextArg(cliArgs, &argIndex)
    case "--limit": limit = Int(nextArg(cliArgs, &argIndex)) ?? limit
    case "--accept-threshold": acceptThreshold = Double(nextArg(cliArgs, &argIndex)) ?? acceptThreshold
    case "--margin": marginThreshold = Double(nextArg(cliArgs, &argIndex)) ?? marginThreshold
    case "--top-n": topN = Int(nextArg(cliArgs, &argIndex)) ?? topN
    case "--json": wantJSON = true
    default:
        if query == nil { query = arg } else { fail("Unknown argument: \(arg)") }
    }
    argIndex += 1
}

guard let query, let dbPath else {
    fail("""
        Usage: swift run book-match "<query>" --db <catalog.sqlite> \
        [--limit 50] [--accept-threshold 90] [--margin 8] [--top-n 5] [--json]
        """)
}

let catalog: BookCatalog
do {
    catalog = try BookCatalog(path: URL(fileURLWithPath: dbPath).standardizedFileURL.path)
} catch {
    fail("Could not open catalog: \(error)")
}

let candidates: [CatalogCandidate]
do {
    candidates = try catalog.retrieveCandidates(forQuery: query, limit: limit)
} catch {
    fail("Retrieval failed: \(error)")
}

let normalizedQuery = normalizeForSearch(query)
let scored = candidates.map {
    ScoredCandidate(candidate: $0, score: tokenSetRatio(normalizedQuery, normalizeForSearch($0.searchableText)))
}
let policy = AcceptPolicy(acceptThreshold: acceptThreshold, marginThreshold: marginThreshold, topN: topN)
let decision = policy.decide(scored)

struct CandidateJSON: Codable {
    let id: Int64
    let title: String
    let author: String
    let isbn: String?
    let workKey: String
    let score: Double
}

struct ResultJSON: Codable {
    let query: String
    let retrievedCount: Int
    let decision: String
    let winner: CandidateJSON?
    let topCandidates: [CandidateJSON]
}

func toJSON(_ sc: ScoredCandidate<CatalogCandidate>) -> CandidateJSON {
    CandidateJSON(
        id: sc.candidate.id, title: sc.candidate.title, author: sc.candidate.author,
        isbn: sc.candidate.isbn, workKey: sc.candidate.workKey, score: (sc.score * 100).rounded() / 100
    )
}

var result: ResultJSON
switch decision {
case .autoAccept(let winner):
    result = ResultJSON(
        query: query, retrievedCount: candidates.count, decision: "auto-accept",
        winner: toJSON(winner), topCandidates: [toJSON(winner)]
    )
case .ambiguous(let top):
    result = ResultJSON(
        query: query, retrievedCount: candidates.count, decision: "ambiguous",
        winner: nil, topCandidates: top.map(toJSON)
    )
case .noMatch:
    result = ResultJSON(query: query, retrievedCount: candidates.count, decision: "no-match", winner: nil, topCandidates: [])
}

if wantJSON {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try! encoder.encode(result)
    print(String(data: data, encoding: .utf8)!)
} else {
    print("query: \(result.query)")
    print("retrieved: \(result.retrievedCount) candidates")
    print("decision: \(result.decision)")
    for c in result.topCandidates {
        print("  [\(c.score)] \(c.title) -- \(c.author) (workKey=\(c.workKey), isbn=\(c.isbn ?? "-"))")
    }
}
