import Foundation

// Authors-only `customWords` lexicon builder per the locked "Book ID OCR
// gains" plan §C: "At catalog-build: popularity-ordered works; split
// author on ',' / '&' / ' and '; score by work-count (tie-break better
// rank); top K=1500, hard cap 2000; filter stopwords / 1-letter /
// all-digit / len>40; no titles/series."
public enum CustomWordsBuilder {
    public static let targetCount = 1500
    public static let hardCap = 2000
    public static let maxWordLength = 40

    public struct AuthorEntry {
        public let author: String
        /// Lower is more popular; used only as a tie-break within equal
        /// work-counts, so any large sentinel is fine when unknown.
        public let popularityRank: Int

        public init(author: String, popularityRank: Int) {
            self.author = author
            self.popularityRank = popularityRank
        }
    }

    /// Builds the ranked, filtered word list. Order doesn't matter to
    /// `BookCatalog.insertCustomWords` (stored in a plain `PRIMARY KEY`
    /// table), but is kept deterministic (score desc, rank asc, word asc)
    /// for reproducible catalog builds.
    public static func build(from entries: [AuthorEntry]) -> [String] {
        var bestRank: [String: Int] = [:]
        var workCount: [String: Int] = [:]

        for entry in entries {
            let words = Set(individualWords(from: entry.author))
            for word in words {
                workCount[word, default: 0] += 1
                bestRank[word] = min(bestRank[word] ?? Int.max, entry.popularityRank)
            }
        }

        let candidates = workCount.keys
            .filter(isEligible)
            .sorted { a, b in
                let countA = workCount[a] ?? 0, countB = workCount[b] ?? 0
                if countA != countB { return countA > countB }
                let rankA = bestRank[a] ?? Int.max, rankB = bestRank[b] ?? Int.max
                if rankA != rankB { return rankA < rankB }
                return a < b
            }

        guard candidates.count > targetCount else { return candidates }

        // Ties straddling the target cutoff all get included (up to
        // hardCap) rather than arbitrarily truncating mid-tie.
        var result = Array(candidates.prefix(targetCount))
        let boundaryCount = workCount[candidates[targetCount - 1]] ?? 0
        var i = targetCount
        while i < candidates.count, result.count < hardCap, (workCount[candidates[i]] ?? 0) == boundaryCount {
            result.append(candidates[i])
            i += 1
        }
        return result
    }

    /// Splits a (possibly multi-author) display string into individual
    /// name-word tokens: first splits on `,` / `&` / ` and ` to separate
    /// distinct people, then on whitespace for each person's own words
    /// (so "Stephen King and Peter Straub" -> ["Stephen", "King", "Peter",
    /// "Straub"], each an independent Vision `customWords` candidate).
    static func individualWords(from author: String) -> [String] {
        let normalized = author.replacingOccurrences(of: " and ", with: ",", options: .caseInsensitive)
        let people = normalized.components(separatedBy: CharacterSet(charactersIn: ",&"))
        var words: [String] = []
        for person in people {
            for word in person.split(separator: " ") {
                let cleaned = word.trimmingCharacters(in: .punctuationCharacters)
                guard !cleaned.isEmpty else { continue }
                words.append(cleaned)
            }
        }
        return words
    }

    static func isEligible(_ word: String) -> Bool {
        guard word.count > 1, word.count <= maxWordLength else { return false }
        guard !word.allSatisfy(\.isNumber) else { return false }
        guard !englishStopwordsForCustomWords.contains(word.lowercased()) else { return false }
        return true
    }
}
