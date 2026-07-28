import Foundation

// Role-aware query assembly per the locked "Book ID OCR gains" plan §B:
// turns a reading-order-sorted list of `SpineTextLine`s (one winning OCR
// pass's observations) into the title/author/general token pools the
// 3-pass FTS retrieval consumes, plus the canonical `titleQuery`/
// `authorQuery`/`allNormalizedOCR` strings the field-aware rerank (§B) and
// AcceptPolicy stay pinned to -- n-best/confusion variants (§A) only ever
// widen the token *pools*, never these canonical strings.

/// One token entered into a retrieval pool, carrying the priority used to
/// rank pool members before the top-N cut (`tokenPriority` below).
public struct SpineRoleToken: Equatable {
    public let token: String
    public let priority: Double
}

/// Everything the role-aware retrieval + rerank stages need for one
/// spine's winning OCR pass.
public struct SpineRoleQueries: Equatable {
    /// Normalized, position-ordered join of lines with `Ptitle >= 0.35`
    /// (or every title-leaning line -- `Ptitle >= Pauthor` -- if none clear
    /// that bar, falling back further to the single max-`Ptitle` line only
    /// when literally every line leans author) -- the rerank/AcceptPolicy
    /// title-field input. Canonical (C1) text only.
    public let titleQuery: String
    /// Normalized, position-ordered join of lines with `Pauthor >= 0.40`;
    /// may be empty. Canonical (C1) text only.
    public let authorQuery: String
    /// Normalized join of every line's canonical text, in reading order --
    /// the "mashed blob" fallback field the rerank's `blobScore` compares
    /// against.
    public let allNormalizedOCR: String
    /// Top-8 tokens by `tokenPriority(Ptitle)`, canonical + n-best/confusion
    /// variants merged in before the cut. Feeds the FTS title-column pass.
    public let titleTokens: [SpineRoleToken]
    /// Top-6 tokens by `tokenPriority(Pauthor)`. Feeds the FTS
    /// author-column pass.
    public let authorTokens: [SpineRoleToken]
    /// Top-8 tokens by `tokenPriority(max(Ptitle, Pauthor))`. Feeds the
    /// FTS general (both-column) pass.
    public let generalTokens: [SpineRoleToken]
    /// Top-12 tokens by `tokenPriority(max(Ptitle, Pauthor))` -- the wider
    /// pool the empty-shortlist fallback pass draws from (unscoped OR
    /// across both columns, no per-pass LIMIT-80/50/30 narrowing).
    public let fallbackTokens: [SpineRoleToken]
    /// `true` when neither a confident title line nor a confident author
    /// line stands out from the geometry alone -- signals the caller
    /// should consider FM escalation (§G), text-only.
    public let rolesAmbiguous: Bool
    /// `true` when `titleQuery` came from the no-line-clears-threshold
    /// fallback path rather than a confident `Ptitle >= 0.35` line --
    /// signals to `FieldAwareScore` that `titleQuery` is comparatively
    /// less trustworthy than `allNormalizedOCR` for this spine.
    public let titleQueryIsFallback: Bool
}

public enum SpineRoleQueryBuilder {
    static let titleQueryThreshold = 0.35
    static let authorQueryThreshold = 0.40
    static let titlePoolSize = 8
    static let authorPoolSize = 6
    static let generalPoolSize = 8
    static let fallbackPoolSize = 12
    /// §A: "FTS token budget <= 24 unique normalized tokens length >= 3
    /// for the expansion/alternate pool" -- variant (non-canonical) tokens
    /// only; canonical C1 tokens are uncapped here (the per-pool top-8/6/8
    /// cuts below bound them instead).
    static let expansionBudget = 24
    static let ambiguityConfidenceFloor = 0.60
    static let ambiguityMargin = 0.15
    static let authorCandidateFloor = 0.35

    /// Builds role queries from `lines`, which callers must supply already
    /// sorted in reading order (see `SpinePerception.assembleReadingOrder`)
    /// -- `allNormalizedOCR`/`titleQuery`/`authorQuery` join in list order.
    public static func build(lines: [SpineTextLine]) -> SpineRoleQueries {
        guard !lines.isEmpty else {
            return SpineRoleQueries(
                titleQuery: "", authorQuery: "", allNormalizedOCR: "",
                titleTokens: [], authorTokens: [], generalTokens: [], fallbackTokens: [], rolesAmbiguous: false,
                titleQueryIsFallback: false
            )
        }

        let scored = lines.map { line in (line: line, role: SpineRoleScoring.roleScores(for: line)) }

        let allNormalizedOCR = normalizeForSearch(lines.map(\.text).joined(separator: " "))

        let (titleQuery, titleQueryIsFallback) = canonicalQuery(
            scored, threshold: titleQueryThreshold, select: \.title, fallbackToMax: true
        )
        // A line only qualifies for `authorQuery` when author is its most
        // likely role, not merely above the floor -- without this, a line
        // whose title/author scores are both near their respective
        // thresholds (a single ambiguous line reporting *both*, e.g. a
        // two-word title read as one line whose height/position sit
        // between the title and author bands) would otherwise leak its
        // own title words into `authorQuery`, tanking `authorScore`
        // against the real (unrelated) catalog author even though
        // `titleQuery` alone would have matched cleanly.
        let (authorQuery, _) = canonicalQuery(
            scored, threshold: authorQueryThreshold, select: \.author, fallbackToMax: false,
            dominantOver: \.title
        )

        var expansionBudgetRemaining = expansionBudget
        var seenVariantTokens = Set<String>()
        // Canonical tokens are tracked too so a variant that happens to
        // collide with an already-present canonical token never double
        // counts against the budget or the pools.
        for line in lines {
            for token in searchTokens(normalizeForSearch(line.text)) where token.count >= 3 {
                seenVariantTokens.insert(token)
            }
        }

        var titlePool: [String: Double] = [:]
        var authorPool: [String: Double] = [:]
        var generalPool: [String: Double] = [:]

        func addToken(_ token: String, role: SpineRoleScores) {
            guard token.count >= 3 else { return }
            addMax(&titlePool, token, tokenPriority(token: token, roleProbability: role.title))
            addMax(&authorPool, token, tokenPriority(token: token, roleProbability: role.author))
            addMax(&generalPool, token, tokenPriority(token: token, roleProbability: max(role.title, role.author)))
        }

        for (line, role) in scored {
            let canonicalTokens = searchTokens(normalizeForSearch(line.text)).filter { $0.count >= 3 }
            for token in canonicalTokens { addToken(token, role: role) }

            guard expansionBudgetRemaining > 0 else { continue }

            // Alternate (kept C2/C3) whole-line readings: tokenize and
            // treat any token not already seen as a retrieval-only variant.
            var variantCandidates: [(token: String, confidence: Float)] = []
            for alt in line.alternates {
                for token in searchTokens(normalizeForSearch(alt.text)) where token.count >= 3 {
                    variantCandidates.append((token, alt.confidence))
                }
            }
            // Confusion expansions of the canonical tokens themselves.
            for token in canonicalTokens {
                for expanded in OCRConfusion.expand(token: token, confidence: line.confidence) {
                    variantCandidates.append((expanded, line.confidence))
                }
            }
            // Confusion expansions of the kept alternates' own tokens.
            for alt in line.alternates {
                for token in searchTokens(normalizeForSearch(alt.text)) where token.count >= 3 {
                    for expanded in OCRConfusion.expand(token: token, confidence: alt.confidence) {
                        variantCandidates.append((expanded, alt.confidence))
                    }
                }
            }

            for (token, _) in variantCandidates {
                guard expansionBudgetRemaining > 0 else { break }
                guard !seenVariantTokens.contains(token) else { continue }
                seenVariantTokens.insert(token)
                expansionBudgetRemaining -= 1
                addToken(token, role: role)
            }
        }

        let rolesAmbiguous = isRolesAmbiguous(scored)

        return SpineRoleQueries(
            titleQuery: titleQuery,
            authorQuery: authorQuery,
            allNormalizedOCR: allNormalizedOCR,
            titleTokens: topN(titlePool, titlePoolSize),
            authorTokens: topN(authorPool, authorPoolSize),
            generalTokens: topN(generalPool, generalPoolSize),
            fallbackTokens: topN(generalPool, fallbackPoolSize),
            rolesAmbiguous: rolesAmbiguous,
            titleQueryIsFallback: titleQueryIsFallback
        )
    }

    /// Builds role queries directly from a Foundation Model's cleaned
    /// title/author (§G: FM escalation is text-only in, text-only out) --
    /// skips geometry scoring and n-best/confusion expansion entirely,
    /// since FM already produced the canonical answer from the raw OCR
    /// text. Feeds the exact same `retrieveRoleAware`/`FieldAwareScore`/
    /// `AcceptPolicy` path as geometry-derived queries (plan diagram:
    /// "roles -[rolesAmbiguous]-> fm -> retrieve"), just with every
    /// token at equal (max) priority within its own field and
    /// `rolesAmbiguous` forced `false` (FM's whole point was resolving
    /// that ambiguity).
    public static func build(fmTitle: String, fmAuthor: String) -> SpineRoleQueries {
        let titleQuery = normalizeForSearch(fmTitle)
        let authorQuery = normalizeForSearch(fmAuthor)
        let allNormalizedOCR = authorQuery.isEmpty ? titleQuery : "\(titleQuery) \(authorQuery)"

        func tokens(_ text: String) -> [SpineRoleToken] {
            searchTokens(text).filter { $0.count >= 3 }.map { SpineRoleToken(token: $0, priority: 1.0) }
        }
        let titleTokens = tokens(titleQuery)
        let authorTokens = tokens(authorQuery)
        var generalPool: [String: Double] = [:]
        for token in titleTokens + authorTokens { addMax(&generalPool, token.token, token.priority) }

        return SpineRoleQueries(
            titleQuery: titleQuery,
            authorQuery: authorQuery,
            allNormalizedOCR: allNormalizedOCR,
            titleTokens: Array(titleTokens.prefix(titlePoolSize)),
            authorTokens: Array(authorTokens.prefix(authorPoolSize)),
            generalTokens: topN(generalPool, generalPoolSize),
            fallbackTokens: topN(generalPool, fallbackPoolSize),
            rolesAmbiguous: false,
            titleQueryIsFallback: false
        )
    }

    /// `Prole * (0.70 + 0.03 * min(len, 10))` -- longer tokens (more
    /// distinctive substrings) get a mild boost within the same role.
    static func tokenPriority(token: String, roleProbability: Double) -> Double {
        roleProbability * (0.70 + 0.03 * Double(min(token.count, 10)))
    }

    private static func addMax(_ pool: inout [String: Double], _ token: String, _ priority: Double) {
        if let existing = pool[token], existing >= priority { return }
        pool[token] = priority
    }

    private static func topN(_ pool: [String: Double], _ n: Int) -> [SpineRoleToken] {
        pool
            .map { SpineRoleToken(token: $0.key, priority: $0.value) }
            .sorted { $0.priority > $1.priority }
            .prefix(n)
            .map { $0 }
    }

    /// Returns the canonical query string plus whether the no-line-clears-
    /// `threshold` fallback path was taken (only ever `true` when
    /// `fallbackToMax` is; author's own call always gets `false` back).
    private static func canonicalQuery(
        _ scored: [(line: SpineTextLine, role: SpineRoleScores)],
        threshold: Double,
        select: (SpineRoleScores) -> Double,
        fallbackToMax: Bool,
        dominantOver: ((SpineRoleScores) -> Double)? = nil
    ) -> (query: String, isFallback: Bool) {
        let qualifying = scored.filter { entry in
            guard select(entry.role) >= threshold else { return false }
            if let dominantOver { return select(entry.role) > dominantOver(entry.role) }
            return true
        }
        let chosen: [(line: SpineTextLine, role: SpineRoleScores)]
        let isFallback: Bool
        if qualifying.isEmpty {
            guard fallbackToMax else { return ("", false) }
            chosen = fallbackSelection(scored, select: select, dominantOver: dominantOver)
            isFallback = !chosen.isEmpty
        } else {
            chosen = qualifying
            isFallback = false
        }
        let ordered = chosen.sorted { SpineRoleScoring.spinePosition(of: $0.line) < SpineRoleScoring.spinePosition(of: $1.line) }
        return (normalizeForSearch(ordered.map(\.line.text).joined(separator: " ")), isFallback)
    }

    /// No line clears `threshold` outright. Rather than betting the entire
    /// canonical query on one arbitrary highest-scoring line -- on a
    /// multi-line vertical spine where every line reads as one or two
    /// words, a stray author first name, edition/publisher text, or a
    /// country-name subtitle is just as likely to "win" as the real title
    /// -- keep every line that at least *leans* this way over the
    /// counter-role (`select(role) >= counterRole(role)`) *and* isn't
    /// itself predominantly `other`-role boilerplate (ISBN/publisher/
    /// edition text); joining them in position order approximates the real
    /// title text far better than any single line does, and is a no-op
    /// when there's only one line anyway. Only when literally every line
    /// leans the other way (or is `other`-dominant) does this fall back
    /// further to the single max-scoring line, as before.
    ///
    /// The `other`-dominance check was added after measuring a real
    /// regression against the actual Swift pipeline (not just the Python
    /// proxy): without it, a numeric/edition/publisher line that happens
    /// to score marginally above `author` on the title/author comparison
    /// alone -- despite `other` being its clear top role -- got joined
    /// into `titleQuery` too, diluting it with boilerplate tokens (e.g. a
    /// spine's "THIRD EDITION" or ISBN-ish digit-run line leaking in
    /// alongside the real title lines) and flipping a couple of previously
    /// correct matches to wrong ones.
    private static func fallbackSelection(
        _ scored: [(line: SpineTextLine, role: SpineRoleScores)],
        select: (SpineRoleScores) -> Double,
        dominantOver: ((SpineRoleScores) -> Double)?
    ) -> [(line: SpineTextLine, role: SpineRoleScores)] {
        let counterRole = dominantOver ?? { $0.author }
        let leaning = scored.filter { select($0.role) >= counterRole($0.role) && select($0.role) >= $0.role.other }
        if !leaning.isEmpty { return leaning }
        guard let best = scored.max(by: { select($0.role) < select($1.role) }) else { return [] }
        return [best]
    }

    private static func isRolesAmbiguous(_ scored: [(line: SpineTextLine, role: SpineRoleScores)]) -> Bool {
        guard let titleLine = scored.max(by: { $0.role.title < $1.role.title }) else { return false }
        let titleAmbiguous = titleLine.role.title < ambiguityConfidenceFloor
            || (titleLine.role.title - max(titleLine.role.author, titleLine.role.other)) < ambiguityMargin

        guard let authorLine = scored.max(by: { $0.role.author < $1.role.author }) else { return titleAmbiguous }
        let authorIsCandidate = authorLine.role.author >= authorCandidateFloor
        let authorAmbiguous = authorIsCandidate && (
            authorLine.role.author < ambiguityConfidenceFloor
                || (authorLine.role.author - max(authorLine.role.title, authorLine.role.other)) < ambiguityMargin
        )

        return titleAmbiguous || authorAmbiguous
    }
}
