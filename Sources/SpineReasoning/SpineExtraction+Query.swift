import FoundationModels

// Turns an FM extraction into the same shape of query string
// `normalizeForSearch` + `token_set_ratio` already expect from raw OCR
// text, per docs/BOOK_ID_IOS_PIPELINE.md's pipeline diagram: "[optional FM
// @Generable parse] -> Normalize -> FTS5 trigram shortlist ->
// token_set_ratio rerank". FM cleanup replaces the *input text*, not the
// matching algorithm -- title+author concatenated is exactly what
// `SpineMatching`'s scorer already handles for raw OCR (see
// `FuzzyMatch.swift`), so no new matching logic is needed.
@available(iOS 26.0, macOS 26.0, *)
extension SpineExtraction {
    /// `"title author"`, the same concatenated shape a legible OCR
    /// reading of a spine (title line(s) then author line) would already
    /// produce -- kept as the query text so an FM-assisted read and a
    /// clean Vision-only read are matched identically downstream.
    public var matchQueryText: String {
        author.isEmpty ? title : "\(title) \(author)"
    }
}
