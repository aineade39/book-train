import Foundation

/// Where a match's identifying string ultimately came from — surfaced in
/// the UI/telemetry so barcode hits (bypass fuzzy match entirely) are
/// distinguishable from OCR-only and FM-assisted matches. Mirrors
/// docs/BOOK_ID_IOS_PIPELINE.md §Data model's `MatchResult` shape.
public enum MatchSource: String, Codable {
    case barcode
    case ocr
    case fmAssisted = "fm-assisted"
}

/// One resolved (or ambiguous) match for a single spine detection.
/// `spineDetectionId` ties back to `SpineCore.OBBDetection.id` without this
/// module depending on `SpineCore` (kept a plain `UUID` here).
public struct MatchResult<BookID: Hashable & Codable>: Codable {
    public let bookId: BookID
    public let score: Double
    public let margin: Double
    public let source: MatchSource
    public let spineDetectionId: UUID

    public init(bookId: BookID, score: Double, margin: Double, source: MatchSource, spineDetectionId: UUID) {
        self.bookId = bookId
        self.score = score
        self.margin = margin
        self.source = source
        self.spineDetectionId = spineDetectionId
    }
}
