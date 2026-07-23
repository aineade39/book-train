import FoundationModels

// Structured hard-case extraction per docs/BOOK_ID_IOS_PIPELINE.md
// §Foundation Models integration: "Language layer on top of Vision
// output — disambiguate title vs author vs publisher noise" via
// `LanguageModelSession` + **`@Generable` structured output**, not
// unstructured image prompting or FM-as-OCR. Matches the doc's "Correct
// heavy-path shape" sample exactly (`title`/`author`/`isbn?`).
//
// `@available`-gated as a whole: every FoundationModels type this touches
// (`@Generable`'s generated conformance, `LanguageModelSession.respond`)
// requires iOS/macOS 26, so there is no useful subset of this type usable
// below that even though the package's deployment target is lower.
@available(iOS 26.0, macOS 26.0, *)
@Generable
public struct SpineExtraction {
    @Guide(description: "The book's title, cleaned up from noisy OCR text read off a single spine. Fix obvious OCR letter substitutions; do not invent words not plausibly present in the source text.")
    public let title: String

    @Guide(description: "The author's full name if legible in the OCR text.")
    public let author: String

    @Guide(description: "An ISBN-10 or ISBN-13 if one literally appears in the OCR text (rare on a spine); omit otherwise -- never guess or look one up.")
    public let isbn: String?
}
