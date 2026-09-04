import Foundation

// Fixed English stopword list, originally for `CustomWordsBuilder` per the
// locked "Book ID OCR gains" plan §C: "ship a fixed English list of <= 200
// words in package resources." Kept as a Swift source constant (not a
// bundled text resource) -- same effect, no `Bundle.module`
// resource-loading surface for a fixed ~150-word list that only ever
// changes alongside this file.
//
// Also reused (as of the catalog match-speed fix) by
// `BookCatalogRoleRetrieval.columnRetrieve` to drop near-zero-information
// tokens from FTS `MATCH` expressions before querying -- a token like
// "the" can match millions of rows in a large catalog, and filtering it
// out shrinks the row count `ORDER BY rank` has to score, on top of that
// fix. Deliberately the *same* list rather than a second one to maintain:
// a word too common to be useful in the `custom_words` Vision lexicon is
// equally too common to be a useful catalog search term.
public let englishStopwords: Set<String> = [
    "a", "an", "the", "and", "or", "but", "nor", "for", "so", "yet",
    "of", "in", "on", "at", "by", "to", "from", "with", "without", "into",
    "onto", "over", "under", "up", "down", "off", "out", "about", "above",
    "below", "between", "among", "through", "during", "before", "after",
    "as", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "has", "have", "had", "having",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "than", "too", "very",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "if", "then", "else", "because", "while", "until", "unless", "though",
    "again", "further", "once", "also", "just", "now", "one", "two", "three",
    "mr", "mrs", "ms", "dr", "jr", "sr", "de", "van", "von", "la", "le",
    "editor", "author", "translator", "illustrator", "et", "al",
]
