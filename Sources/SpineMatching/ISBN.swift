import Foundation

// ISBN normalization/validation per the locked "Book ID OCR gains" plan
// §D (ISBN / barcode): "Strip separators; uppercase X; ISBN-10 or 13 only;
// checksum required; ISBN-10 -> ISBN-13 (978); invalid discarded (no
// banner)." Lives in `SpineMatching` (not `SpinePerception`) because it's
// pure catalog-identity logic with no Vision dependency, and both the app
// and `spine-id` CLI already depend on this module for matching.

/// A validated, canonicalized ISBN-13 -- the only shape callers should
/// persist or query the catalog with. Construction always checks the
/// checksum; an ISBN-10 input is converted to its canonical ISBN-13 form
/// (prefix `978`, recomputed check digit) rather than kept as ISBN-10.
public struct ISBN13: Equatable, Hashable {
    public let value: String

    /// Strips separators (`-`, spaces), uppercases a trailing `X`, accepts
    /// only a 10-character (ISBN-10, optionally `X`-terminated) or
    /// 13-character (ISBN-13, all-digit) payload, verifies its checksum,
    /// and normalizes ISBN-10 to ISBN-13. Returns `nil` for anything else
    /// -- callers must silently discard invalid payloads (no banner, no
    /// partial match) per the locked decision.
    public init?(rawPayload: String) {
        let cleaned = ISBN13.strip(rawPayload)
        if cleaned.count == 13, cleaned.allSatisfy(\.isNumber) {
            guard ISBN13.isValidISBN13Checksum(cleaned) else { return nil }
            value = cleaned
        } else if cleaned.count == 10 {
            guard ISBN13.isValidISBN10Checksum(cleaned) else { return nil }
            guard let converted = ISBN13.convertISBN10ToISBN13(cleaned) else { return nil }
            value = converted
        } else {
            return nil
        }
    }

    /// Internal fast-path init for already-known-valid ISBN-13 strings
    /// (e.g. catalog build pipeline reading pre-validated OL data) --
    /// skips checksum recomputation but still requires the right shape.
    static func preValidated(_ isbn13: String) -> ISBN13? {
        guard isbn13.count == 13, isbn13.allSatisfy(\.isNumber) else { return nil }
        return ISBN13(uncheckedValue: isbn13)
    }

    private init(uncheckedValue: String) {
        value = uncheckedValue
    }

    private static func strip(_ raw: String) -> String {
        var out = ""
        for ch in raw {
            if ch.isNumber {
                out.append(ch)
            } else if ch == "x" || ch == "X" {
                out.append("X")
            }
            // separators (-, space, etc.) and anything else are dropped.
        }
        return out
    }

    /// `sum(d_i * (11 - i))` for `i = 1...10` (1-indexed), `X` = 10; valid
    /// iff divisible by 11. Standard ISBN-10 checksum.
    private static func isValidISBN10Checksum(_ digits10: String) -> Bool {
        let chars = Array(digits10)
        guard chars.count == 10 else { return false }
        var sum = 0
        for (i, ch) in chars.enumerated() {
            let digit: Int
            if ch == "X" {
                guard i == 9 else { return false } // X only valid as the final check digit
                digit = 10
            } else if let d = ch.wholeNumberValue, ch.isNumber {
                digit = d
            } else {
                return false
            }
            sum += digit * (10 - i)
        }
        return sum % 11 == 0
    }

    /// Alternating 1/3 weights starting at weight 1; valid iff the total
    /// (including the check digit) is divisible by 10. Standard ISBN-13 /
    /// EAN-13 checksum.
    private static func isValidISBN13Checksum(_ digits13: String) -> Bool {
        let digits = digits13.compactMap(\.wholeNumberValue)
        guard digits.count == 13 else { return false }
        var sum = 0
        for (i, d) in digits.enumerated() {
            sum += d * (i % 2 == 0 ? 1 : 3)
        }
        return sum % 10 == 0
    }

    /// Drops the ISBN-10 check digit, prepends `978`, and recomputes a
    /// fresh ISBN-13 check digit over the resulting 12 digits.
    private static func convertISBN10ToISBN13(_ digits10: String) -> String? {
        let first9 = Array(digits10.prefix(9)).compactMap(\.wholeNumberValue)
        guard first9.count == 9 else { return nil }
        let twelve = [9, 7, 8] + first9
        var sum = 0
        for (i, d) in twelve.enumerated() {
            sum += d * (i % 2 == 0 ? 1 : 3)
        }
        let check = (10 - (sum % 10)) % 10
        return twelve.map(String.init).joined() + String(check)
    }
}
