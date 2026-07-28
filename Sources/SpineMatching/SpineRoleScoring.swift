import CoreGraphics
import Foundation

// Soft title/author/other geometry roles per the locked "Book ID OCR
// gains" plan §B (Geometry title / author / other): line height +
// vertical position on the spine predict whether a line is more likely
// the title, the author, or neither ("other": publisher, ISBN, tagline,
// series blurb, ...), without ever hard-committing to "tallest line only".

/// Soft per-line role probabilities, always summing to `1`.
public struct SpineRoleScores: Equatable {
    public let title: Double
    public let author: Double
    public let other: Double
}

public enum SpineRoleScoring {
    /// `lineHeight = A / max(w, 1)` where `A` is the quad's polygon area
    /// (shoelace) and `w` is the longer of the two candidate "width" edge
    /// pairs -- i.e. treat the quad as an approximate parallelogram and
    /// solve `height = area / width` using whichever edge pair is longer
    /// (handles both a landscape line, where the top/bottom edges are the
    /// long ones, and a portrait/rotated line, where the left/right edges
    /// are). Deliberately `A / w`, not `2A / w`: for an axis-aligned
    /// rectangle the shoelace formula already returns the true `w * h`
    /// area (it isn't a "half sum" that still needs doubling), so a `2A/w`
    /// term would silently double every line's apparent height.
    static func lineHeight(of line: SpineTextLine) -> Double {
        let q0 = line.topLeft, q1 = line.topRight, q2 = line.bottomRight, q3 = line.bottomLeft
        let area = polygonArea([q0, q1, q2, q3])
        let widthCandidate1 = (dist(q1, q0) + dist(q2, q3)) / 2
        let widthCandidate2 = (dist(q3, q0) + dist(q2, q1)) / 2
        let w = max(widthCandidate1, widthCandidate2)
        return area / max(w, 1)
    }

    /// `lineHeight` as a fraction of the crop's own height.
    public static func relativeHeight(of line: SpineTextLine) -> Double {
        clamp(lineHeight(of: line) / max(line.cropHeight, 1), 0, 1)
    }

    /// Mean corner `y` (crop-local, top-left origin) as a fraction of crop
    /// height -- `0` at the top of the spine crop, `1` at the bottom.
    public static func spinePosition(of line: SpineTextLine) -> Double {
        let meanY = (line.topLeft.y + line.topRight.y + line.bottomRight.y + line.bottomLeft.y) / 4
        return clamp(Double(meanY) / max(line.cropHeight, 1), 0, 1)
    }

    /// Trapezoid membership function: `0` at/below `a` and at/above `d`,
    /// ramping linearly to `1` over `[a, b]`, flat `1` over `[b, c]`,
    /// ramping back down to `0` over `[c, d]`.
    private static func band(_ x: Double, _ a: Double, _ b: Double, _ c: Double, _ d: Double) -> Double {
        if x <= a || x >= d { return 0 }
        if x < b { return (x - a) / (b - a) }
        if x <= c { return 1 }
        return (d - x) / (d - c)
    }

    /// Raw (pre-normalization) title/author/other scores from line height
    /// and vertical position alone, before the text-content boosts below.
    private static func rawScores(relativeHeight h: Double, spinePosition p: Double) -> (title: Double, author: Double, other: Double) {
        let titleRaw = 0.05
            + 0.65 * band(h, 0.020, 0.060, 0.160, 0.260)
            + 0.30 * band(p, 0, 0.08, 0.70, 0.92)
        let authorRaw = 0.05
            + 0.60 * band(h, 0.012, 0.032, 0.105, 0.170)
            + 0.30 * band(p, 0.28, 0.48, 0.88, 0.98)
        let otherRaw = 0.05
            + 0.55 * band(h, 0.004, 0.010, 0.045, 0.085)
            + 0.30 * max(band(p, 0, 0.02, 0.20, 0.38), band(p, 0.72, 0.90, 0.98, 1))
        return (titleRaw, authorRaw, otherRaw)
    }

    /// `true` when `text` looks like non-title/author boilerplate (ISBN,
    /// URL, copyright mark, or a digit run of 4+) -- boosts `other`.
    private static func looksLikeExclusion(_ text: String) -> Bool {
        let lowered = text.lowercased()
        if lowered.contains("isbn") || lowered.contains("http") || lowered.contains("www") || text.contains("\u{00A9}") {
            return true
        }
        var run = 0
        for ch in text {
            if ch.isNumber {
                run += 1
                if run >= 4 { return true }
            } else {
                run = 0
            }
        }
        return false
    }

    /// `true` when `text` reads like a person's name -- an optional
    /// "by"/"written by"/"author" prefix followed by 2-4 alphabetic words
    /// (each 2-24 chars, no digits), where either the prefix was present
    /// or at least 2 of those words are Capitalized and the line isn't
    /// ALL CAPS. Boosts `author`.
    private static func looksLikePersonName(_ text: String) -> Bool {
        var words = text.split(separator: " ").map(String.init)
        guard !words.isEmpty else { return false }

        var hadPrefix = false
        let prefixes = ["by", "written by", "author"]
        let lowered = text.lowercased()
        for prefix in prefixes where lowered.hasPrefix(prefix) {
            hadPrefix = true
            let remainder = String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
            words = remainder.split(separator: " ").map(String.init)
            break
        }

        guard (2...4).contains(words.count) else { return false }
        guard words.allSatisfy({ word in
            (2...24).contains(word.count) && word.allSatisfy(\.isLetter)
        }) else { return false }

        if hadPrefix { return true }

        let isAllCaps = text == text.uppercased() && text != text.lowercased()
        let capitalizedCount = words.filter { $0.first.map(\.isUppercase) ?? false }.count
        return capitalizedCount >= 2 && !isAllCaps
    }

    /// Full soft role computation for one line: raw geometric scores,
    /// text-content boosts, then normalized to sum to `1`.
    public static func roleScores(for line: SpineTextLine) -> SpineRoleScores {
        let h = relativeHeight(of: line)
        let p = spinePosition(of: line)
        var (titleRaw, authorRaw, otherRaw) = rawScores(relativeHeight: h, spinePosition: p)

        if looksLikeExclusion(line.text) {
            otherRaw *= 1.50
        }
        if looksLikePersonName(line.text) {
            authorRaw *= 1.25
        }

        let total = titleRaw + authorRaw + otherRaw
        guard total > 0 else { return SpineRoleScores(title: 1.0 / 3, author: 1.0 / 3, other: 1.0 / 3) }
        return SpineRoleScores(title: titleRaw / total, author: authorRaw / total, other: otherRaw / total)
    }

    private static func clamp(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
        min(max(x, lo), hi)
    }

    private static func polygonArea(_ pts: [CGPoint]) -> Double {
        guard pts.count >= 3 else { return 0 }
        var area = 0.0
        for i in 0..<pts.count {
            let p = pts[i], q = pts[(i + 1) % pts.count]
            area += Double(p.x * q.y - q.x * p.y)
        }
        return abs(area) / 2
    }

    private static func dist(_ a: CGPoint, _ b: CGPoint) -> Double {
        Foundation.hypot(Double(a.x - b.x), Double(a.y - b.y))
    }
}
