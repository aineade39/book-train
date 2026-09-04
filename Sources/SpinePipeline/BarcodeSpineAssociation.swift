import CoreGraphics
import Foundation
import SpineCore

/// Geometric association of full-frame barcode detections to spine OBBs,
/// per the locked "Book ID OCR gains" plan §D (ISBN / barcode) "Spine
/// association": "scene-pixel barcode center; attach if inside OBB or
/// distance <= clamp(0.5 * spineShortEdge, 24, 96) px; ties within 5px ->
/// unassociated." Only meaningful once spine detection has actually run --
/// `SpineIdentificationEngine.run`'s frame-level unique-ISBN short circuit
/// skips detection entirely and never calls this.
///
/// Pure geometry (no Vision/SQLite), so it's exercised directly by
/// `SpinePipelineTests` without a live catalog or camera frame.
enum BarcodeSpineAssociation {
    /// `clamp(0.5 * shortEdge, 24, 96)` -- the max attach distance (px) for
    /// a barcode center that doesn't land inside any spine's OBB.
    static func attachRadius(shortEdge: Double) -> Double {
        min(max(0.5 * shortEdge, 24), 96)
    }

    /// Maps each `detections` entry (by `OBBDetection.id`) to the single
    /// `Payload` whose scene-pixel point best associates with it. A point
    /// inside a spine's OBB wins outright over any distance-based
    /// candidate; otherwise the nearest spine within its own
    /// `attachRadius` wins, unless a second spine centroid is within 5px
    /// of that same distance (a genuine tie is left unassociated rather
    /// than guessed).
    static func associate<Payload>(
        points: [(payload: Payload, scenePoint: CGPoint)],
        detections: [OBBDetection]
    ) -> [UUID: Payload] {
        var result: [UUID: Payload] = [:]
        for entry in points {
            guard let winner = bestMatch(for: entry.scenePoint, among: detections) else { continue }
            result[winner.id] = entry.payload
        }
        return result
    }

    private static func bestMatch(for point: CGPoint, among detections: [OBBDetection]) -> OBBDetection? {
        if let contained = detections.first(where: { contains($0, point) }) {
            return contained
        }

        let distances = detections
            .map { ($0, dist(point, CGPoint(x: $0.cx, y: $0.cy))) }
            .sorted { $0.1 < $1.1 }
        guard let nearest = distances.first else { return nil }

        let radius = attachRadius(shortEdge: min(nearest.0.w, nearest.0.h))
        guard nearest.1 <= radius else { return nil }

        if distances.count > 1, abs(distances[1].1 - nearest.1) <= 5 {
            return nil
        }
        return nearest.0
    }

    /// Whether `point` (scene pixels) falls within `detection`'s rotated
    /// rect -- rotates `point` into the OBB's local frame and tests
    /// against its half-extents.
    private static func contains(_ detection: OBBDetection, _ point: CGPoint) -> Bool {
        let dx = Double(point.x) - detection.cx
        let dy = Double(point.y) - detection.cy
        let c = cos(-detection.angle), s = sin(-detection.angle)
        let localX = c * dx - s * dy
        let localY = s * dx + c * dy
        return abs(localX) <= detection.w / 2 && abs(localY) <= detection.h / 2
    }
}
