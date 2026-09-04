import Foundation

// Per-detection-id result cache per docs/BOOK_ID_IOS_PIPELINE.md
// §Non-functional requirements: "cache OCR + match results keyed by
// detection id so pan/zoom and re-scan don't re-OCR" and §Detection &
// isolation: "Preserve stable detection identity (id) through multi-pass
// merge for UI tracking and re-scan."
//
// An `actor` (not a plain dictionary) because the capped-concurrency OCR
// scheduler's `onResult` callback can fire from different task-group
// child tasks — a cache callers check-then-populate around concurrent OCR
// needs its own serialization, not just the scheduler's.

/// Thread-safe cache from a stable detection `id` to its OCR result,
/// generic over the value so the app layer can either cache
/// `SpineOCRResult` alone or a richer record (OCR + match decision)
/// without this type needing to depend on `SpineMatching`/`SpineCatalog`.
public actor SpineResultCache<Value> {
    private var storage: [UUID: Value] = [:]

    public init() {}

    public func value(for id: UUID) -> Value? {
        storage[id]
    }

    public func store(_ value: Value, for id: UUID) {
        storage[id] = value
    }

    /// Returns the cached value for `id` if present; otherwise computes it
    /// with `compute`, stores it, and returns it. `compute` runs outside
    /// the actor's isolation (it's a synchronous closure evaluated inline,
    /// not scheduled onto the actor), so cache population never blocks
    /// other detections' lookups on this one's work.
    public func value(for id: UUID, computeIfMissing compute: () throws -> Value) rethrows -> Value {
        if let cached = storage[id] { return cached }
        let computed = try compute()
        storage[id] = computed
        return computed
    }

    public var count: Int { storage.count }

    public func removeAll() {
        storage.removeAll()
    }
}
