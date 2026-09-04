import CoreGraphics
import Foundation
import SpineCore

// Lazy/visible-first + capped-concurrency OCR scheduling per
// docs/BOOK_ID_IOS_PIPELINE.md §Non-functional requirements: "OCR lazily
// (visible / user-tapped spines first, not all at once), cap concurrent
// VNImageRequestHandlers ... so pan/zoom and re-scan don't re-OCR."
//
// The still-image app pipeline has no live viewport to report which
// spines are actually visible, so "visible-first" is approximated by
// detection area (bigger box -> more likely to be a prominent, in-focus
// spine worth reading first) — a reasonable stand-in that also happens to
// be the same signal a live camera's "closest to center / largest on
// screen" heuristic would use.

/// One spine ready for OCR: its detection (for orientation routing) and
/// its already-computed `uprightWarp` crop.
public struct OCRJob {
    public let detection: OBBDetection
    public let crop: CGImage

    public init(detection: OBBDetection, crop: CGImage) {
        self.detection = detection
        self.crop = crop
    }
}

/// Runs `OCROrientationRouter.recognize` over many spine crops with a
/// capped number of concurrently in-flight `VNImageRequestHandler`s
/// (bounding memory/CPU on a dense shelf of 50-100 spines) and a
/// visible-first launch order (largest detections first). Streams each
/// result to `onResult` as soon as it's ready, so a caller can render
/// incrementally instead of blocking on the whole batch.
public struct CappedConcurrencyOCRScheduler {
    public var router: OCROrientationRouter
    public var maxConcurrency: Int

    public init(router: OCROrientationRouter, maxConcurrency: Int = 4) {
        self.router = router
        self.maxConcurrency = max(1, maxConcurrency)
    }

    /// Visible-first priority order: descending detection area. Exposed
    /// standalone so callers/tests can inspect the intended launch order
    /// without running OCR.
    public static func prioritized(_ jobs: [OCRJob]) -> [OCRJob] {
        jobs.sorted { $0.detection.w * $0.detection.h > $1.detection.w * $1.detection.h }
    }

    /// Runs every job. `onResult` is called from a task-group child task
    /// (not necessarily the caller's task) as each job completes; jobs
    /// whose OCR throws are silently skipped, matching the `try?` used at
    /// existing (sequential) call sites. Launch order follows
    /// `prioritized(_:)`; with `maxConcurrency == 1` this makes completion
    /// order deterministic and equal to priority order too.
    public func run(jobs: [OCRJob], onResult: @escaping @Sendable (UUID, SpineOCRResult) -> Void) async {
        let ordered = Self.prioritized(jobs)
        let router = self.router
        var iterator = ordered.makeIterator()

        await withTaskGroup(of: Void.self) { group in
            func launchNext() {
                guard let job = iterator.next() else { return }
                let detectionId = job.detection.id
                group.addTask {
                    guard let result = try? router.recognize(crop: job.crop, detection: job.detection) else { return }
                    onResult(detectionId, result)
                }
            }
            for _ in 0..<maxConcurrency { launchNext() }
            while await group.next() != nil {
                launchNext()
            }
        }
    }
}
