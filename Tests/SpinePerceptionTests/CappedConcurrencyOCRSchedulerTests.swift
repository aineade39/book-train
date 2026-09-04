import CoreGraphics
import ImageIO
import XCTest

import SpineCore
@testable import SpinePerception

/// Thread-safe recorder shared across concurrent task-group children so
/// tests can assert both *launch order* and *peak concurrency* without
/// depending on real Vision timing.
private final class OrderAndConcurrencyRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var order: [Int] = []
    private var active = 0
    private var peakConcurrent = 0

    func begin(_ tag: Int) {
        lock.lock()
        active += 1
        peakConcurrent = max(peakConcurrent, active)
        order.append(tag)
        lock.unlock()
    }

    func end() {
        lock.lock()
        active -= 1
        lock.unlock()
    }

    var callOrder: [Int] { lock.lock(); defer { lock.unlock() }; return order }
    var maxConcurrent: Int { lock.lock(); defer { lock.unlock() }; return peakConcurrent }
}

/// Thread-safe counter/collector for `onResult` callbacks, which may run
/// on different task-group child tasks.
private final class ResultCollector<Element>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [Element] = []

    func append(_ item: Element) {
        lock.lock()
        items.append(item)
        lock.unlock()
    }

    var count: Int { lock.lock(); defer { lock.unlock() }; return items.count }
    var all: [Element] { lock.lock(); defer { lock.unlock() }; return items }
}

/// Recognizer that tags every call by the crop's width (each test gives
/// every job a distinctly-sized crop so calls are attributable to a job)
/// and optionally sleeps to simulate real OCR latency for concurrency-cap
/// assertions.
private struct RecordingRecognizer: TextRecognizer {
    let recorder: OrderAndConcurrencyRecorder
    let delaySeconds: Double

    func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation] {
        recorder.begin(image.width)
        if delaySeconds > 0 { Thread.sleep(forTimeInterval: delaySeconds) }
        recorder.end()
        return [fullCropObservation(text: "spine\(image.width)", confidence: 0.9)]
    }
}

final class CappedConcurrencyOCRSchedulerTests: XCTestCase {
    private func det(w: Double, h: Double) -> OBBDetection {
        OBBDetection(cx: 0, cy: 0, w: w, h: h, angle: 0, conf: 0.9)
    }

    private func job(area side: Int, w: Double, h: Double) -> OCRJob {
        OCRJob(detection: det(w: w, h: h), crop: makeSolidCGImage(width: side, height: side))
    }

    // MARK: - Priority order

    func testPrioritizedSortsLargestDetectionAreaFirst() {
        let small = job(area: 10, w: 10, h: 40)   // area 400
        let big = job(area: 30, w: 100, h: 100)   // area 10000
        let medium = job(area: 20, w: 50, h: 50)  // area 2500
        let ordered = CappedConcurrencyOCRScheduler.prioritized([small, big, medium])
        XCTAssertEqual(ordered.map { Int($0.detection.w * $0.detection.h) }, [10000, 2500, 400])
    }

    func testSequentialRunVisitsJobsInPriorityOrder() async {
        let recorder = OrderAndConcurrencyRecorder()
        let router = OCROrientationRouter(recognizer: RecordingRecognizer(recorder: recorder, delaySeconds: 0))
        let scheduler = CappedConcurrencyOCRScheduler(router: router, maxConcurrency: 1)

        let jobs = [
            job(area: 11, w: 10, h: 10),   // smallest, area 100
            job(area: 31, w: 100, h: 100), // largest, area 10000
            job(area: 21, w: 50, h: 50),   // middle, area 2500
        ]

        let results = ResultCollector<UUID>()
        await scheduler.run(jobs: jobs) { id, _ in results.append(id) }

        XCTAssertEqual(results.count, 3)
        // Every job's crop width appears; the *first* appearance of each
        // width tells us launch order, which for maxConcurrency == 1 is
        // also completion order.
        let firstOccurrences = recorder.callOrder.reduce(into: [Int]()) { acc, tag in
            if !acc.contains(tag) { acc.append(tag) }
        }
        XCTAssertEqual(firstOccurrences, [31, 21, 11], "largest-area job should be scheduled first")
        XCTAssertEqual(recorder.maxConcurrent, 1)
    }

    // MARK: - Concurrency cap

    func testConcurrencyNeverExceedsMaxConcurrencyEvenWithManySlowJobs() async {
        let recorder = OrderAndConcurrencyRecorder()
        let router = OCROrientationRouter(recognizer: RecordingRecognizer(recorder: recorder, delaySeconds: 0.05))
        let scheduler = CappedConcurrencyOCRScheduler(router: router, maxConcurrency: 3)

        let jobs = (0..<9).map { i in job(area: 40 + i, w: Double(10 + i), h: Double(10 + i)) }

        let results = ResultCollector<UUID>()
        await scheduler.run(jobs: jobs) { id, _ in results.append(id) }

        XCTAssertEqual(results.count, 9)
        XCTAssertGreaterThan(recorder.maxConcurrent, 1, "should actually run concurrently, not degrade to sequential")
        XCTAssertLessThanOrEqual(recorder.maxConcurrent, 3, "must never exceed the configured cap")
    }

    // MARK: - Failing jobs are skipped, not fatal

    func testThrowingRecognizerSkipsThatJobWithoutFailingOthers() async {
        struct ThrowingRecognizer: TextRecognizer {
            func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> [RecognizedTextObservation] {
                throw NSError(domain: "test", code: 1)
            }
        }
        let router = OCROrientationRouter(recognizer: ThrowingRecognizer())
        let scheduler = CappedConcurrencyOCRScheduler(router: router, maxConcurrency: 2)
        let jobs = [job(area: 10, w: 10, h: 10), job(area: 20, w: 20, h: 20)]

        let results = ResultCollector<UUID>()
        await scheduler.run(jobs: jobs) { id, _ in results.append(id) }
        XCTAssertTrue(results.all.isEmpty)
    }
}
