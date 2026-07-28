import CoreGraphics
import Foundation
import SpineCatalog
import SpineCore
import SpineMatching
import SpinePerception
import SpineReasoning

// Shared detect -> isolate -> read -> normalize -> match -> accept
// orchestration per the locked "Book ID OCR gains" plan §H ("spine-id: In
// scope -- shared package APIs for ISBN, role retrieve/rerank, accept; CLI
// and app call the same code") -- this is that shared code. Both `spine-id`
// (this package's own macOS CLI) and the iOS/macOS app's
// `SpineIdentificationPipeline` build a `SpineIdentificationEngine` and
// adapt `SpinePipelineResult` to their own output shape (JSON vs. SwiftUI
// models) instead of re-implementing the control flow twice.
//
// Deliberately has no AppKit/UIKit/SwiftUI dependency (mirrors every other
// library in this package) so it stays usable from both.

/// Mirrors `spine-id`'s JSON `source` field. `.barcode` covers both the
/// frame-level unique-ISBN short circuit (§D: "skip detect/OCR/FM/match
/// entirely") and the per-spine override `BarcodeSpineAssociation` applies
/// once a barcode is geometrically associated with a detected spine (§D:
/// "vs OCR: Unique ISBN always wins... OCR never overrides").
public enum SpinePipelineSource: String, Equatable {
    case ocr
    case fmAssisted = "fm-assisted"
    case barcode
}

/// One candidate presented on `.needsConfirmation`, with whatever score
/// motivated its inclusion -- a fuzzy `matchRoleAware` rerank score, or a
/// flat `100` for every candidate in a barcode-sourced multi-work ISBN hit
/// (§D: "no popularity auto-pick" -- there's no fuzzy signal to rank them
/// by, so all tie).
public struct SpinePipelineCandidate: Equatable {
    public let candidate: CatalogCandidate
    public let score: Double

    public init(candidate: CatalogCandidate, score: Double) {
        self.candidate = candidate
        self.score = score
    }
}

/// The engine's own initial decision states -- deliberately narrower than
/// the app's `SpineMatchDecision`, which layers `.userConfirmed`/
/// `.userRejected` on top of `.needsConfirmation` once a person acts in
/// `ConfirmSheet`. That's UI-session state, not something this engine (or
/// `spine-id`) ever produces.
public enum SpinePipelineDecision: Equatable {
    case autoAccepted(title: String, author: String, score: Double)
    case needsConfirmation(candidates: [SpinePipelineCandidate])
    case noMatch
    case didNotPassQualityGate
}

/// One detected spine, all the way through read -> normalize -> match.
public struct SpinePipelineSpine: Identifiable, Equatable {
    public let id: UUID
    public let detection: OBBDetection
    public let assembledText: String
    public let ocrQualityScore: Double
    public let decision: SpinePipelineDecision
    public let source: SpinePipelineSource
    /// Score gap between the winning work and the best distinct-work
    /// runner-up (`BookCatalog.matchRoleAware`'s `AcceptOutcome.margin`,
    /// §rerank-telemetry: "emit margin") -- `nil` when there was no match
    /// attempt, no distinct-work runner-up to measure against, or the
    /// decision came from a barcode short circuit/override rather than
    /// fuzzy rerank.
    public let matchMargin: Double?

    public static func == (lhs: SpinePipelineSpine, rhs: SpinePipelineSpine) -> Bool {
        lhs.id == rhs.id && lhs.assembledText == rhs.assembledText && lhs.ocrQualityScore == rhs.ocrQualityScore
            && lhs.decision == rhs.decision && lhs.source == rhs.source && lhs.matchMargin == rhs.matchMargin
    }
}

/// One end-to-end `SpineIdentificationEngine.run` over a single captured
/// frame.
public struct SpinePipelineResult {
    public let captureSharpness: Double
    public let captureExposure: Double
    /// Whether `CaptureQualityGate`'s cheap frame-level proxies (sharpness,
    /// exposure) cleared their floors -- **advisory only**. `false` does
    /// *not* mean detection was skipped; `spines` may still be non-empty
    /// (and often is -- see `SpineIdentificationEngine.run`'s doc comment
    /// for the real-world regression that made this advisory rather than
    /// a hard veto). Callers use this to show a "photo may be low
    /// quality" hint alongside whatever spines were actually found, never
    /// to suppress results.
    public let capturePassed: Bool
    public let isbnBarcodes: [String]
    public let spines: [SpinePipelineSpine]
}

/// A Foundation Model's cleaned title/author for one escalated spine (§G).
private struct FMOverride {
    let title: String
    let author: String
}

public final class SpineIdentificationEngine {
    private let detector: SpineDetector
    private let barcodeReader: BarcodeReader
    private let captureGate: CaptureQualityGate
    private let ocrQualityGate: OCRQualityGate
    /// Per-instance cache from a spine's stable detection `id` to its OCR
    /// result, per docs/BOOK_ID_IOS_PIPELINE.md §Non-functional
    /// requirements: "cache OCR + match results keyed by detection id so
    /// pan/zoom and re-scan don't re-OCR."
    private let ocrCache = SpineResultCache<SpineOCRResult>()

    public var detectionOptions: DetectionOptions
    public var denseShelfOptions: DenseShelfDetectionOptions
    public var acceptPolicy: AcceptPolicy
    /// Bounds how many `VNImageRequestHandler`s run concurrently on a dense
    /// shelf (see `CappedConcurrencyOCRScheduler`).
    public var maxConcurrentOCR: Int
    /// Whether to escalate a rate-limited subset of hard-case (marginal
    /// OCR quality score) spines to the on-device Foundation Model before
    /// matching (see `runFMEscalation`).
    public var useFM: Bool
    public var fmEscalationPolicy: FMEscalationPolicy
    /// §C: production `recognitionLanguages` pinned to `["en-US"]` -- a
    /// caller only overrides this for the OCR Mac vs iOS parity harness,
    /// never for production matching.
    public var recognitionLanguages: [String]

    public init(
        detector: SpineDetector,
        barcodeReader: BarcodeReader = VisionBarcodeReader(),
        captureGate: CaptureQualityGate = .default,
        ocrQualityGate: OCRQualityGate = .default,
        detectionOptions: DetectionOptions = DetectionOptions(confidenceThreshold: 0.15, iouThreshold: 0.45, maxDetections: 500),
        denseShelfOptions: DenseShelfDetectionOptions = .default,
        acceptPolicy: AcceptPolicy = AcceptPolicy(acceptThreshold: 90, marginThreshold: 8, topN: 5),
        maxConcurrentOCR: Int = 4,
        useFM: Bool = true,
        fmEscalationPolicy: FMEscalationPolicy = .default,
        recognitionLanguages: [String] = ["en-US"]
    ) {
        self.detector = detector
        self.barcodeReader = barcodeReader
        self.captureGate = captureGate
        self.ocrQualityGate = ocrQualityGate
        self.detectionOptions = detectionOptions
        self.denseShelfOptions = denseShelfOptions
        self.acceptPolicy = acceptPolicy
        self.maxConcurrentOCR = maxConcurrentOCR
        self.useFM = useFM
        self.fmEscalationPolicy = fmEscalationPolicy
        self.recognitionLanguages = recognitionLanguages
    }

    public func run(on image: CGImage, catalog: BookCatalog?) async throws -> SpinePipelineResult {
        // §D/§F: full-frame barcode detect runs first, ahead of the capture
        // gate -- "even if capture gate would fail" a unique ISBN hit must
        // still short-circuit straight to an accepted result.
        let isbnHits = detectISBNHits(in: image, catalog: catalog)

        if let shortCircuit = uniqueWorkShortCircuit(isbnHits: isbnHits, image: image) {
            return shortCircuit
        }

        let captureScore = captureGate.score(image)
        let capturePassed = captureGate.passes(image)

        // `capturePassed`/`captureScore` are advisory only -- surfaced to
        // the caller (UI banner, telemetry) but never used to skip
        // detection. A real-world regression proved a hard veto here is
        // unsafe: two genuine bookshelf photos (dim-but-legible rooms,
        // exposure ~0.06-0.09 against the 0.12 floor, sharpness > 0.98)
        // were silently reduced to zero spines even though the identical
        // frames produced dozens of correctly-matched spines once
        // detect/OCR/match actually ran. `CaptureQualityGate`'s cheap
        // frame-level proxies (global mean luminance, Laplacian variance)
        // are useful hints but not reliable predictors of whether the
        // *content* is legible -- see `runOCROnly`'s doc comment, which
        // made the same call for the OCR parity harness before this class
        // did for production. Single-shot detect, escalating to the
        // tiled/jigsaw re-infer pass only once the shelf is dense enough
        // to risk neighbor bleed or missed spines (see `denseShelfDetect`
        // / docs/BOOK_ID_IOS_PIPELINE.md §Delivery sequencing step 2).
        let denseResult = try denseShelfDetect(
            image: image,
            predict: { [detector, detectionOptions] img in try detector.predict(img, options: detectionOptions) },
            options: detectionOptions,
            denseOptions: denseShelfOptions
        )
        let detections = denseResult.detections

        // Build one OCR job per detection up front so the scheduler can
        // launch its visible-first (largest-area-first), capped-concurrency
        // pass over all of them at once instead of one spine at a time.
        var jobsByID: [UUID: OCRJob] = [:]
        for detection in detections {
            guard let crop = uprightWarp(of: detection, in: image) else { continue }
            jobsByID[detection.id] = OCRJob(detection: detection, crop: crop)
        }

        let ocrResults = await runOCR(jobs: Array(jobsByID.values), catalog: catalog)
        let fmQueryOverrides = await runFMEscalation(ocrResults: ocrResults)

        var spines: [SpinePipelineSpine] = []
        for detection in detections {
            guard let ocr = ocrResults[detection.id] else { continue }

            guard ocr.passedQualityGate, !ocr.assembledText.isEmpty else {
                spines.append(SpinePipelineSpine(
                    id: detection.id, detection: detection, assembledText: ocr.assembledText,
                    ocrQualityScore: ocr.qualityScore, decision: .didNotPassQualityGate, source: .ocr, matchMargin: nil
                ))
                continue
            }

            // §G: an FM override replaces the geometry-derived role
            // queries with FM's own clean title/author reading, but still
            // flows through the identical retrieve/rerank/accept path
            // (`matchDecision` below) as any other spine.
            let fmOverride = fmQueryOverrides[detection.id]
            let queries = fmOverride.map { SpineRoleQueryBuilder.build(fmTitle: $0.title, fmAuthor: $0.author) } ?? ocr.roleQueries
            let (decision, margin) = matchDecision(for: queries, catalog: catalog)
            spines.append(SpinePipelineSpine(
                id: detection.id, detection: detection, assembledText: ocr.assembledText,
                ocrQualityScore: ocr.qualityScore, decision: decision,
                source: fmOverride != nil ? .fmAssisted : .ocr, matchMargin: margin
            ))
        }

        spines = applyBarcodeSpineOverrides(isbnHits: isbnHits, detections: detections, image: image, spines: spines)

        return SpinePipelineResult(
            captureSharpness: captureScore.sharpness, captureExposure: captureScore.exposure,
            capturePassed: capturePassed, isbnBarcodes: isbnHits.map(\.isbn13), spines: spines
        )
    }

    /// A full-frame barcode payload that decoded to a checksum-valid ISBN,
    /// with whatever the catalog knows about that ISBN already looked up
    /// once up front -- shared by both the frame-level short circuit and
    /// the per-spine override pass so `lookupISBN` never runs twice for
    /// the same barcode.
    private struct ISBNHit {
        let barcode: DetectedBarcode
        let isbn13: String
        let candidates: [CatalogCandidate]
    }

    /// §D: "Strip separators; uppercase X; ISBN-10 or 13 only; checksum
    /// required; ... invalid discarded (no banner)" -- runs before the
    /// capture gate so a unique hit can bypass it entirely.
    private func detectISBNHits(in image: CGImage, catalog: BookCatalog?) -> [ISBNHit] {
        let barcodes = (try? barcodeReader.detectBarcodes(in: image))?.filter(\.looksLikeISBN) ?? []
        return barcodes.compactMap { barcode in
            guard let isbn13 = ISBN13(rawPayload: barcode.payload) else { return nil }
            let candidates = (try? catalog?.lookupISBN(isbn13.value)) ?? nil ?? []
            return ISBNHit(barcode: barcode, isbn13: isbn13.value, candidates: candidates)
        }
    }

    /// §D "Unique work": when the *entire frame* carries exactly one
    /// distinct valid ISBN and it maps to exactly one catalog work,
    /// short-circuit to `autoAccepted` and skip detect/OCR/FM/match
    /// entirely -- "even if capture gate would fail" (§F). A frame with no
    /// barcode, an unrecognized ISBN, or one mapping to multiple works
    /// (§D "Multi-work same ISBN") falls through to the normal pipeline,
    /// where `applyBarcodeSpineOverrides` gets another chance to resolve
    /// it once real spine geometry exists to associate against.
    private func uniqueWorkShortCircuit(isbnHits: [ISBNHit], image: CGImage) -> SpinePipelineResult? {
        let distinctISBNs = Set(isbnHits.map(\.isbn13))
        guard distinctISBNs.count == 1, let hit = isbnHits.first, hit.candidates.count == 1 else { return nil }
        let winner = hit.candidates[0]

        let fullFrame = OBBDetection(
            cx: Double(image.width) / 2, cy: Double(image.height) / 2,
            w: Double(image.width), h: Double(image.height), angle: 0, conf: 1
        )
        let spine = SpinePipelineSpine(
            id: fullFrame.id, detection: fullFrame, assembledText: "", ocrQualityScore: 1,
            decision: .autoAccepted(title: winner.title, author: winner.author, score: 100),
            source: .barcode, matchMargin: nil
        )
        return SpinePipelineResult(
            captureSharpness: 1, captureExposure: 1, capturePassed: true, isbnBarcodes: [hit.isbn13], spines: [spine]
        )
    }

    /// §D "Spine association" + "vs OCR": once real detections exist,
    /// geometrically associates each barcode's scene-pixel center to the
    /// nearest/containing spine OBB and overrides *that spine's* decision
    /// -- a unique-work ISBN always wins over whatever OCR/FM concluded
    /// ("OCR never overrides"); a multi-work ISBN narrows confirmation to
    /// just those candidates ("no popularity auto-pick"). A barcode with
    /// no catalog match, or one that doesn't associate to any spine
    /// (outside every attach radius, or a genuine 5px tie), leaves that
    /// spine's OCR-derived decision untouched.
    private func applyBarcodeSpineOverrides(
        isbnHits: [ISBNHit], detections: [OBBDetection], image: CGImage, spines: [SpinePipelineSpine]
    ) -> [SpinePipelineSpine] {
        guard !isbnHits.isEmpty, !detections.isEmpty else { return spines }
        let points = isbnHits.map { hit in
            (payload: hit, scenePoint: hit.barcode.sceneCenter(imageWidth: image.width, imageHeight: image.height))
        }
        let associations = BarcodeSpineAssociation.associate(points: points, detections: detections)
        guard !associations.isEmpty else { return spines }

        return spines.map { spine in
            guard let hit = associations[spine.id], !hit.candidates.isEmpty else { return spine }
            let decision: SpinePipelineDecision
            if hit.candidates.count == 1 {
                let winner = hit.candidates[0]
                decision = .autoAccepted(title: winner.title, author: winner.author, score: 100)
            } else {
                decision = .needsConfirmation(candidates: hit.candidates.map { SpinePipelineCandidate(candidate: $0, score: 100) })
            }
            return SpinePipelineSpine(
                id: spine.id, detection: spine.detection, assembledText: spine.assembledText,
                ocrQualityScore: spine.ocrQualityScore, decision: decision, source: .barcode, matchMargin: nil
            )
        }
    }

    /// Runs OCR over `jobs`, checking/populating the per-id cache around a
    /// capped-concurrency, visible-first scheduler pass for whatever isn't
    /// already cached.
    private func runOCR(jobs: [OCRJob], catalog: BookCatalog?) async -> [UUID: SpineOCRResult] {
        var results: [UUID: SpineOCRResult] = [:]
        var uncached: [OCRJob] = []
        for job in jobs {
            if let cached = await ocrCache.value(for: job.detection.id) {
                results[job.detection.id] = cached
            } else {
                uncached.append(job)
            }
        }

        guard !uncached.isEmpty else { return results }
        let fresh = await runOCRUncached(jobs: uncached, router: makeProductionRouter(catalog: catalog))
        for (id, result) in fresh {
            results[id] = result
            await ocrCache.store(result, for: id)
        }
        return results
    }

    /// Production router per §C (Vision knobs + customWords): pinned
    /// `en-US` (no on-device language auto-detect), authors-only
    /// `customWords` loaded from the bundled catalog, language correction
    /// always on. Rebuilt per call rather than cached -- `customWords()`
    /// is one small SQLite read, and this keeps the pipeline correct if
    /// the catalog is ever swapped mid-session.
    private func makeProductionRouter(catalog: BookCatalog?) -> OCROrientationRouter {
        let customWords = (try? catalog?.customWords()) ?? []
        let recognizer = VisionTextRecognizer(
            usesLanguageCorrection: true, recognitionLanguages: recognitionLanguages, customWords: customWords
        )
        return OCROrientationRouter(recognizer: recognizer, qualityGate: ocrQualityGate)
    }

    /// Capped-concurrency, visible-first scheduler pass with no cache
    /// read/write -- factored out of `runOCR` so a possibly-differently
    /// configured `router` (e.g. the OCR Mac vs iOS parity harness) can
    /// run without contaminating, or being contaminated by, the
    /// per-instance `ocrCache` `run()` relies on for pan/rescan continuity.
    private func runOCRUncached(jobs: [OCRJob], router: OCROrientationRouter) async -> [UUID: SpineOCRResult] {
        let scheduler = CappedConcurrencyOCRScheduler(router: router, maxConcurrency: maxConcurrentOCR)
        let collected = ResultBox<[UUID: SpineOCRResult]>([:])
        await scheduler.run(jobs: jobs) { id, result in
            collected.mutate { $0[id] = result }
        }
        return collected.value
    }

    /// Escalates hard-case spines to the on-device Foundation Model, per
    /// docs/BOOK_ID_IOS_PIPELINE.md §Foundation Models integration and the
    /// locked "Book ID OCR gains" plan §G -- runs *after* the OCR quality
    /// gate but *before* matching. Two independent triggers, unioned (§G:
    /// "Trigger when existing FMEscalationPolicy fires or rolesAmbiguous"):
    /// the existing rate-limited marginal-OCR-quality budget, and any
    /// spine geometry couldn't confidently assign a title/author role to.
    ///
    /// Both triggers still share `fmEscalationPolicy`'s one budget
    /// (worst-quality-first) -- docs/BOOK_ID_IOS_PIPELINE.md's own
    /// non-functional requirement is "rate-limited (top ~5% of hard
    /// cases)" for FM escalation *overall*, not per trigger, and
    /// `rolesAmbiguous` alone carries no rate limit of its own: a dense,
    /// hard-to-read shelf can trip it for most of the frame's spines at
    /// once. Each `SpineReasoningService.extract` call already races its
    /// own 12s timeout, but an unbounded escalation set -- even run at
    /// bounded concurrency -- would still turn into minutes of FM round
    /// trips for one capture.
    private func runFMEscalation(ocrResults: [UUID: SpineOCRResult]) async -> [UUID: FMOverride] {
        guard useFM else { return [:] }
        guard #available(iOS 26.0, macOS 26.0, *) else { return [:] }
        let service = SpineReasoningService()
        guard service.isAvailable else { return [:] }

        let passed = ocrResults.filter { $0.value.passedQualityGate && !$0.value.assembledText.isEmpty }
        guard !passed.isEmpty else { return [:] }
        let budget = fmEscalationPolicy.escalationBudget(forSpineCount: passed.count)
        guard budget > 0 else { return [:] }

        let escalationWorthy = passed.filter { _, ocr in
            fmEscalationPolicy.isHardCase(qualityScore: ocr.qualityScore) || ocr.roleQueries.rolesAmbiguous
        }
        let escalate = Set(
            escalationWorthy
                .sorted { $0.value.qualityScore < $1.value.qualityScore }
                .prefix(budget)
                .map(\.key)
        )
        guard !escalate.isEmpty else { return [:] }

        var pending = escalate.makeIterator()
        return await withTaskGroup(of: (UUID, FMOverride?).self) { group in
            func addNext() {
                guard let id = pending.next(), let ocr = passed[id] else { return }
                group.addTask {
                    guard let extraction = await service.extract(from: ocr.assembledText) else { return (id, nil) }
                    return (id, FMOverride(title: extraction.title, author: extraction.author))
                }
            }
            for _ in 0..<Self.maxConcurrentFM { addNext() }

            var overrides: [UUID: FMOverride] = [:]
            while let (id, override) = await group.next() {
                if let override { overrides[id] = override }
                addNext()
            }
            return overrides
        }
    }

    /// Bounds how many `SpineReasoningService.extract` calls run at once --
    /// separate from `maxConcurrentOCR` since FM escalation is a much
    /// smaller, rate-limited-by-policy subset of spines, not every
    /// detection on the shelf.
    private static let maxConcurrentFM = 3

    /// Retrieve -> field-aware rerank -> accept, via the single shared
    /// `BookCatalog.matchRoleAware` package API -- `AcceptPolicy` stays
    /// 90/8 by default; the margin `matchRoleAware` reports is surfaced
    /// for telemetry (§rerank-telemetry), not consumed by the decision
    /// itself.
    private func matchDecision(for queries: SpineRoleQueries, catalog: BookCatalog?) -> (SpinePipelineDecision, Double?) {
        guard let catalog else { return (.noMatch, nil) }
        guard let outcome = try? catalog.matchRoleAware(queries, acceptPolicy: acceptPolicy) else {
            return (.noMatch, nil)
        }
        switch outcome.decision {
        case .autoAccept(let winner):
            return (.autoAccepted(title: winner.candidate.title, author: winner.candidate.author, score: winner.score), outcome.margin)
        case .ambiguous(let top):
            let candidates = top.map { SpinePipelineCandidate(candidate: $0.candidate, score: $0.score) }
            return (.needsConfirmation(candidates: candidates), outcome.margin)
        case .noMatch:
            return (.noMatch, outcome.margin)
        }
    }
}

/// Lock-protected mutable box so `CappedConcurrencyOCRScheduler.run`'s
/// `onResult` callback -- which fires from concurrent task-group child
/// tasks, not just the calling task -- can safely accumulate into a
/// dictionary without each call needing to hop back onto an actor.
private final class ResultBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value

    init(_ initial: Value) { storage = initial }

    func mutate(_ body: (inout Value) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        body(&storage)
    }

    var value: Value { lock.lock(); defer { lock.unlock() }; return storage }
}
