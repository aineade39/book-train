import Foundation
import FoundationModels

// Availability-gated FM entry point per docs/BOOK_ID_IOS_PIPELINE.md
// §Foundation Models integration: "Availability gate: branch on
// `SystemLanguageModel.default.availability` (the runtime source of
// truth) ... Fallback: Vision-only + user pick from top matches on
// unsupported devices." Callers never need to check OS version or device
// tier themselves -- `isAvailable` folds all of that in, and `extract`
// degrades to `nil` (never throws) so a failed/unavailable FM call always
// looks exactly like "FM wasn't consulted for this spine".
@available(iOS 26.0, macOS 26.0, *)
public struct SpineReasoningService: Sendable {
    public var model: SystemLanguageModel

    public init(model: SystemLanguageModel = .default) {
        self.model = model
    }

    /// The gate every other member exists behind -- `false` folds in OS
    /// version, device eligibility, Apple Intelligence enablement, and
    /// region, per the doc's "not a hardcoded chip check" guidance.
    public var isAvailable: Bool {
        model.availability == .available
    }

    static let instructions = """
        You clean up noisy OCR text read from a single book spine, \
        photographed at an angle and possibly rotated. Extract the book's \
        title and author. Fix obvious OCR letter substitutions, but do not \
        invent or translate text that is not plausibly present in the \
        source. Only report an ISBN if one is literally present in the OCR \
        text -- never guess or look one up.
        """

    /// Runs one FM call over `assembledText` (the OCR orientation router's
    /// reading-order output for a single spine). Returns `nil` -- never
    /// throws -- when the model isn't available, the call fails for any
    /// reason (guardrail violation, malformed response, ...), or it
    /// doesn't finish within `timeout`; per the doc's Vision-only
    /// fallback, callers should treat that exactly like "FM wasn't
    /// consulted" and keep using the raw OCR text.
    ///
    /// The `timeout` race is deliberately implemented so the *caller*
    /// never waits past it, even if `LanguageModelSession.respond` itself
    /// doesn't return promptly on cancellation (observed in practice: a
    /// contended/busy on-device model can leave a `respond` call hung far
    /// longer than its own guardrail/latency budget) -- a hang here would
    /// be strictly worse than a slow miss, since it stalls the entire
    /// per-spine result the caller is waiting on.
    public func extract(from assembledText: String, timeout: Duration = .seconds(12)) async -> SpineExtraction? {
        guard isAvailable else { return nil }
        let session = LanguageModelSession(model: model, instructions: Self.instructions)
        let prompt = "OCR text from one book spine: \"\(assembledText)\""

        return await withCheckedContinuation { (continuation: CheckedContinuation<SpineExtraction?, Never>) in
            let once = ResumeOnce(continuation)
            // Unstructured (not a task group child) so a loser task that
            // ignores cancellation can't hold up this function's return --
            // see the doc comment above.
            Task {
                do {
                    let response = try await session.respond(to: prompt, generating: SpineExtraction.self)
                    once.resume(with: response.content)
                } catch {
                    once.resume(with: nil)
                }
            }
            Task {
                try? await Task.sleep(for: timeout)
                once.resume(with: nil)
            }
        }
    }
}

/// Ensures a `CheckedContinuation` is resumed exactly once even when two
/// independent, uncoordinated tasks (the real call and its timeout) race
/// to resume it -- resuming twice is a fatal error.
private final class ResumeOnce<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var didResume = false
    private let continuation: CheckedContinuation<T, Never>

    init(_ continuation: CheckedContinuation<T, Never>) {
        self.continuation = continuation
    }

    func resume(with value: T) {
        lock.lock()
        defer { lock.unlock() }
        guard !didResume else { return }
        didResume = true
        continuation.resume(returning: value)
    }
}
