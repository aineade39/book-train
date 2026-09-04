// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "book-train-swift",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [
        .library(name: "SpineCore", targets: ["SpineCore"]),
        .library(name: "SpinePerception", targets: ["SpinePerception"]),
        .library(name: "SpineMatching", targets: ["SpineMatching"]),
        .library(name: "SpineCatalog", targets: ["SpineCatalog"]),
        .library(name: "SpineReasoning", targets: ["SpineReasoning"]),
        .library(name: "SpinePipeline", targets: ["SpinePipeline"]),
        .executable(name: "bookspines", targets: ["bookspines"]),
        .executable(name: "layout-crops", targets: ["layout-crops"]),
        .executable(name: "spine-read", targets: ["spine-read"]),
        .executable(name: "catalog-build", targets: ["catalog-build"]),
        .executable(name: "book-match", targets: ["book-match"]),
        .executable(name: "spine-id", targets: ["spine-id"]),
        .executable(name: "ocr-quality-sweep", targets: ["ocr-quality-sweep"]),
        .executable(name: "zoom-compare", targets: ["zoom-compare"]),
    ],
    dependencies: [
        // FTS5 trigram catalog (SpineCatalog) — see docs/BOOK_ID_IOS_PIPELINE.md §Catalog matching.
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.9.0"),
    ],
    targets: [
        // MARK: - Libraries (macOS + iOS; no AppKit/UIKit — kept device-agnostic
        // per GEOMETRY.md and AGENTS.md so the iOS app and macOS CLIs/tests
        // share exactly one implementation).

        .target(name: "SpineCore"),

        // Pure-Swift normalization + fuzzy rerank + accept policy. No
        // dependencies, so it is trivially unit-testable and reusable from
        // SpineCatalog without pulling in GRDB/Vision.
        .target(name: "SpineMatching"),

        // Vision-backed OCR orientation routing, reading-order assembly,
        // barcode, and capture/OCR quality gates. Depends on SpineMatching
        // (still no GRDB/FoundationModels) so it can hand off
        // `SpineTextLine`s for role scoring / n-best confusion expansion
        // without a third module owning the glue.
        .target(name: "SpinePerception", dependencies: ["SpineCore", "SpineMatching"]),

        // GRDB + FTS5 trigram local catalog.
        .target(
            name: "SpineCatalog",
            dependencies: [
                "SpineMatching",
                .product(name: "GRDB", package: "GRDB.swift"),
            ]
        ),

        // FoundationModels `@Generable` hard-case extraction, availability-gated.
        .target(name: "SpineReasoning", dependencies: ["SpineMatching"]),

        // End-to-end detect -> isolate -> read -> normalize -> match ->
        // accept orchestration -- the locked "Book ID OCR gains" plan §H
        // ("spine-id: shared package APIs ... CLI and app call the same
        // code"). Depends on every other library so `spine-id` and the
        // iOS/macOS app's `SpineIdentificationPipeline` can both build a
        // `SpineIdentificationEngine` instead of each re-implementing the
        // barcode short circuit / detect / OCR / FM / match control flow.
        .target(
            name: "SpinePipeline",
            dependencies: ["SpineCore", "SpinePerception", "SpineMatching", "SpineCatalog", "SpineReasoning"]
        ),

        // MARK: - macOS CLIs (validation harnesses; mirror the existing
        // bookspines/layout-crops shape so every pipeline stage is
        // exercisable and testable on Mac before any iOS app work).

        .executableTarget(name: "bookspines", dependencies: ["SpineCore"]),
        .executableTarget(name: "layout-crops", dependencies: ["SpineCore"]),
        .executableTarget(name: "spine-read", dependencies: ["SpineCore", "SpinePerception"]),
        .executableTarget(name: "catalog-build", dependencies: ["SpineCatalog", "SpineMatching"]),
        .executableTarget(name: "book-match", dependencies: ["SpineCatalog", "SpineMatching"]),
        .executableTarget(
            name: "spine-id",
            dependencies: ["SpineCore", "SpinePerception", "SpineCatalog", "SpineMatching", "SpineReasoning", "SpinePipeline"]
        ),
        .executableTarget(name: "ocr-quality-sweep", dependencies: ["SpineCore", "SpinePerception"]),
        // Jigsaw-zoom Stage 0 comparison harness (single-shot vs layout-crops
        // vs, later, jigsaw-zoom) over the scenes corpus.
        .executableTarget(name: "zoom-compare", dependencies: ["SpineCore"]),

        // MARK: - Tests

        .testTarget(name: "SpineCoreTests", dependencies: ["SpineCore"]),
        .testTarget(name: "SpineMatchingTests", dependencies: ["SpineMatching"]),
        .testTarget(name: "SpinePerceptionTests", dependencies: ["SpinePerception"]),
        .testTarget(
            name: "SpineCatalogTests",
            dependencies: ["SpineCatalog", .product(name: "GRDB", package: "GRDB.swift")]
        ),
        .testTarget(name: "SpineReasoningTests", dependencies: ["SpineReasoning"]),
        .testTarget(name: "SpinePipelineTests", dependencies: ["SpinePipeline"]),
    ]
)
