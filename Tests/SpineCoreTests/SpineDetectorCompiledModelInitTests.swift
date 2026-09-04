import CoreML
import Foundation
import XCTest

@testable import SpineCore

/// Opt-in test for `SpineDetector.init(compiledModelURL:)` -- the loading
/// path an app-bundled `.mlpackage` resource needs (Xcode's Core ML
/// resource build phase ships only the compiled `.mlmodelc` output, so
/// on-device code can't route through `init(modelURL:)`'s
/// `MLModel.compileModel(at:)` step, which requires an uncompiled source).
/// `XCTSkip`s itself when the local production model isn't present.
final class SpineDetectorCompiledModelInitTests: XCTestCase {
    func testCompiledModelURLInitProducesTheSameLoadedShapeAsModelURLInit() throws {
        let modelURL = defaultModelURL()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }

        let fromSource = try SpineDetector(modelURL: modelURL)
        let compiledURL = try cachedCompiledModelURL(for: modelURL)
        let fromCompiled = try SpineDetector(compiledModelURL: compiledURL)

        XCTAssertEqual(fromSource.inputWidth, fromCompiled.inputWidth)
        XCTAssertEqual(fromSource.inputHeight, fromCompiled.inputHeight)
        XCTAssertEqual(fromSource.inputName, fromCompiled.inputName)
        XCTAssertEqual(fromSource.outputName, fromCompiled.outputName)
        XCTAssertEqual(fromSource.layout, fromCompiled.layout)
    }

    func testCompiledModelURLInitThrowsOnMissingPath() {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("nope-\(UUID().uuidString).mlmodelc")
        XCTAssertThrowsError(try SpineDetector(compiledModelURL: missing))
    }
}
