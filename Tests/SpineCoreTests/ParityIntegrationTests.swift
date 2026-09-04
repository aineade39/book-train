import CoreGraphics
import Foundation
import XCTest

@testable import SpineCore

/// Opt-in integration tests -- each `XCTSkip`s itself when its prerequisite
/// (a Python venv, or the local model/data root) isn't present, so `swift
/// test` stays fully offline and dependency-free by default. Run explicitly
/// with e.g. `swift test --filter ParityIntegrationTests` once
/// `.venv`/`$BOOK_SPINES_DATA` are set up.
final class ParityIntegrationTests: XCTestCase {

    private var repoRoot: URL {
        // Tests/SpineCoreTests/ParityIntegrationTests.swift -> repo root.
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    // MARK: - Test 1: Swift vs Python plan_crops on a shared synthetic fixture

    /// Bypasses the YOLO/Core ML model entirely: feeds the *same*
    /// hand-authored `Det` list (kept numerically in sync with
    /// `tools/layout_crop_parity_fixture.py`) into both `planCrops`/
    /// `verifyPlan` here and the Python reference implementation, then
    /// diffs the two plans. This is the strongest available check that the
    /// port is numerically faithful, since it needs no model and is fully
    /// deterministic (a uniform-gray scene makes pixel-texture seam search
    /// deterministic in both languages).
    func testPlanCropsMatchesPythonReferenceOnSharedFixture() throws {
        let pythonURL = repoRoot.appendingPathComponent(".venv/bin/python3")
        let scriptURL = repoRoot.appendingPathComponent("tools/layout_crop_parity_fixture.py")
        guard FileManager.default.fileExists(atPath: pythonURL.path) else {
            throw XCTSkip("no .venv/bin/python3 -- run tools/build_spines_dataset.py's venv setup first")
        }
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw XCTSkip("missing tools/layout_crop_parity_fixture.py")
        }

        let process = Process()
        process.executableURL = pythonURL
        process.arguments = [scriptURL.path]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let msg = String(data: errData, encoding: .utf8) ?? "<no stderr>"
            throw XCTSkip("Python fixture script failed (missing cv2/numpy in .venv?): \(msg)")
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let pyPayload = try decoder.decode(PyParityPayload.self, from: outData)

        // Must mirror tools/layout_crop_parity_fixture.py's DETS exactly.
        func deg(_ d: Double) -> Double { d * .pi / 180 }
        let dets = [
            OBBDetection(cx: 80, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 150, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 220, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 500, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 570, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 640, cy: 160, w: 50, h: 200, angle: deg(0), conf: 0.90),
            OBBDetection(cx: 80, cy: 400, w: 50, h: 200, angle: deg(0), conf: 0.85),
            OBBDetection(cx: 150, cy: 400, w: 50, h: 200, angle: deg(0), conf: 0.85),
            OBBDetection(cx: 220, cy: 400, w: 50, h: 200, angle: deg(80), conf: 0.85),
            OBBDetection(cx: 290, cy: 400, w: 50, h: 200, angle: deg(80), conf: 0.85),
        ]
        let imgW = pyPayload.imgW, imgH = pyPayload.imgH
        XCTAssertEqual(imgW, 900)
        XCTAssertEqual(imgH, 500)

        let raster = SceneRaster(cgImage: makeSolidCGImage(width: imgW, height: imgH))!
        let swiftPlans = planCrops(
            dets: dets, imgW: imgW, imgH: imgH, raster: raster,
            angleTolDeg: 25.0, rowGapK: 0.2, colGapK: 1.0,
            minBlockMembers: 2, imgsz: 1024, maxCropDimK: 1.5
        )
        let swiftRules = verifyPlan(dets: dets, plans: swiftPlans, imgW: imgW, imgH: imgH, angleTolDeg: 25.0)

        XCTAssertEqual(hardRulesOK(swiftRules), pyPayload.rulesOk, "hard-rule verdict diverged from Python")

        var pyOkByRule: [String: Bool] = [:]
        for r in pyPayload.rules { pyOkByRule[r.rule] = r.ok }
        for r in swiftRules {
            XCTAssertEqual(r.ok, pyOkByRule[r.rule], "\(r.rule) ok-flag diverged from Python (\(r.detail))")
        }

        XCTAssertEqual(swiftPlans.count, pyPayload.plans.count, "planned crop count diverged from Python")
        for (swiftPlan, pyPlan) in zip(swiftPlans, pyPayload.plans) {
            XCTAssertEqual(swiftPlan.shelfId, pyPlan.shelfId)
            XCTAssertEqual(swiftPlan.blockId, pyPlan.blockId)
            XCTAssertEqual(Set(swiftPlan.memberIndices), Set(pyPlan.memberIndices), "\(swiftPlan.name) membership diverged from Python")
            XCTAssertEqual(swiftPlan.angleDeg, pyPlan.angleDeg, accuracy: 0.1, "\(swiftPlan.name) angle diverged from Python")
            XCTAssertEqual(swiftPlan.quad.count, pyPlan.quad.count)
            for (sp, pp) in zip(swiftPlan.quad, pyPlan.quad) {
                XCTAssertEqual(Double(sp.x), pp[0], accuracy: 0.05, "\(swiftPlan.name) quad.x diverged from Python")
                XCTAssertEqual(Double(sp.y), pp[1], accuracy: 0.05, "\(swiftPlan.name) quad.y diverged from Python")
            }
        }
    }

    // MARK: - Test 2: full layout-crops CLI against real local data

    /// Smoke-tests the compiled `layout-crops` executable end to end against
    /// the local model + a real shelf photo, when both happen to be present
    /// on this machine (`$BOOK_SPINES_DATA` / `~/ml/book-spines` production
    /// model, and a fixture photo at the repo root). Does not attempt exact
    /// numeric parity with the Python `.pt` pipeline -- those predictions can
    /// legitimately differ (different runtime, different weights format).
    func testLayoutCropsCLIPlanOnlyAgainstRealModelAndPhoto() throws {
        let modelURL = defaultModelURL()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("no production Core ML model at \(modelURL.path)")
        }
        let imageURL = repoRoot.appendingPathComponent("bookcase.books.spines.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("no fixture photo at \(imageURL.path)")
        }
        guard let cliURL = Self.layoutCropsExecutableURL() else {
            throw XCTSkip("could not locate the built layout-crops executable next to the test bundle")
        }

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("layout-crops-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // --- plan-only pass ---
        let planJSON = tmpDir.appendingPathComponent("plan.json")
        let planResult = try Self.run(cliURL, args: [imageURL.path, "--plan-only", "--json", planJSON.path])
        XCTAssertEqual(planResult.exitCode, 0, "plan-only run should exit 0 (hard rules ok); stderr: \(planResult.stderr)")
        let planPayload = try JSONDecoder().decode(CLIPayload.self, from: Data(contentsOf: planJSON))
        XCTAssertTrue(planPayload.rulesOk, "hard rules should pass on the real fixture photo")
        XCTAssertGreaterThan(planPayload.firstPassCount, 0, "expected at least one first-pass detection")
        XCTAssertGreaterThan(planPayload.plannedCrops, 0)

        // --- full materialize + re-infer + overlay pass ---
        let cropsDir = tmpDir.appendingPathComponent("crops")
        let fullJSON = tmpDir.appendingPathComponent("full.json")
        let overlayPNG = tmpDir.appendingPathComponent("overlay.png")
        let fullResult = try Self.run(cliURL, args: [
            imageURL.path,
            "--write-crops", cropsDir.path,
            "--infer-crops",
            "--overlay-crops",
            "--overlay-dets",
            "--json", fullJSON.path,
            "--out", overlayPNG.path,
        ])
        XCTAssertEqual(fullResult.exitCode, 0, "full pipeline run should exit 0; stderr: \(fullResult.stderr)")
        let fullPayload = try JSONDecoder().decode(CLIPayload.self, from: Data(contentsOf: fullJSON))
        XCTAssertTrue(fullPayload.rulesOk)
        XCTAssertFalse(fullPayload.crops.isEmpty)
        for crop in fullPayload.crops {
            guard let path = crop.path else {
                XCTFail("crop \(crop.name) missing a written path")
                continue
            }
            XCTAssertTrue(FileManager.default.fileExists(atPath: path), "expected crop PNG at \(path)")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: overlayPNG.path), "expected scene overlay PNG")
    }

    // MARK: - Helpers

    private static func layoutCropsExecutableURL() -> URL? {
        let candidate = Bundle(for: ParityIntegrationTests.self).bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("layout-crops")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    private struct RunResult { let exitCode: Int32; let stdout: String; let stderr: String }

    private static func run(_ executable: URL, args: [String]) throws -> RunResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = args
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return RunResult(
            exitCode: process.terminationStatus,
            stdout: String(data: outData, encoding: .utf8) ?? "",
            stderr: String(data: errData, encoding: .utf8) ?? ""
        )
    }
}

// MARK: - JSON payload shapes (mirror tools/layout_crop_parity_fixture.py and Sources/layout-crops/main.swift)

private struct PyParityRule: Decodable {
    let rule: String
    let hard: Bool
    let ok: Bool
    let detail: String
}

private struct PyParityPlan: Decodable {
    let shelfId: Int
    let blockId: Int
    let angleDeg: Double
    let quad: [[Double]]
    let memberIndices: [Int]
}

private struct PyParityPayload: Decodable {
    let imgW: Int
    let imgH: Int
    let rulesOk: Bool
    let rules: [PyParityRule]
    let plans: [PyParityPlan]
}

private struct CLICropMeta: Decodable {
    let name: String
    let path: String?
}

private struct CLIPayload: Decodable {
    let rulesOk: Bool
    let firstPassCount: Int
    let plannedCrops: Int
    let crops: [CLICropMeta]

    enum CodingKeys: String, CodingKey {
        case rulesOk = "rules_ok"
        case firstPassCount = "first_pass_count"
        case plannedCrops = "planned_crops"
        case crops
    }
}
