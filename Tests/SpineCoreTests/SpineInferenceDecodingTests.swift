import CoreML
import XCTest

@testable import SpineCore

/// Decode-path tests for `SpineDetector.decodeDetections`, exercised
/// directly against hand-built `MLMultiArray` fixtures -- no Core ML model
/// required. Covers the legacy `[1,6,N]` layout, the YOLO26 end-to-end
/// `[1,maxDet,7]` layout, its transposed `[1,7,maxDet]` form, and a
/// non-contiguous-stride buffer to prove the decoder doesn't assume
/// row-major layout.
final class SpineInferenceDecodingTests: XCTestCase {
    /// Identity letterbox (gain 1, no padding) so decoded coordinates equal
    /// the raw model outputs, keeping fixture math trivial.
    let identityLetterbox = LetterboxParams(gain: 1, padX: 0, padY: 0, newUnpadW: 640, newUnpadH: 640)

    private func multiArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        for (i, v) in values.enumerated() {
            arr[i] = NSNumber(value: v)
        }
        return arr
    }

    // MARK: - legacyChannelsFirst [1, 6, N]

    func testDecodeLegacyChannelsFirstFiltersByConfidenceAndMapsBoxes() throws {
        let anchors = 3
        // Channel-major layout: [cx x N][cy x N][w x N][h x N][conf x N][angle x N].
        let cx: [Float] = [100, 200, 300]
        let cy: [Float] = [50, 60, 70]
        let w: [Float] = [20, 30, 25]
        let h: [Float] = [10, 15, 12]
        let conf: [Float] = [0.9, 0.05, 0.5] // anchor 1 filtered out
        let angle: [Float] = [0.1, 0.0, -0.2]
        let values = cx + cy + w + h + conf + angle
        let arr = try multiArray(shape: [1, 6, anchors], values: values)

        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 6, anchors], layout: .legacyChannelsFirst,
            confidenceThreshold: 0.15, letterbox: identityLetterbox, originX: 0, originY: 0
        )

        XCTAssertEqual(dets.count, 2)
        let byConf = dets.sorted { $0.conf > $1.conf }
        XCTAssertEqual(byConf[0].cx, 100, accuracy: 1e-4)
        XCTAssertEqual(byConf[0].cy, 50, accuracy: 1e-4)
        XCTAssertEqual(byConf[0].w, 20, accuracy: 1e-4)
        XCTAssertEqual(byConf[0].h, 10, accuracy: 1e-4)
        XCTAssertEqual(byConf[0].angle, 0.1, accuracy: 1e-6)
        XCTAssertEqual(byConf[0].conf, 0.9, accuracy: 1e-6)
        XCTAssertEqual(byConf[1].cx, 300, accuracy: 1e-4)
        XCTAssertEqual(byConf[1].conf, 0.5, accuracy: 1e-6)
    }

    func testDecodeLegacyChannelsFirstAppliesLetterboxAndOrigin() throws {
        // gain 2, pad (10, 5): scene = (raw - pad) / gain + origin.
        let letterbox = LetterboxParams(gain: 2, padX: 10, padY: 5, newUnpadW: 320, newUnpadH: 320)
        let values: [Float] = [110, 65, 40, 20, 0.8, 0.0] // cx cy w h conf angle for 1 anchor
        let arr = try multiArray(shape: [1, 6, 1], values: values)
        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 6, 1], layout: .legacyChannelsFirst,
            confidenceThreshold: 0.15, letterbox: letterbox, originX: 1000, originY: 2000
        )
        XCTAssertEqual(dets.count, 1)
        XCTAssertEqual(dets[0].cx, (110.0 - 10) / 2 + 1000, accuracy: 1e-4)
        XCTAssertEqual(dets[0].cy, (65.0 - 5) / 2 + 2000, accuracy: 1e-4)
        XCTAssertEqual(dets[0].w, 40.0 / 2, accuracy: 1e-4)
        XCTAssertEqual(dets[0].h, 20.0 / 2, accuracy: 1e-4)
    }

    func testDecodeLegacyChannelsFirstDropsDegenerateTinyBoxes() throws {
        // w/h <= 1 after letterbox scaling must be dropped (guard in mapModelBoxToImage).
        let values: [Float] = [50, 50, 0.5, 0.5, 0.9, 0.0]
        let arr = try multiArray(shape: [1, 6, 1], values: values)
        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 6, 1], layout: .legacyChannelsFirst,
            confidenceThreshold: 0.15, letterbox: identityLetterbox, originX: 0, originY: 0
        )
        XCTAssertEqual(dets.count, 0)
    }

    // MARK: - end2endDetections [1, maxDet, 7]

    func testDecodeEnd2EndDetectionsRowMajorFiltersByConfidence() throws {
        // Row-major [cx, cy, w, h, conf, cls, angle] per row.
        let row0: [Float] = [150, 80, 40, 20, 0.8, 0, 0.3]
        let row1: [Float] = [250, 90, 35, 18, 0.05, 0, -0.1] // filtered
        let arr = try multiArray(shape: [1, 2, 7], values: row0 + row1)
        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 2, 7], layout: .end2endDetections,
            confidenceThreshold: 0.15, letterbox: identityLetterbox, originX: 0, originY: 0
        )
        XCTAssertEqual(dets.count, 1)
        XCTAssertEqual(dets[0].cx, 150, accuracy: 1e-4)
        XCTAssertEqual(dets[0].cy, 80, accuracy: 1e-4)
        XCTAssertEqual(dets[0].w, 40, accuracy: 1e-4)
        XCTAssertEqual(dets[0].h, 20, accuracy: 1e-4)
        XCTAssertEqual(dets[0].angle, 0.3, accuracy: 1e-6)
        XCTAssertEqual(dets[0].conf, 0.8, accuracy: 1e-6)
    }

    func testDecodeEnd2EndDetectionsTransposedLayout() throws {
        // Transposed physical layout [1, 7, maxDet]: contiguous strides give
        // feature stride == maxDet, row stride == 1 (column-major-ish).
        let maxDet = 2
        // feature order: cx, cy, w, h, conf, cls, angle
        let features: [[Float]] = [
            [150, 250], // cx  (row0, row1)
            [80, 90], // cy
            [40, 35], // w
            [20, 18], // h
            [0.8, 0.05], // conf (row1 filtered)
            [0, 0], // cls
            [0.3, -0.1], // angle
        ]
        var values = [Float](repeating: 0, count: 7 * maxDet)
        for (f, row) in features.enumerated() {
            for (i, v) in row.enumerated() {
                values[i + f * maxDet] = v
            }
        }
        let arr = try multiArray(shape: [1, 7, maxDet], values: values)
        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 7, maxDet], layout: .end2endDetections,
            confidenceThreshold: 0.15, letterbox: identityLetterbox, originX: 0, originY: 0
        )
        XCTAssertEqual(dets.count, 1)
        XCTAssertEqual(dets[0].cx, 150, accuracy: 1e-4)
        XCTAssertEqual(dets[0].cy, 80, accuracy: 1e-4)
        XCTAssertEqual(dets[0].angle, 0.3, accuracy: 1e-6)
    }

    // MARK: - Non-contiguous strides

    func testDecodeLegacyChannelsFirstWithAnchorMajorNonContiguousStrides() throws {
        // Physical layout is anchor-major (interleaved channels per anchor:
        // [cx,cy,w,h,conf,angle] repeated for every anchor), the opposite of
        // the usual channel-major [1,6,N] contiguous layout. Custom strides
        // describe this without moving any data, exercising the decoder's
        // `strideN != 1` slow path.
        let anchors = 3
        struct Row { let cx, cy, w, h, conf, angle: Float }
        let rows = [
            Row(cx: 100, cy: 50, w: 20, h: 10, conf: 0.9, angle: 0.1),
            Row(cx: 200, cy: 60, w: 30, h: 15, conf: 0.05, angle: 0.0), // filtered
            Row(cx: 300, cy: 70, w: 25, h: 12, conf: 0.5, angle: -0.2),
        ]
        var buffer = [Float](repeating: 0, count: anchors * 6)
        for (i, r) in rows.enumerated() {
            let base = i * 6
            buffer[base + 0] = r.cx
            buffer[base + 1] = r.cy
            buffer[base + 2] = r.w
            buffer[base + 3] = r.h
            buffer[base + 4] = r.conf
            buffer[base + 5] = r.angle
        }

        let byteCount = buffer.count * MemoryLayout<Float>.stride
        let rawPointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<Float>.alignment)
        buffer.withUnsafeBytes { rawPointer.copyMemory(from: $0.baseAddress!, byteCount: byteCount) }

        // shape [1, 6, N]; strides (batch, channel, anchor) = (6*N, 1, 6):
        // channel stride 1, anchor stride 6 -- i.e. anchor-major physical layout.
        let arr = try MLMultiArray(
            dataPointer: rawPointer,
            shape: [1, 6, anchors].map { NSNumber(value: $0) },
            dataType: .float32,
            strides: [NSNumber(value: 6 * anchors), NSNumber(value: 1), NSNumber(value: 6)]
        ) { ptr in ptr.deallocate() }

        XCTAssertNotEqual(arr.strides[2].intValue, 1, "fixture must exercise the non-contiguous stride path")

        let dets = try SpineDetector.decodeDetections(
            output: arr, shape: [1, 6, anchors], layout: .legacyChannelsFirst,
            confidenceThreshold: 0.15, letterbox: identityLetterbox, originX: 0, originY: 0
        )

        XCTAssertEqual(dets.count, 2)
        let byConf = dets.sorted { $0.conf > $1.conf }
        XCTAssertEqual(byConf[0].cx, 100, accuracy: 1e-4)
        XCTAssertEqual(byConf[0].angle, 0.1, accuracy: 1e-6)
        XCTAssertEqual(byConf[1].cx, 300, accuracy: 1e-4)
        XCTAssertEqual(byConf[1].angle, -0.2, accuracy: 1e-6)
    }

    // MARK: - Layout detection

    func testOutputLayoutDetectFromShape() {
        XCTAssertEqual(OBBOutputLayout.detect(shape: [1, 6, 8400]), .legacyChannelsFirst)
        XCTAssertEqual(OBBOutputLayout.detect(shape: [1, 300, 7]), .end2endDetections)
        XCTAssertEqual(OBBOutputLayout.detect(shape: [1, 7, 300]), .end2endDetections)
        XCTAssertNil(OBBOutputLayout.detect(shape: [1, 6])) // wrong rank
        XCTAssertNil(OBBOutputLayout.detect(shape: [1, 5, 8400])) // unrecognized
    }
}
