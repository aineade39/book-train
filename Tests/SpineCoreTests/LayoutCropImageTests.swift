import CoreGraphics
import XCTest

@testable import SpineCore

/// Image-transform tests: homography corner mapping and inverse round trip,
/// warp/pad geometry (including a no-y-flip check via a checkerboard),
/// minimum-area rectangle fitting, and the crop-to-scene detection remap.
final class LayoutCropImageTests: XCTestCase {

    // MARK: - Homography

    func testHomographyIdentityMapsPointsUnchanged() {
        let square = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10)]
        let h = Homography(from: square, to: square)
        XCTAssertNotNil(h)
        for p in [CGPoint(x: 3, y: 4), CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 10)] {
            XCTAssertPointsEqual(h!.apply(p), p, accuracy: 1e-6)
            XCTAssertPointsEqual(h!.applyInverse(p), p, accuracy: 1e-6)
        }
    }

    func testHomographyMapsFourCornersExactly() {
        // A skewed source quad (mild trapezoid, as a warped scene quad would
        // be) mapped onto a 200x100 upright rectangle.
        let src = [CGPoint(x: 10, y: 20), CGPoint(x: 210, y: 5), CGPoint(x: 220, y: 130), CGPoint(x: 0, y: 110)]
        let dst = [CGPoint(x: 0, y: 0), CGPoint(x: 199, y: 0), CGPoint(x: 199, y: 99), CGPoint(x: 0, y: 99)]
        guard let h = Homography(from: src, to: dst) else {
            return XCTFail("expected a valid homography for a non-degenerate quad")
        }
        for (s, d) in zip(src, dst) {
            XCTAssertPointsEqual(h.apply(s), d, accuracy: 1e-3)
        }
    }

    func testHomographyInverseRoundTripStaysUnderHundredthPixel() {
        let src = [CGPoint(x: 12, y: 34), CGPoint(x: 305, y: 18), CGPoint(x: 290, y: 240), CGPoint(x: 5, y: 260)]
        let dst = [CGPoint(x: 0, y: 0), CGPoint(x: 255, y: 0), CGPoint(x: 255, y: 255), CGPoint(x: 0, y: 255)]
        guard let h = Homography(from: src, to: dst) else {
            return XCTFail("expected a valid homography")
        }
        let probes = [
            CGPoint(x: 50, y: 50), CGPoint(x: 200, y: 10), CGPoint(x: 130, y: 130),
            CGPoint(x: 250, y: 250), CGPoint(x: 1, y: 254),
        ]
        for p in probes {
            let forward = h.apply(p)
            let back = h.applyInverse(forward)
            XCTAssertPointsEqual(back, p, accuracy: 0.01, "round trip drifted for \(p)")
        }
    }

    func testHomographyIsNilForDegenerateCollinearSource() {
        let collinear = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 20, y: 0), CGPoint(x: 30, y: 0)]
        let dst = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]
        XCTAssertNil(Homography(from: collinear, to: dst))
    }

    // MARK: - quadOutputSize / warpQuad / padNoUpsize

    func testQuadOutputSizeOfAxisAlignedRectangle() {
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 0), CGPoint(x: 100, y: 50), CGPoint(x: 0, y: 50)]
        let (w, h) = quadOutputSize(quad)
        XCTAssertEqual(w, 100)
        XCTAssertEqual(h, 50)
    }

    func testWarpQuadIdentityCropPreservesTopLeftPixelExactly() {
        // A checkerboard raster; warping the top-left quadrant (axis-aligned,
        // no rotation) as its own crop must reproduce that region unchanged
        // and, crucially, must NOT flip vertically (top-left stays top-left).
        let img = makeCheckerboardCGImage(width: 100, height: 100, cols: 4, rows: 4)
        guard let raster = SceneRaster(cgImage: img) else { return XCTFail("raster init failed") }
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 50, y: 0), CGPoint(x: 50, y: 50), CGPoint(x: 0, y: 50)]
        guard let warped = warpQuad(raster, quad: quad, maxSide: nil) else { return XCTFail("warp failed") }
        XCTAssertEqual(warped.outputWidth, 50)
        XCTAssertEqual(warped.outputHeight, 50)
        guard let warpedRaster = SceneRaster(cgImage: warped.image) else { return XCTFail("warped raster init failed") }
        // Top-left cell of a 4x4 board (r=0,c=0) is black; sample well inside it.
        let (r, g, b) = warpedRaster.rgb(x: 5, y: 5)
        XCTAssertEqual(r, 0); XCTAssertEqual(g, 0); XCTAssertEqual(b, 0)
    }

    func testWarpQuadNoUpsizeRespectsMaxSide() {
        let img = makeSolidCGImage(width: 400, height: 200)
        guard let raster = SceneRaster(cgImage: img) else { return XCTFail("raster init failed") }
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 399, y: 0), CGPoint(x: 399, y: 199), CGPoint(x: 0, y: 199)]
        guard let warped = warpQuad(raster, quad: quad, maxSide: 100) else { return XCTFail("warp failed") }
        XCTAssertEqual(max(warped.outputWidth, warped.outputHeight), 100)
        XCTAssertEqual(warped.outputWidth, 100)
        XCTAssertEqual(warped.outputHeight, 50)
    }

    func testPadNoUpsizeCentersAndNeverUpscales() {
        let crop = makeSolidCGImage(width: 40, height: 20, gray: 200)
        guard let pad = padNoUpsize(crop, canvas: 100, padValue: 114) else { return XCTFail("pad failed") }
        // Scale limited by the long side (40) to fit in 100 -> gain 2.5, but
        // "never upsize" means we only ever scale *down*; a small crop like
        // this is upscaled here only because canvas > crop -- gain must stay
        // <= 1.0 per the "no upsize" contract.
        XCTAssertLessThanOrEqual(pad.gain, 1.0)
        XCTAssertEqual(pad.image.width, 100)
        XCTAssertEqual(pad.image.height, 100)
    }

    func testPadNoUpsizeOfExactCanvasSizeCropHasNoPadding() {
        let crop = makeSolidCGImage(width: 100, height: 100, gray: 50)
        guard let pad = padNoUpsize(crop, canvas: 100, padValue: 114) else { return XCTFail("pad failed") }
        XCTAssertEqual(pad.gain, 1.0, accuracy: 1e-9)
        XCTAssertEqual(pad.padX, 0, accuracy: 1e-9)
        XCTAssertEqual(pad.padY, 0, accuracy: 1e-9)
    }

    // MARK: - minAreaRect

    func testMinAreaRectOfAxisAlignedSquareCorners() {
        let pts = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10)]
        guard let rect = minAreaRect(pts) else { return XCTFail("expected a rect") }
        XCTAssertEqual(rect.width * rect.height, 100.0, accuracy: 1e-6)
        XCTAssertEqual(rect.cx, 5, accuracy: 1e-6)
        XCTAssertEqual(rect.cy, 5, accuracy: 1e-6)
    }

    func testMinAreaRectOfRotatedRectangleRecoversDimensions() {
        // A 20x10 rectangle rotated 30 deg about the origin.
        let det = OBBDetection(cx: 0, cy: 0, w: 20, h: 10, angle: .pi / 6, conf: 1)
        guard let rect = minAreaRect(det.corners) else { return XCTFail("expected a rect") }
        let dims = [rect.width, rect.height].sorted()
        XCTAssertEqual(dims[0], 10, accuracy: 1e-3)
        XCTAssertEqual(dims[1], 20, accuracy: 1e-3)
        XCTAssertEqual(rect.width * rect.height, 200, accuracy: 1e-2)
    }

    // MARK: - mapDetFromCrop

    func testMapDetFromCropRoundTripsThroughIdentityHomography() {
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 99, y: 0), CGPoint(x: 99, y: 99), CGPoint(x: 0, y: 99)]
        guard let homography = Homography(from: quad, to: quad) else { return XCTFail("expected homography") }
        // No padding, gain 1: mapping a crop-space det back to scene space
        // through an identity transform should reproduce it unchanged.
        let det = OBBDetection(cx: 50, cy: 40, w: 20, h: 8, angle: 0.1, conf: 0.7)
        guard let mapped = mapDetFromCrop(det, homography: homography, gain: 1, padX: 0, padY: 0) else {
            return XCTFail("expected a mapped detection")
        }
        XCTAssertEqual(mapped.cx, det.cx, accuracy: 1e-3)
        XCTAssertEqual(mapped.cy, det.cy, accuracy: 1e-3)
        XCTAssertEqual(min(mapped.w, mapped.h), min(det.w, det.h), accuracy: 1e-2)
        XCTAssertEqual(max(mapped.w, mapped.h), max(det.w, det.h), accuracy: 1e-2)
    }

    func testMapDetFromCropUndoesPaddingAndGain() {
        // Crop-space det sits inside a 2x-padded canvas (gain 0.5) offset by
        // (10, 10); undoing gain/pad should land it back at (cxC, cyC) before
        // any homography is applied (identity homography here).
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 199, y: 0), CGPoint(x: 199, y: 199), CGPoint(x: 0, y: 199)]
        guard let homography = Homography(from: quad, to: quad) else { return XCTFail("expected homography") }
        let gain = 0.5
        let padX = 10.0, padY = 10.0
        let cxCrop = 60.0, cyCrop = 50.0, wCrop = 20.0, hCrop = 8.0
        let det = OBBDetection(cx: padX + cxCrop * gain, cy: padY + cyCrop * gain, w: wCrop * gain, h: hCrop * gain, angle: 0, conf: 1)
        guard let mapped = mapDetFromCrop(det, homography: homography, gain: gain, padX: padX, padY: padY) else {
            return XCTFail("expected a mapped detection")
        }
        XCTAssertEqual(mapped.cx, cxCrop, accuracy: 1e-2)
        XCTAssertEqual(mapped.cy, cyCrop, accuracy: 1e-2)
        XCTAssertEqual(max(mapped.w, mapped.h), max(wCrop, hCrop), accuracy: 1e-1)
        XCTAssertEqual(min(mapped.w, mapped.h), min(wCrop, hCrop), accuracy: 1e-1)
    }

    func testMapDetFromCropReturnsNilForZeroGain() {
        let quad = [CGPoint(x: 0, y: 0), CGPoint(x: 9, y: 0), CGPoint(x: 9, y: 9), CGPoint(x: 0, y: 9)]
        guard let homography = Homography(from: quad, to: quad) else { return XCTFail("expected homography") }
        let det = OBBDetection(cx: 5, cy: 5, w: 4, h: 2, angle: 0, conf: 1)
        XCTAssertNil(mapDetFromCrop(det, homography: homography, gain: 0, padX: 0, padY: 0))
    }
}
