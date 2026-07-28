import CoreGraphics
import Foundation
import Vision

// Barcode fast path per docs/BOOK_ID_IOS_PIPELINE.md §Text extraction:
// "VNDetectBarcodesRequest on the *full frame* or a back-cover capture;
// ISBN/EAN hits bypass fuzzy title matching. Spines rarely carry a barcode,
// so do not run it per spine crop -- it is wasted work there." Callers
// should invoke this on the full captured frame, never on a per-spine
// `uprightWarp` crop.

public struct DetectedBarcode: Equatable {
    public let payload: String
    public let symbology: String
    /// Vision's normalized bounding box (origin bottom-left, `[0, 1]` on
    /// both axes) -- `.zero` for callers/fixtures that don't need §D's
    /// "Spine association" (scene-pixel barcode center vs. spine OBBs),
    /// which is the only consumer of this field.
    public let boundingBox: CGRect
    /// `true` for ISBN-shaped payloads (EAN-13 starting `978`/`979`, or a
    /// raw ISBN-10/13 string) -- these should bypass fuzzy title matching
    /// entirely per spec.
    public var looksLikeISBN: Bool {
        DetectedBarcode.isISBNLike(payload)
    }

    public init(payload: String, symbology: String, boundingBox: CGRect = .zero) {
        self.payload = payload
        self.symbology = symbology
        self.boundingBox = boundingBox
    }

    static func isISBNLike(_ payload: String) -> Bool {
        let digits = payload.filter(\.isNumber)
        guard digits.count == 10 || digits.count == 13 else { return false }
        if digits.count == 13 { return digits.hasPrefix("978") || digits.hasPrefix("979") }
        return true
    }
}

extension DetectedBarcode {
    /// `boundingBox`'s center converted from Vision's normalized
    /// bottom-left/y-up convention to full-scene top-left/y-down pixels --
    /// the same convention `OBBDetection.cx/cy` uses -- so callers can
    /// measure it against spine OBBs for §D's "Spine association".
    public func sceneCenter(imageWidth: Int, imageHeight: Int) -> CGPoint {
        let nx = boundingBox.midX
        let ny = boundingBox.midY
        return CGPoint(x: nx * CGFloat(imageWidth), y: (1 - ny) * CGFloat(imageHeight))
    }
}

public protocol BarcodeReader {
    func detectBarcodes(in image: CGImage) throws -> [DetectedBarcode]
}

public struct VisionBarcodeReader: BarcodeReader {
    public init() {}

    public func detectBarcodes(in image: CGImage) throws -> [DetectedBarcode] {
        let request = VNDetectBarcodesRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])
        return (request.results ?? []).compactMap { observation in
            guard let payload = observation.payloadStringValue else { return nil }
            return DetectedBarcode(payload: payload, symbology: observation.symbology.rawValue, boundingBox: observation.boundingBox)
        }
    }
}
