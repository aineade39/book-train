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
    /// `true` for ISBN-shaped payloads (EAN-13 starting `978`/`979`, or a
    /// raw ISBN-10/13 string) -- these should bypass fuzzy title matching
    /// entirely per spec.
    public var looksLikeISBN: Bool {
        DetectedBarcode.isISBNLike(payload)
    }

    public init(payload: String, symbology: String) {
        self.payload = payload
        self.symbology = symbology
    }

    static func isISBNLike(_ payload: String) -> Bool {
        let digits = payload.filter(\.isNumber)
        guard digits.count == 10 || digits.count == 13 else { return false }
        if digits.count == 13 { return digits.hasPrefix("978") || digits.hasPrefix("979") }
        return true
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
            return DetectedBarcode(payload: payload, symbology: observation.symbology.rawValue)
        }
    }
}
