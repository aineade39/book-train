import Foundation
import Vision
import AppKit

// 1. Validate command line arguments
guard CommandLine.arguments.count > 1 else {
    print("Usage: swift ocr.swift <path-to-image>")
    exit(1)
}

let imagePath = CommandLine.arguments[1]
let imageURL = URL(fileURLWithPath: imagePath)

// 2. Load the image into a CGImage (macOS native graphics format)
guard let nsImage = NSImage(contentsOf: imageURL),
      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("Error: Could not load image at \(imagePath)")
    exit(1)
}

struct CropRectPx: Codable {
    let x0: Int
    let y0: Int
    let x1: Int
    let y1: Int
}

struct OCRObservation: Codable {
    let text: String
    let confidence: Float
    let cropRectPx: CropRectPx
    let corners: [[Int]]
}

struct OCRImage: Codable {
    let w: Int
    let h: Int
    let path: String
}

struct OCRDocument: Codable {
    let image: OCRImage
    let observations: [OCRObservation]
}

let imageWidth = Double(cgImage.width)
let imageHeight = Double(cgImage.height)

func toPixelPoint(_ point: CGPoint) -> [Int] {
    let x = Int(round(point.x * imageWidth))
    let y = Int(round((1.0 - point.y) * imageHeight))
    return [x, y]
}

func toCropRectPx(_ box: CGRect) -> CropRectPx {
    let x0 = Int(round(box.origin.x * imageWidth))
    let y0 = Int(round((1.0 - box.origin.y - box.height) * imageHeight))
    let x1 = Int(round((box.origin.x + box.width) * imageWidth))
    let y1 = Int(round((1.0 - box.origin.y) * imageHeight))
    return CropRectPx(
        x0: min(x0, x1),
        y0: min(y0, y1),
        x1: max(x0, x1),
        y1: max(y0, y1)
    )
}

// 3. Define the Vision text recognition request
let request = VNRecognizeTextRequest(completionHandler: { request, error in
    if let error = error {
        FileHandle.standardError.write(Data("Error recognizing text: \(error.localizedDescription)\n".utf8))
        return
    }

    guard let observations = request.results as? [VNRecognizedTextObservation] else {
        print("[]")
        return
    }

    // Vision coordinates are normalized (0–1) with origin at bottom-left.
    // Convert to scene pixels with origin at top-left.
    let ocrObservations: [OCRObservation] = observations.compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        return OCRObservation(
            text: candidate.string,
            confidence: candidate.confidence,
            cropRectPx: toCropRectPx(observation.boundingBox),
            corners: [
                toPixelPoint(observation.topLeft),
                toPixelPoint(observation.topRight),
                toPixelPoint(observation.bottomRight),
                toPixelPoint(observation.bottomLeft)
            ]
        )
    }

    let doc = OCRDocument(
        image: OCRImage(
            w: cgImage.width,
            h: cgImage.height,
            path: imageURL.path
        ),
        observations: ocrObservations
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

    do {
        let data = try encoder.encode(doc)
        if let json = String(data: data, encoding: .utf8) {
            print(json)
        }
    } catch {
        FileHandle.standardError.write(Data("Failed to encode JSON: \(error.localizedDescription)\n".utf8))
    }
})

// Configure the request for maximum accuracy and language correction
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

// 4. Create a handler and execute the request
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

do {
    // This performs the request synchronously
    try handler.perform([request])
} catch {
    print("Failed to process image: \(error.localizedDescription)")
}