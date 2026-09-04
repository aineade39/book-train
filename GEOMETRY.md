# Detection geometry

Canonical contract for oriented bounding boxes (OBBs), coordinate spaces, and
where conversions are allowed. Applies to Python tools, `SpineCore`, CLIs, and
the future iOS app. Read this before inventing a new box format or writing
`imageH - cy` in a new file.

Implementation detail lives in source comments; this file is policy.

## Canonical type

**Swift:** `OBBDetection` in `Sources/SpineCore/SpineGeometry.swift`

**Python:** `Det` in `tools/tiled_predict_obb.py` (same fields; used by
`tools/layout_crop_predict.py`)

| Field | Type | Meaning |
|---|---|---|
| `cx`, `cy` | float | Center in **scene pixels** (see below) |
| `w`, `h` | float | Side lengths in **absolute pixels**, not normalized to image size |
| `angle` | float | Rotation in **radians** (x-right / y-down) |
| `conf` | float | Detection confidence |
| `id` | UUID (Swift only) | Stable identity for two-pass merge tracking |

**Derived (never canonical):**

- `corners` — four `CGPoint`s / `(x, y)` tuples from `(cx, cy, w, h, angle)`
- `longAxisAngle()` — spine orientation; **do not assume `w` is the long edge**

After Core ML decode, every detection in planner, NMS, rules, and merge logic
must be in this shape. Do not store YOLO-normalized `[0, 1]` coords or Vision
normalized boxes as the internal representation.

## Coordinate spaces

Three spaces appear in the pipeline. Only **scene** is canonical for
`OBBDetection`.

### Scene (canonical)

Full photo pixels: origin **top-left**, **y increases downward**. Matches
`CGImage` width/height and Python OpenCV image coords.

All `OBBDetection` values after `SpineDetector.predict` (with tile/crop
`originX`/`originY` applied) live here.

### Model letterbox (transient)

Core ML input canvas (e.g. 1024×1024) after Ultralytics letterbox. Raw tensor
values are `(cx, cy, w, h, angle)` in this space. Converted to scene once in
`SpineInference` — do not persist or pass letterbox boxes to the planner.

### Crop-local (transient)

Pixels inside a warped/padded crop image before `mapDetFromCrop` remaps to
scene. Used only inside the layout-crop materialize path. There is no runtime
tag; never mix crop-local and scene points in the same geometry call.

## Boundary conversions

Conversions happen **only** at listed boundaries. Internal code (IoU, planner,
rules, homography) operates in scene pixels with no extra transforms.

| From | To | Owner | Status |
|---|---|---|---|
| Model letterbox tensor | scene `OBBDetection` | `Sources/SpineCore/SpineInference.swift` | **implemented** |
| Tile/crop image + offset | scene `OBBDetection` | `SpineInference.predict(originX:originY:)` | **implemented** |
| Crop pad/gain/homography | scene `OBBDetection` | `Sources/SpineCore/LayoutCropImage.swift` (`mapDetFromCrop`) | **implemented** |
| scene `OBBDetection` | `CGPath` / overlay draw | `SpineCore` (planned shared helper) | **planned** |
| scene `OBBDetection` | upright `CGImage` crop | `Sources/SpineCore/UprightWarp.swift` (`uprightWarp`) | **implemented** |
| upright crop pixel | scene `OBBDetection` space | `Sources/SpineCore/UprightWarp.swift` (`cropPixelToScene`) | **implemented** |
| Vision normalized OCR box (post-orientation) | scene pixels | `Sources/SpineCore/OCRGeometry.swift` (`ocrQuadToScene`) | **implemented** |
| scene pixels | SwiftUI view coords | iOS app (`GeometryReader`, content mode) | **planned** — app-owned; `SpineCore` may offer primitives only |

Before adding a new conversion (e.g. another `imageH - cy` flip), check this
table. Extend the table when you add a boundary; do not scatter undocumented
transforms in CLI or app code.

## Non-goals

Do **not** use as canonical internal format:

- Vision `VNRecognizedObjectObservation.boundingBox` (normalized, bottom-left,
  axis-aligned only)
- YOLO training normalized `[cx, cy, w, h]` in `[0, 1]` after decode
- 4-corner quad as the stored detection (fine as a derived view or
  post-homography intermediate)
- Core Image bottom-left / y-up except at a narrow render boundary, immediately
  converted back to scene top-left

## JSON export (not canonical)

CLI and eval JSON are **export views**, not the source of truth.

| Shape | Where | Notes |
|---|---|---|
| `OBBDetectionJSON` | `layout-crops`, parity fixtures | `{cx, cy, w, h, angle, conf}` — matches Python `asdict(Det)` |
| `SpineJSON` / `DocumentJSON` | `bookspines` CLI | int pixels, `angleDeg`, `cornersPx` — legacy CLI shape |
| `layout-crops` payload | `Sources/layout-crops/main.swift` | pipeline report + `first_pass` / `detections_merged` |

New consumers should prefer `OBBDetectionJSON` (or encode from `OBBDetection`
directly). Unifying `bookspines` JSON is optional tech debt.

## iOS app notes

- **Canonical in memory:** `OBBDetection` in scene pixels (efficient on device;
  no per-use denormalization).
- **Draw boundary:** convert scene → view once per frame in the UI layer;
  depends on zoom and `contentMode` — not fully specifiable in this repo yet.
- **OCR:** align Vision text boxes to scene pixels before comparing to spine
  OBBs via `ocrQuadToScene` (see the boundary table above); this is a data
  boundary, not just a draw boundary — reading-order assembly depends on it.

## Source map

| Concern | Primary file |
|---|---|
| `OBBDetection`, angles, IoU, convex polygons | `Sources/SpineCore/SpineGeometry.swift` |
| Non-convex polygons, staircase-seam splits | `Sources/SpineCore/PolygonPartition.swift` |
| Letterbox + decode → scene | `Sources/SpineCore/SpineInference.swift` |
| Homography, crop remap | `Sources/SpineCore/LayoutCropImage.swift` |
| Jigsaw planner (quad cutter) | `Sources/SpineCore/LayoutCropPlanner.swift` |
| Jigsaw-zoom recursion + free-form cutter | `Sources/SpineCore/JigsawZoomEngine.swift`, `Sources/SpineCore/PieceCutter.swift` |
| Python `Det` reference | `tools/tiled_predict_obb.py` |
