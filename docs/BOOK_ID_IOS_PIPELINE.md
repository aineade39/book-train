# Book ID iOS pipeline — functional spec & architecture

On-device spine identification for the book ID iOS app: detect → isolate →
read → normalize → match → confirm. This document captures the recommended
approach derived from review of the Core ML / Vision / Foundation Models stack
against the existing `book-train` / `SpineCore` pipeline.

Related repo docs:

| Doc | Role |
|---|---|
| [`GEOMETRY.md`](../GEOMETRY.md) | OBB canonical type, coordinate spaces, conversion boundaries |
| [`MODELS.md`](../MODELS.md) | Production spine detector, promotion / acceptance |
| [`AGENTS.md`](../AGENTS.md) | Train/export commands, Swift package layout |

---

## 1. Functional specifications

**Starting point:** The current **OBB detection output** (`OBBDetection` in scene
pixels from the Core ML spine model) is the canonical input to all downstream
behavior. No parallel axis-aligned detection path; no full-shelf OCR.

### Scope

Build the on-device book ID spine identification pipeline. All processing
runs locally. Cloud is optional fallback only where Apple Intelligence /
Foundation Models are unavailable.

### Detection & isolation

- Accept a camera frame or library photo.
- **Capture-quality pre-gate:** before detection, cheaply score the frame for
  focus / blur / exposure. Reject-and-prompt-retake on clearly bad frames rather
  than spending the full detect→OCR→match budget on unreadable input. This is the
  first quality gate; the OCR quality gate below is the second.
- Run **YOLO-OBB Core ML** inference via `VNCoreMLRequest` to produce oriented
  spine detections in **scene pixel coordinates** (top-left origin, y-down).
- **Single-shot OBB is the v1 path.** For dense shelves only, support **tiled
  first-pass detection** and the **layout-crops jigsaw planner** (plan → verify
  rules → optional per-crop re-inference → global rotated NMS). Tiling/jigsaw is a
  phase-2 hardening step (see Delivery sequencing), not required to ship.
- Convert each `OBBDetection` to an **upright, perspective-corrected spine
  crop** (warp undoing box rotation). This is the sole crop primitive for OCR;
  axis-aligned bounding boxes are out of scope.
- Preserve stable detection identity (`id`) through multi-pass merge for UI
  tracking and re-scan.

### Text extraction

- **Primary path (all supported devices):** `VNRecognizeTextRequest` at
  `.accurate`, with language correction and locale-appropriate
  `recognitionLanguages`.
- **Orientation strategy:** After upright warp, run OCR with aspect-ratio-guided
  orientation and select by **summed candidate confidence, not observation
  count**:
  - Tall crops: primary `.right` (or `.left`), then `.up`
  - Wide crops: primary `.up`, then `.right`
  - Run `.down` (180°) only when both passes fail the quality gate.
  - Note: the current CLI reference (`ocrSpine`) is a placeholder — fixed
    `.up`/`.right` selected by observation *count*. The confidence-sum,
    aspect-guided form above is the target and supersedes it (see §2 OCR
    orientation design).
- **Reading-order assembly:** a spine yields several text observations. Before
  normalization/match, concatenate them **ordered along the spine's long axis**
  (`longAxisAngle()`), not in raw Vision result order, so title/author tokens
  stay coherent for the fuzzy scorer.
- **Barcode path (capture/cover mode, not spine crops):**
  `VNDetectBarcodesRequest` on the *full frame* or a back-cover capture; ISBN/EAN
  hits bypass fuzzy title matching. Spines rarely carry a barcode, so do not run
  it per spine crop — it is wasted work there.
- **Quality gate** before catalog lookup: joint score from OCR confidence,
  orientation agreement, string plausibility (length, charset, token shape), and
  optional detection confidence. Low scores do not auto-match.
- **Heavy path (iOS 26+, A17 Pro+, Apple Intelligence enabled):** Escalate
  only hard cases. Use `LanguageModelSession` with **`@Generable` structured
  output** to parse noisy OCR into `{title, author, isbn?}` — not unstructured
  image prompting. On iOS 27+, FM may orchestrate `OCRTool` /
  `BarcodeReaderTool`, understanding these are Vision-backed and will not fix
  preprocessing failures.
- **User confirmation:** Auto-accept catalog matches only with high rerank
  score **and** clear margin over runner-up; otherwise present top-N
  candidates.

### Catalog matching

- Maintain local book catalog in **SQLite via GRDB**.
- **Stage 1 — retrieval:** FTS5 virtual table with `tokenize='trigram'` and
  `detail='none'`, backed by a content table (not contentless FTS) for `LIKE`
  verification. Return ~50 candidates in milliseconds. Trigram needs ≥3
  characters, so for very short reads (initials, one-word titles) fall back to a
  prefix / `LIKE` query on the content table instead of returning zero
  candidates.
- **Stage 2 — rerank:** **token-set-ratio-style fuzzy match** (`token_set_ratio`
  / `WRatio`, RapidFuzz-equivalent) over OCR string vs candidate title/author
  fields. Tolerates mashed title+author+publisher strings. **Do not use
  `partial_token_set_ratio`:** it saturates at 100 whenever the OCR tokens are a
  subset of a candidate, producing many tied 100s that collapse the accept
  *margin* test and manufacture false auto-matches.
- **Normalization:** Light search-form normalization (lowercase, Unicode fold,
  collapse whitespace, strip decorative punctuation). Do **not** delete
  whitespace or all punctuation pre-match; fuzzy rerank handles variation.

### Non-functional requirements

- **Privacy:** Default on-device; no image or OCR text leaves device in core
  flow.
- **Device coverage:** Full pipeline on all target iPhones; FM escalation
  gracefully degrades on unsupported hardware / regions.
- **Performance budget:** Vision OCR is ms–low-seconds *per crop*, but a dense
  shelf (50–100 spines × ≥2 orientation passes) is the real cost. Bound the
  *aggregate*: OCR lazily (visible / user-tapped spines first, not all at once),
  cap concurrent `VNImageRequestHandler`s, and cache OCR + match results keyed by
  detection `id` so pan/zoom and re-scan don't re-OCR. FM escalation is seconds
  per spine and must be rate-limited (top ~5% of hard cases).
- **Geometry contract:** All internal geometry in scene pixels per
  [`GEOMETRY.md`](../GEOMETRY.md); Vision normalized boxes converted to scene
  pixels only at the documented boundaries — this includes a **data** boundary
  (OCR text boxes → scene, to associate text with a spine OBB), not just the
  UI/draw boundary.
- **Evaluation:** Reuse existing CLI parity harness (`layout-crops` ↔ Python)
  and rotation-bucket eval before changing detection or crop behavior.

### Out of scope

- Training/export pipeline changes (owned by this repo's train scripts).
- Cloud catalog sync design (orthogonal; local FTS index assumed).
- Raw VLM "describe this spine image" prompting as primary OCR.

---

## 2. Architecture and design

### Layered pipeline

```
Photo → CaptureQualityGate → SpineDetector (Core ML OBB) → [optional LayoutCropPlanner]
     → UprightWarp → Vision OCR (aspect-guided orientations) → assemble-along-long-axis
     → QualityGate → [optional FM @Generable parse]
     → Normalize → FTS5 trigram shortlist → token_set_ratio rerank → UI confirm
     (full-frame/cover Barcode is a parallel fast path that bypasses fuzzy match)
```

### Module boundaries

| Layer | Owner | Responsibility |
|---|---|---|
| **SpineCore** (Swift package) | Shared library | `OBBDetection`, inference decode, NMS, layout planner, upright warp*, geometry conversions |
| **Perception** | App service | Vision OCR, barcode scan, orientation routing, quality gate |
| **Reasoning** (optional) | App service | `LanguageModelSession` + `@Generable`; availability gating |
| **Catalog** | App persistence | GRDB + FTS5 trigram index + sync to content table |
| **Matching** | App service | Candidate retrieval + `token_set_ratio` rerank + accept/reject policy |
| **UI** | App | Camera capture, spine overlay (scene→view transform), match confirmation |

`SpineCore` stays model- and UI-agnostic. The iOS app consumes the same types
and crop logic validated by the `layout-crops` CLI.

*Upright warp lives in `Sources/bookspines/main.swift` (`uprightCrop`) today;
the boundary table in [`GEOMETRY.md`](../GEOMETRY.md) marks it **partial —
move to `SpineCore` when the iOS app starts**. The row above is the target home.

### API choices (Apple stack)

| Concern | API | Rationale |
|---|---|---|
| Detection | `VNCoreMLRequest` + exported OBB model | ANE-optimized; matches training/eval lineage |
| Dense shelves | `planCrops` + optional re-infer | Handles missed spines and neighbor bleed better than naive per-box crop |
| Crop | Core Image affine warp from OBB | Axis-aligned crops bleed neighbors; OBB warp is required |
| OCR | `VNRecognizeTextRequest` (`.accurate`) | Fixed CV task; real-time capable; Apple's recommended specialist |
| Barcode | `VNDetectBarcodesRequest` | Highest-precision identifier when present |
| Hard-case reasoning | `LanguageModelSession.respond(to:generating:)` | Structured extraction from OCR text |
| FM tools (iOS 27+) | `OCRTool`, `BarcodeReaderTool` | Orchestration only; same Vision backend |
| Catalog search | SQLite FTS5 trigram | Substring candidate retrieval at scale |
| Fuzzy match | Swift-native `token_set_ratio` (RapidFuzz-equivalent) | Typo-tolerant rerank on short shortlists; ~50 lines of Swift avoids the C++ interop / build-toolchain tax for one scorer |

**Explicit anti-patterns:** axis-aligned YOLO crops; full-shelf OCR;
unstructured FM image prompts as primary text extraction; aggressive string
mangling before fuzzy match; FTS5 treated as fuzzy search without stage-2
rerank.

### OCR orientation design

After OBB upright warp, `CGImagePropertyOrientation` selects reading direction
(90° increments only). Strategy is **guided two-pass with gated third pass**,
not brute-force 4-way on every spine:

1. Aspect ratio picks primary/secondary orientation (`.up` vs `.right`).
2. Score by **confidence sum** and plausibility, **not observation count**.
3. Run `.down` only on fallback (both passes below the quality gate).
4. Residual skew requires pixel deskew or better OBB — not more orientation
   enums.

**Current vs target:** the crop primitive `uprightCrop` in
`Sources/bookspines/main.swift` is reusable as-is. Its companion `ocrSpine` is
**not** the target — it runs a fixed `.up`/`.right` pair and picks the winner by
observation *count* (`up.count >= rotated.count`), which is exactly the
anti-pattern step 2 forbids. Treat `ocrSpine` as a placeholder to replace with
the aspect-guided, confidence-summed form above when the app path is built.

### Foundation Models integration

- **Availability gate:** branch on `SystemLanguageModel.default.availability`
  (the runtime source of truth), not a hardcoded chip check — it already folds in
  device support, Apple Intelligence enabled, and region. Treat "A17 Pro+ / newer"
  as guidance for the *expected* device tier, not the gating condition.
- **Role:** Language layer on top of Vision output — disambiguate title vs
  author vs publisher noise.
- **Memory:** `SystemLanguageModel` runs out-of-process; does not count against
  app RAM like bundled Core ML models (per Apple guidance). Still subject to
  latency and system scheduling.
- **Fallback:** Vision-only + user pick from top matches on unsupported
  devices.

Correct heavy-path shape:

```swift
// Structured extraction from OCR text — not unstructured image prompting.
@Generable
struct SpineExtraction {
    let title: String
    let author: String
    let isbn: String?
}

let extraction = try await session.respond(
    to: "OCR text: \(ocrBlob)",
    generating: SpineExtraction.self
).content
```

### Data model

- **Canonical in-memory:** `OBBDetection` (scene pixels, radians, UUID).
- **Transient:** model letterbox coords, crop-local coords — never persisted or
  passed to planner.
- **Catalog:** `books` content table + `books_fts` trigram index (synced via
  triggers); store both display and search-normalized fields.
- **Match result:**
  `{bookId, score, margin, source: barcode|ocr|fm-assisted, spineDetectionId}`.
- **Editions:** title+author maps to many editions. Retrieve/rank at the *work*
  level (or dedupe editions before applying the margin test) so the runner-up in
  the accept policy isn't just another printing of the same book.

### Matching design

- FTS5 trigram narrows catalog → ~50 via shared character triplets (substring
  retrieval, not edit distance); short-read fallback to prefix/`LIKE` per Stage 1.
- `token_set_ratio` / `WRatio` (not `partial_token_set_ratio`) handles OCR
  drop/merge errors and title+author concatenation without saturating at 100 on
  subset matches.
- Accept policy: `score ≥ T` **and** `score - runnerUp ≥ Δ`; otherwise surface
  UI. The margin test only works if the scorer doesn't tie at ceiling — hence the
  scorer choice above.

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Unreadable capture (blur / glare / low light) | Capture-quality pre-gate + retake prompt before spending the detect→OCR→match budget |
| Vertical / stylized spine text | OBB warp + orientation routing; pixel deskew for hard cases; FM parse, not FM OCR |
| Neighbor bleed in crops | OBB not AABB; layout-crops jigsaw seams; tighter crop rules |
| FM device / regional fragmentation | Feature-flagged tier keyed off `SystemLanguageModel.availability`; Vision-only fallback |
| OCR latency on large shelves | Aspect-guided 2-pass OCR; lazy/visible-first OCR; capped concurrency; per-`id` result cache; FM only on gated subset |
| False auto-matches | `token_set_ratio`/`WRatio` (no ceiling-saturating scorer) + margin-based accept + user confirmation default |
| Geometry bugs | Single scene-pixel contract (`GEOMETRY.md`); conversions only at documented data/draw boundaries |

### Delivery sequencing

1. **Ship:** capture-quality gate → single-shot OBB detect → upright warp →
   Vision OCR (aspect-guided, confidence-summed orientations) → full-frame
   barcode → FTS5 + `token_set_ratio` rerank → confirm UI.
2. **Harden:** tiled + layout-crops jigsaw for dense shelves (not needed for v1);
   lazy/concurrent OCR + per-`id` caching; quality gate tuning on real rotation
   buckets.
3. **Enhance:** FM `@Generable` structuring for hard cases (iOS 26+).
4. **Optional:** `RecognizeDocumentsRequest` evaluation for multi-line / vertical
   reading order where platform support allows.

This architecture reuses the existing `book-train` / `SpineCore` investment,
aligns with Apple's Vision-first / FM-for-reasoning guidance, and scopes
Foundation Models as a gated enhancement rather than a core dependency.
