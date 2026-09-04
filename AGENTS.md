# book-train

YOLO-OBB book-spine detection → Core ML exports consumed by a Swift package
(`Package.swift`: `SpineCore` library, `bookspines` + `layout-crops` executables).
This repo is **code and docs only**; large datasets, runs, and weights live outside git.

Canonical detail lives in existing docs — read them instead of inventing policy:

| Doc | Role |
|---|---|
| [`DATA.md`](DATA.md) | Data root, layout, naming, sidecars, rclone backup |
| [`MODELS.md`](MODELS.md) | Scoreboard, current best, promotion / acceptance |
| [`GEOMETRY.md`](GEOMETRY.md) | OBB canonical type, coordinate spaces, boundary conversions |

Data root: `$BOOK_SPINES_DATA` (default `$HOME/ml/book-spines`). Script defaults
come from `tools/paths.py`. Raw dumps are Drive zips fetched by
`tools/fetch_raw.py` (see `DATA.md`).

## Outside-world questions go to coldframe

Platform rules, store policy, market and competitors, general engineering
practice: [coldframe](https://github.com/aineade39/coldframe) holds these, in a
separate repo that never reads this one. Cite its pages; do not re-derive the
answer here.

Asking "is this standard practice?" in this window returns a description of
*this* design, restated as industry practice and sounding well-sourced. That
has been measured, not assumed — a controlled comparison had this window cite
these docs as evidence about the industry and state something false about a
platform API that the isolated repo, holding the vendor's own page, got right.

## Python environment

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Recreate
a venv any time:

```bash
# Training / Ultralytics stack (Python 3.14)
uv sync

# Core ML export / rotation-sweep stack (Python 3.12, separate venv)
UV_PROJECT_ENVIRONMENT=.venv-export uv sync --python 3.12 --group export
```

After `uv sync`, use `.venv/bin/python` (or `uv run python`) in the commands
below. Commit changes to `pyproject.toml` when adding deps (`uv add …`); commit
`uv.lock` after every lock update (`uv lock`).

## Commands

Two venvs: `.venv` (train / Ultralytics), `.venv-export` (Core ML export / eval
that needs the export stack). Prefer the matching one.

```bash
# Rebuild / refresh combined YOLO dataset (4TU+IEEE+open-shelves+roboflow)
.venv/bin/python tools/build_spines_dataset.py

# Smoke train + export (fast pipeline check)
.venv/bin/python tools/train_combined_obb.py --smoke

# Full combined train + Core ML export (defaults: yolo26s-obb, imgsz=1024, deg90, 120 ep)
.venv/bin/python tools/train_combined_obb.py

# Export only from existing weights
.venv/bin/python tools/train_combined_obb.py --skip-train --weights <path/to/best.pt>

# Resume interrupted run (same run dir). Finished runs: finetune with --model best.pt instead.
.venv/bin/python tools/train_combined_obb.py --resume --model $BOOK_SPINES_DATA/runs/<run>/weights/last.pt

# Per-angle rotation sweep (required before claiming a win)
.venv-export/bin/python tools/eval_rotation_sweep.py \
  --weights $BOOK_SPINES_DATA/runs/<run_name>/weights/best.pt \
  --weights-name "<run_name>" \
  --compare $BOOK_SPINES_DATA/models/production/SpineDetectorOBB-aug.mlpackage \
  --compare-name "aug (current best)" \
  --angles 30,45,60,90 \
  --json $BOOK_SPINES_DATA/eval/sweep.json

# Promote latest run: export if needed, rotation sweep vs aug,
# production alias + MODELS.md (use --skip-sweep to skip the slow eval)
.venv-export/bin/python tools/promote_coreml.py --latest

# Core ML inference / OCR harness (default: production/SpineDetectorOBB.mlpackage alias)
swift bookspines.swift <image> [--model path.mlpackage] [--ocr]

# Compare single-shot vs tiled .pt inference on a dense shelf photo
.venv/bin/python tools/tiled_predict_obb.py <image> [--tiles 2x2]

# Full-scene jigsaw crops (OBB-gap seams; empty cells for missed spines; verified by rules)
.venv/bin/python tools/layout_crop_predict.py <image> --overlay-plan
.venv/bin/python tools/layout_crop_predict.py <image> --write-crops /tmp/crops --infer-crops --overlay-crops
```

Older 4TU-only path: `tools/train_4tu_obb.py`. Prefer `train_combined_obb.py`
unless explicitly working on the 4TU-only lineage.

### Swift package (`Package.swift`)

The Swift/Core ML on-device side lives in a `SpineCore` library plus two
executables — the Python commands above stay the train/eval reference path
(`.pt` weights, full argparse surface); Swift is the exported-model
equivalent used for on-device inference and jigsaw cropping.

```bash
swift build -c release

# Inference / OCR harness (default: production/SpineDetectorOBB.mlpackage alias)
swift run -c release bookspines <image> [--model path.mlpackage] [--ocr]

# Full-scene jigsaw crops — same shape as tools/layout_crop_predict.py
swift run -c release layout-crops <image> --overlay-plan
swift run -c release layout-crops <image> \
  --write-crops /tmp/crops --infer-crops --overlay-crops

# Unit tests (geometry/rules/planner/image-transform/decoder fixtures — no
# model or dataset dependency, safe to run anywhere)
swift test

# Opt-in integration tests (skip themselves via XCTSkip when their
# prerequisite is missing): a Python-vs-Swift plan_crops parity check
# against a shared synthetic detection fixture (needs .venv/bin/python3 with
# numpy+cv2 — no model), and a full layout-crops CLI smoke test against the
# local production model + a real photo (needs $BOOK_SPINES_DATA)
swift test --filter ParityIntegrationTests
```

### Book catalog + match CLIs

Canonical detail: [`docs/BOOK_CATALOG.md`](docs/BOOK_CATALOG.md).

```bash
# Fixture smoke (no OL download)
python tools/build_book_catalog.py --profile dev_smoke --fixture Tests/fixtures/ol-mini

# Full automated pipeline
python tools/build_book_catalog.py --all --install-ios

# CSV catalog (small lists)
swift run -c release catalog-build catalog.csv --db /tmp/catalog.sqlite

# Match harness (no photo/model)
swift run -c release book-match "dune frank herbert" --db /tmp/catalog.sqlite --json

# Full detect → OCR → match
swift run -c release spine-id <image> --db /tmp/catalog.sqlite [--fm] [--json out.json]

# ISBN scrape loop — detached (never as a long-lived Agent shell)
python tools/run_book_show_api_loop_detached.py start
```

Root [`bookspines.swift`](bookspines.swift) predates this package and is
kept only as a compatibility reference; prefer `swift run bookspines`.
Never commit `.build/`, `.swiftpm/`, generated crop images, overlays, or any
model/eval artifacts the executables write out.

## Hard rules

1. **Never commit** weights, runs, datasets, images, or exports: `*.pt`,
   `runs/`, `models/`, `out/`, `*.mlpackage/`, large images (see `.gitignore`).
2. **Never reuse a bare run name** (e.g. `spine-obb`) with `exist_ok=True` for a
   different config — that silently overwrites weights/`args.yaml`. Prefer the
   auto-generated stem from `train_combined_obb.py` (`--name` / `--tag` only when intentional).
3. **Same stem** for run dir, `best.pt`, `.mlpackage`, and sidecar `.json`.
4. Every `.mlpackage` needs a **same-stem `.json` sidecar** (provenance).
5. **Acceptance**: a new full run must beat **`SpineDetectorOBB-aug`** on
   **every** rotation bucket in `eval_rotation_sweep.py` (orig / 30 / 45 / 60 / 90),
   not just pooled mAP. Numbers and the comparison recipe live in `MODELS.md`.
6. After train/export: run `tools/promote_coreml.py --latest` (runs the
   rotation sweep vs aug by default, copies into `models/production/`,
   retargets `SpineDetectorOBB.mlpackage`, updates **Current best** in
   `MODELS.md`). Do not paste full sidecars into that file.
   Leave `SpineDetectorOBB-aug` on disk as the frozen compare baseline.
7. Raw data under the data root is **irreplaceable**; derived + runs are
   rebuildable. Prefer `rclone copy` for Drive archives (see `DATA.md`).
8. YOLO26 Core ML output layout differs from YOLO11 — decode paths live in
   `Sources/SpineCore/SpineInference.swift`; do not assume `[1, 6, N]` for
   every export.
9. **Never attach a multi-hour job to an Agent-spawned shell.** Cursor aborts
   those terminals after ~3h. Start the ISBN scrape with
   `python tools/run_book_show_api_loop_detached.py start` (returns immediately;
   `docker compose up -d` plus host `caffeinate`) — not `caffeinate -i tools/run_book_show_api_loop.sh`
   as a background Agent command. One-time gate: `... prepare` after
   `disk-cleanup nas-copy/gdrive-sync --item ml-goodreads-scrape --execute`.

## Agent habits

- Prefer editing scripts/docs over inventing new train entry points.
- Before changing train defaults, check existing smoke / full run names and
  `MODELS.md` so comparisons stay apples-to-apples.
- Do not “clean up” large local artifacts unless asked; they are outside git on purpose.
- When unsure about layout or naming, open `DATA.md` first.
- When unsure about OBB fields, coordinate spaces, or conversion boundaries,
  open `GEOMETRY.md` first — do not add ad-hoc `imageH - cy` / Vision-flip
  paths without updating the boundary table there.
- **Scope anchoring:** for compare / validate / benchmark / run requests that
  could apply to more than one program or pipeline in the repo (or session),
  default to the **primary deliverable of the current work** — not the simplest
  or most familiar path. State the chosen scope in one sentence before executing
  (e.g. “comparing `layout-crops --infer-crops` ↔
  `layout_crop_predict.py --infer-crops`”). Ask only when scopes would produce
  materially different answers and a quick repo or session check cannot
  disambiguate.
