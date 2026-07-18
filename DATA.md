# Data and artifacts

This repo is **code and docs only**. Large files live outside git, under a
project data root, with Google Drive as cold backup via rclone.

## Data root

```bash
export BOOK_SPINES_DATA="${BOOK_SPINES_DATA:-$HOME/ml/book-spines}"
```

Layout:

```text
$BOOK_SPINES_DATA/
  raw/<id>/SOURCE.md   # provenance only (cold)
  tmp/                 # fetch/unpack scratch (short TTL)
  derived/             # rebuildable YOLO / Create ML datasets
  runs/                # Ultralytics run dirs (weights, args.yaml, logs)
  models/
    production/        # promoted exports worth keeping
    candidates/        # trial exports
  eval/                # sweep JSON, overlays (short TTL)
```

Script defaults resolve through `tools/paths.py` / `tools/fetch_raw.py`
(honors `$BOOK_SPINES_DATA`):

| Role | Default |
|---|---|
| Raw dumps | Drive zip + `SOURCE.md`; local `raw/<id>/SOURCE.md` only |
| 4TU YOLO-OBB | `derived/4tu-spines_yolo-obb` |
| 4TU+IEEE YOLO-OBB | `derived/4tu-ieee_yolo-obb` |
| Create ML export | `derived/4tu-spines_createml` |
| Training runs | `runs/` |
| Trial Core ML | `models/candidates/` |
| Promoted Core ML | `models/production/` |

Nothing under the data root (or repo-local `models/`, `runs/`, `out/`, `*.pt`)
is committed. See `.gitignore`.

## Raw datasets (provenance)

Each dump has id `raw/<id>/` with a **`SOURCE.md`** sidecar in **both** places
(local + Drive). Required sidecar fields for fetch:

| Field | Meaning |
|---|---|
| `drive_path` | rclone remote dir, e.g. `gdrive:dev/ml/raw/ieee-book-spine` |
| `archive` | zip filename inside that dir |
| `content_root` | optional path inside the zip after unpack (omit if zip root is the dataset) |

**Unified prep path:** every ingest script uses `tools/fetch_raw.unpacked_raw()` —
rclone zip → temp unpack → process a **directory** → delete temp. No script
reads zip members in place; no permanent unpacked twin on Drive or local raw.

| id | DOI / URL | Drive archive |
|---|---|---|
| `4tu-spines` | [10.4121/uuid:33f2a166-de13-4505-b359-2b202c491fd8](https://doi.org/10.4121/uuid:33f2a166-de13-4505-b359-2b202c491fd8) | `gdrive:dev/ml/raw/4tu-spines/4tu-spines.zip` |
| `ieee-book-spine` | [10.21227/g82y-gt86](https://doi.org/10.21227/g82y-gt86) | `gdrive:dev/ml/raw/ieee-book-spine/book_spine.zip` |
| `roboflow-book-spine-obb` | [universe.roboflow.com/-b6bdz/book-spine-obb](https://universe.roboflow.com/-b6bdz/book-spine-obb) | `gdrive:dev/ml/raw/roboflow-book-spine-obb/book-spine-obb.v1-obb.yolov8-obb.zip` |

When adding a raw source: write `SOURCE.md`, upload the zip to `drive_path`,
keep only the sidecar under local `raw/<id>/`, add a row here.

## Derived naming

Rebuildable trees under `derived/`. Pattern: `{lineage}_{format}`.

| id | Built from | Script |
|---|---|---|
| `4tu-spines_yolo-obb` | raw `4tu-spines` | `tools/train_4tu_obb.py` |
| `4tu-spines_createml` | raw `4tu-spines` | `tools/convert_4tu_to_createml.py` |
| `4tu-ieee_yolo-obb` | `4tu-spines_yolo-obb` + raw `ieee-book-spine` | `tools/build_spines_dataset.py` |

Each derived folder has a **`SOURCE.md`** (sources, script, git commit, flags,
`built_at`) written by the prep script. Prefer encoding rot/tile/limit in that
sidecar, not in the folder name, unless you keep multiple variants on disk.

Run-name `{dataset}_…` prefix is the lineage stem (`4tu-ieee`, `4tu-spines`),
not a vague `combined`.

## What goes where

| Kind | Local (`~/ml`) | Drive (`gdrive:dev/ml`) | Git |
|---|---|---|---|
| Raw dumps | `SOURCE.md` only | **zip + `SOURCE.md`** (no unpack) | no |
| Derived YOLO data | yes | optional / skip | no |
| Training runs | yes | `best.pt` only if useful | no |
| Promoted Core ML | yes | yes | no |
| Candidate exports | yes | rarely | no |
| Eval dumps | yes (short TTL) | no | no |
| Train scripts, docs | this repo | optional | yes |

**Raw is irreplaceable. Derived and runs are rebuildable from raw + this repo.**

## Drive (rclone)

Remote: `gdrive:dev/ml/`

- Drive = **archives**; local = **working trees** (`derived/`, `runs/`, models).
- Prefer `rclone copy`, not `sync`, for `raw/` and `models/production/`.
- Do not keep an unpacked twin of a raw zip on Drive, or a zip under local `raw/`.

```bash
rclone lsd gdrive:dev/ml/raw
# push sidecar after edits
rclone copy "$BOOK_SPINES_DATA/raw/ieee-book-spine/SOURCE.md" \
  gdrive:dev/ml/raw/ieee-book-spine/
# push archive (one-shot)
rclone copy "$BOOK_SPINES_DATA/raw/ieee-book-spine/book_spine.zip" \
  gdrive:dev/ml/raw/ieee-book-spine/
```

Prep examples (fetch happens automatically):

```bash
.venv/bin/python tools/build_spines_dataset.py --limit 20
.venv/bin/python tools/train_4tu_obb.py --limit 3 --skip-train --skip-export
```

## Naming

Run and export names encode the key config:

```text
{dataset}_{arch}_{imgsz}px_deg{degrees}_ep{epochs}_frac{pct}_{YYYYMMDD-HHMM}[_{tag}]
```

Example: `4tu-ieee_yolo26s-obb_1024px_deg90_ep20_frac15_20260716-2054_smoke`

| Token | Meaning |
|---|---|
| `dataset` | lineage stem from derived folder (`4tu-ieee`, `4tu-spines`, …) |
| `arch` | base checkpoint stem (`yolo26s-obb`, …) |
| `imgsz` | training image size |
| `deg` | rotation augmentation degrees (`0` = none) |
| `ep` | epoch budget (may early-stop) |
| `frac` | % of train split used (`100` = full) |
| timestamp | run start; avoids name collisions |
| `tag` | optional (`smoke`, …) |

`tools/train_combined_obb.py` generates this; override with `--name` / `--tag`.
**Never reuse a bare name like `spine-obb` with `exist_ok=True` for a different
config** — that silently overwrites weights and `args.yaml`.

Use the same stem for the run dir, `best.pt`, `.mlpackage`, and sidecar.

## Sidecar manifests

Every exported `.mlpackage` should have a same-stem `.json` next to it
(written by `tools/train_combined_obb.py`):

```text
foo_fp16.mlpackage
foo_fp16.json          ← sidecar
```

Typical fields: `run_name`, `export_name`, `exported_at`, `git_commit`,
`source_weights`, `run_dir`, `data_yaml`, `model_base`, `imgsz`, `epochs`,
`degrees`, `fraction`, `quantize`, plus optional eval hooks.

The sidecar travels with the binary (e.g. into Xcode or Drive) so provenance
survives after the Ultralytics run directory is gone. Full train config also
lives in `runs/<run_name>/args.yaml` while that dir exists.

## Model registry

Curated “what matters / what’s best” lives in [`MODELS.md`](MODELS.md), not
here. This file is policy; that file is the scoreboard.
