# Model registry

Every trained/exported spine-detector checkpoint should get a row here. Filenames
now encode the key config (see naming convention below), but this file is the
place for anything a filename can't hold: eval results, notes, and what actually
happened.

Full config for any run is always in `~/data/yolo-obb-runs/<run_name>/args.yaml`
(written automatically by Ultralytics), when that run directory still exists.
Full per-angle eval results come from `tools/eval_rotation_sweep.py --json`.
Exported `.mlpackage` files get a sidecar `<name>.json` manifest (written
automatically by `tools/train_combined_obb.py`) with a copy of the same info, so
the artifact is traceable even after it's been copied out of this repo (e.g. into
the Xcode project).

## Naming convention

```
{dataset}_{arch}_{imgsz}px_deg{degrees}_ep{epochs}_frac{pct}_{YYYYMMDD-HHMM}[_{tag}]
```

Example: `combined_yolo26s-obb_1024px_deg90_ep20_frac15_20260716-2054_smoke`

- `dataset` — data root tag (`combined` = 4TU+IEEE merged, `spines` = 4TU-only legacy)
- `arch` — base checkpoint (`yolo11n-obb`, `yolo26s-obb`, ...)
- `imgsz` — training image size
- `deg` — `degrees` rotation augmentation (0 = none, 90 = full rotation)
- `ep` — epoch budget requested (not necessarily reached, if early-stopped)
- `frac` — % of train split actually used (100 = full dataset)
- timestamp — when the run started, guarantees no two runs ever collide
- `tag` — optional free-form suffix (`smoke`, `full`, experiment name, ...)

`tools/train_combined_obb.py` generates this automatically; pass `--name`/`--tag`
to override. **Never reuse a bare, unparameterized name like `"spine-obb"` with
`exist_ok=True`** for a different config — collisions silently overwrite the
previous run's weights/`args.yaml`. (An earlier version of this doc claimed this
had actually happened between `spine-obb` and `spine-obb-aug` below — it hadn't;
see "Corrections" at the bottom. But the risk is real, hence the convention.)

## Runs

All mAP50 columns are per-bucket from `tools/eval_rotation_sweep.py` (original /
rot30 / rot45 / rot60 / rot90) on the val split, at `conf=0.15, iou=0.5`.

| Run name / artifact | Arch | Dataset | imgsz | deg | epochs | frac | mAP50 (orig/30/45/60/90) | Notes |
|---|---|---|---|---|---|---|---|---|
| `spine-obb` (legacy, `tools/train_4tu_obb.py`), trained 2026-07-15 | yolo11n-obb | 4TU only | 1024 | 0 | 100 | 100% | 0.976 / 0.948 / 0.813 / 0.234 / 0.041 | Plain baseline, no rotation augmentation. As expected, collapses at 60°/90° (recall 0.08 at 90°) — this is the *un-augmented* model, correctly named, never touched by anything else. |
| `spine-obb-aug` (legacy run dir now gone) → `models/SpineDetectorOBB-aug.mlpackage`, trained/exported 2026-07-16 | yolo11n-obb | 4TU only | 1024 | 90 | 20 | 100% | 0.973 / 0.966 / 0.976 / 0.968 / 0.969 | **The real degrees=90 fine-tune.** Massive, confirmed improvement over the `deg0` baseline at 60°/90° (mAP50 +0.73 / +0.93, recall +0.60 / +0.88), ~flat elsewhere. This validates the original "7→24 detections" finding with a rigorous rotated-IoU metric, not just a box count. The raw `.pt`/`args.yaml`/`results.csv` for this run are no longer on disk (run dir `~/data/yolo-obb-runs/spine-obb-aug/` is gone; cause unknown, not an overwrite — see Corrections), but the Core ML export survived and is fully usable for eval/inference. |
| `spine-obb-v2-smoke` → should be `combined_yolo26s-obb_1024px_deg90_ep20_frac15_*_smoke` under the new convention, trained 2026-07-16 | yolo26s-obb | combined (4TU+IEEE) | 1024 | 90 | 20 | 15% | 0.907 / 0.876 / 0.811 / 0.874 / 0.919 | Smoke test only (pipeline/label sanity check), reinitialized head (nc 15→1). Currently **worse than `spine-obb-aug` across every bucket** (5-20 pts mAP50) — expected, since it's undertrained (15% data, 20 epochs, fresh head) vs. a properly fine-tuned model. Core ML export format changed vs yolo11n-obb: image input + pre-decoded `[1,300,7]` `(cx,cy,w,h,conf,cls,angle)` instead of raw `[1,6,N]` grid — `bookspines.swift` decode path will need updating once the full model ships. |
| _(full run — pending)_ | yolo26s-obb | combined (4TU+IEEE) | 1024 | 90 | 120 | 100% | TBD | Planned: cloud GPU (T4/L4 spot on `gemini-proxy`), ~$3-12, ~10-17h. Acceptance bar: beat `spine-obb-aug`'s numbers above on every bucket, not just pooled mAP. |

## How to add a row

After training + export:

```bash
.venv-export/bin/python tools/eval_rotation_sweep.py \
  --weights ~/data/yolo-obb-runs/<run_name>/weights/best.pt \
  --weights-name "<run_name>" \
  --compare models/SpineDetectorOBB-aug.mlpackage \
  --compare-name "aug (current best)" \
  --angles 30,45,60,90 \
  --json /tmp/sweep.json
```

Copy the per-bucket mAP50 numbers and the run name into a new row above.

## Corrections

- **2026-07-16, during this session:** an earlier version of this file claimed
  `train_4tu_obb.py`'s hardcoded `name="spine-obb"` + `exist_ok=True` had
  overwritten an earlier degrees=90 fine-tune's weights. Checked file timestamps
  (`spine-obb/args.yaml` mtime == birthtime, i.e. written exactly once, Jul 15
  16:03) and the original transcript: the degrees=90 fine-tune was actually saved
  under a **different** name, `spine-obb-aug`, and never collided with
  `spine-obb` at all. Its Core ML export (`models/SpineDetectorOBB-aug.mlpackage`)
  is still on disk and usable — nothing important was actually lost. Only the raw
  PyTorch run directory for `spine-obb-aug` is gone, for unknown reasons unrelated
  to naming collisions.
