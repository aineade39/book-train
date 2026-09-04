# Model registry

Curated spine-detector runs and exports. Layout, naming, sidecars, and backup
policy: [`DATA.md`](DATA.md).

Filenames encode config; this file holds eval results and judgment calls.
Per-angle numbers are from `tools/eval_rotation_sweep.py` on val at
`conf=0.15, iou=0.5` (mAP50: orig / 30° / 45° / 60° / 90°).

## Current best

**`4tu-ieee-shelves_yolo26s-obb_1024px_deg90_ep120_frac100_20260719-0627_fp16`** (yolo26s-obb, 4tu-ieee-shelves, deg90, 120 ep) — production default via `SpineDetectorOBB.mlpackage` symlink. Acceptance baseline remains `SpineDetectorOBB-aug`.

## Runs

| Artifact / run | Arch | Data | imgsz | deg | ep | frac | mAP50 | Notes |
|---|---|---|---|---|---|---|---|---|
| `4tu-ieee-shelves_yolo26s-obb_1024px_deg90_ep120_frac100_20260719-0627_fp16.mlpackage` | yolo26s-obb | 4tu-ieee-shelves | 1024 | 90 | 120 | 100% | TBD | **Current best.** production default from latest full run; rotation-sweep acceptance TBD |
| `spine-obb` → legacy export | yolo11n-obb | 4TU | 1024 | 0 | 100 | 100% | 0.976 / 0.948 / 0.813 / 0.234 / 0.041 | No rotation aug; collapses at 60°/90°. |
| `SpineDetectorOBB-aug.mlpackage` | yolo11n-obb | 4TU | 1024 | 90 | 20 | 100% | 0.973 / 0.966 / 0.976 / 0.968 / 0.969 | Run dir gone; Core ML + this row remain. |
| smoke `…_frac15_*_smoke` | yolo26s-obb | 4tu-ieee | 1024 | 90 | 20 | 15% | 0.907 / 0.876 / 0.811 / 0.874 / 0.919 | Pipeline check only; undertrained vs aug. yolo26 export layout differs from yolo11 — see `bookspines.swift`. |
| `4tu-ieee_yolo26s-obb_…_ep120_…` (was `combined_…`) | yolo26s-obb | 4tu-ieee | 1024 | 90 | 120 | 100% | TBD | Acceptance: beat aug on every bucket. Existing run dirs may still use the old `combined_` prefix. |

## After a new train + export

```bash
# Promote latest run (export if needed, rotation sweep vs aug, alias + MODELS.md)
.venv-export/bin/python tools/promote_coreml.py --latest
```

Keep the sidecar next to the `.mlpackage`; don’t paste full provenance here.
`SpineDetectorOBB-aug` stays on disk as the frozen compare baseline.

