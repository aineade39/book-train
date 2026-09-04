#!/usr/bin/env python3
"""Per-angle rotation sweep evaluation for OBB spine-detection checkpoints.

Aggregate mAP over the whole val split can look great while the model is
actually blind at intermediate rotation angles, because rotated copies are a
small minority of val images. This script buckets the val split by the
rotation suffix baked in by build_spines_dataset.py (original / rot30 /
rot45 / rot60 / rot90) and runs Ultralytics' own OBB val() on each bucket
separately, so precision/recall/mAP50/mAP50-95 are reported per angle
instead of pooled.

Usage:
    tools/eval_rotation_sweep.py --weights best.pt
    tools/eval_rotation_sweep.py --weights new.pt --compare old.pt --compare-name "old (yolo11n)"
    tools/eval_rotation_sweep.py --weights best.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import derived_dir  # noqa: E402

BUCKET_RE = re.compile(r"_rot(\d+)(?=\.[A-Za-z0-9]+$)")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. `pip install ultralytics`", file=sys.stderr)
        sys.exit(1)
    return YOLO


def bucket_val_images(images_val_dir: Path, angles: list[int]) -> dict[str, list[Path]]:
    wanted = {f"rot{a}" for a in angles}
    buckets: dict[str, list[Path]] = {"original": []}
    for a in angles:
        buckets[f"rot{a}"] = []

    for p in sorted(images_val_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = BUCKET_RE.search(p.name)
        if m:
            key = f"rot{m.group(1)}"
            if key in wanted:
                buckets[key].append(p)
            # rotation angles not requested are silently skipped
        else:
            buckets["original"].append(p)
    return {k: v for k, v in buckets.items() if v}


def count_instances(images: list[Path], images_val_dir: Path, labels_val_dir: Path) -> int:
    total = 0
    for img in images:
        label_path = labels_val_dir / (img.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                total += sum(1 for line in f if line.strip())
    return total


def run_bucket_val(
    YOLO: Any,
    weights: Path,
    base_yaml: dict[str, Any],
    data_root: Path,
    bucket_name: str,
    images: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix=f"rotsweep_{bucket_name}_") as tmp:
        tmp_dir = Path(tmp)
        list_path = tmp_dir / "images.txt"
        list_path.write_text("\n".join(str(p.resolve()) for p in images) + "\n")

        yaml_path = tmp_dir / "data.yaml"
        cfg = dict(base_yaml)
        cfg["path"] = str(data_root)
        cfg["val"] = str(list_path)
        cfg.setdefault("train", cfg["val"])  # unused for val-only runs, but required by schema
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        model = YOLO(str(weights), task="obb")
        results = model.val(
            data=str(yaml_path),
            split="val",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            plots=False,
            batch=8,
        )
        box = results.box
        return {
            "n_images": len(images),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help="Model checkpoint (.pt or .mlpackage)")
    ap.add_argument("--weights-name", default=None, help="Display label for --weights (default: filename)")
    ap.add_argument("--compare", type=Path, default=None, help="Second checkpoint to compare against")
    ap.add_argument("--compare-name", default=None, help="Display label for --compare")
    ap.add_argument("--data-root", type=Path, default=derived_dir("4tu-ieee_yolo-obb"))
    ap.add_argument("--angles", default="30,45,60,90", help="Comma-separated rotation angles to evaluate")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--json", type=Path, default=None, help="Optional path to dump raw results as JSON")
    args = ap.parse_args()

    angles = [int(a) for a in args.angles.split(",") if a.strip()]

    images_val_dir = args.data_root / "images" / "val"
    labels_val_dir = args.data_root / "labels" / "val"
    data_yaml_path = args.data_root / "spines.yaml"
    if not images_val_dir.is_dir():
        print(f"val images dir not found: {images_val_dir}", file=sys.stderr)
        sys.exit(1)
    base_yaml = yaml.safe_load(data_yaml_path.read_text())

    buckets = bucket_val_images(images_val_dir, angles)
    if not buckets:
        print("No val images found.", file=sys.stderr)
        sys.exit(1)

    YOLO = ensure_ultralytics()

    weights_list = [(args.weights, args.weights_name or args.weights.stem)]
    if args.compare:
        weights_list.append((args.compare, args.compare_name or args.compare.stem))

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for weights, label in weights_list:
        print(f"\n=== Evaluating: {label} ({weights}) ===", flush=True)
        bucket_results: dict[str, dict[str, float]] = {}
        for bucket_name, images in buckets.items():
            n_inst = count_instances(images, images_val_dir, labels_val_dir)
            print(f"  {bucket_name}: {len(images)} images, {n_inst} GT instances ...", flush=True)
            metrics = run_bucket_val(
                YOLO, weights, base_yaml, args.data_root, bucket_name, images,
                args.imgsz, args.conf, args.iou, args.device,
            )
            metrics["n_instances"] = n_inst
            bucket_results[bucket_name] = metrics
        all_results[label] = bucket_results

    # Report
    bucket_order = list(buckets.keys())
    header = f"{'model':<24s} {'bucket':<10s} {'n_img':>6s} {'n_inst':>7s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s}"
    print("\n" + header)
    print("-" * len(header))
    for label, bucket_results in all_results.items():
        for bucket_name in bucket_order:
            m = bucket_results[bucket_name]
            print(
                f"{label:<24s} {bucket_name:<10s} {m['n_images']:>6d} {m['n_instances']:>7d} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['map50']:>7.3f} {m['map50_95']:>9.3f}"
            )

    if len(weights_list) == 2:
        (new_label, _), (old_label, _) = weights_list[0], weights_list[1]
        new_label, old_label = weights_list[0][1], weights_list[1][1]
        print(f"\nDelta ({new_label} - {old_label}), mAP50 / recall per bucket:")
        for bucket_name in bucket_order:
            new_m = all_results[new_label][bucket_name]
            old_m = all_results[old_label][bucket_name]
            d_map = new_m["map50"] - old_m["map50"]
            d_rec = new_m["recall"] - old_m["recall"]
            flag = "  <-- regression" if d_map < -0.05 or d_rec < -0.05 else ""
            print(f"  {bucket_name:<10s} mAP50 {d_map:+.3f}   recall {d_rec:+.3f}{flag}")

    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved raw results to {args.json}")


if __name__ == "__main__":
    main()
