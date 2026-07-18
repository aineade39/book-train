#!/usr/bin/env python3
"""
Convert the 4TU / Shenzhen book-spine LabelMe JSONs into a Create ML
object-detection dataset.

Dataset: https://doi.org/10.4121/uuid:33f2a166-de13-4505-b359-2b202c491fd8

Usage:
  python3 tools/convert_4tu_to_createml.py \\
      --input ~/data/4tu-spines \\
      --output ~/data/createml-spines \\
      --split 0.8

Then in Create ML → Object Detection, drag in the `train` folder
(images + annotations.json). Optionally use `valid` as the test set.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any


CREATE_ML_LABEL = "spine"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert 4TU LabelMe spine JSONs to Create ML object detection format."
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Folder containing LabelMe .json files (recursive).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output folder (will create train/ and optionally valid/).",
    )
    p.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train fraction (rest goes to valid). Use 1.0 for train-only. Default: 0.8",
    )
    p.add_argument(
        "--label",
        default=CREATE_ML_LABEL,
        help=f'Class name written into Create ML annotations. Default: "{CREATE_ML_LABEL}"',
    )
    p.add_argument(
        "--min-size",
        type=float,
        default=4.0,
        help="Drop boxes with width or height below this many pixels. Default: 4",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/valid split.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N JSON files (0 = all). Useful for a smoke test.",
    )
    return p.parse_args()


def find_json_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.json"))
    # Skip Create ML / COCO style sidecars if re-running into the same tree.
    return [f for f in files if f.name.lower() != "annotations.json"]


def decode_image_bytes(doc: dict[str, Any], json_path: Path) -> tuple[bytes, str]:
    """Return (image_bytes, file_extension_without_dot)."""
    image_data = doc.get("imageData")
    if image_data:
        raw = base64.b64decode(image_data)
        ext = sniff_image_ext(raw)
        return raw, ext

    image_path = doc.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_path}: no imageData or imagePath")

    src = (json_path.parent / image_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"{json_path}: image not found at {src}")

    raw = src.read_bytes()
    ext = src.suffix.lstrip(".").lower() or sniff_image_ext(raw)
    return raw, ext


def sniff_image_ext(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    # LabelMe often stores JPEG even without a clear path.
    return "jpg"


def polygon_to_box(points: list[list[float]]) -> tuple[float, float, float, float] | None:
    """Axis-aligned box from polygon → (x_min, y_min, width, height)."""
    if not points or len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return x0, y0, w, h


def shape_to_createml(
    shape: dict[str, Any],
    label: str,
    min_size: float,
) -> dict[str, Any] | None:
    shape_type = (shape.get("shape_type") or "polygon").lower()
    points = shape.get("points") or []

    if shape_type in ("rectangle", "polygon", ""):
        box = polygon_to_box(points)
    else:
        # Circles / lines are not useful spine labels; skip.
        return None

    if box is None:
        return None

    x0, y0, w, h = box
    if w < min_size or h < min_size:
        return None

    # Create ML defaults: top-left origin, center anchor.
    return {
        "label": label,
        "coordinates": {
            "x": round(x0 + w / 2.0, 2),
            "y": round(y0 + h / 2.0, 2),
            "width": round(w, 2),
            "height": round(h, 2),
        },
    }


def convert_one(
    json_path: Path,
    out_dir: Path,
    index: int,
    label: str,
    min_size: float,
) -> dict[str, Any] | None:
    with json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    shapes = doc.get("shapes") or []
    annotations: list[dict[str, Any]] = []
    for shape in shapes:
        ann = shape_to_createml(shape, label=label, min_size=min_size)
        if ann is not None:
            annotations.append(ann)

    if not annotations:
        return None

    image_bytes, ext = decode_image_bytes(doc, json_path)
    if ext not in ("jpg", "jpeg", "png"):
        # Create ML is happiest with JPEG/PNG.
        ext = "jpg"

    filename = f"spine_{index:04d}.{ext if ext != 'jpeg' else 'jpg'}"
    (out_dir / filename).write_bytes(image_bytes)

    return {
        "imagefilename": filename,
        "annotation": annotations,
    }


def write_split(
    src_files: list[Path],
    out_root: Path,
    split_name: str,
    label: str,
    min_size: float,
) -> tuple[int, int]:
    """Write images + annotations.json. Returns (image_count, box_count)."""
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []
    box_count = 0

    for i, json_path in enumerate(src_files, start=1):
        try:
            record = convert_one(
                json_path,
                out_dir=out_dir,
                index=i,
                label=label,
                min_size=min_size,
            )
        except Exception as exc:  # noqa: BLE001 — keep batching; report and continue
            print(f"  skip {json_path.name}: {exc}", file=sys.stderr)
            continue

        if record is None:
            print(f"  skip {json_path.name}: no usable boxes", file=sys.stderr)
            continue

        annotations.append(record)
        box_count += len(record["annotation"])

    ann_path = out_dir / "annotations.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
        f.write("\n")

    return len(annotations), box_count


def main() -> int:
    args = parse_args()

    if not 0.0 < args.split <= 1.0:
        print("--split must be in (0, 1]", file=sys.stderr)
        return 1

    if not args.input.is_dir():
        print(f"Input folder not found: {args.input}", file=sys.stderr)
        return 1

    json_files = find_json_files(args.input)
    if not json_files:
        print(f"No .json files under {args.input}", file=sys.stderr)
        return 1

    if args.limit > 0:
        json_files = json_files[: args.limit]

    print(f"Found {len(json_files)} LabelMe JSON file(s)")

    if args.output.exists():
        print(f"Clearing existing output: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    rng = random.Random(args.seed)
    shuffled = list(json_files)
    rng.shuffle(shuffled)

    if args.split >= 1.0:
        train_files, valid_files = shuffled, []
    else:
        n_train = max(1, int(round(len(shuffled) * args.split)))
        # Keep at least one valid sample when possible.
        if len(shuffled) > 1 and n_train >= len(shuffled):
            n_train = len(shuffled) - 1
        train_files = shuffled[:n_train]
        valid_files = shuffled[n_train:]

    train_n, train_boxes = write_split(
        train_files, args.output, "train", args.label, args.min_size
    )
    print(f"train: {train_n} images, {train_boxes} boxes → {args.output / 'train'}")

    if valid_files:
        valid_n, valid_boxes = write_split(
            valid_files, args.output, "valid", args.label, args.min_size
        )
        print(f"valid: {valid_n} images, {valid_boxes} boxes → {args.output / 'valid'}")
    else:
        print("valid: skipped (--split 1.0)")

    print(
        "\nNext: open Create ML → Object Detection → "
        "drag the train folder in as Training Data "
        "(and valid as Testing Data if present)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
