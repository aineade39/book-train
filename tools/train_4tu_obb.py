#!/usr/bin/env python3
"""
End-to-end: 4TU book-spine LabelMe polygons → YOLO-OBB → Core ML.

Dataset: https://doi.org/10.4121/uuid:33f2a166-de13-4505-b359-2b202c491fd8

Setup (once):

  cd /Users/joebr/dev/book-train
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip ultralytics opencv-python-headless

Full run (your 4TU dump is under ~/Downloads):

  source .venv/bin/activate
  python tools/train_4tu_obb.py --source ~/Downloads

Smoke test (3 images, convert only):

  python tools/train_4tu_obb.py --source ~/Downloads --limit 3 --skip-train --skip-export
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

CLASS_NAME = "spine"
CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(
        description="Convert 4TU spines to YOLO-OBB, train, and export Core ML."
    )
    p.add_argument(
        "--source",
        type=Path,
        default=home / "Downloads",
        help="Folder that contains (or is) the 4TU dump. "
        "Also accepts ~/dowload or a path ending in dataset_661. "
        f"Default: {home / 'Downloads'}",
    )
    p.add_argument(
        "--dataset-out",
        type=Path,
        default=home / "data" / "yolo-obb-spines",
        help="Where to write YOLO images/labels/yaml.",
    )
    p.add_argument(
        "--runs-out",
        type=Path,
        default=home / "data" / "yolo-obb-runs",
        help="Ultralytics project directory for training runs.",
    )
    p.add_argument(
        "--export-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Folder for the exported .mlpackage (default: repo models/).",
    )
    p.add_argument("--split", type=float, default=0.8, help="Train fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024, help="OBB models often use 1024.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--model",
        default="yolo11n-obb.pt",
        help="Base checkpoint (yolo11n-obb.pt or yolo26n-obb.pt, etc.).",
    )
    p.add_argument(
        "--device",
        default="mps",
        help="Ultralytics device: mps (Apple GPU), cpu, or 0 for CUDA.",
    )
    p.add_argument("--limit", type=int, default=0, help="Max JSONs (0=all). Smoke-test with 20.")
    p.add_argument("--min-size", type=float, default=4.0, help="Min box side in pixels.")
    p.add_argument("--skip-convert", action="store_true", help="Reuse existing dataset-out.")
    p.add_argument("--skip-train", action="store_true", help="Only convert / export.")
    p.add_argument("--skip-export", action="store_true", help="Train but do not export Core ML.")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Existing .pt to export (skips train). Useful after a prior run.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Locate + read 4TU LabelMe JSONs
# ---------------------------------------------------------------------------

def is_labelme_dir(path: Path) -> bool:
    """True if folder looks like the 4TU LabelMe dump (not random Downloads JSON)."""
    samples = list(path.glob("IMG_*.json"))[:3]
    if not samples:
        samples = list(path.glob("*.json"))[:5]
    if not samples:
        return False
    ok = 0
    for sample in samples:
        try:
            doc = load_labelme(sample)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, dict) and ("shapes" in doc) and (
            doc.get("imageData") or doc.get("imagePath")
        ):
            ok += 1
    return ok > 0


def resolve_json_root(source: Path) -> Path:
    """Find the directory that actually holds LabelMe JSON files."""
    candidates: list[Path] = []
    source = source.expanduser().resolve()

    for alt in (
        source,
        source / "dataset_661",
        Path.home() / "dowload",
        Path.home() / "dowload" / "dataset_661",
        Path.home() / "download",
        Path.home() / "Downloads",
        Path.home() / "data" / "4tu-spines",
        Path.home() / "data" / "4tu-spines" / "dataset_661",
    ):
        if alt.exists():
            candidates.append(alt.resolve())

    for root in list(candidates):
        if not root.is_dir():
            continue
        direct = root / "dataset_661"
        if direct.is_dir():
            candidates.append(direct)
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            name = child.name.lower().replace(" ", "-")
            if child.is_dir() and ("book-spine" in name or "dataset_661" in name):
                candidates.append(child)
                if (child / "dataset_661").is_dir():
                    candidates.append(child / "dataset_661")
            # Unpacked 4TU folder often nests dataset_661
            if child.is_dir():
                nested = child / "dataset_661"
                if nested.is_dir():
                    candidates.append(nested)

    seen: set[Path] = set()
    labelme_hits: list[Path] = []
    for cand in candidates:
        if cand in seen or not cand.is_dir():
            continue
        seen.add(cand)
        if is_labelme_dir(cand):
            labelme_hits.append(cand)

    # Prefer dataset_661 / folders with the most IMG_*.json files
    if labelme_hits:
        labelme_hits.sort(
            key=lambda p: (
                0 if p.name == "dataset_661" else 1,
                -len(list(p.glob("IMG_*.json"))),
                str(p),
            )
        )
        return labelme_hits[0]

    raise FileNotFoundError(
        f"No LabelMe *.json files found under {source}. "
        "Unpack the 4TU zip so dataset_661/*.json is reachable, then pass "
        "--source ~/data/4tu-spines/dataset_661"
    )


def load_labelme(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise ValueError(f"Could not decode {path}: {last_err}")


def sniff_ext(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return "jpg"


def image_size_from_bytes(raw: bytes) -> tuple[int, int]:
    """Width, height without requiring OpenCV (JPEG/PNG headers)."""
    # PNG
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        w = int.from_bytes(raw[16:20], "big")
        h = int.from_bytes(raw[20:24], "big")
        return w, h
    # JPEG: scan for SOF0/SOF2
    if raw.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF
                h = int.from_bytes(raw[i + 5 : i + 7], "big")
                w = int.from_bytes(raw[i + 7 : i + 9], "big")
                return w, h
            if marker == 0xD9:
                break
            if marker == 0xD8 or marker == 0x01 or (0xD0 <= marker <= 0xD7):
                i += 2
                continue
            length = int.from_bytes(raw[i + 2 : i + 4], "big")
            i += 2 + length
    raise ValueError("unsupported or corrupt image header")


def extract_image(doc: dict[str, Any], json_path: Path) -> tuple[bytes, str, int, int]:
    w = int(doc.get("imageWidth") or 0)
    h = int(doc.get("imageHeight") or 0)
    if doc.get("imageData"):
        raw = base64.b64decode(doc["imageData"])
        ext = sniff_ext(raw)
        if not w or not h:
            w, h = image_size_from_bytes(raw)
        return raw, ext, w, h

    image_path = doc.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_path}: no imageData/imagePath")
    stem = json_path.stem
    for cand in (
        json_path.with_suffix(".jpg"),
        json_path.with_suffix(".jpeg"),
        json_path.with_suffix(".png"),
        json_path.parent / Path(str(image_path).replace("\\", "/")).name,
    ):
        if cand.is_file():
            raw = cand.read_bytes()
            if not w or not h:
                w, h = image_size_from_bytes(raw)
            return raw, cand.suffix.lstrip(".") or sniff_ext(raw), w, h
    raise FileNotFoundError(f"{json_path}: image for {stem} not found")


# ---------------------------------------------------------------------------
# Polygon → YOLO OBB (normalized 4 corners)
# ---------------------------------------------------------------------------

def order_corners_ccw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def polygon_to_obb_corners(points: list[list[float]]) -> list[tuple[float, float]] | None:
    """Return 4 corners (pixel space), CCW."""
    if len(points) < 3:
        return None
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) == 4:
        return order_corners_ccw(pts)

    try:
        import cv2
        import numpy as np
    except ImportError:
        # Fallback AABB when OpenCV isn't installed yet (still valid OBB quads).
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return order_corners_ccw([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(arr)
    box = cv2.boxPoints(rect)
    return order_corners_ccw([(float(x), float(y)) for x, y in box])


def corners_to_yolo_line(
    corners: list[tuple[float, float]],
    image_w: int,
    image_h: int,
    min_size: float,
) -> str | None:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if (max(xs) - min(xs)) < min_size or (max(ys) - min(ys)) < min_size:
        return None

    norm: list[float] = []
    for x, y in corners:
        nx = min(1.0, max(0.0, x / image_w))
        ny = min(1.0, max(0.0, y / image_h))
        norm.extend((nx, ny))

    if math.isclose(min(norm[0::2]), max(norm[0::2])) or math.isclose(
        min(norm[1::2]), max(norm[1::2])
    ):
        return None

    body = " ".join(f"{v:.6f}" for v in norm)
    return f"{CLASS_ID} {body}"


# ---------------------------------------------------------------------------
# Convert dataset
# ---------------------------------------------------------------------------

def convert_dataset(
    json_root: Path,
    out_root: Path,
    split: float,
    seed: int,
    limit: int,
    min_size: float,
) -> Path:
    json_files = sorted(json_root.glob("*.json"))
    if not json_files:
        json_files = [
            f for f in sorted(json_root.rglob("*.json")) if f.name != "annotations.json"
        ]
    if not json_files:
        raise FileNotFoundError(f"No JSON under {json_root}")

    if limit > 0:
        json_files = json_files[:limit]

    rng = random.Random(seed)
    shuffled = list(json_files)
    rng.shuffle(shuffled)

    n_train = max(1, int(round(len(shuffled) * split)))
    if len(shuffled) > 1 and n_train >= len(shuffled):
        n_train = len(shuffled) - 1
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:] or shuffled[-1:],
    }

    if out_root.exists():
        shutil.rmtree(out_root)

    for split_name in ("train", "val"):
        (out_root / "images" / split_name).mkdir(parents=True)
        (out_root / "labels" / split_name).mkdir(parents=True)

    total_boxes = 0
    total_images = 0
    skipped = 0

    for split_name, files in splits.items():
        for i, jp in enumerate(files, start=1):
            try:
                doc = load_labelme(jp)
                image_bytes, ext, w, h = extract_image(doc, jp)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {jp.name}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            lines: list[str] = []
            for shape in doc.get("shapes") or []:
                st = (shape.get("shape_type") or "polygon").lower()
                if st not in ("polygon", "rectangle", ""):
                    continue
                corners = polygon_to_obb_corners(shape.get("points") or [])
                if corners is None:
                    continue
                line = corners_to_yolo_line(corners, w, h, min_size)
                if line:
                    lines.append(line)

            if not lines:
                print(f"  skip {jp.name}: no usable spines", file=sys.stderr)
                skipped += 1
                continue

            stem = f"spine_{split_name}_{i:04d}"
            if ext == "jpeg":
                ext = "jpg"
            img_path = out_root / "images" / split_name / f"{stem}.{ext}"
            lbl_path = out_root / "labels" / split_name / f"{stem}.txt"
            img_path.write_bytes(image_bytes)
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            total_images += 1
            total_boxes += len(lines)

    yaml_path = out_root / "spines.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        f"Converted {total_images} images / {total_boxes} OBB boxes "
        f"(skipped {skipped}) → {out_root}"
    )
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Train + export
# ---------------------------------------------------------------------------

def ensure_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in this Python.\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install ultralytics opencv-python-headless"
        ) from exc
    return YOLO


def train_obb(
    yaml_path: Path,
    model_name: str,
    runs_out: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    YOLO = ensure_ultralytics()
    runs_out.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_out),
        name="spine-obb",
        exist_ok=True,
    )
    best = runs_out / "spine-obb" / "weights" / "best.pt"
    if not best.is_file():
        best = runs_out / "spine-obb" / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"No weights found under {runs_out / 'spine-obb' / 'weights'}")
    print(f"Best weights: {best}")
    return best


def export_coreml(weights: Path, export_out: Path, imgsz: int) -> Path:
    YOLO = ensure_ultralytics()
    export_out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = model.export(format="coreml", imgsz=imgsz, nms=False)
    exported_path = Path(str(exported))
    dest = export_out / "SpineDetectorOBB.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    if exported_path.is_dir():
        shutil.copytree(exported_path, dest)
    else:
        shutil.copy2(exported_path, dest)
    print(f"Core ML package: {dest}")
    return dest


def main() -> int:
    args = parse_args()

    if not 0.0 < args.split <= 1.0:
        print("--split must be in (0, 1]", file=sys.stderr)
        return 1

    yaml_path = args.dataset_out / "spines.yaml"
    if not args.skip_convert:
        json_root = resolve_json_root(args.source)
        print(f"LabelMe JSON root: {json_root}")
        yaml_path = convert_dataset(
            json_root=json_root,
            out_root=args.dataset_out,
            split=args.split,
            seed=args.seed,
            limit=args.limit,
            min_size=args.min_size,
        )
    elif not yaml_path.is_file():
        print(f"--skip-convert set but missing {yaml_path}", file=sys.stderr)
        return 1

    weights = args.weights
    if not args.skip_train and weights is None:
        weights = train_obb(
            yaml_path=yaml_path,
            model_name=args.model,
            runs_out=args.runs_out,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    elif weights is None and not args.skip_export:
        guess = args.runs_out / "spine-obb" / "weights" / "best.pt"
        if guess.is_file():
            weights = guess
        else:
            print("No weights to export. Train first or pass --weights.", file=sys.stderr)
            return 1

    if not args.skip_export:
        assert weights is not None
        export_coreml(weights, args.export_out, args.imgsz)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
