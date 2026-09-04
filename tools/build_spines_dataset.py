#!/usr/bin/env python3
"""
Merge 4TU + IEEE + cleaned open-shelves + roboflow into one YOLO-OBB dataset,
and add a rotated subset of the (4TU+IEEE) validation split so checkpoint
selection rewards angle-robust models.

Sources:
  4TU : already converted at --tu-dataset (images/labels/{train,val}).
  IEEE: raw COCO under ``ieee-book-spine`` (fetched via tools/fetch_raw.py).
        train2017 -> train, test2017 -> val.
        Frames failing shared label-QA are skipped.
  open-shelves / roboflow: cleaned derived trees (active labels only;
        quarantine_* folders are ignored). Included by default.

Rotation strategy: training rotation is online (degrees=90). Static rotated
copies are added only to val from 4TU+IEEE records (acceptance buckets).

Usage:
  python tools/build_spines_dataset.py
  # legacy 4TU+IEEE only:
  python tools/build_spines_dataset.py --no-shelves --out $BOOK_SPINES_DATA/derived/4tu-ieee_yolo-obb

Dev smoke:
  python tools/build_spines_dataset.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_obb_labels import QuadBox, label_qa_reject_reason, shoelace  # noqa: E402
from derived_meta import write_derived_source  # noqa: E402
from fetch_raw import unpacked_raw  # noqa: E402
from paths import derived_dir  # noqa: E402
from train_4tu_obb import CLASS_NAME, corners_to_yolo_line, polygon_to_obb_corners  # noqa: E402

Quad = list[tuple[float, float]]
RotationRecord = tuple[str, bytes, int, int, list[Quad]]


def quads_to_audit_boxes(quads: list[Quad]) -> list[QuadBox]:
    """Pixel quads → audit QuadBox list for shared label-QA rules."""
    out: list[QuadBox] = []
    for q in quads:
        pts = np.array(q, dtype=np.float32)
        edges = []
        for i in range(4):
            d = pts[(i + 1) % 4] - pts[i]
            L = float(np.hypot(float(d[0]), float(d[1])))
            ang = math.degrees(math.atan2(float(d[1]), float(d[0]))) % 180
            if ang > 90:
                ang -= 180
            edges.append((L, ang))
        edges.sort(key=lambda t: -t[0])
        long_L, ang = edges[0]
        shorts = sorted(e[0] for e in edges)[:2]
        short_L = max(1e-6, 0.5 * (shorts[0] + shorts[1]))
        out.append(QuadBox(pts=pts, area=shoelace(pts), aspect=long_L / short_L, angle=ang))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ieee-id",
        default="ieee-book-spine",
        help="Raw dataset id (SOURCE.md under raw/<id>/). Fetched/unpacked via fetch_raw.",
    )
    p.add_argument(
        "--ieee-root",
        type=Path,
        default=None,
        help="Optional already-unpacked IEEE book_spine/ directory (skips fetch).",
    )
    p.add_argument(
        "--tu-dataset",
        type=Path,
        default=derived_dir("4tu-spines_yolo-obb"),
        help="Already-converted 4TU YOLO-OBB dataset (images/labels/{train,val}).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=derived_dir("4tu-ieee-shelves_yolo-obb"),
        help="Output derived tree (default: 4TU+IEEE+open-shelves+roboflow).",
    )
    p.add_argument(
        "--open-shelves",
        type=Path,
        default=derived_dir("open-shelves_yolo-obb"),
        help="Cleaned open-shelves YOLO-OBB derived tree.",
    )
    p.add_argument(
        "--roboflow",
        type=Path,
        default=derived_dir("roboflow-book-spine-obb_yolo-obb"),
        help="Cleaned roboflow-book-spine-obb YOLO-OBB derived tree.",
    )
    p.add_argument(
        "--no-shelves",
        action="store_true",
        help="Skip open-shelves + roboflow (legacy 4TU+IEEE only).",
    )
    p.add_argument("--min-size", type=float, default=4.0, help="Min OBB side in pixels.")
    p.add_argument(
        "--rotate-val-fraction",
        type=float,
        default=0.25,
        help="Fraction of the combined val split to also add as rotated copies.",
    )
    p.add_argument("--rotate-angles", default="30,45,60,90", help="Comma-separated degrees, cycled across picks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Cap images per source/split (0 = all). For dev iteration.")
    p.add_argument(
        "--tile",
        type=int,
        default=0,
        help="If >0, tile train images larger than --tile-threshold-mult * this into "
        "overlapping squares of this size (px). Off by default; matches on-device "
        "tiled-inference scale for the final production run.",
    )
    p.add_argument("--tile-threshold-mult", type=float, default=2.0)
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep $BOOK_SPINES_DATA/tmp/<fetch> after IEEE unpack (debug).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry: rotate-without-cropping + simple AABB-based tile clipping
# ---------------------------------------------------------------------------


def rotate_image_and_quads(
    img: np.ndarray, quads: list[Quad], angle_deg: float
) -> tuple[np.ndarray, list[Quad], int, int]:
    """Rotate `img` about its center by `angle_deg`, expanding the canvas so
    nothing is cropped, and apply the same affine transform to every quad."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - center[0]
    m[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(img, m, (new_w, new_h), borderValue=(114, 114, 114))

    new_quads: list[Quad] = []
    for quad in quads:
        new_quads.append(
            [(m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]) for x, y in quad]
        )
    return rotated, new_quads, new_w, new_h


def tile_image_and_quads(
    img: np.ndarray, quads: list[Quad], tile_size: int, overlap: float
) -> list[tuple[np.ndarray, list[Quad]]]:
    """Split `img` into overlapping tile_size x tile_size crops. A quad is kept
    in a tile if >=30% of its (pre-clip) AABB area falls inside that tile;
    surviving quads are translated + clamped into tile-local pixel space."""
    h, w = img.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))
    xs = sorted(set(list(range(0, max(w - tile_size, 0) + 1, step)) + [max(w - tile_size, 0)]))
    ys = sorted(set(list(range(0, max(h - tile_size, 0) + 1, step)) + [max(h - tile_size, 0)]))

    tiles: list[tuple[np.ndarray, list[Quad]]] = []
    for y0 in ys:
        for x0 in xs:
            tw, th = min(tile_size, w), min(tile_size, h)
            crop = img[y0 : y0 + th, x0 : x0 + tw]
            tile_quads: list[Quad] = []
            for quad in quads:
                local = [(px - x0, py - y0) for px, py in quad]
                lxs, lys = [p[0] for p in local], [p[1] for p in local]
                aabb_x0, aabb_x1 = min(lxs), max(lxs)
                aabb_y0, aabb_y1 = min(lys), max(lys)
                inter_w = max(0.0, min(aabb_x1, tw) - max(aabb_x0, 0.0))
                inter_h = max(0.0, min(aabb_y1, th) - max(aabb_y0, 0.0))
                full_area = max(1e-6, (aabb_x1 - aabb_x0) * (aabb_y1 - aabb_y0))
                if (inter_w * inter_h) / full_area < 0.3:
                    continue
                tile_quads.append([(min(max(px, 0.0), tw), min(max(py, 0.0), th)) for px, py in local])
            tiles.append((crop, tile_quads))
    return tiles


# ---------------------------------------------------------------------------
# 4TU: copy already-converted YOLO-OBB files with a source prefix
# ---------------------------------------------------------------------------


def process_4tu(
    tu_dataset: Path, split: str, out_split: str, out_root: Path, limit: int
) -> tuple[int, int, list[RotationRecord]]:
    img_dir = tu_dataset / "images" / split
    lbl_dir = tu_dataset / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"4TU dataset missing {img_dir}. Run tools/train_4tu_obb.py first.")

    img_paths = sorted(p for p in img_dir.glob("*") if p.is_file())
    if limit:
        img_paths = img_paths[:limit]

    count_images = count_boxes = 0
    rotation_records: list[RotationRecord] = []
    skipped_qa = 0

    for img_path in img_paths:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue
        lines = [line for line in lbl_path.read_text().splitlines() if line.strip()]
        if not lines:
            continue

        arr = cv2.imread(str(img_path))
        if arr is None:
            continue
        height, width = arr.shape[:2]
        quads: list[Quad] = []
        for line in lines:
            coords = [float(v) for v in line.split()[1:]]
            quads.append([(coords[i] * width, coords[i + 1] * height) for i in range(0, 8, 2)])
        if label_qa_reject_reason(quads_to_audit_boxes(quads), width, height):
            skipped_qa += 1
            continue

        stem = f"4tu_{out_split}_{img_path.stem}"
        img_out = out_root / "images" / out_split / f"{stem}{img_path.suffix.lower()}"
        lbl_out = out_root / "labels" / out_split / f"{stem}.txt"
        shutil.copy2(img_path, img_out)
        lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count_images += 1
        count_boxes += len(lines)

        if out_split == "val":
            rotation_records.append((stem, img_path.read_bytes(), width, height, quads))

    if skipped_qa:
        print(f"  4TU {split}: skipped {skipped_qa} label-QA fails", flush=True)
    return count_images, count_boxes, rotation_records


def process_yolo_derived(
    src_root: Path,
    prefix: str,
    out_root: Path,
    limit: int,
) -> tuple[int, int, int, int]:
    """Copy active train/val from a cleaned YOLO-OBB derived tree (skip quarantine)."""
    if not src_root.is_dir():
        print(f"  skip {prefix}: missing {src_root}", flush=True)
        return 0, 0, 0, 0

    totals = {"train": (0, 0), "val": (0, 0)}
    for split in ("train", "val"):
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        paths = sorted(p for p in img_dir.glob("*") if p.is_file())
        if limit:
            paths = paths[:limit]
        n_img = n_box = 0
        skipped_qa = 0
        for img_path in paths:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue
            lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            arr = cv2.imread(str(img_path))
            if arr is None:
                continue
            h, w = arr.shape[:2]
            quads: list[Quad] = []
            for line in lines:
                coords = [float(v) for v in line.split()[1:]]
                quads.append([(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)])
            if label_qa_reject_reason(quads_to_audit_boxes(quads), w, h):
                skipped_qa += 1
                continue
            stem = f"{prefix}_{split}_{img_path.stem}"
            shutil.copy2(img_path, out_root / "images" / split / f"{stem}{img_path.suffix.lower()}")
            (out_root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1
            n_box += len(lines)
        if skipped_qa:
            print(f"  {prefix} {split}: skipped {skipped_qa} label-QA fails", flush=True)
        totals[split] = (n_img, n_box)
    return totals["train"][0], totals["train"][1], totals["val"][0], totals["val"][1]


# ---------------------------------------------------------------------------
# IEEE: COCO polygons from an unpacked book_spine/ directory
# ---------------------------------------------------------------------------


def load_ieee_split(
    data_root: Path, ann_name: str
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads((data_root / "annotations" / ann_name).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return images, anns_by_image


def process_ieee(
    data_root: Path,
    ann_name: str,
    image_subdir: str,
    out_split: str,
    out_root: Path,
    min_size: float,
    limit: int,
) -> tuple[int, int, list[RotationRecord]]:
    images, anns_by_image = load_ieee_split(data_root, ann_name)
    items = sorted(images.items())
    if limit:
        items = items[:limit]

    count_images = count_boxes = 0
    rotation_records: list[RotationRecord] = []

    for image_id, img_meta in items:
        file_name = img_meta["file_name"]
        width, height = int(img_meta["width"]), int(img_meta["height"])

        quads: list[Quad] = []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd"):
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            flat = seg[0]
            pts = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
            corners = polygon_to_obb_corners(pts)
            if corners is not None:
                quads.append(corners)

        lines = [line for corners in quads if (line := corners_to_yolo_line(corners, width, height, min_size))]
        if not lines:
            continue

        audit_quads = [
            corners
            for corners in quads
            if corners_to_yolo_line(corners, width, height, min_size)
        ]
        qa = label_qa_reject_reason(quads_to_audit_boxes(audit_quads), width, height)
        if qa:
            continue

        stem = f"ieee_{out_split}_{image_id:05d}"
        ext = (Path(file_name).suffix.lstrip(".") or "jpg").lower()
        img_path = data_root / "images" / image_subdir / file_name
        raw = img_path.read_bytes()

        img_out = out_root / "images" / out_split / f"{stem}.{ext}"
        lbl_out = out_root / "labels" / out_split / f"{stem}.txt"
        img_out.write_bytes(raw)
        lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count_images += 1
        count_boxes += len(lines)

        if out_split == "val":
            rotation_records.append((stem, raw, width, height, audit_quads))

    return count_images, count_boxes, rotation_records


# ---------------------------------------------------------------------------
# Rotated validation subset + optional train tiling
# ---------------------------------------------------------------------------


def add_rotated_val(
    records: list[RotationRecord],
    angles: list[int],
    fraction: float,
    out_root: Path,
    min_size: float,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    picks = pool[: int(round(len(pool) * fraction))]

    added_images = added_boxes = 0
    for i, (stem, raw, _width, _height, quads) in enumerate(picks):
        angle = angles[i % len(angles)]
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        rotated, new_quads, new_w, new_h = rotate_image_and_quads(arr, quads, angle)
        lines = [line for corners in new_quads if (line := corners_to_yolo_line(corners, new_w, new_h, min_size))]
        if not lines:
            continue

        out_stem = f"{stem}_rot{angle}"
        cv2.imwrite(str(out_root / "images" / "val" / f"{out_stem}.jpg"), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_root / "labels" / "val" / f"{out_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        added_images += 1
        added_boxes += len(lines)

    return added_images, added_boxes


def apply_tiling(out_root: Path, tile_size: int, threshold_mult: float, overlap: float, min_size: float) -> None:
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    threshold = tile_size * threshold_mult

    added_images = added_boxes = removed_images = 0
    for img_path in sorted(img_dir.glob("*")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue
        arr = cv2.imread(str(img_path))
        if arr is None:
            continue
        h, w = arr.shape[:2]
        if max(w, h) <= threshold:
            continue

        quads: list[Quad] = []
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            coords = [float(v) for v in line.split()[1:]]
            quads.append([(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)])

        for i, (crop, tile_quads) in enumerate(tile_image_and_quads(arr, quads, tile_size, overlap)):
            th, tw = crop.shape[:2]
            lines = [line for corners in tile_quads if (line := corners_to_yolo_line(corners, tw, th, min_size))]
            if not lines:
                continue
            stem = f"{img_path.stem}_tile{i:02d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            added_images += 1
            added_boxes += len(lines)

        img_path.unlink()
        lbl_path.unlink()
        removed_images += 1

    print(f"Tiling: replaced {removed_images} oversized train images with {added_images} tiles ({added_boxes} boxes)")


# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    out_root = args.out.expanduser().resolve()
    tu_dataset = args.tu_dataset.expanduser().resolve()

    if not tu_dataset.is_dir():
        print(f"4TU dataset not found: {tu_dataset}", file=sys.stderr)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True)
        (out_root / "labels" / split).mkdir(parents=True)

    angles = [int(a) for a in args.rotate_angles.split(",") if a.strip()]

    tu_train_imgs, tu_train_boxes, _ = process_4tu(tu_dataset, "train", "train", out_root, args.limit)
    tu_val_imgs, tu_val_boxes, tu_val_records = process_4tu(tu_dataset, "val", "val", out_root, args.limit)
    print(f"4TU  train: {tu_train_imgs:5d} images / {tu_train_boxes:6d} boxes")
    print(f"4TU  val:   {tu_val_imgs:5d} images / {tu_val_boxes:6d} boxes")

    with unpacked_raw(args.ieee_id, keep_tmp=args.keep_tmp, override_root=args.ieee_root) as ieee_root:
        ieee_train_imgs, ieee_train_boxes, _ = process_ieee(
            ieee_root, "instances_train2017.json", "train2017", "train", out_root, args.min_size, args.limit
        )
        ieee_val_imgs, ieee_val_boxes, ieee_val_records = process_ieee(
            ieee_root, "instances_test2017.json", "test2017", "val", out_root, args.min_size, args.limit
        )
    print(f"IEEE train: {ieee_train_imgs:5d} images / {ieee_train_boxes:6d} boxes")
    print(f"IEEE val:   {ieee_val_imgs:5d} images / {ieee_val_boxes:6d} boxes")

    os_train = os_val = rf_train = rf_val = 0
    os_train_b = os_val_b = rf_train_b = rf_val_b = 0
    sources = ["4tu-spines_yolo-obb", "ieee-book-spine"]
    if not args.no_shelves:
        os_train, os_train_b, os_val, os_val_b = process_yolo_derived(
            args.open_shelves.expanduser().resolve(), "os", out_root, args.limit
        )
        rf_train, rf_train_b, rf_val, rf_val_b = process_yolo_derived(
            args.roboflow.expanduser().resolve(), "rf", out_root, args.limit
        )
        print(f"OS   train: {os_train:5d} images / {os_train_b:6d} boxes")
        print(f"OS   val:   {os_val:5d} images / {os_val_b:6d} boxes")
        print(f"RF   train: {rf_train:5d} images / {rf_train_b:6d} boxes")
        print(f"RF   val:   {rf_val:5d} images / {rf_val_b:6d} boxes")
        sources.extend(["open-shelves_yolo-obb", "roboflow-book-spine-obb_yolo-obb"])

    added_images, added_boxes = add_rotated_val(
        tu_val_records + ieee_val_records, angles, args.rotate_val_fraction, out_root, args.min_size, args.seed
    )
    print(f"Rotated val additions: {added_images:5d} images / {added_boxes:6d} boxes (angles {angles})")

    if args.tile > 0:
        apply_tiling(out_root, args.tile, args.tile_threshold_mult, args.tile_overlap, args.min_size)

    yaml_path = out_root / "spines.yaml"
    yaml_path.write_text(
        "\n".join([f"path: {out_root}", "train: images/train", "val: images/val", "names:", f"  0: {CLASS_NAME}", ""]),
        encoding="utf-8",
    )

    # Stratified unrotated subset for every-epoch train val (full val kept for sweeps).
    try:
        from make_val_train_subset import build_val_train_subset

        build_val_train_subset(out_root, count=375, seed=args.seed)
    except Exception as exc:  # noqa: BLE001 — build should still succeed if subset helper fails
        print(f"warning: val_train subset skipped ({exc})", file=sys.stderr)

    final_train = len(list((out_root / "images" / "train").glob("*")))
    final_val = len(list((out_root / "images" / "val").glob("*")))
    title = (
        "4TU + IEEE + open-shelves + roboflow YOLO-OBB merge"
        if not args.no_shelves
        else "4TU + IEEE YOLO-OBB merge with optional rotated val / tiling"
    )
    write_derived_source(
        out_root,
        derived_id=out_root.name,
        title=title,
        sources=sources,
        script="tools/build_spines_dataset.py",
        flags={
            "ieee_id": args.ieee_id,
            "tu_dataset": str(tu_dataset),
            "include_shelves": not args.no_shelves,
            "open_shelves": str(args.open_shelves),
            "roboflow": str(args.roboflow),
            "rotate_val_fraction": args.rotate_val_fraction,
            "rotate_angles": args.rotate_angles,
            "tile": args.tile,
            "tile_threshold_mult": args.tile_threshold_mult,
            "tile_overlap": args.tile_overlap,
            "min_size": args.min_size,
            "limit": args.limit,
            "seed": args.seed,
            "label_qa": "label_qa_reject_reason on 4TU/IEEE/OS/RF",
            "train_images": final_train,
            "val_images": final_val,
        },
        notes=(
            "Rotated val copies (4TU+IEEE only) are for checkpoint selection; "
            "train rotation is online (degrees=90). Shelves use cleaned active labels."
        ),
    )
    print(f"Combined dataset: {final_train} train images, {final_val} val images -> {out_root}")
    print(f"Dataset YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
