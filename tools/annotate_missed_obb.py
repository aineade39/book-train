#!/usr/bin/env python3
"""Find missed spines in cleaned Roboflow-derived sets and annotate them.

Fixes two problems left by ``clean_roboflow_obb.py``:
  1. ``upright_score`` preferred 90° Roboflow augs, so horizontal-pile photos
     were stored rotated (labels still present, but natural horiz piles were
     replaced by the rotated twin — and some bases kept a thinner RF label set).
  2. True gaps (spines never labeled in any export) — complete via an ensemble
     of teachers (combined yolo26s + spine-obb + production Core ML when swift
     is available).

Updates derived trees **in place**:
  derived/open-shelves_yolo-obb
  derived/roboflow-book-spine-obb_yolo-obb

Example:
  .venv/bin/python tools/annotate_missed_obb.py
  .venv/bin/python tools/annotate_missed_obb.py --dataset open-shelves --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import derived_dir, raw_dir, runs_dir  # noqa: E402
from train_4tu_obb import corners_to_yolo_line  # noqa: E402

Quad = list[tuple[float, float]]


@dataclass
class Box:
    pts: Quad
    source: str
    conf: float = 1.0


@dataclass
class Variant:
    path: Path
    label_path: Path
    n: int
    n_h: int
    n_v: int
    shelf: float
    size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("open-shelves", "roboflow-book-spine-obb", "all"), default="all")
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--iou-match", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--preview", type=int, default=40)
    p.add_argument("--skip-swift", action="store_true")
    return p.parse_args()


def base_name(stem: str) -> str:
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def norm_base(b: str) -> str:
    b = b.lower()
    for s in (
        "_jpg",
        "_jpeg",
        "_png",
        "-jpg",
        "-jpeg",
        "-large",
        "_compressed_jpeg",
        "-jpg_compressed_jpeg",
        "_mov",
    ):
        b = b.replace(s, "")
    return b


def shelf_score(path: Path) -> float:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 0.0
    im = cv2.GaussianBlur(im, (3, 3), 0)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gy)) - np.mean(np.abs(gx)))


def box_angle_stats(pts: Quad) -> tuple[float, bool, bool]:
    best_L, best_ang = 0.0, 0.0
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        L = math.hypot(x1 - x0, y1 - y0)
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
        if ang > 90:
            ang -= 180
        if L > best_L:
            best_L, best_ang = L, ang
    return best_ang, abs(best_ang) < 35, abs(best_ang) > 55


def valid_quad(pts: Quad, min_edge: float = 3.0) -> bool:
    if len(pts) != 4:
        return False
    edges = []
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        edges.append(math.hypot(x1 - x0, y1 - y0))
    edges = sorted(edges)
    if edges[0] < min_edge:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) >= min_edge and (max(ys) - min(ys)) >= min_edge


def parse_boxes(label_path: Path, w: int, h: int, source: str = "gt") -> list[Box]:
    out: list[Box] = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        vals = list(map(float, parts[1:9]))
        pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]
        if valid_quad(pts):
            out.append(Box(pts=pts, source=source, conf=1.0))
    return out


def quad_iou(a: Quad, b: Quad) -> float:
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    ay0, ay1 = min(p[1] for p in a), max(p[1] for p in a)
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    by0, by1 = min(p[1] for p in b), max(p[1] for p in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-6, aa + bb - inter)


def index_raw_variants() -> dict[str, list[Variant]]:
    roots = [
        raw_dir("open-shelves/open-shelves.v9i.yolov8-obb"),
        raw_dir("roboflow-book-spine-obb/book-spine-obb.v1-obb.yolov8-obb"),
    ]
    by: dict[str, list[Variant]] = defaultdict(list)
    for root in roots:
        if not root.is_dir():
            continue
        for split in ("train", "valid", "test"):
            img_dir = root / split / "images"
            if not img_dir.is_dir():
                continue
            for ip in img_dir.iterdir():
                if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                with Image.open(ip) as im:
                    w, h = im.size
                lp = root / split / "labels" / f"{ip.stem}.txt"
                boxes = parse_boxes(lp, w, h)
                nh = sum(1 for b in boxes if box_angle_stats(b.pts)[1])
                nv = sum(1 for b in boxes if box_angle_stats(b.pts)[2])
                by[norm_base(base_name(ip.stem))].append(
                    Variant(
                        path=ip,
                        label_path=lp,
                        n=len(boxes),
                        n_h=nh,
                        n_v=nv,
                        shelf=shelf_score(ip),
                        size=(w, h),
                    )
                )
    return by


def pick_best_variant(variants: list[Variant]) -> Variant:
    """Most boxes; prefer shelf-aligned orientation (shelf_score sign matches label dom)."""

    def key(v: Variant) -> tuple:
        label_h = v.n_h >= v.n_v
        image_h = v.shelf >= 0
        agree = label_h == image_h
        # Prefer natural horizontal structure when tied (keeps pile photos unrotated)
        return (v.n, agree, v.shelf, v.size[0] * v.size[1])

    return max(variants, key=key)


def near_existing(pts: Quad, existing: list[Box], dilate_px: float) -> bool:
    """True if proposal AABB is within dilate_px of any existing box (missed neighbors)."""
    if not existing:
        return True
    ax0 = min(p[0] for p in pts) - dilate_px
    ax1 = max(p[0] for p in pts) + dilate_px
    ay0 = min(p[1] for p in pts) - dilate_px
    ay1 = max(p[1] for p in pts) + dilate_px
    for e in existing:
        bx0, bx1 = min(p[0] for p in e.pts), max(p[0] for p in e.pts)
        by0, by1 = min(p[1] for p in e.pts), max(p[1] for p in e.pts)
        if ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1:
            continue
        return True
    return False


def accept_proposal(p: Box, existing: list[Box], w: int, h: int) -> bool:
    """Drop wood/carpet/frame false positives from teachers."""
    if not valid_quad(p.pts):
        return False
    if p.conf < 0.10:
        return False
    xs = [q[0] for q in p.pts]
    ys = [q[1] for q in p.pts]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bw < 3 or bh < 3:
        return False
    area = bw * bh
    if area > 0.15 * w * h:
        return False  # huge region (frame / carpet)
    # AABB aspect — spine quads are elongated in image axes or diagonally
    aspect = max(bw, bh) / max(1e-6, min(bw, bh))
    if aspect < 1.8:
        return False
    dilate = 0.12 * max(w, h)
    if existing and not near_existing(p.pts, existing, dilate):
        return False
    # Reject boxes mostly outside the image
    cx, cy = sum(xs) / 4, sum(ys) / 4
    if cx < 0 or cy < 0 or cx > w or cy > h:
        return False
    return True


def merge_new(existing: list[Box], proposals: list[Box], iou_thr: float, w: int, h: int) -> list[Box]:
    out = list(existing)
    for p in proposals:
        if not accept_proposal(p, existing, w, h):
            continue
        if any(quad_iou(p.pts, e.pts) >= iou_thr for e in out):
            continue
        out.append(p)
    return out


def teacher_predict(model, paths: list[Path], imgsz: int, conf: float) -> list[list[Box]]:
    results = model.predict([str(p) for p in paths], imgsz=imgsz, conf=conf, verbose=False)
    out: list[list[Box]] = []
    for r in results:
        boxes: list[Box] = []
        if r.obb is not None and len(r.obb):
            xy = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy() if r.obb.conf is not None else np.ones(len(xy))
            for pts_arr, c in zip(xy, confs):
                pts = [(float(pts_arr[i][0]), float(pts_arr[i][1])) for i in range(4)]
                if valid_quad(pts):
                    boxes.append(Box(pts=pts, source="teacher", conf=float(c)))
        out.append(boxes)
    return out


def swift_predict(image: Path, model: Path, conf: float) -> list[Box]:
    """Run production Core ML via bookspines.swift; parse JSON stdout."""
    script = Path(__file__).resolve().parent.parent / "bookspines.swift"
    if not script.exists() or not model.exists():
        return []
    try:
        proc = subprocess.run(
            ["swift", str(script), str(image), "--model", str(model), "--conf", str(conf)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    # stdout is JSON; stderr is logs
    text = proc.stdout.strip()
    if not text.startswith("{"):
        # sometimes mixed — find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0:
            return []
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    boxes: list[Box] = []
    for sp in data.get("spines", []):
        corners = sp.get("cornersPx") or []
        if len(corners) != 4:
            continue
        pts = [(float(c[0]), float(c[1])) for c in corners]
        if valid_quad(pts):
            boxes.append(Box(pts=pts, source="aug-ml", conf=float(sp.get("confidence", 0.5))))
    # remove accidental .spines.png next to training images
    png = Path(str(image) + ".spines.png")
    if png.exists():
        png.unlink()
    return boxes


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def meta_stem(path: Path) -> str:
    name = path.name
    for suf in (".meta.json", ".json"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem


def process_dataset(
    name: str,
    variants_by_norm: dict[str, list[Variant]],
    args: argparse.Namespace,
    models: list[tuple[str, object]],
    aug_model: Path | None,
) -> dict:
    out = derived_dir(f"{name}_yolo-obb")
    meta_files = sorted((out / "meta").rglob("*.json"))
    if args.limit:
        meta_files = meta_files[: args.limit]

    report = {
        "dataset": name,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replaced_variant": 0,
        "boxes_added_teacher": 0,
        "images_touched": 0,
        "per_image": [],
    }
    preview_dir = out / "preview_misses"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    n_prev = 0

    # Batch teacher on current (post-replace) images
    jobs: list[tuple[Path, Path, Path, dict, list[Box], np.ndarray]] = []

    for mp in meta_files:
        m = load_meta(mp)
        stem = meta_stem(mp)
        split = mp.parent.name
        img_path = out / "images" / split / f"{stem}.jpg"
        lab_path = out / "labels" / split / f"{stem}.txt"
        if not img_path.exists() or not lab_path.exists():
            continue

        key = norm_base(m.get("base", stem))
        variants = variants_by_norm.get(key, [])
        replaced = False
        if variants:
            best = pick_best_variant(variants)
            with Image.open(img_path) as im:
                cw, ch = im.size
                cur_boxes = parse_boxes(lab_path, cw, ch)
                cur_n = len(cur_boxes)
            cur_shelf = shelf_score(img_path)
            cur_nh = sum(1 for b in cur_boxes if box_angle_stats(b.pts)[1])
            cur_nv = sum(1 for b in cur_boxes if box_angle_stats(b.pts)[2])
            # Replace when denser labels exist, OR recover natural pile orientation
            # (current is rotated vert-dominant, best is horiz-dominant with shelf>0).
            orient_fix = (
                best.n >= cur_n - 1
                and best.shelf > 5
                and best.n_h > best.n_v
                and (cur_shelf < 0 or cur_nv > cur_nh)
            )
            denser = best.n >= cur_n + 2
            if denser or orient_fix:
                img = cv2.imread(str(best.path))
                if img is not None:
                    h, w = img.shape[:2]
                    boxes = parse_boxes(best.label_path, w, h, source="gt")
                    lines = [ln for b in boxes if (ln := corners_to_yolo_line(b.pts, w, h, min_size=3.0))]
                    if len(lines) >= 3:
                        cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        lab_path.write_text("\n".join(lines) + "\n")
                        m["src"] = str(best.path)
                        m["n_gt"] = len(boxes)
                        m["replaced_variant"] = True
                        mp.write_text(json.dumps(m, indent=2) + "\n")
                        replaced = True
                        report["replaced_variant"] += 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = parse_boxes(lab_path, w, h, source="gt")
        jobs.append((img_path, lab_path, mp, m, boxes, img))
        if replaced:
            report["images_touched"] += 1

    # Ensemble teacher completion in batches
    for i in range(0, len(jobs), args.batch):
        batch = jobs[i : i + args.batch]
        paths = [j[0] for j in batch]
        ensemble: list[list[Box]] = [[] for _ in batch]
        for _name, model in models:
            preds = teacher_predict(model, paths, args.imgsz, args.conf)
            for bi, pb in enumerate(preds):
                ensemble[bi].extend(pb)

        if aug_model and not args.skip_swift:
            for bi, (img_path, *_rest) in enumerate(batch):
                ensemble[bi].extend(swift_predict(img_path, aug_model, args.conf))

        for bi, (img_path, lab_path, mp, m, boxes, img) in enumerate(batch):
            before = len(boxes)
            h, w = img.shape[:2]
            merged = merge_new(boxes, ensemble[bi], args.iou_match, w, h)
            added = len(merged) - before
            if added <= 0 and not m.get("replaced_variant"):
                continue
            # reload image if replaced earlier
            img2 = cv2.imread(str(img_path))
            if img2 is not None:
                img = img2
                h, w = img.shape[:2]
                # if we reloaded after replace, boxes should match disk
                if m.get("replaced_variant"):
                    boxes = parse_boxes(lab_path, w, h, source="gt")
                    merged = merge_new(boxes, ensemble[bi], args.iou_match, w, h)
                    added = len(merged) - len(boxes)
            lines = [ln for b in merged if (ln := corners_to_yolo_line(b.pts, w, h, min_size=3.0))]
            lab_path.write_text("\n".join(lines) + "\n")
            m["n_final"] = len(lines)
            m["n_added_miss_pass"] = added
            m["n_gt"] = sum(1 for b in merged if b.source == "gt")
            mp.write_text(json.dumps(m, indent=2) + "\n")
            report["boxes_added_teacher"] += added
            report["images_touched"] += 1
            report["per_image"].append(
                {
                    "stem": img_path.stem,
                    "added": added,
                    "final": len(lines),
                    "replaced_variant": bool(m.get("replaced_variant")),
                }
            )
            if n_prev < args.preview and added > 0:
                vis = img.copy()
                for b in merged:
                    color = (0, 220, 255) if b.source == "gt" else (0, 255, 80)
                    pts = np.array([(int(p[0]), int(p[1])) for p in b.pts], np.int32)
                    cv2.polylines(vis, [pts], True, color, 2)
                cv2.putText(
                    vis,
                    f"+{added} -> {len(lines)}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(preview_dir / f"{img_path.stem}.jpg"), vis)
                n_prev += 1
        print(f"  {name}: {min(i + args.batch, len(jobs))}/{len(jobs)}", flush=True)

    (out / "miss_annotate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # append note to SOURCE.md
    src = out / "SOURCE.md"
    if src.exists():
        note = (
            f"\n\n## Miss-annotate pass\n\n"
            f"- at: `{report['updated_at']}`\n"
            f"- replaced_variant: `{report['replaced_variant']}`\n"
            f"- boxes_added_teacher: `{report['boxes_added_teacher']}`\n"
            f"- images_touched: `{report['images_touched']}`\n"
            f"- script: `tools/annotate_missed_obb.py`\n"
        )
        text = src.read_text()
        if "## Miss-annotate pass" in text:
            text = text.split("## Miss-annotate pass")[0].rstrip() + "\n"
        src.write_text(text + note)
    return report


def main() -> int:
    args = parse_args()
    print("Indexing raw variants …", flush=True)
    variants = index_raw_variants()
    print(f"  {len(variants)} unique bases", flush=True)

    from ultralytics import YOLO

    weights = [
        (
            "combined",
            runs_dir(
                "combined_yolo26s-obb_1024px_deg90_ep120_frac100_20260717-0755",
                "weights",
                "best.pt",
            ),
        ),
        ("spine-obb", runs_dir("spine-obb", "weights", "best.pt")),
    ]
    models = []
    for name, path in weights:
        if path.exists():
            models.append((name, YOLO(str(path))))
            print(f"  teacher {name}: {path}", flush=True)
    if not models:
        raise SystemExit("no teacher weights found")

    aug = Path.home() / "ml/book-spines/models/production/SpineDetectorOBB-aug.mlpackage"
    if not aug.exists():
        aug = None
    else:
        print(f"  teacher aug-ml: {aug}", flush=True)

    names = ["open-shelves", "roboflow-book-spine-obb"] if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        rep = process_dataset(name, variants, args, models, aug)
        print(
            f"  replaced={rep['replaced_variant']} "
            f"added_boxes={rep['boxes_added_teacher']} "
            f"touched={rep['images_touched']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
