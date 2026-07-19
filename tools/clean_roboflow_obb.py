#!/usr/bin/env python3
"""Clean / complete Roboflow OBB exports (open-shelves + book-spine-obb).

Problems in the raw exports:
  - Heavy Roboflow aug duplicates (rotations, flips) of the same photo
  - Incomplete labels (adjacent spines unlabeled)
  - open-shelves: 640×640 *stretch* (distorts thin spines)
  - Class names ``book`` / ``0`` instead of ``spine``
  - open-shelves ⊂ roboflow by filename (same personal shelves)

This script:
  1. Dedupes to one upright-ish version per original stem
  2. For open-shelves bases, prefers the roboflow 512 "fit" pixels over 640 stretch
  3. Keeps valid GT OBBs; drops near-identical duplicate GT quads (poly IoU)
  4. Drops label-QA failures via shared ``label_qa_reject_reason`` (incl. degen)
  5. Optionally adds high-conf teacher predictions that don't overlap GT
  6. Drops near-empty / degenerate / still-hopeless frames
  7. Writes derived YOLO-OBB trees with class ``spine``

Examples:
  .venv/bin/python tools/clean_roboflow_obb.py --dataset open-shelves
  .venv/bin/python tools/clean_roboflow_obb.py --dataset roboflow-book-spine-obb
  .venv/bin/python tools/clean_roboflow_obb.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derived_meta import write_derived_source  # noqa: E402
from paths import derived_dir, raw_dir, runs_dir  # noqa: E402
from train_4tu_obb import CLASS_NAME, corners_to_yolo_line  # noqa: E402
from audit_obb_labels import (  # noqa: E402
    QuadBox,
    label_qa_reject_reason,
    shoelace,
)

Quad = list[tuple[float, float]]


DATASETS = {
    "open-shelves": {
        "raw_subdir": "open-shelves/open-shelves.v9i.yolov8-obb",
        "out_id": "open-shelves_yolo-obb",
        "title": "Open Shelves cleaned YOLO-OBB",
        "prefer_rf_pixels": True,  # avoid 640 stretch when RF twin exists
    },
    "roboflow-book-spine-obb": {
        "raw_subdir": "roboflow-book-spine-obb/book-spine-obb.v1-obb.yolov8-obb",
        "out_id": "roboflow-book-spine-obb_yolo-obb",
        "title": "Roboflow book-spine-obb cleaned YOLO-OBB",
        "prefer_rf_pixels": False,
        # When cleaning RF, skip bases already emitted into open-shelves derived
        # so a later merge does not double-count the same shelves.
        "exclude_open_shelves_bases": True,
    },
}


@dataclass
class Box:
    pts: Quad  # pixel corners
    source: str  # "gt" | "teacher"
    conf: float = 1.0


@dataclass
class Candidate:
    base: str
    norm: str
    path: Path
    label_path: Path
    split: str
    n_gt: int
    upright_score: float
    size: tuple[int, int]


@dataclass
class CleanResult:
    base: str
    keep: bool
    reason: str
    n_gt: int = 0
    n_added: int = 0
    n_final: int = 0
    n_teacher: int = 0
    n_unmatched_teacher: int = 0
    src_path: str = ""
    boxes: list[Box] = field(default_factory=list)
    image: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    p.add_argument(
        "--weights",
        type=Path,
        default=runs_dir(
            "combined_yolo26s-obb_1024px_deg90_ep120_frac100_20260717-0755",
            "weights",
            "best.pt",
        ),
        help="Teacher OBB weights for completing missing labels.",
    )
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.18, help="Teacher conf for new boxes.")
    p.add_argument("--iou-match", type=float, default=0.25, help="IoU to treat pred as matching GT.")
    p.add_argument("--min-boxes", type=int, default=3, help="Drop images with fewer final boxes.")
    p.add_argument(
        "--max-unmatched-ratio",
        type=float,
        default=2.5,
        help="If unmatched_teacher / max(gt,1) exceeds this after merge attempt, drop "
        "(still massively incomplete / wrong domain for teacher).",
    )
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Cap unique bases (0=all).")
    p.add_argument("--preview", type=int, default=24)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--no-teacher",
        action="store_true",
        help="Keep human GT only — do not merge teacher predictions into train labels.",
    )
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


def parse_gt_boxes(label_path: Path, w: int, h: int) -> list[Box]:
    if not label_path.exists():
        return []
    out: list[Box] = []
    for ln in label_path.read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        vals = list(map(float, parts[1:9]))
        pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]
        if _valid_quad(pts):
            out.append(Box(pts=pts, source="gt", conf=1.0))
    return out


def _valid_quad(pts: Quad, min_edge: float = 3.0, max_aspect: float = 50.0) -> bool:
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
    aspect = edges[-1] / max(1e-6, edges[-2])
    if aspect > max_aspect:
        return False
    # Reject collapsed quads
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) - min(xs) < min_edge or max(ys) - min(ys) < min_edge:
        return False
    return True


def upright_score(boxes: list[Box]) -> float:
    """Fraction of boxes that look like vertical spines (good for unaugmented)."""
    if not boxes:
        return 0.0
    n = 0
    for b in boxes:
        best_L, best_ang = 0.0, 0.0
        for i in range(4):
            x0, y0 = b.pts[i]
            x1, y1 = b.pts[(i + 1) % 4]
            L = math.hypot(x1 - x0, y1 - y0)
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
            if ang > 90:
                ang -= 180
            if L > best_L:
                best_L, best_ang = L, ang
        if abs(best_ang) > 50:  # long edge near vertical
            n += 1
    return n / len(boxes)


def quad_iou(a: Quad, b: Quad) -> float:
    """Approx IoU via AABB (fast) — good enough for match/no-match."""
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    ay0, ay1 = min(p[1] for p in a), max(p[1] for p in a)
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    by0, by1 = min(p[1] for p in b), max(p[1] for p in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-6, area_a + area_b - inter)


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def poly_iou(a: Quad, b: Quad) -> float:
    """True polygon IoU for near-duplicate GT detection."""
    pa = np.array(a, dtype=np.float32)
    pb = np.array(b, dtype=np.float32)
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    if inter_area is None:
        inter = 0.0
    else:
        inter = float(inter_area) if np.isscalar(inter_area) else float(cv2.contourArea(inter_area))
    aa, bb = _shoelace(pa), _shoelace(pb)
    union = aa + bb - inter
    return inter / union if union > 1e-6 else 0.0


def dedupe_near_identical(boxes: list[Box], iou_thr: float = 0.90) -> list[Box]:
    """Drop near-copy quads (same spine labeled twice). Keep first of each cluster."""
    if len(boxes) < 2:
        return boxes
    keep: list[Box] = []
    for b in boxes:
        if any(poly_iou(b.pts, k.pts) >= iou_thr for k in keep):
            continue
        keep.append(b)
    return keep


def index_export(root: Path) -> dict[str, list[Candidate]]:
    by: dict[str, list[Candidate]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for ip in img_dir.iterdir():
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lp = root / split / "labels" / f"{ip.stem}.txt"
            with Image.open(ip) as im:
                w, h = im.size
            boxes = parse_gt_boxes(lp, w, h)
            b = base_name(ip.stem)
            by[norm_base(b)].append(
                Candidate(
                    base=b,
                    norm=norm_base(b),
                    path=ip,
                    label_path=lp,
                    split=split,
                    n_gt=len(boxes),
                    upright_score=upright_score(boxes),
                    size=(w, h),
                )
            )
    return by


def pick_best(cands: list[Candidate]) -> Candidate:
    # Prefer more labels. Do NOT maximize upright_score — that selects 90° Roboflow
    # augs of horizontal piles (spines become "vertical" after rotation).
    # Prefer positive shelf-edge score so pile photos stay in natural orientation.
    def key(c: Candidate) -> tuple:
        shelf = 0.0
        try:
            # lazy import path already has cv2
            im = cv2.imread(str(c.path), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                im = cv2.GaussianBlur(im, (3, 3), 0)
                gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
                shelf = float(np.mean(np.abs(gy)) - np.mean(np.abs(gx)))
        except Exception:
            shelf = 0.0
        return (c.n_gt, shelf, c.size[0] * c.size[1], c.split == "train")

    return max(cands, key=key)


def load_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"failed to read {path}")
    return im


def teacher_predict_batch(model, paths: list[Path], imgsz: int, conf: float) -> list[list[Box]]:
    results = model.predict(
        [str(p) for p in paths],
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )
    out: list[list[Box]] = []
    for r in results:
        boxes: list[Box] = []
        if r.obb is None or len(r.obb) == 0:
            out.append(boxes)
            continue
        # xyxyxyxy pixel corners
        xy = r.obb.xyxyxyxy.cpu().numpy()
        confs = r.obb.conf.cpu().numpy() if r.obb.conf is not None else np.ones(len(xy))
        for pts_arr, c in zip(xy, confs):
            pts = [(float(pts_arr[i][0]), float(pts_arr[i][1])) for i in range(4)]
            if _valid_quad(pts):
                boxes.append(Box(pts=pts, source="teacher", conf=float(c)))
        out.append(boxes)
    return out


def boxes_to_quadboxes(boxes: list[Box]) -> list[QuadBox]:
    """Convert clean Box list into audit QuadBox for shared QA rules."""
    out: list[QuadBox] = []
    for b in boxes:
        pts = np.array(b.pts, dtype=np.float32)
        edges = []
        for i in range(4):
            d = pts[(i + 1) % 4] - pts[i]
            L = float(np.hypot(d[0], d[1]))
            ang = math.degrees(math.atan2(d[1], d[0])) % 180
            if ang > 90:
                ang -= 180
            edges.append((L, ang))
        edges.sort(key=lambda t: -t[0])
        long_L, ang = edges[0]
        shorts = sorted(e[0] for e in edges)[:2]
        short_L = max(1e-6, 0.5 * (shorts[0] + shorts[1]))
        out.append(QuadBox(pts=pts, area=shoelace(pts), aspect=long_L / short_L, angle=ang))
    return out


def merge_boxes(gt: list[Box], pred: list[Box], iou_thr: float) -> tuple[list[Box], int]:
    """Keep all GT; add preds that don't match any GT."""
    final = list(gt)
    unmatched = 0
    for p in pred:
        if any(quad_iou(p.pts, g.pts) >= iou_thr for g in gt):
            continue
        unmatched += 1
        final.append(p)
    return final, unmatched


def clean_one(
    cand: Candidate,
    teacher_boxes: list[Box],
    args: argparse.Namespace,
) -> CleanResult:
    img = load_bgr(cand.path)
    h, w = img.shape[:2]
    gt = dedupe_near_identical(parse_gt_boxes(cand.label_path, w, h))
    if getattr(args, "no_teacher", False):
        merged, n_unmatched = list(gt), 0
    else:
        merged, n_unmatched = merge_boxes(gt, teacher_boxes, args.iou_match)

    # Drop tiny / edge-degenerate after merge already filtered
    merged = dedupe_near_identical([b for b in merged if _valid_quad(b.pts)])

    n_gt = len(gt)
    n_added = sum(1 for b in merged if b.source == "teacher")
    n_final = len(merged)
    n_teacher = len(teacher_boxes)

    if n_final < args.min_boxes:
        return CleanResult(
            base=cand.base,
            keep=False,
            reason=f"too_few_boxes:{n_final}",
            n_gt=n_gt,
            n_added=n_added,
            n_final=n_final,
            n_teacher=n_teacher,
            n_unmatched_teacher=n_unmatched,
            src_path=str(cand.path),
        )

    qa = label_qa_reject_reason(boxes_to_quadboxes(merged), w, h)
    if qa:
        return CleanResult(
            base=cand.base,
            keep=False,
            reason=qa,
            n_gt=n_gt,
            n_added=n_added,
            n_final=n_final,
            n_teacher=n_teacher,
            n_unmatched_teacher=n_unmatched,
            src_path=str(cand.path),
        )

    # Still hopelessly incomplete relative to teacher (and GT was sparse)
    if n_gt <= 4 and n_unmatched > max(8, args.max_unmatched_ratio * max(n_gt, 1)):
        # Keep anyway if we successfully added most unmatched (merge did add them)
        if n_added < 0.5 * n_unmatched:
            return CleanResult(
                base=cand.base,
                keep=False,
                reason="incomplete_unfixed",
                n_gt=n_gt,
                n_added=n_added,
                n_final=n_final,
                n_teacher=n_teacher,
                n_unmatched_teacher=n_unmatched,
                src_path=str(cand.path),
            )

    return CleanResult(
        base=cand.base,
        keep=True,
        reason="ok",
        n_gt=n_gt,
        n_added=n_added,
        n_final=n_final,
        n_teacher=n_teacher,
        n_unmatched_teacher=n_unmatched,
        src_path=str(cand.path),
        boxes=merged,
        image=img,
    )


def write_dataset(
    results: list[CleanResult],
    out: Path,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> dict[str, int]:
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    kept = [r for r in results if r.keep and r.image is not None]
    rng.shuffle(kept)
    n_train = max(1, int(len(kept) * args.train_frac)) if kept else 0
    splits = {"train": kept[:n_train], "val": kept[n_train:]}
    if not splits["val"] and kept:
        splits["val"] = [kept[-1]]
        splits["train"] = kept[:-1] or kept

    counts = {"train": 0, "val": 0}
    preview_n = 0
    for split, items in splits.items():
        img_dir = out / "images" / split
        lab_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(items):
            assert r.image is not None
            stem = f"spine_{split}_{i:04d}"
            h, w = r.image.shape[:2]
            lines = []
            for b in r.boxes:
                ln = corners_to_yolo_line(b.pts, w, h, min_size=3.0)
                if ln:
                    lines.append(ln)
            if len(lines) < args.min_boxes:
                continue
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), r.image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            meta_dir = out / "meta" / split
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "base": r.base,
                        "src": r.src_path,
                        "n_gt": r.n_gt,
                        "n_added": r.n_added,
                        "n_final": r.n_final,
                    },
                    indent=2,
                )
                + "\n"
            )
            counts[split] += 1
            if preview_n < args.preview:
                vis = r.image.copy()
                for b in r.boxes:
                    color = (0, 220, 255) if b.source == "gt" else (0, 255, 80)
                    pts = np.array([(int(p[0]), int(p[1])) for p in b.pts], dtype=np.int32)
                    cv2.polylines(vis, [pts], True, color, 2)
                prev = out / "preview"
                prev.mkdir(exist_ok=True)
                cv2.imwrite(str(prev / f"{stem}.jpg"), vis)
                preview_n += 1

    (out / "spines.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n"
    )
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "meta": meta,
        "dropped": [
            {
                "base": r.base,
                "reason": r.reason,
                "n_gt": r.n_gt,
                "n_teacher": r.n_teacher,
                "n_unmatched_teacher": r.n_unmatched_teacher,
            }
            for r in results
            if not r.keep
        ],
        "kept_summary": {
            "n": len(kept),
            "mean_gt": float(np.mean([r.n_gt for r in kept])) if kept else 0,
            "mean_added": float(np.mean([r.n_added for r in kept])) if kept else 0,
            "mean_final": float(np.mean([r.n_final for r in kept])) if kept else 0,
            "frac_with_adds": float(np.mean([r.n_added > 0 for r in kept])) if kept else 0,
        },
    }
    (out / "clean_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return counts


def run_dataset(name: str, args: argparse.Namespace, open_shelves_norms: set[str] | None) -> set[str]:
    cfg = DATASETS[name]
    root = raw_dir(cfg["raw_subdir"])
    if not root.is_dir():
        raise SystemExit(f"missing raw export: {root}")

    rf_root = raw_dir("roboflow-book-spine-obb/book-spine-obb.v1-obb.yolov8-obb")
    os_index = index_export(root)
    rf_index = index_export(rf_root) if rf_root.is_dir() else {}

    print(f"\n=== {name} ===")
    print(f"  raw unique bases: {len(os_index)}")

    # Choose image source per base
    chosen: list[Candidate] = []
    for norm, cands in os_index.items():
        if cfg.get("exclude_open_shelves_bases") and open_shelves_norms and norm in open_shelves_norms:
            continue
        primary = pick_best(cands)
        if cfg.get("prefer_rf_pixels") and norm in rf_index:
            rf_best = pick_best(rf_index[norm])
            # Prefer RF pixels (no stretch) but keep whichever has more GT if RF is much worse
            if rf_best.n_gt + 2 >= primary.n_gt:
                primary = rf_best
        chosen.append(primary)

    chosen.sort(key=lambda c: c.norm)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"  after dedupe/filter: {len(chosen)}")

    results: list[CleanResult] = []
    if args.no_teacher:
        print("  --no-teacher: writing human GT only (no model merge)", flush=True)
        for cand in chosen:
            results.append(clean_one(cand, [], args))
    else:
        if not args.weights.exists():
            raise SystemExit(f"teacher weights not found: {args.weights}")
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        for i in range(0, len(chosen), args.batch):
            batch = chosen[i : i + args.batch]
            preds = teacher_predict_batch(model, [c.path for c in batch], args.imgsz, args.conf)
            for cand, pred in zip(batch, preds):
                results.append(clean_one(cand, pred, args))
            kept_so_far = sum(1 for r in results if r.keep)
            print(f"  … {min(i + args.batch, len(chosen))}/{len(chosen)} kept={kept_so_far}", flush=True)

    out = derived_dir(cfg["out_id"])
    meta = {
        "dataset": name,
        "raw": str(root),
        "weights": str(args.weights) if not args.no_teacher else "none",
        "conf": args.conf,
        "iou_match": args.iou_match,
        "min_boxes": args.min_boxes,
        "no_teacher": bool(args.no_teacher),
        "prefer_rf_pixels": bool(cfg.get("prefer_rf_pixels")),
        "exclude_open_shelves_bases": bool(cfg.get("exclude_open_shelves_bases")),
        "unique_in": len(chosen),
        "kept": sum(1 for r in results if r.keep),
        "dropped": sum(1 for r in results if not r.keep),
    }
    counts = write_dataset(results, out, args, meta)
    if args.no_teacher:
        notes = (
            f"Deduped Roboflow augs → one variant per original (shelf-score pick). "
            f"**Human GT only** (`--no-teacher`); no teacher boxes merged. "
            f"Class renamed to `{CLASS_NAME}`.\n\n"
            f"Split counts: train={counts['train']} val={counts['val']}. "
            f"See `clean_report.json` for drops."
        )
    else:
        notes = (
            f"Deduped Roboflow augs → unique upright-ish originals. "
            f"GT OBBs kept; teacher (`{args.weights.name}`) added non-overlapping boxes "
            f"(cyan=GT, green=teacher in `preview/`). Class renamed to `{CLASS_NAME}`.\n\n"
            f"Split counts: train={counts['train']} val={counts['val']}. "
            f"See `clean_report.json` for drops."
        )
    if cfg.get("prefer_rf_pixels"):
        notes += (
            "\n\nFor bases shared with roboflow-book-spine-obb, prefers RF 512×512 "
            "fit-within pixels over open-shelves 640×640 stretch."
        )
    write_derived_source(
        out,
        derived_id=cfg["out_id"],
        title=cfg["title"] + (" (GT only)" if args.no_teacher else ""),
        sources=[name] if args.no_teacher else [name, "teacher:" + args.weights.name],
        script="tools/clean_roboflow_obb.py",
        flags={
            "conf": args.conf if not args.no_teacher else "n/a",
            "imgsz": args.imgsz,
            "min_boxes": args.min_boxes,
            "train_frac": args.train_frac,
            "limit": args.limit or "all",
            "no_teacher": args.no_teacher,
        },
        notes=notes,
    )
    print(f"  wrote {out} train={counts['train']} val={counts['val']}")

    # Return norms kept for open-shelves so RF can exclude them
    return {norm_base(r.base) for r in results if r.keep}


def main() -> int:
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    # Always do open-shelves before roboflow when both, so RF can exclude OS bases
    if "open-shelves" in names and "roboflow-book-spine-obb" in names:
        names = ["open-shelves", "roboflow-book-spine-obb"]

    os_norms: set[str] | None = None
    for name in names:
        norms = run_dataset(name, args, os_norms)
        if name == "open-shelves":
            os_norms = norms
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
