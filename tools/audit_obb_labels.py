#!/usr/bin/env python3
"""Audit OBB label quality WITHOUT modifying labels.

Implements the expert-recommended layered audit:
  L0 = human GT in a derived tree (prefer --no-teacher rebuild)
  L1 = teacher proposals written only to a sidecar folder (optional)

Uses true polygon IoU (not AABB), OBB/shoelace area (not AABB "huge"),
duplicate graphs, and the single hard-drop gate ``label_qa_reject_reason``
(shared with clean + build). Emits quarantine manifests + ranked review.

Does NOT rewrite train labels.

Example:
  .venv/bin/python tools/clean_roboflow_obb.py --dataset all --no-teacher
  .venv/bin/python tools/audit_obb_labels.py \\
      --derived $BOOK_SPINES_DATA/derived/open-shelves_yolo-obb \\
      --out $BOOK_SPINES_DATA/eval/obb-audit-open-shelves
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import derived_dir, eval_dir  # noqa: E402


@dataclass
class QuadBox:
    pts: np.ndarray  # (4,2) float
    area: float
    aspect: float
    angle: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", type=Path, required=True, help="Derived YOLO-OBB root to audit.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dup-iou", type=float, default=0.55)
    p.add_argument("--soft-iou", type=float, default=0.15)
    p.add_argument("--top", type=int, default=100, help="Write overlays for top-N priority images.")
    return p.parse_args()


def shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def parse_label(path: Path, w: int, h: int) -> list[QuadBox]:
    out: list[QuadBox] = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        vals = list(map(float, ln.split()[1:9]))
        pts = np.array([(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)], dtype=np.float32)
        # long-edge angle + aspect from consecutive edges
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
        short_L = edges[2][0] if len(edges) > 2 else edges[-1][0]
        # opposite edges: use shorter of the two long-ish
        shorts = sorted(e[0] for e in edges)[:2]
        short_L = max(1e-6, 0.5 * (shorts[0] + shorts[1]))
        area = shoelace(pts)
        aspect = long_L / short_L
        out.append(QuadBox(pts=pts, area=area, aspect=aspect, angle=ang))
    return out


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32)
    pb = b.astype(np.float32)
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    if inter_area is None:
        inter = 0.0
    else:
        inter = float(inter_area) if np.isscalar(inter_area) else float(cv2.contourArea(inter_area))
    aa, bb = shoelace(pa), shoelace(pb)
    union = aa + bb - inter
    return inter / union if union > 1e-6 else 0.0


def aabb_fill(pts: np.ndarray) -> float:
    """OBB area / axis-aligned envelope. 1.0 ⇒ label is an AABB."""
    ax0, ax1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    ay0, ay1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    return shoelace(pts) / max(1e-6, (ax1 - ax0) * (ay1 - ay0))


def is_axis_aligned_quad(pts: np.ndarray, tol: float = 2.0) -> bool:
    for i in range(4):
        d = pts[(i + 1) % 4] - pts[i]
        if abs(float(d[0])) > tol and abs(float(d[1])) > tol:
            return False
    return True


def aabb_fat_stats(boxes: list[QuadBox]) -> tuple[float, float, float]:
    """frac_axis_aligned, mean aabb_fill, median aspect."""
    n = len(boxes)
    if n == 0:
        return 0.0, 0.0, 0.0
    frac_aa = sum(1 for b in boxes if is_axis_aligned_quad(b.pts)) / n
    mean_fill = float(np.mean([aabb_fill(b.pts) for b in boxes]))
    med_aspect = float(np.median([b.aspect for b in boxes]))
    return frac_aa, mean_fill, med_aspect


def is_aabb_fat_frame(
    boxes: list[QuadBox],
    *,
    min_boxes: int = 3,
    frac_aa: float = 0.8,
    mean_fill: float = 0.95,
    max_med_aspect: float = 5.0,
) -> bool:
    """AABB labels that are too fat for upright spines → tilted shelf with AABB GT.

    Upright vertical spines labeled as AABB stay thin (median aspect ≫ 5).
    Tilted shelves force AABB envelopes to fatten (0406: fill=1, aspect≈3).
    """
    if len(boxes) < min_boxes:
        return False
    fa, mf, ma = aabb_fat_stats(boxes)
    return fa >= frac_aa and mf >= mean_fill and ma < max_med_aspect


def count_degen(boxes: list[QuadBox]) -> int:
    """Boxes with near-zero area or non-convex corner order."""
    n = 0
    for b in boxes:
        if b.area < 1.0 or not cv2.isContourConvex(b.pts.astype(np.float32)):
            n += 1
    return n


def label_qa_reject_reason(boxes: list[QuadBox], w: int, h: int) -> str | None:
    """Single hard-drop gate for train labels (audit / clean / build all use this).

    Returns a short reason string, or None if the frame is OK to keep.
    """
    if not boxes:
        return "empty"
    if count_degen(boxes) > 0:
        return "degen"
    if is_aabb_fat_frame(boxes):
        return "aabb_fat"
    max_iou = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            max_iou = max(max_iou, poly_iou(boxes[i].pts, boxes[j].pts))
    if max_iou >= 0.80:
        return "near_dup"
    if max_iou >= 0.55:
        return "overlap"
    if max_iou >= 0.40:
        return "touch"
    _, _, ma = aabb_fat_stats(boxes)
    n_fat = sum(1 for b in boxes if b.aspect < 1.6)
    n_huge = sum(1 for b in boxes if b.area > 0.08 * w * h and b.aspect < 3.0)
    n_big = sum(1 for b in boxes if b.area > 0.05 * w * h and b.aspect < 4.0)
    if n_huge >= 1:
        return "huge"
    if n_big >= 1:
        return "large_fat"
    if n_fat / len(boxes) >= 0.15:
        return "squatish"
    n_soft = sum(1 for b in boxes if b.aspect < 2.5)
    if n_soft / len(boxes) >= 0.35 and ma < 4.0:
        return "low_aspect"
    return None


def dup_frac(boxes: list[QuadBox], thr: float) -> tuple[float, int]:
    n = len(boxes)
    if n < 2:
        return 0.0, 0
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if poly_iou(boxes[i].pts, boxes[j].pts) >= thr:
                union(i, j)
                pairs += 1
    comps: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        comps[r] = comps.get(r, 0) + 1
    in_multi = sum(c for c in comps.values() if c >= 2)
    return in_multi / n, pairs


def priority(row: dict) -> float:
    p = 0.0
    if row.get("aabb_fat"):
        p += 5.0
    if row["dup_frac"] >= 0.08:
        p += 3.0
    if row["n_degen"] > 0:
        p += 3.0
    if row["frac_fat"] > 0.15:
        p += 1.0
    if row["frac_huge_obb"] > 0.1:
        p += 1.0
    if row["area_cv"] > 1.8 and row["n"] >= 8:
        p += 1.0
    if row["n"] < 3:
        p += 2.0
    p += min(2.0, row["dup_pairs"] / 20.0)
    return p


def main() -> int:
    args = parse_args()
    derived = args.derived
    out = args.out or eval_dir(f"obb-audit-{derived.name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)

    rows = []
    for lab in sorted((derived / "labels").rglob("*.txt")):
        split = lab.parent.name
        img = derived / "images" / split / f"{lab.stem}.jpg"
        if not img.exists():
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = parse_label(lab, w, h)
        n_degen = count_degen(boxes)
        fat = huge = 0
        areas = []
        for b in boxes:
            if b.aspect < 1.6:
                fat += 1
            # huge only if large AND not elongated (frame/carpet)
            if b.area > 0.08 * w * h and b.aspect < 3.0:
                huge += 1
            areas.append(b.area)
        df, pairs = dup_frac(boxes, args.dup_iou)
        fa, mf, ma = aabb_fat_stats(boxes)
        aabb_fat = is_aabb_fat_frame(boxes)
        qa_reason = label_qa_reject_reason(boxes, w, h)
        mu = float(np.mean(areas)) if areas else 0.0
        sd = float(np.std(areas)) if areas else 0.0
        row = {
            "stem": lab.stem,
            "split": split,
            "n": len(boxes),
            "dup_frac": round(df, 4),
            "dup_pairs": pairs,
            "n_degen": n_degen,
            "frac_fat": round(fat / max(len(boxes), 1), 4),
            "frac_huge_obb": round(huge / max(len(boxes), 1), 4),
            "frac_aa": round(fa, 4),
            "mean_aabb_fill": round(mf, 4),
            "med_aspect": round(ma, 4),
            "aabb_fat": aabb_fat,
            "qa_reason": qa_reason or "",
            "area_cv": round(sd / mu, 4) if mu > 0 else 0.0,
            "path": str(img),
            "label": str(lab),
        }
        row["priority"] = round(priority(row), 3)
        rows.append(row)

    rows.sort(key=lambda r: -r["priority"])
    quarantine = [r for r in rows if r["qa_reason"]]
    soft = [r for r in rows if not r["qa_reason"] and r["priority"] >= 2.0][: max(0, args.top)]

    with (out / "audit.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["stem"])
        wri.writeheader()
        wri.writerows(rows)

    (out / "quarantine_hard.json").write_text(json.dumps(quarantine, indent=2) + "\n")
    (out / "review_soft.json").write_text(json.dumps(soft, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(
        f"""# OBB audit — `{derived.name}`

Read-only audit. Labels were **not** modified.

| Metric | Value |
|---|---|
| images | {len(rows)} |
| hard quarantine | {len(quarantine)} |
| aabb-fat frames | {sum(1 for r in rows if r['aabb_fat'])} |
| soft review (priority≥2, capped) | {len(soft)} |
| mean dup_frac | {np.mean([r['dup_frac'] for r in rows]):.3f} |
| images with dup_frac≥0.08 | {sum(1 for r in rows if r['dup_frac']>=0.08)} |

## Policy
- Train on human GT only (`clean_roboflow_obb.py --no-teacher`) until spot-check proves teacher adds help.
- Hard quarantine / clean / build drop: **only** `label_qa_reject_reason`
  (empty, degen, AABB-fat, near-dup, overlap, touch, huge, large_fat, squatish,
  low_aspect). No parallel rules.
- Soft review: remaining priority≥2 frames (if any).
- Teacher proposals (if any) must live in sidecars — never overwrite L0.

See `audit.csv`, `quarantine_hard.json`, `review_soft.json`, `overlays/`.
"""
    )

    for i, r in enumerate(rows[: args.top]):
        im = cv2.imread(r["path"])
        h, w = im.shape[:2]
        boxes = parse_label(Path(r["label"]), w, h)
        # mark duplicates in magenta
        n = len(boxes)
        is_dup = [False] * n
        for a in range(n):
            for b in range(a + 1, n):
                if poly_iou(boxes[a].pts, boxes[b].pts) >= args.dup_iou:
                    is_dup[a] = is_dup[b] = True
        for j, b in enumerate(boxes):
            color = (255, 0, 255) if is_dup[j] else (0, 220, 255)
            if b.area > 0.08 * w * h and b.aspect < 3.0:
                color = (0, 0, 255)
            cv2.polylines(im, [b.pts.astype(np.int32)], True, color, 2)
        cv2.putText(
            im,
            f"P={r['priority']} dup={r['dup_frac']:.2f} n={r['n']}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(out / "overlays" / f"{i:03d}_{r['stem']}.jpg"), im)

    print(f"Audited {len(rows)} images → {out}")
    print(f"  hard quarantine: {len(quarantine)}")
    print(f"  soft review listed: {len(soft)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
