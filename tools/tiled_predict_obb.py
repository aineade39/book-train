#!/usr/bin/env python3
"""Compare single-shot vs tiled OBB inference on dense shelf photos.

Ultralytics weights (.pt) — quick check before Core ML export, or to benchmark
tiling gain on wide bookcases. Tile grid + overlap match ``bookspines.swift``.

Examples:
  .venv/bin/python tools/tiled_predict_obb.py bookcase.books.png
  .venv/bin/python tools/tiled_predict_obb.py bookcase.books.png --tiles 2x2
  .venv/bin/python tools/tiled_predict_obb.py img.jpg --weights ~/ml/book-spines/runs/.../best.pt
  .venv/bin/python tools/tiled_predict_obb.py img.jpg --mode tiled --out img.spines.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import eval_dir, runs_dir  # noqa: E402


@dataclass
class Det:
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    conf: float

    def corners(self) -> list[tuple[float, float]]:
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        v1x, v1y = c * self.w / 2, s * self.w / 2
        v2x, v2y = -s * self.h / 2, c * self.h / 2
        return [
            (self.cx + v1x + v2x, self.cy + v1y + v2y),
            (self.cx + v1x - v2x, self.cy + v1y - v2y),
            (self.cx - v1x - v2x, self.cy - v1y - v2y),
            (self.cx - v1x + v2x, self.cy - v1y + v2y),
        ]

    def offset(self, dx: float, dy: float) -> Det:
        return Det(self.cx + dx, self.cy + dy, self.w, self.h, self.angle, self.conf)


@dataclass
class Tile:
    x0: int
    y0: int
    w: int
    h: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="Shelf photo to run inference on.")
    p.add_argument("--weights", type=Path, default=None, help="YOLO-OBB .pt (default: newest runs/*/best.pt).")
    p.add_argument(
        "--mode",
        choices=("both", "single", "tiled"),
        default="both",
        help="Run full image, tiled, or print both for comparison.",
    )
    p.add_argument("--tiles", default=None, help='Tile grid like "2x2". Default: auto (2x2 if max side > 3000).')
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.45, help="Rotated IoU threshold for cross-tile NMS.")
    p.add_argument("--max-det", type=int, default=500)
    p.add_argument("--device", default="mps", help="Ultralytics device (mps, cpu, 0, …).")
    p.add_argument("--out", type=Path, default=None, help="Annotated PNG (tiled result if mode=both).")
    p.add_argument("--json", type=Path, default=None, help="Write detection summary JSON.")
    return p.parse_args()


def latest_best_pt() -> Path:
    root = runs_dir()
    hits: list[tuple[float, Path]] = []
    if root.is_dir():
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            best = run_dir / "weights" / "best.pt"
            if best.is_file():
                hits.append((best.stat().st_mtime, best))
    if not hits:
        raise FileNotFoundError(f"No runs/*/weights/best.pt under {root}")
    hits.sort(key=lambda t: t[0], reverse=True)
    return hits[0][1]


def compute_tiles(image_w: int, image_h: int, spec: str | None, overlap: float) -> list[Tile]:
    cols, rows = 1, 1
    if spec:
        parts = spec.lower().split("x")
        if len(parts) == 2:
            c, r = int(parts[0]), int(parts[1])
            if c > 0 and r > 0:
                cols, rows = c, r
            else:
                print(f'Warning: bad --tiles "{spec}", using 1x1', file=sys.stderr)
        else:
            print(f'Warning: could not parse --tiles "{spec}", using 1x1', file=sys.stderr)
    elif max(image_w, image_h) > 3000:
        cols, rows = 2, 2

    if cols == 1 and rows == 1:
        return [Tile(0, 0, image_w, image_h)]

    ow = max(0.0, min(overlap, 0.9))
    tile_w = image_w / (cols - (cols - 1) * ow)
    tile_h = image_h / (rows - (rows - 1) * ow)
    step_x = tile_w * (1 - ow)
    step_y = tile_h * (1 - ow)

    tiles: list[Tile] = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(min(max(0, col * step_x), max(0, image_w - tile_w)))
            y0 = int(min(max(0, row * step_y), max(0, image_h - tile_h)))
            tw = int(min(tile_w, image_w))
            th = int(min(tile_h, image_h))
            tiles.append(Tile(x0, y0, tw, th))
    return tiles


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def _ensure_ccw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    signed = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    return list(reversed(pts)) if signed < 0 else pts


def _clip_polygon(subject: list[tuple[float, float]], edge_a: tuple[float, float], edge_b: tuple[float, float]) -> list[tuple[float, float]]:
    def side(p: tuple[float, float]) -> float:
        return (edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0])

    def intersect(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
        a1 = edge_b[1] - edge_a[1]
        b1 = edge_a[0] - edge_b[0]
        c1 = a1 * edge_a[0] + b1 * edge_a[1]
        a2 = q[1] - p[1]
        b2 = p[0] - q[0]
        c2 = a2 * p[0] + b2 * p[1]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return p
        return ((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)

    out: list[tuple[float, float]] = []
    if not subject:
        return out
    for i, current in enumerate(subject):
        previous = subject[(i + len(subject) - 1) % len(subject)]
        cur_in = side(current) >= 0
        prev_in = side(previous) >= 0
        if cur_in:
            if not prev_in:
                out.append(intersect(previous, current))
            out.append(current)
        elif prev_in:
            out.append(intersect(previous, current))
    return out


def rotated_iou(a: Det, b: Det) -> float:
    reach = (math.hypot(a.w, a.h) + math.hypot(b.w, b.h)) / 2
    if math.hypot(a.cx - b.cx, a.cy - b.cy) >= reach:
        return 0.0
    quad_a = _ensure_ccw(a.corners())
    quad_b = _ensure_ccw(b.corners())
    inter = quad_a
    for i in range(len(quad_b)):
        if not inter:
            break
        inter = _clip_polygon(inter, quad_b[i], quad_b[(i + 1) % len(quad_b)])
    inter_area = _polygon_area(inter)
    if inter_area <= 0:
        return 0.0
    union = _polygon_area(quad_a) + _polygon_area(quad_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_rotated(candidates: list[Det], iou: float, max_det: int) -> list[Det]:
    kept: list[Det] = []
    for cand in sorted(candidates, key=lambda d: d.conf, reverse=True):
        if len(kept) >= max_det:
            break
        if all(rotated_iou(cand, k) <= iou for k in kept):
            kept.append(cand)
    return kept


def ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Use .venv/bin/python", file=sys.stderr)
        sys.exit(1)
    return YOLO


def predict_crop(
    model: Any,
    crop_bgr: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    device: str,
    offset_x: float,
    offset_y: float,
) -> list[Det]:
    results = model.predict(
        source=crop_bgr,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    dets: list[Det] = []
    for r in results:
        obb = r.obb
        if obb is None or len(obb) == 0:
            continue
        xywhr = obb.xywhr.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for i in range(len(confs)):
            cx, cy, w, h, angle = (float(v) for v in xywhr[i])
            if w <= 1 or h <= 1:
                continue
            dets.append(Det(cx + offset_x, cy + offset_y, w, h, angle, float(confs[i])))
    return dets


def run_single(
    model: Any,
    image_bgr: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    device: str,
    iou: float,
    max_det: int,
) -> list[Det]:
    raw = predict_crop(model, image_bgr, imgsz=imgsz, conf=conf, device=device, offset_x=0, offset_y=0)
    return nms_rotated(raw, iou, max_det)


def run_tiled(
    model: Any,
    image_bgr: np.ndarray,
    tiles: list[Tile],
    *,
    imgsz: int,
    conf: float,
    device: str,
    iou: float,
    max_det: int,
) -> list[Det]:
    candidates: list[Det] = []
    for tile in tiles:
        crop = image_bgr[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w]
        if crop.size == 0:
            continue
        candidates.extend(
            predict_crop(
                model,
                crop,
                imgsz=imgsz,
                conf=conf,
                device=device,
                offset_x=float(tile.x0),
                offset_y=float(tile.y0),
            )
        )
    return nms_rotated(candidates, iou, max_det)


def draw_detections(image_bgr: np.ndarray, dets: list[Det], tiles: list[Tile] | None = None) -> np.ndarray:
    out = image_bgr.copy()
    if tiles and len(tiles) > 1:
        for tile in tiles:
            cv2.rectangle(out, (tile.x0, tile.y0), (tile.x0 + tile.w, tile.y0 + tile.h), (255, 128, 0), 2)
    for det in dets:
        pts = np.array(det.corners(), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return out


def summarize(name: str, dets: list[Det], tiles: list[Tile]) -> dict[str, Any]:
    return {
        "mode": name,
        "count": len(dets),
        "tiles": len(tiles),
        "mean_conf": round(float(np.mean([d.conf for d in dets])), 4) if dets else 0.0,
    }


def default_json_path(image: Path) -> Path:
    eval_dir().mkdir(parents=True, exist_ok=True)
    stem = image.stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return eval_dir(f"{stem}_tiled_predict_{ts}.json")


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    weights = args.weights.expanduser().resolve() if args.weights else latest_best_pt()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 1

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return 1
    h, w = image_bgr.shape[:2]

    tiles = compute_tiles(w, h, args.tiles, args.tile_overlap)
    if len(tiles) > 1:
        spec = args.tiles or "auto"
        print(f"Tiling {w}x{h} -> {len(tiles)} tiles ({spec}, overlap {args.tile_overlap})")

    YOLO = ensure_ultralytics()
    model = YOLO(str(weights), task="obb")
    print(f"Weights: {weights}")

    payload: dict[str, Any] = {
        "image": str(image_path),
        "image_size": {"width": w, "height": h},
        "weights": str(weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "tiles_spec": args.tiles,
        "tile_overlap": args.tile_overlap,
        "results": {},
    }

    draw_dets: list[Det] = []
    draw_tiles: list[Tile] | None = None

    if args.mode in ("both", "single"):
        single = run_single(
            model,
            image_bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            iou=args.iou,
            max_det=args.max_det,
        )
        summary = summarize("single", single, [Tile(0, 0, w, h)])
        print(f"single: {summary['count']} spines (mean conf {summary['mean_conf']})")
        payload["results"]["single"] = {**summary, "detections": [asdict(d) for d in single]}
        if args.mode == "single":
            draw_dets = single

    if args.mode in ("both", "tiled"):
        tiled = run_tiled(
            model,
            image_bgr,
            tiles,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            iou=args.iou,
            max_det=args.max_det,
        )
        summary = summarize("tiled", tiled, tiles)
        print(f"tiled:  {summary['count']} spines (mean conf {summary['mean_conf']}, {summary['tiles']} tiles)")
        payload["results"]["tiled"] = {**summary, "detections": [asdict(d) for d in tiled]}
        draw_dets = tiled
        draw_tiles = tiles if len(tiles) > 1 else None

    if args.mode == "both" and "single" in payload["results"] and "tiled" in payload["results"]:
        delta = payload["results"]["tiled"]["count"] - payload["results"]["single"]["count"]
        print(f"delta:  {delta:+d} spines (tiled - single)")

    out_path = args.out
    if out_path is None and draw_dets:
        out_path = image_path.with_name(f"{image_path.stem}.spines.png")
    if out_path and draw_dets is not None:
        annotated = draw_detections(image_bgr, draw_dets, draw_tiles)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Wrote {out_path}")
        payload["annotated_image"] = str(out_path.resolve())

    json_path = args.json or default_json_path(image_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
