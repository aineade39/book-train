#!/usr/bin/env python3
"""Draw OCR-parity overlays: Mac OBBs, optional iOS-native OBBs, oracle centers.

Colors (BGR):
  Mac OBB          cyan-blue
  iOS-native OBB   magenta
  Oracle paired    green
  Oracle unpaired  red
  Pair link        amber (oracle point -> paired Mac box center)

Example:
  .venv/bin/python3 tools/overlay_ocr_parity.py \\
      --mac-dir /tmp/ocr-mac-export --ios-dir /tmp/ocr-ios-export \\
      --out-dir ~/ml/book-spines/eval/ocr-parity/overlays
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ocr_parity import discover_exports, load_json, obb_from_spine, point_in_polygon  # noqa: E402

COLOR_MAC_OBB = (255, 180, 0)
COLOR_IOS_OBB = (255, 0, 255)
COLOR_ORACLE_PAIRED = (0, 220, 0)
COLOR_ORACLE_UNPAIRED = (0, 0, 255)
COLOR_LINK = (0, 200, 255)

SCENE_FILES = {
    "bedroom1": "bedroom1.jpeg",
    "bookcase": "bookcase.jpg",
    "office1": "office1.jpeg",
    "office2": "office2.jpeg",
    "office3": "office3.jpeg",
}


def draw_obb(img, spine, color, thickness):
    pts = np.array(obb_from_spine(spine).corners(), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)


def draw_point(img, xy, color, radius):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 2, lineType=cv2.LINE_AA)


def legend(img, lines, scale):
    x0, y0 = int(24 * scale), int(36 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7 * scale
    th = max(1, int(2 * scale))
    pad = int(10 * scale)
    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t, _ in lines]
    box_w = max(w for w, _h in sizes) + 2 * pad + int(28 * scale)
    line_h = max(h for _w, h in sizes) + pad
    box_h = pad + line_h * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x0 - pad, y0 - int(28 * scale)),
        (x0 + box_w, y0 - int(28 * scale) + box_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, (text, color) in enumerate(lines):
        y = y0 + i * line_h
        cv2.circle(img, (x0 + int(8 * scale), y - int(4 * scale)), int(7 * scale), color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (x0 + int(24 * scale), y), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def pair_indices(oracle: dict, spines: list[dict]) -> dict[int, int]:
    books = oracle["books"]
    w, h = oracle["image"]["w"], oracle["image"]["h"]
    obbs = [obb_from_spine(s) for s in spines]
    avg_diag = sum(o.diag for o in obbs) / len(obbs) if obbs else 0.0
    radius = avg_diag if avg_diag > 0 else 200.0
    candidates = []
    for bi, book in enumerate(books):
        y1000, x1000 = book["point_2d"]
        px, py = (x1000 / 1000.0) * w, (y1000 / 1000.0) * h
        for si, obb in enumerate(obbs):
            contained = point_in_polygon(px, py, obb.corners())
            dist = math.hypot(px - obb.cx, py - obb.cy)
            if contained or dist <= radius:
                candidates.append((not contained, dist, bi, si))
    candidates.sort()
    used_b: set[int] = set()
    used_s: set[int] = set()
    assign: dict[int, int] = {}
    for _, _, bi, si in candidates:
        if bi in used_b or si in used_s:
            continue
        used_b.add(bi)
        used_s.add(si)
        assign[bi] = si
    return assign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mac-dir", type=Path, required=True)
    p.add_argument("--ios-dir", type=Path, default=None)
    p.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "optimize-gemini" / "fixtures" / "assets" / "scenes",
    )
    p.add_argument(
        "--oracles-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "optimize-gemini" / "fixtures" / "oracles",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "ml/book-spines/eval/ocr-parity/overlays",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mac_exports = discover_exports(args.mac_dir)
    ios_exports = discover_exports(args.ios_dir) if args.ios_dir else {}

    for scene_id, filename in SCENE_FILES.items():
        if scene_id not in mac_exports or "mac-native" not in mac_exports[scene_id]:
            print(f"skip {scene_id}: no mac-native export", file=sys.stderr)
            continue
        img_path = args.scenes_dir / filename
        oracle = load_json(args.oracles_dir / f"{scene_id}.json")
        mac = load_json(mac_exports[scene_id]["mac-native"])
        ios_native = None
        if scene_id in ios_exports and "ios-native" in ios_exports[scene_id]:
            ios_native = load_json(ios_exports[scene_id]["ios-native"])

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"failed to read {img_path}")
        h, w = img.shape[:2]
        scale = max(w, h) / 2000.0
        lw = max(2, int(round(2.5 * scale)))
        pr = max(6, int(round(8 * scale)))

        if ios_native:
            for s in ios_native["spines"]:
                draw_obb(img, s, COLOR_IOS_OBB, max(1, lw - 1))
        for s in mac["spines"]:
            draw_obb(img, s, COLOR_MAC_OBB, lw)

        assign = pair_indices(oracle, mac["spines"])
        ow, oh = oracle["image"]["w"], oracle["image"]["h"]
        if (ow, oh) != (w, h):
            print(f"WARN {scene_id}: image {w}x{h} vs oracle {ow}x{oh}", file=sys.stderr)

        paired = unpaired = 0
        for bi, book in enumerate(oracle["books"]):
            y1000, x1000 = book["point_2d"]
            px, py = (x1000 / 1000.0) * ow, (y1000 / 1000.0) * oh
            if bi in assign:
                paired += 1
                spine = mac["spines"][assign[bi]]
                cv2.line(
                    img,
                    (int(round(px)), int(round(py))),
                    (int(round(spine["cx"])), int(round(spine["cy"]))),
                    COLOR_LINK,
                    max(1, lw // 2),
                    lineType=cv2.LINE_AA,
                )
                draw_point(img, (px, py), COLOR_ORACLE_PAIRED, pr)
            else:
                unpaired += 1
                draw_point(img, (px, py), COLOR_ORACLE_UNPAIRED, pr)

        lines = [(f"Mac OBB ({len(mac['spines'])})", COLOR_MAC_OBB)]
        if ios_native:
            lines.append((f"iOS-native OBB ({len(ios_native['spines'])})", COLOR_IOS_OBB))
        lines.extend([
            (f"Oracle paired ({paired})", COLOR_ORACLE_PAIRED),
            (f"Oracle unpaired ({unpaired})", COLOR_ORACLE_UNPAIRED),
            ("Pair link", COLOR_LINK),
        ])
        legend(img, lines, scale)

        out_path = args.out_dir / f"{scene_id}.overlay.jpg"
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"{scene_id}: {out_path}  paired={paired} unpaired={unpaired}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
