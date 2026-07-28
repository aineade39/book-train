#!/usr/bin/env python3
"""Draw spine-id decision overlays on scene photos.

Colors match the BookID app (`ResultsView` / `SpineMatchDecision.overlayColor`):
  auto-accept / barcode / fm-assisted  green
  ambiguous (needs confirmation)       yellow
  no-match / OCR quality fail          gray

Header: scene title + capture-gate advisory + color legend.

Example:
  .venv/bin/python3 tools/overlay_spine_id.py \\
      --json-dir ~/ml/book-spines/eval/spine-id-scenes/json \\
      --scenes-dir ../optimize-gemini/fixtures/assets/scenes \\
      --out-dir ~/ml/book-spines/eval/spine-id-scenes/overlays
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ocr_parity import OBB  # noqa: E402

# BGR — aligned with ResultsView overlay colors.
COLOR_AUTO = (0, 200, 0)
COLOR_AMBIG = (0, 220, 255)
COLOR_NOMATCH = (140, 140, 140)
COLOR_QUALITY_FAIL = (90, 90, 90)

SCENE_FILES = {
    "bedroom1": "bedroom1.jpeg",
    "bookcase": "bookcase.jpg",
    "office1": "office1.jpeg",
    "office2": "office2.jpeg",
    "office3": "office3.jpeg",
}


def spine_color(spine: dict) -> tuple[int, int, int]:
    decision = spine.get("decision", "no-match")
    if decision == "auto-accept":
        return COLOR_AUTO
    if decision == "ambiguous":
        return COLOR_AMBIG
    if not spine.get("passedOCRQualityGate", True):
        return COLOR_QUALITY_FAIL
    return COLOR_NOMATCH


def spine_obb(spine: dict) -> OBB:
    return OBB(
        float(spine["cx"]),
        float(spine["cy"]),
        float(spine["w"]),
        float(spine["h"]),
        math.radians(float(spine["angleDeg"])),
    )


def draw_obb(img: np.ndarray, obb: OBB, color: tuple[int, int, int], thickness: int) -> None:
    pts = np.array(obb.corners(), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)


def draw_header(canvas: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(0.9, w / 2200))
    thickness = max(1, int(round(scale * 2)))
    line_h = int(round(24 * scale + 10))
    bar_h = line_h * len(lines) + int(20 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    y = int(16 * scale + 14)
    for line in lines:
        cv2.putText(canvas, line, (14, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
        y += line_h
    return canvas


def legend_segment(counts: dict[str, int]) -> str:
    parts = []
    if counts.get("auto", 0):
        parts.append(f"green auto-accept ({counts['auto']})")
    if counts.get("ambig", 0):
        parts.append(f"yellow ambiguous ({counts['ambig']})")
    if counts.get("nomatch", 0):
        parts.append(f"gray no-match ({counts['nomatch']})")
    if counts.get("quality", 0):
        parts.append(f"gray quality-fail ({counts['quality']})")
    return "  |  ".join(parts) if parts else "no spines"


def overlay_scene(scene_id: str, payload: dict, image_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"failed to read image: {image_path}")

    h, w = img.shape[:2]
    lw = max(2, int(round(max(w, h) / 900)))

    counts = {"auto": 0, "ambig": 0, "nomatch": 0, "quality": 0}
    for spine in payload.get("spines", []):
        color = spine_color(spine)
        if color == COLOR_AUTO:
            counts["auto"] += 1
        elif color == COLOR_AMBIG:
            counts["ambig"] += 1
        elif color == COLOR_QUALITY_FAIL:
            counts["quality"] += 1
        else:
            counts["nomatch"] += 1
        draw_obb(img, spine_obb(spine), color, lw)

    gate = "PASS" if payload.get("capturePassed", True) else "ADVISORY FAIL"
    sharp = payload.get("captureSharpness")
    expo = payload.get("captureExposure")
    gate_line = ""
    if sharp is not None and expo is not None:
        gate_line = (
            f"capture gate: {gate}  sharpness={sharp:.3f}  exposure={expo:.3f}  "
            f"spines={len(payload.get('spines', []))}"
        )

    header = [
        f"{scene_id}  —  spine-id pipeline (advisory capture gate)",
        gate_line or f"spines={len(payload.get('spines', []))}",
        f"legend: {legend_segment(counts)}",
    ]
    if payload.get("isbnBarcodes"):
        header.append(f"ISBN barcodes: {', '.join(payload['isbnBarcodes'])}")

    out = draw_header(img, header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"{scene_id}: {out_path}  {legend_segment(counts)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", type=Path, required=True, help="Directory of <scene>.json spine-id outputs.")
    p.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "optimize-gemini" / "fixtures" / "assets" / "scenes",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scene", action="append", default=[], help="Limit to scene id(s); default all five oracle scenes.")
    args = p.parse_args()

    scenes = args.scene or list(SCENE_FILES)
    for scene_id in scenes:
        filename = SCENE_FILES.get(scene_id)
        if not filename:
            print(f"skip unknown scene: {scene_id}", file=sys.stderr)
            continue
        json_path = args.json_dir / f"{scene_id}.json"
        if not json_path.exists():
            print(f"skip {scene_id}: missing {json_path}", file=sys.stderr)
            continue
        payload = json.loads(json_path.read_text())
        image_path = args.scenes_dir / filename
        overlay_scene(scene_id, payload, image_path, args.out_dir / f"{scene_id}.overlay.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
