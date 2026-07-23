#!/usr/bin/env python3
"""Draw OBB overlays from Swift or Python prediction JSON.

Supports layout-crops / tiled_predict / bookspines JSON shapes. In compare
mode, draws only detections present in one file but not the other (symmetric
diff at a rotated-IoU threshold) using high-contrast colors.

Examples:
  .venv/bin/python tools/overlay_obb_json.py scenes/bookcase.jpg preds.json \\
      --field detections_merged --out /tmp/obb.png

  .venv/bin/python tools/overlay_obb_json.py scenes/bookcase.jpg \\
      --json-a swift.json --json-b python.json --compare --field first_pass \\
      --label-a swift --label-b python --out /tmp/diff.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_predict_obb import Det, rotated_iou  # noqa: E402

# BGR — chosen for maximum contrast on shelf photos.
COLOR_A_ONLY = (255, 0, 255)   # magenta
COLOR_B_ONLY = (0, 255, 255)   # yellow (BGR)
COLOR_SINGLE = (4, 42, 255)    # orange — matches layout_crop_predict first-pass


@dataclass
class LoadedPredictions:
    path: Path
    label: str
    field: str
    dets: list[Det]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, nargs="?", default=None, help="Scene image (optional if JSON has image path).")
    p.add_argument("json", type=Path, nargs="?", default=None, help="Single prediction JSON (omit in --compare mode).")
    p.add_argument("--json-a", type=Path, default=None, help="First JSON for --compare.")
    p.add_argument("--json-b", type=Path, default=None, help="Second JSON for --compare.")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Symmetric diff overlay: only OBBs matched in one file, not both.",
    )
    p.add_argument(
        "--field",
        default="auto",
        choices=(
            "auto",
            "first_pass",
            "detections_merged",
            "spines",
            "tiled",
            "detections",
        ),
        help="Which detection list to read (auto picks the best match).",
    )
    p.add_argument("--iou", type=float, default=0.5, help="Rotated-IoU threshold for --compare matching.")
    p.add_argument("--label-a", type=str, default=None, help="Legend label for --json-a (default: stem).")
    p.add_argument("--label-b", type=str, default=None, help="Legend label for --json-b (default: stem).")
    p.add_argument("--out", type=Path, required=True, help="Output overlay image path.")
    p.add_argument("--thickness", type=int, default=0, help="Line thickness (0 = auto from image size).")
    return p.parse_args()


def _det_from_layout(d: dict[str, Any]) -> Det:
    return Det(
        float(d["cx"]),
        float(d["cy"]),
        float(d["w"]),
        float(d["h"]),
        float(d["angle"]),
        float(d["conf"]),
    )


def _det_from_bookspines(s: dict[str, Any]) -> Det:
    return Det(
        float(s["cxPx"]),
        float(s["cyPx"]),
        float(s["wPx"]),
        float(s["hPx"]),
        math.radians(float(s["angleDeg"])),
        float(s["confidence"]),
    )


def _pick_field(doc: dict[str, Any], field: str) -> tuple[str, list[dict[str, Any]]]:
    if field != "auto":
        if field == "spines":
            return field, list(doc.get("spines") or [])
        if field == "tiled":
            return field, list(doc.get("results", {}).get("tiled", {}).get("detections") or [])
        if field == "detections":
            raw = doc.get("detections")
            if isinstance(raw, list):
                return field, raw
            raise KeyError("--field detections requires top-level 'detections' list")
        raw = doc.get(field)
        if not isinstance(raw, list):
            raise KeyError(f"JSON has no list field {field!r}")
        return field, raw

    if isinstance(doc.get("detections_merged"), list):
        return "detections_merged", doc["detections_merged"]
    if isinstance(doc.get("first_pass"), list):
        return "first_pass", doc["first_pass"]
    if isinstance(doc.get("spines"), list):
        return "spines", doc["spines"]
    tiled = doc.get("results", {}).get("tiled", {}).get("detections")
    if isinstance(tiled, list):
        return "tiled", tiled
    if isinstance(doc.get("detections"), list):
        return "detections", doc["detections"]
    raise KeyError("Could not auto-detect detection list in JSON")


def load_predictions(path: Path, field: str, label: str | None) -> LoadedPredictions:
    doc = json.loads(path.expanduser().resolve().read_text())
    picked, raw = _pick_field(doc, field)
    dets: list[Det] = []
    for item in raw:
        if "cxPx" in item or "confidence" in item:
            dets.append(_det_from_bookspines(item))
        else:
            dets.append(_det_from_layout(item))
    return LoadedPredictions(
        path=path.expanduser().resolve(),
        label=label or path.stem,
        field=picked,
        dets=dets,
    )


def resolve_image_path(cli_image: Path | None, *json_paths: Path) -> Path:
    if cli_image is not None:
        p = cli_image.expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Image not found: {p}")
    for jp in json_paths:
        doc = json.loads(jp.read_text())
        img = doc.get("image")
        if isinstance(img, str) and Path(img).is_file():
            return Path(img).resolve()
        if isinstance(img, dict) and "path" in img:
            p = Path(img["path"])
            if p.is_file():
                return p.resolve()
    raise FileNotFoundError("No image: pass <image> or use JSON with a valid image path")


def greedy_match_indices(a: list[Det], b: list[Det], iou_thresh: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    cands: list[tuple[float, int, int]] = []
    for i, da in enumerate(a):
        for j, db in enumerate(b):
            iou = rotated_iou(da, db)
            if iou >= iou_thresh:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    for iou, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, iou))
    unmatched_a = [i for i in range(len(a)) if i not in used_a]
    unmatched_b = [j for j in range(len(b)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def line_width(h: int, w: int, thickness: int) -> int:
    if thickness > 0:
        return thickness
    return max(2, int(round((w + h) / 2 * 0.0025)))


def draw_obbs(
    image_bgr: np.ndarray,
    dets: list[Det],
    color: tuple[int, int, int],
    lw: int,
) -> None:
    for det in dets:
        pts = np.array(det.corners(), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=lw)


def mean_conf(dets: list[Det]) -> float:
    return float(np.mean([d.conf for d in dets])) if dets else 0.0


def draw_header(
    canvas: np.ndarray,
    lines: list[str],
    *,
    bar_h: int = 0,
) -> np.ndarray:
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.85, w / 2200))
    thickness = max(1, int(round(scale * 2)))
    line_h = int(round(22 * scale + 10))
    bar_h = bar_h or (line_h * len(lines) + 16)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    y = int(14 * scale + 12)
    for line in lines:
        cv2.putText(canvas, line, (12, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
        y += line_h
    return canvas


def compare_lines(
    image: Path,
    field_a: str,
    field_b: str,
    iou: float,
    a: LoadedPredictions,
    b: LoadedPredictions,
    pairs: list[tuple[int, int, float]],
    only_a: list[int],
    only_b: list[int],
) -> list[str]:
    dets_a = [a.dets[i] for i in only_a]
    dets_b = [b.dets[j] for j in only_b]
    return [
        f"OBB compare (symmetric diff)  IoU>={iou:.2f}  image: {image.name}",
        f"A [{a.label}] field={field_a}  total={len(a.dets)}  exclusive={len(only_a)}  "
        f"(magenta)  mean_conf={mean_conf(dets_a):.3f}",
        f"B [{b.label}] field={field_b}  total={len(b.dets)}  exclusive={len(only_b)}  "
        f"(yellow)  mean_conf={mean_conf(dets_b):.3f}",
        f"matched both: {len(pairs)}  hidden on overlay",
    ]


def single_lines(image: Path, loaded: LoadedPredictions) -> list[str]:
    return [
        f"OBB overlay  image: {image.name}",
        f"source: {loaded.path.name}  field: {loaded.field}  count: {len(loaded.dets)}  "
        f"mean_conf: {mean_conf(loaded.dets):.3f}",
    ]


def main() -> int:
    args = parse_args()
    compare = args.compare or (args.json_a is not None and args.json_b is not None)
    if compare:
        if args.json_a is None or args.json_b is None:
            print("error: --compare requires --json-a and --json-b", file=sys.stderr)
            return 2
        path_a = args.json_a.expanduser().resolve()
        path_b = args.json_b.expanduser().resolve()
        loaded_a = load_predictions(path_a, args.field, args.label_a)
        loaded_b = load_predictions(path_b, args.field, args.label_b)
        image_path = resolve_image_path(args.image, path_a, path_b)
    else:
        if args.json is None:
            print("error: pass <json> or use --json-a/--json-b with --compare", file=sys.stderr)
            return 2
        path = args.json.expanduser().resolve()
        loaded_a = load_predictions(path, args.field, None)
        image_path = resolve_image_path(args.image, path)
        loaded_b = None

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"error: could not read image {image_path}", file=sys.stderr)
        return 1
    h, w = image_bgr.shape[:2]
    lw = line_width(h, w, args.thickness)
    out = image_bgr.copy()

    if compare:
        assert loaded_b is not None
        pairs, only_a_idx, only_b_idx = greedy_match_indices(loaded_a.dets, loaded_b.dets, args.iou)
        only_a = [loaded_a.dets[i] for i in only_a_idx]
        only_b = [loaded_b.dets[j] for j in only_b_idx]
        draw_obbs(out, only_a, COLOR_A_ONLY, lw + 1)
        draw_obbs(out, only_b, COLOR_B_ONLY, lw + 1)
        header = compare_lines(
            image_path,
            loaded_a.field,
            loaded_b.field,
            args.iou,
            loaded_a,
            loaded_b,
            pairs,
            only_a_idx,
            only_b_idx,
        )
    else:
        draw_obbs(out, loaded_a.dets, COLOR_SINGLE, lw)
        header = single_lines(image_path, loaded_a)

    out = draw_header(out, header)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), out):
        print(f"error: could not write {args.out}", file=sys.stderr)
        return 1

    print("=" * 72)
    for line in header:
        print(line)
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
