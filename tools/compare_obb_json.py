#!/usr/bin/env python3
"""Compare Swift `bookspines` JSON vs Python `tiled_predict_obb.py` (tiled mode).

Runs both CLIs on each image (or reads cached JSON if --swift-json/--py-json
are supplied), then reports count deltas and greedy rotated-IoU matching at
several thresholds. Intended for apples-to-apples checks: same conf/iou/max-det,
Python --mode tiled vs Swift auto-tile (2x2 when max side > 3000).

Example:
  .venv/bin/python tools/compare_obb_json.py scenes/bookcase.jpg
  .venv/bin/python tools/compare_obb_json.py scenes/*.jpeg --weights $BOOK_SPINES_DATA/runs/.../best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_predict_obb import Det, rotated_iou  # noqa: E402


@dataclass
class OBB:
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians
    conf: float

    @classmethod
    def from_swift(cls, s: dict) -> OBB:
        return cls(
            cx=float(s["cxPx"]),
            cy=float(s["cyPx"]),
            w=float(s["wPx"]),
            h=float(s["hPx"]),
            angle=math.radians(float(s["angleDeg"])),
            conf=float(s["confidence"]),
        )

    @classmethod
    def from_python(cls, d: dict) -> OBB:
        return cls(
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            w=float(d["w"]),
            h=float(d["h"]),
            angle=float(d["angle"]),
            conf=float(d["conf"]),
        )

    def as_det(self) -> Det:
        return Det(self.cx, self.cy, self.w, self.h, self.angle, self.conf)


def greedy_match(a: list[OBB], b: list[OBB], iou_thresh: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching by descending IoU (a=swift, b=python)."""
    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for i, oa in enumerate(a):
        da = oa.as_det()
        for j, ob in enumerate(b):
            iou = rotated_iou(da, ob.as_det())
            if iou >= iou_thresh:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, iou))
    unmatched_a = [i for i in range(len(a)) if i not in used_a]
    unmatched_b = [j for j in range(len(b)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def match_stats(a: list[OBB], b: list[OBB]) -> dict:
    out: dict = {"swift": len(a), "python": len(b), "delta_py_minus_swift": len(b) - len(a)}
    for thr in (0.5, 0.45, 0.3):
        pairs, ua, ub = greedy_match(a, b, thr)
        cx_err: list[float] = []
        cy_err: list[float] = []
        conf_err: list[float] = []
        for i, j, _ in pairs:
            cx_err.append(abs(a[i].cx - b[j].cx))
            cy_err.append(abs(a[i].cy - b[j].cy))
            conf_err.append(abs(a[i].conf - b[j].conf))
        out[f"match_iou{thr:.2f}"] = {
            "matched": len(pairs),
            "swift_only": len(ua),
            "python_only": len(ub),
            "mean_cx_err_px": round(sum(cx_err) / len(cx_err), 2) if cx_err else None,
            "mean_cy_err_px": round(sum(cy_err) / len(cy_err), 2) if cy_err else None,
            "mean_conf_err": round(sum(conf_err) / len(conf_err), 4) if conf_err else None,
            "max_cx_err_px": round(max(cx_err), 2) if cx_err else None,
            "max_cy_err_px": round(max(cy_err), 2) if cy_err else None,
        }
    return out


def run_swift(repo: Path, image: Path, model: Path | None) -> dict:
    cmd = ["swift", "run", "-c", "release", "bookspines", str(image)]
    if model:
        cmd.extend(["--model", str(model)])
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"swift bookspines failed ({image.name}):\n{proc.stderr}")
    return json.loads(proc.stdout)


def run_python(py: Path, image: Path, weights: Path, out_json: Path) -> dict:
    cmd = [
        str(py),
        str(Path(__file__).resolve().parent / "tiled_predict_obb.py"),
        str(image),
        "--mode",
        "tiled",
        "--weights",
        str(weights),
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tiled_predict_obb failed ({image.name}):\n{proc.stderr}")
    return json.loads(out_json.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", type=Path, default=Path(__file__).resolve().parents[1] / ".venv/bin/python3")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path.home() / "ml/book-spines/runs/4tu-ieee-shelves_yolo26s-obb_1024px_deg90_ep120_frac100_20260719-0627/weights/best.pt",
    )
    p.add_argument("--model", type=Path, default=None, help="Optional .mlpackage for Swift (default: production alias).")
    p.add_argument("--out-dir", type=Path, default=None, help="Cache per-image JSON here.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "swift_schema": {
            "top_level": ["image", "model", "tiles", "count", "spines"],
            "spine_fields": ["confidence", "cxPx", "cyPx", "wPx", "hPx", "angleDeg", "cornersPx", "texts?"],
            "angle_units": "degrees",
        },
        "python_schema": {
            "top_level": ["image", "image_size", "weights", "imgsz", "conf", "iou", "tiles_spec", "tile_overlap", "results"],
            "detection_fields": ["cx", "cy", "w", "h", "angle", "conf"],
            "angle_units": "radians",
            "note": "compare uses results.tiled.detections (tiled mode only)",
        },
        "scenes": {},
    }

    for image in args.images:
        image = image.expanduser().resolve()
        stem = image.stem
        swift_json_path = args.out_dir / f"{stem}.swift.json" if args.out_dir else None
        py_json_path = args.out_dir / f"{stem}.python.json" if args.out_dir else (image.parent / f"{stem}.compare_py.json")

        swift_doc = run_swift(args.repo, image, args.model)
        if swift_json_path:
            swift_json_path.write_text(json.dumps(swift_doc, indent=2) + "\n")

        py_doc = run_python(args.python, image, args.weights, py_json_path)
        py_dets = py_doc["results"]["tiled"]["detections"]
        swift_dets = swift_doc["spines"]

        swift_obbs = [OBB.from_swift(s) for s in swift_dets]
        py_obbs = [OBB.from_python(d) for d in py_dets]

        scene = {
            "image": str(image),
            "image_size": swift_doc["image"],
            "swift": {
                "model": swift_doc.get("model"),
                "tiles": swift_doc.get("tiles"),
                "count": swift_doc.get("count"),
            },
            "python": {
                "weights": Path(py_doc["weights"]).name,
                "tiles": py_doc["results"]["tiled"]["tiles"],
                "count": py_doc["results"]["tiled"]["count"],
                "mean_conf": py_doc["results"]["tiled"].get("mean_conf"),
            },
            "comparison": match_stats(swift_obbs, py_obbs),
        }
        report["scenes"][stem] = scene

        c = scene["comparison"]
        m = c["match_iou0.50"]
        print(
            f"{stem:22s}  swift={c['swift']:3d}  python={c['python']:3d}  "
            f"delta={c['delta_py_minus_swift']:+3d}  "
            f"matched@0.50={m['matched']:3d}  swift_only={m['swift_only']:2d}  python_only={m['python_only']:2d}  "
            f"mean_err=({m['mean_cx_err_px']},{m['mean_cy_err_px']})px conf={m['mean_conf_err']}"
        )

    if args.out_dir:
        summary_path = args.out_dir / "compare_report.json"
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
