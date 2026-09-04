#!/usr/bin/env python3
"""Interactive OBB label viewer (YOLO-OBB, LabelMe JSON, Create ML JSON).

Draws ground-truth quads on the image. Default mode is an OpenCV window you
can step through with the keyboard. Optionally write overlay PNGs + an HTML
index for browser review.

Two-color compare (e.g. manual corrections vs training labels):

  .venv/bin/python tools/view_obb_labels.py path/to/img.jpg \\
      --labels path/to/manual.json --labels-b path/to/train.txt \\
      --name-a manual --name-b train --write /tmp/compare --no-show

Manual JSON shape: ``{"boxes":[{"corners_px":[[x,y],...]}, ...]}``
(also accepts YOLO-OBB ``.txt`` for either side).

Examples:
  # open-shelves YOLO-OBB train split
  .venv/bin/python tools/view_obb_labels.py \\
      $BOOK_SPINES_DATA/raw/open-shelves/open-shelves.v9i.yolov8-obb/train

  # single image + label
  .venv/bin/python tools/view_obb_labels.py path/to/img.jpg --labels path/to/img.txt

  # LabelMe sidecar JSON (4TU-style)
  .venv/bin/python tools/view_obb_labels.py path/to/IMG_0001.json

  # Create ML folder (images + annotations.json)
  .venv/bin/python tools/view_obb_labels.py \\
      $BOOK_SPINES_DATA/derived/4tu-spines_createml/train

  # dump overlays + browse
  .venv/bin/python tools/view_obb_labels.py <dataset> --write /tmp/obb-view --html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import book_spines_data  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
COLOR_BOX = (0, 255, 0)  # BGR — single-source default
COLOR_VERT = (0, 0, 255)
COLOR_TEXT = (255, 255, 0)
# Two-color compare (high contrast on shelf photos)
COLOR_A = (255, 0, 255)  # magenta — typically manual
COLOR_B = (0, 220, 255)  # yellow/amber — typically train


@dataclass
class Sample:
    image: Path
    label: Path | None
    kind: str  # yolo | labelme | createml
    boxes: list[np.ndarray] | None = None  # filled after load; (N,4,2) px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "path",
        type=Path,
        help="Dataset root, image, LabelMe JSON, or Create ML folder.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Primary labels: YOLO-OBB .txt or manual corners JSON (default: sibling labels/<stem>.txt).",
    )
    p.add_argument(
        "--labels-b",
        type=Path,
        default=None,
        help="Second label set for two-color compare (YOLO-OBB .txt or manual JSON).",
    )
    p.add_argument("--name-a", default="A", help="Legend name for --labels (default A).")
    p.add_argument("--name-b", default="B", help="Legend name for --labels-b (default B).")
    p.add_argument(
        "--split",
        default=None,
        help="When path is a YOLO root with train/val/test, pick one split (default: first found).",
    )
    p.add_argument("--limit", type=int, default=0, help="View at most N samples (0 = all).")
    p.add_argument("--start", type=int, default=0, help="0-based index to start at.")
    p.add_argument("--no-index", action="store_true", help="Hide per-vertex index labels.")
    p.add_argument("--thickness", type=int, default=0, help="Line thickness (0 = auto).")
    p.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write overlay images here instead of / in addition to interactive view.",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="With --write, also emit index.html and open it.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window (use with --write).",
    )
    p.add_argument(
        "--max-display",
        type=int,
        default=1400,
        help="Max window width/height in interactive mode (default 1400).",
    )
    return p.parse_args()


def line_width(h: int, w: int, thickness: int) -> int:
    if thickness > 0:
        return max(1, int(round(thickness * 0.5)))
    # Auto thickness scaled 0.5× vs prior default for lighter overlays.
    return max(1, int(round((w + h) / 2 * 0.0025 * 0.5)))


def parse_yolo_obb_txt(path: Path, w: int, h: int) -> list[np.ndarray]:
    """YOLO-OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized, may be outside [0,1])."""
    if not path.is_file():
        return []
    out: list[np.ndarray] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        vals = list(map(float, parts[1:9]))
        pts = np.array([(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)], dtype=np.float32)
        out.append(pts)
    return out


def parse_manual_boxes_json(path: Path) -> list[np.ndarray]:
    """Manual / correction JSON: ``{"boxes":[{"corners_px":[[x,y],x4]}, ...]}``."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    raw = doc.get("boxes") if isinstance(doc, dict) else doc
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected list of boxes or {{boxes: [...]}}")
    out: list[np.ndarray] = []
    for item in raw:
        if isinstance(item, dict):
            corners = item.get("corners_px") or item.get("corners") or item.get("pts")
        else:
            corners = item
        if not corners or len(corners) < 3:
            continue
        pts = np.array([(float(p[0]), float(p[1])) for p in corners[:4]], dtype=np.float32)
        out.append(pts)
    return out


def load_boxes_file(path: Path | None, w: int, h: int) -> list[np.ndarray]:
    """Load YOLO-OBB .txt or manual corners JSON. Missing path → empty."""
    if path is None:
        return []
    path = path.expanduser().resolve()
    if not path.is_file():
        return []
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and "shapes" in doc:
            return parse_labelme_shapes(doc)
        return parse_manual_boxes_json(path)
    return parse_yolo_obb_txt(path, w, h)


def parse_labelme_shapes(doc: dict[str, Any]) -> list[np.ndarray]:
    boxes: list[np.ndarray] = []
    for shape in doc.get("shapes") or []:
        st = (shape.get("shape_type") or "polygon").lower()
        if st not in ("polygon", "rectangle", "rotation", ""):
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        if st == "rectangle" and len(points) == 2:
            (x0, y0), (x1, y1) = points
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            pts = [(float(p[0]), float(p[1])) for p in points]
            if len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            elif len(pts) > 4:
                # Fit min-area rect for >4-gon LabelMe polys.
                arr = np.array(pts, dtype=np.float32)
                rect = cv2.minAreaRect(arr)
                pts = cv2.boxPoints(rect).tolist()
            elif len(pts) == 3:
                # Duplicate last to make a quad-ish poly for drawing.
                pts = pts + [pts[0]]
        if len(pts) >= 3:
            boxes.append(np.array(pts[:4] if len(pts) >= 4 else pts, dtype=np.float32))
    return boxes


def parse_createml_boxes(anns: list[dict[str, Any]]) -> list[np.ndarray]:
    """Create ML center/width/height AABB → corner quads."""
    boxes: list[np.ndarray] = []
    for ann in anns:
        c = ann.get("coordinates") or {}
        try:
            cx, cy = float(c["x"]), float(c["y"])
            bw, bh = float(c["width"]), float(c["height"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0 = cx - bw / 2.0, cy - bh / 2.0
        x1, y1 = cx + bw / 2.0, cy + bh / 2.0
        boxes.append(
            np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float32)
        )
    return boxes


def resolve_yolo_split_root(root: Path, split: str | None) -> Path | None:
    """Return a dir that contains images/ (+ labels/), or None."""
    root = root.expanduser().resolve()
    if (root / "images").is_dir():
        return root
    splits = ["train", "val", "valid", "test"]
    if split:
        splits = [split] + [s for s in splits if s != split]
    for name in splits:
        cand = root / name
        if (cand / "images").is_dir():
            return cand
    return None


def iter_yolo_samples(split_root: Path) -> Iterator[Sample]:
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        yield Sample(image=img, label=lbl if lbl.is_file() else None, kind="yolo")


def iter_createml_samples(folder: Path) -> Iterator[Sample]:
    ann_path = folder / "annotations.json"
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"{ann_path}: expected a list of image annotations")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in doc:
        name = item.get("imagefilename") or item.get("imageFilename")
        if not name:
            continue
        by_name[str(name)] = list(item.get("annotation") or [])
    for name, anns in sorted(by_name.items()):
        img = folder / name
        if not img.is_file():
            continue
        boxes = parse_createml_boxes(anns)
        yield Sample(image=img, label=ann_path, kind="createml", boxes=boxes)


def iter_labelme_dir(folder: Path) -> Iterator[Sample]:
    for jp in sorted(folder.glob("*.json")):
        if jp.name.lower() == "annotations.json":
            continue
        yield Sample(image=jp, label=jp, kind="labelme")  # image resolved on load


def load_sample(sample: Sample) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Return (bgr image, boxes px, title)."""
    if sample.kind == "labelme":
        jp = sample.label if sample.label is not None else sample.image
        assert jp is not None
        doc = json.loads(jp.read_text(encoding="utf-8"))
        boxes = parse_labelme_shapes(doc)
        img_path = None
        image_data = doc.get("imageData")
        if image_data:
            import base64

            raw = base64.b64decode(image_data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            title = jp.name
        else:
            rel = doc.get("imagePath") or ""
            candidates = [
                jp.with_suffix(".jpg"),
                jp.with_suffix(".jpeg"),
                jp.with_suffix(".png"),
                jp.parent / Path(str(rel).replace("\\", "/")).name,
            ]
            for c in candidates:
                if c.is_file():
                    img_path = c
                    break
            if img_path is None:
                raise FileNotFoundError(f"{jp}: no image for LabelMe JSON")
            bgr = cv2.imread(str(img_path))
            title = img_path.name
        if bgr is None:
            raise FileNotFoundError(f"{jp}: could not decode image")
        return bgr, boxes, title

    bgr = cv2.imread(str(sample.image))
    if bgr is None:
        raise FileNotFoundError(f"could not read {sample.image}")
    h, w = bgr.shape[:2]
    if sample.boxes is not None:
        boxes = sample.boxes
    elif sample.kind == "yolo" and sample.label is not None:
        boxes = parse_yolo_obb_txt(sample.label, w, h)
    else:
        boxes = []
    return bgr, boxes, sample.image.name


def draw_dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    lw: int,
    *,
    dash: int = 12,
    gap: int = 8,
) -> None:
    """Closed dashed outline so an underlying solid color stays visible."""
    pts_i = pts.reshape(-1, 2).astype(np.float32)
    n = len(pts_i)
    for i in range(n):
        p0 = pts_i[i]
        p1 = pts_i[(i + 1) % n]
        seg = p1 - p0
        length = float(np.hypot(seg[0], seg[1]))
        if length < 1e-3:
            continue
        u = seg / length
        t = 0.0
        draw = True
        while t < length:
            span = dash if draw else gap
            t1 = min(length, t + span)
            if draw:
                a = (int(round(p0[0] + u[0] * t)), int(round(p0[1] + u[1] * t)))
                b = (int(round(p0[0] + u[0] * t1)), int(round(p0[1] + u[1] * t1)))
                cv2.line(img, a, b, color, lw, cv2.LINE_AA)
            t = t1
            draw = not draw


def draw_box_set(
    out: np.ndarray,
    boxes: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    lw: int,
    show_vertex_index: bool,
    label_prefix: str = "",
    style: str = "solid",
    fill_alpha: float = 0.0,
) -> None:
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(0.7, w / 1600))
    for bi, pts in enumerate(boxes):
        pts_i = pts.astype(np.int32)
        if fill_alpha > 0:
            layer = out.copy()
            cv2.fillPoly(layer, [pts_i.reshape(-1, 1, 2)], color)
            cv2.addWeighted(layer, fill_alpha, out, 1.0 - fill_alpha, 0, out)
        if style == "dashed":
            draw_dashed_polyline(out, pts_i, color, lw)
        else:
            cv2.polylines(out, [pts_i.reshape(-1, 1, 2)], True, color, lw, cv2.LINE_AA)
        for vi, (x, y) in enumerate(pts_i):
            cv2.circle(out, (int(x), int(y)), max(2, lw), color, -1)
            if show_vertex_index:
                cv2.putText(
                    out,
                    str(vi),
                    (int(x) + 3, int(y) - 3),
                    font,
                    scale * 0.75,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        tag = f"{label_prefix}{bi}" if label_prefix else str(bi)
        cv2.putText(
            out,
            tag,
            (int(cx) - 6, int(cy) + 4),
            font,
            scale,
            color,
            max(1, lw - 1),
            cv2.LINE_AA,
        )


def draw_overlay(
    bgr: np.ndarray,
    boxes: list[np.ndarray],
    *,
    title: str,
    index: int,
    total: int,
    show_vertex_index: bool,
    thickness: int,
    boxes_b: list[np.ndarray] | None = None,
    name_a: str = "A",
    name_b: str = "B",
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    lw = line_width(h, w, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(0.7, w / 1600))
    compare = boxes_b is not None
    n_a = len(boxes)
    n_b = len(boxes_b or []) if compare else 0

    if compare:
        # Yellow (train): thick solid + light fill. Magenta (manual): dashed on top
        # so overlapping edges still show yellow in the gaps.
        draw_box_set(
            out,
            boxes_b or [],
            COLOR_B,
            lw=max(1, lw + 1),
            show_vertex_index=False,
            label_prefix="B",
            style="solid",
            fill_alpha=0.18,
        )
        draw_box_set(
            out,
            boxes,
            COLOR_A,
            lw=max(1, lw),
            show_vertex_index=show_vertex_index,
            label_prefix="A",
            style="dashed",
            fill_alpha=0.0,
        )
        header = f"[{index + 1}/{total}] {title}  [n/p/q/s]"
    else:
        draw_box_set(out, boxes, COLOR_BOX, lw=lw, show_vertex_index=show_vertex_index)
        for pts in boxes:
            for x, y in pts.astype(np.int32):
                cv2.circle(out, (int(x), int(y)), max(2, lw), COLOR_VERT, -1)
        header = (
            f"[{index + 1}/{total}] {title}  boxes={n_a}  "
            f"[n/→ next  p/← prev  q quit  s save]"
        )

    bar_h = int(28 * max(1.0, scale / 0.5))
    if compare:
        bar_h = int(bar_h * 1.55)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
    cv2.putText(out, header, (10, int(bar_h * 0.42)), font, scale, (245, 245, 245), 1, cv2.LINE_AA)
    if compare:
        y = int(bar_h * 0.82)
        x = 10
        cv2.rectangle(out, (x, y - 11), (x + 18, y + 5), COLOR_A, -1)
        cv2.line(out, (x + 2, y - 3), (x + 16, y - 3), (16, 16, 16), 1, cv2.LINE_AA)
        label_a = f"{name_a}: {n_a}"
        cv2.putText(out, label_a, (x + 24, y + 2), font, scale * 0.9, COLOR_A, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label_a, font, scale * 0.9, 1)
        x = x + 24 + tw + 28
        cv2.rectangle(out, (x, y - 11), (x + 18, y + 5), COLOR_B, -1)
        label_b = f"{name_b}: {n_b}"
        cv2.putText(out, label_b, (x + 24, y + 2), font, scale * 0.9, COLOR_B, 1, cv2.LINE_AA)
    return out


def discover_samples(path: Path, labels: Path | None, split: str | None) -> list[Sample]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix.lower() == ".json":
            return [Sample(image=path, label=path, kind="labelme")]
        if path.suffix.lower() in IMG_EXTS:
            if labels is not None:
                lbl = labels.expanduser().resolve()
            else:
                # sibling .txt or ../labels/<stem>.txt
                sib = path.with_suffix(".txt")
                alt = path.parent.parent / "labels" / f"{path.stem}.txt"
                lbl = sib if sib.is_file() else alt
            return [Sample(image=path, label=lbl if lbl.is_file() else None, kind="yolo")]
        raise ValueError(f"Unsupported file type: {path}")

    # Directory
    if (path / "annotations.json").is_file():
        return list(iter_createml_samples(path))

    yolo_root = resolve_yolo_split_root(path, split)
    if yolo_root is not None:
        return list(iter_yolo_samples(yolo_root))

    jsons = [p for p in path.glob("*.json") if p.name.lower() != "annotations.json"]
    if jsons:
        return list(iter_labelme_dir(path))

    # Flat images + labels/ next to them or beside
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    if images:
        lbl_dir = path / "labels"
        samples: list[Sample] = []
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                lbl = img.with_suffix(".txt")
            samples.append(Sample(image=img, label=lbl if lbl.is_file() else None, kind="yolo"))
        return samples

    raise FileNotFoundError(
        f"No YOLO images/labels, LabelMe JSON, or Create ML annotations.json under {path}"
    )


def fit_display(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def write_html_index(out_dir: Path, entries: list[tuple[str, str, int]]) -> Path:
    """entries: (filename, title, n_boxes)"""
    rows = []
    for fname, title, n in entries:
        rows.append(
            "<div class='card'>"
            f"<a href='{html.escape(fname)}'><img src='{html.escape(fname)}' loading='lazy'></a>"
            f"<div class='cap'>{html.escape(title)} · {n} boxes</div>"
            "</div>"
        )
    body = "\n".join(rows)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OBB label viewer</title>
<style>
body {{ margin:0; font:14px/1.4 system-ui,sans-serif; background:#111; color:#eee; }}
h1 {{ margin:16px; font-size:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; padding:16px; }}
.card {{ background:#1a1a1a; border-radius:8px; overflow:hidden; }}
.card img {{ width:100%; display:block; }}
.cap {{ padding:8px 10px; color:#bbb; }}
</style></head>
<body>
<h1>OBB overlays ({len(entries)})</h1>
<div class="grid">
{body}
</div>
</body></html>
"""
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def interactive_loop(
    samples: list[Sample],
    *,
    start: int,
    show_vertex_index: bool,
    thickness: int,
    max_display: int,
    write_dir: Path | None,
    labels_b: Path | None = None,
    name_a: str = "A",
    name_b: str = "B",
) -> int:
    if not samples:
        print("No samples.", file=sys.stderr)
        return 1
    i = max(0, min(start, len(samples) - 1))
    win = "obb-labels  (n/p/q/s)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        sample = samples[i]
        try:
            bgr, boxes, title = load_sample(sample)
            h, w = bgr.shape[:2]
            boxes_b = load_boxes_file(labels_b, w, h) if labels_b is not None else None
        except Exception as exc:  # noqa: BLE001
            print(f"skip [{i}] {sample.image}: {exc}", file=sys.stderr)
            i = (i + 1) % len(samples)
            continue
        drawn = draw_overlay(
            bgr,
            boxes,
            title=title,
            index=i,
            total=len(samples),
            show_vertex_index=show_vertex_index,
            thickness=thickness,
            boxes_b=boxes_b,
            name_a=name_a,
            name_b=name_b,
        )
        cv2.imshow(win, fit_display(drawn, max_display))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # q / Esc
            break
        if key in (ord("n"), ord("."), 83):  # n / . / right
            i = (i + 1) % len(samples)
        elif key in (ord("p"), ord(","), 81):  # p / , / left
            i = (i - 1) % len(samples)
        elif key == ord("s"):
            dest_dir = write_dir or (book_spines_data() / "eval" / "obb-label-view")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{i:04d}_{Path(title).stem}_overlay.jpg"
            cv2.imwrite(str(dest), drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"saved {dest}")
        elif key == ord("r"):
            continue
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    samples = discover_samples(args.path, args.labels, args.split)
    if args.limit > 0:
        samples = samples[args.start : args.start + args.limit]
    elif args.start > 0:
        samples = samples[args.start :]

    print(f"samples: {len(samples)} from {args.path}")
    if not samples:
        return 1

    # Single-image compare: --labels overrides the discovered primary label file.
    if args.labels is not None and len(samples) == 1 and samples[0].kind == "yolo":
        samples[0] = Sample(
            image=samples[0].image,
            label=args.labels.expanduser().resolve(),
            kind="yolo",
            boxes=None,
        )

    write_dir = args.write.expanduser().resolve() if args.write else None
    html_entries: list[tuple[str, str, int]] = []
    labels_b = args.labels_b.expanduser().resolve() if args.labels_b else None

    if write_dir is not None:
        write_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            try:
                bgr, boxes, title = load_sample(sample)
                h, w = bgr.shape[:2]
                boxes_b = load_boxes_file(labels_b, w, h) if labels_b is not None else None
            except Exception as exc:  # noqa: BLE001
                print(f"skip write [{i}] {sample.image}: {exc}", file=sys.stderr)
                continue
            drawn = draw_overlay(
                bgr,
                boxes,
                title=title,
                index=i,
                total=len(samples),
                show_vertex_index=not args.no_index,
                thickness=args.thickness,
                boxes_b=boxes_b,
                name_a=args.name_a,
                name_b=args.name_b,
            )
            fname = f"{i:04d}_{Path(title).stem}_overlay.jpg"
            out_path = write_dir / fname
            cv2.imwrite(str(out_path), drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            n = len(boxes) + (len(boxes_b) if boxes_b is not None else 0)
            html_entries.append((fname, title, n))
            print(f"wrote {out_path}  A={len(boxes)} B={len(boxes_b) if boxes_b is not None else '-'}")
        if args.html and html_entries:
            index = write_html_index(write_dir, html_entries)
            print(f"wrote {index}")
            webbrowser.open(index.as_uri())

    if args.no_show:
        return 0

    return interactive_loop(
        samples,
        start=0 if (args.limit or args.start) else args.start,
        show_vertex_index=not args.no_index,
        thickness=args.thickness,
        max_display=args.max_display,
        write_dir=write_dir,
        labels_b=labels_b,
        name_a=args.name_a,
        name_b=args.name_b,
    )


if __name__ == "__main__":
    raise SystemExit(main())
