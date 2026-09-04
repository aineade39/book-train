#!/usr/bin/env python3
"""Synthesize shelf photos by pasting labeled spine crops onto empty shelf backgrounds.

Pipeline:
  1. Harvest RGBA spine crops from an existing YOLO-OBB dataset (quad labels).
  2. Paste rows of spines into configured horizontal shelf bands on CC backgrounds.
  3. Write YOLO-OBB labels for every pasted spine.

This is cut-and-paste domain randomization — useful as a supplement, not a
replacement for real labeled scenes. Lighting/perspective will look synthetic.

Example:
  .venv/bin/python tools/synth_shelf_spines.py --per-bg 3 --preview 8
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import derived_dir  # noqa: E402


@dataclass
class SpineCrop:
    rgba: np.ndarray  # HxWx4 uint8, upright (tall) spine
    src: str


@dataclass
class Band:
    """Horizontal shelf band in normalized image coords [0,1]."""

    y0: float
    y1: float
    x0: float = 0.05
    x1: float = 0.95


# Per-background shelf bands after EXIF-correct orientation (visual tune).
BANDS: dict[str, list[Band]] = {
    # High-angle photo: sit crops on the lower part of each bay (shelf floor).
    "skrin_bazaru": [
        Band(0.16, 0.26, 0.20, 0.80),
        Band(0.32, 0.42, 0.20, 0.80),
        Band(0.48, 0.58, 0.20, 0.80),
        Band(0.64, 0.74, 0.20, 0.80),
        Band(0.80, 0.90, 0.20, 0.80),
    ],
    "empty_bookshelves": [
        Band(0.12, 0.32, 0.08, 0.92),
        Band(0.38, 0.58, 0.08, 0.92),
        Band(0.64, 0.84, 0.08, 0.92),
    ],
    "joensuu_shelf_crop": [Band(0.12, 0.88, 0.08, 0.92)],
    "procedural_oak": [
        Band(0.10, 0.26, 0.08, 0.92),
        Band(0.30, 0.46, 0.08, 0.92),
        Band(0.50, 0.66, 0.08, 0.92),
        Band(0.70, 0.86, 0.08, 0.92),
    ],
}

# Poor cut-and-paste targets (wrong domain / already stocked / strong aisle perspective).
SKIP_BGS = {
    "empty_mobile_bookcase",
    "seccio_viatges",
    "unibastions",
    "joensuu_library",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source-obb",
        type=Path,
        default=Path.home() / "ml/book-spines/raw/open-shelves/open-shelves.v9i.yolov8-obb",
        help="YOLO-OBB dataset root with train|valid/{images,labels}.",
    )
    p.add_argument(
        "--bg-dir",
        type=Path,
        default=Path.home() / "ml/book-spines/tmp/synth-bgs",
        help="Directory of empty shelf background images.",
    )
    p.add_argument("--out", type=Path, default=derived_dir("synth-shelf-spines"))
    p.add_argument("--split", default="train", help="Output split name.")
    p.add_argument("--per-bg", type=int, default=3, help="Synthetic images per background.")
    p.add_argument("--limit-bg", type=int, default=0, help="Cap backgrounds (0=all usable).")
    p.add_argument("--max-spines", type=int, default=800, help="Max harvested spine crops.")
    p.add_argument("--min-spine-px", type=int, default=24, help="Min crop short-side before keep.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", type=int, default=8, help="Write N overlay previews under out/preview/.")
    return p.parse_args()


def yolo_obb_corners(line: str, w: int, h: int) -> list[tuple[float, float]] | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    vals = list(map(float, parts[1:9]))
    return [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]


def order_quad_for_warp(box: np.ndarray, prefer_tall: bool = True) -> np.ndarray:
    pts = box.tolist()
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    w = float(np.linalg.norm(ordered[1] - ordered[0]))
    h = float(np.linalg.norm(ordered[3] - ordered[0]))
    if prefer_tall and w > h:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered


def crop_spine_rgba(img_bgr: np.ndarray, corners: list[tuple[float, float]]) -> np.ndarray | None:
    pts = np.array(corners, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    (_cx, _cy), (rw, rh), _angle = rect
    if rw < 2 or rh < 2:
        return None
    box = cv2.boxPoints(rect).astype(np.float32)
    w, h = float(rw), float(rh)
    src = order_quad_for_warp(box, prefer_tall=True)
    dst_w = max(8, int(round(min(w, h))))
    dst_h = max(8, int(round(max(w, h))))
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255
    return rgba


def harvest_spines(source: Path, max_spines: int, min_px: int, rng: random.Random) -> list[SpineCrop]:
    crops: list[SpineCrop] = []
    label_files: list[Path] = []
    for split in ("train", "valid", "test"):
        lab = source / split / "labels"
        if lab.is_dir():
            label_files.extend(lab.glob("*.txt"))
    rng.shuffle(label_files)
    for lp in label_files:
        if len(crops) >= max_spines:
            break
        split = lp.parent.parent.name
        img_path = next((source / split / "images").glob(lp.stem + ".*"), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        for ln in lp.read_text().splitlines():
            if len(crops) >= max_spines:
                break
            corners = yolo_obb_corners(ln, w, h)
            if not corners:
                continue
            rgba = crop_spine_rgba(img, corners)
            if rgba is None:
                continue
            sh, sw = rgba.shape[:2]
            if min(sh, sw) < min_px:
                continue
            if sh < sw * 1.2:
                rgba = np.rot90(rgba, k=1)
                sh, sw = rgba.shape[:2]
            if sh < sw * 1.2:
                continue
            crops.append(SpineCrop(rgba=rgba, src=img_path.name))
    return crops


def default_bands(n: int = 3) -> list[Band]:
    bands = []
    top, bot = 0.12, 0.90
    step = (bot - top) / n
    for i in range(n):
        y0 = top + i * step + 0.02
        y1 = top + (i + 1) * step - 0.02
        bands.append(Band(y0, y1))
    return bands


def load_bg_bgr(path: Path) -> np.ndarray | None:
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


def make_procedural_shelf(path: Path, w: int = 1200, h: int = 1600) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (210, 205, 198)
    x0, x1 = int(0.06 * w), int(0.94 * w)
    y0, y1 = int(0.04 * h), int(0.96 * h)
    wood = (92, 120, 150)
    cv2.rectangle(img, (x0, y0), (x1, y1), wood, thickness=-1)
    margin = int(0.03 * w)
    cv2.rectangle(img, (x0 + margin, y0 + margin), (x1 - margin, y1 - margin), (70, 95, 120), -1)
    for t in (0.26, 0.46, 0.66, 0.86):
        yy = int(y0 + t * (y1 - y0))
        cv2.rectangle(img, (x0 + margin, yy - 8), (x1 - margin, yy + 8), wood, -1)
        cv2.line(img, (x0 + margin, yy - 8), (x1 - margin, yy - 8), (140, 170, 190), 1)
    noise = np.random.default_rng(0).integers(-8, 8, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def paste_rgba(
    canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, angle_deg: float = 0.0
) -> list[tuple[float, float]]:
    h, w = rgba.shape[:2]
    if abs(angle_deg) > 0.1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2
        rgba = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        corners0 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners = (M @ np.hstack([corners0, ones]).T).T
    else:
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    ch, cw = canvas.shape[:2]
    x0, y0 = int(x), int(y)
    minx, miny = float(corners[:, 0].min()), float(corners[:, 1].min())
    corners_shift = corners.copy()
    corners_shift[:, 0] += x0 - minx
    corners_shift[:, 1] += y0 - miny

    rh, rw = rgba.shape[:2]
    px = int(round(x0 - minx))
    py = int(round(y0 - miny))
    x1, y1 = max(0, px), max(0, py)
    x2, y2 = min(cw, px + rw), min(ch, py + rh)
    if x2 > x1 and y2 > y1:
        sx1, sy1 = x1 - px, y1 - py
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        roi = canvas[y1:y2, x1:x2]
        patch = rgba[sy1:sy2, sx1:sx2]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        shadow = roi.astype(np.float32) * (1.0 - 0.22 * alpha)
        rgb = patch[:, :, :3].astype(np.float32)
        canvas[y1:y2, x1:x2] = (shadow * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return [(float(c[0]), float(c[1])) for c in corners_shift]


def corners_to_yolo(corners: list[tuple[float, float]], w: int, h: int) -> str | None:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if max(xs) - min(xs) < 3 or max(ys) - min(ys) < 3:
        return None
    parts = ["0"]
    for x, y in corners:
        parts.append(f"{min(1.0, max(0.0, x / w)):.6f}")
        parts.append(f"{min(1.0, max(0.0, y / h)):.6f}")
    return " ".join(parts)


def fill_band(
    canvas: np.ndarray,
    band: Band,
    spines: list[SpineCrop],
    rng: random.Random,
) -> list[list[tuple[float, float]]]:
    ch, cw = canvas.shape[:2]
    x0, x1 = int(band.x0 * cw), int(band.x1 * cw)
    y0, y1 = int(band.y0 * ch), int(band.y1 * ch)
    band_h = max(8, y1 - y0)
    cursor = x0
    quads: list[list[tuple[float, float]]] = []
    gap = max(0, int(band_h * 0.02))
    while cursor < x1 - 4:
        crop = rng.choice(spines)
        rgba = crop.rgba.copy()
        target_h = int(band_h * rng.uniform(0.88, 0.98))
        scale = target_h / rgba.shape[0]
        target_w = max(4, int(round(rgba.shape[1] * scale * rng.uniform(0.85, 1.15))))
        rgba = cv2.resize(rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if cursor + target_w > x1:
            break
        lean = rng.uniform(-8.0, 8.0)
        paste_y = y1 - target_h - rng.randint(0, max(0, int(band_h * 0.05)))
        quads.append(paste_rgba(canvas, rgba, cursor, paste_y, angle_deg=lean))
        cursor += target_w + gap + rng.randint(0, 2)
        if rng.random() < 0.08:
            cursor += rng.randint(4, max(5, band_h // 4))
    return quads


def draw_preview(img_bgr: np.ndarray, quads: list[list[tuple[float, float]]], path: Path) -> None:
    vis = img_bgr.copy()
    for q in quads:
        pts = np.array(q, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 220, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    bg_dir: Path = args.bg_dir
    bg_dir.mkdir(parents=True, exist_ok=True)

    proc = bg_dir / "procedural_oak.jpg"
    if not proc.exists():
        make_procedural_shelf(proc)

    bgs = sorted(
        [
            p
            for p in bg_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem not in SKIP_BGS
        ]
    )
    if args.limit_bg:
        bgs = bgs[: args.limit_bg]
    if not bgs:
        print(f"No usable backgrounds in {bg_dir}", file=sys.stderr)
        return 1

    print(f"Backgrounds ({len(bgs)}): {[p.stem for p in bgs]}")
    print(f"Harvesting spines from {args.source_obb} ...")
    spines = harvest_spines(args.source_obb, args.max_spines, args.min_spine_px, rng)
    print(f"  harvested {len(spines)} spine crops")
    if len(spines) < 10:
        print("Need more spine crops.", file=sys.stderr)
        return 1

    out: Path = args.out
    img_dir = out / "images" / args.split
    lab_dir = out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    (out / "spines.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: spine\n"
    )

    previews = 0
    n_out = 0
    for bg_path in bgs:
        bg0 = load_bg_bgr(bg_path)
        if bg0 is None:
            continue
        max_side = 1600
        h0, w0 = bg0.shape[:2]
        scale = min(1.0, max_side / max(h0, w0))
        if scale < 1.0:
            bg0 = cv2.resize(bg0, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        bands = BANDS.get(bg_path.stem) or default_bands(3)

        for i in range(args.per_bg):
            canvas = bg0.copy()
            alpha = rng.uniform(0.92, 1.08)
            beta = rng.randint(-10, 10)
            canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            all_quads: list[list[tuple[float, float]]] = []
            use_bands = list(bands)
            rng.shuffle(use_bands)
            n_fill = max(1, len(use_bands) - (1 if len(use_bands) > 2 and rng.random() < 0.35 else 0))
            for band in use_bands[:n_fill]:
                all_quads.extend(fill_band(canvas, band, spines, rng))

            stem = f"synth_{bg_path.stem}_{i:02d}"
            ch, cw = canvas.shape[:2]
            lines = [ln for q in all_quads if (ln := corners_to_yolo(q, cw, ch))]
            if len(lines) < 3:
                continue
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_out += 1
            if previews < args.preview:
                draw_preview(canvas, all_quads, out / "preview" / f"{stem}.jpg")
                previews += 1
            print(f"  wrote {stem} boxes={len(lines)}")

    # Move a couple to val
    train_imgs = sorted((out / "images" / "train").glob("*.jpg"))
    for p in train_imgs[: max(1, len(train_imgs) // 5)]:
        stem = p.stem
        for kind in ("images", "labels"):
            src = out / kind / "train" / (stem + (".jpg" if kind == "images" else ".txt"))
            dst = out / kind / "val" / src.name
            if src.exists():
                src.replace(dst)

    (out / "SOURCE.md").write_text(
        f"""# synth-shelf-spines

Cut-and-paste synthetic shelf images: Wikimedia Commons empty-shelf
backgrounds + spine crops from an existing YOLO-OBB source (plus one
procedural oak bookcase).

| Field | Value |
|---|---|
| **id** | `synth-shelf-spines` |
| **script** | `tools/synth_shelf_spines.py` |
| **spine_source** | `{args.source_obb}` |
| **bg_dir** | `{bg_dir}` |
| **images** | `{n_out}` |
| **note** | Survey / domain randomization only. Labels are exact for pasted crops; lighting and perspective are synthetic. Not a substitute for real labeled scenes. |
"""
    )
    print(f"Done: {n_out} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
