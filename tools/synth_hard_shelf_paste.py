#!/usr/bin/env python3
"""Hard-case shelf synthesis v2.

Unlike the naive vertical-spine sprinkler:
  1. Hosts are cluttered (non-empty) shelf photos with non-book objects.
  2. Clear 1–2 shelf bays by painting empty backboard from border wood colors.
  3. Mine hard donor *regions* from labeled OBB data (horizontal piles,
     horizontal-on-vertical mixes) by reading annotation geometry.
  4. Tile those regions flush into the cleared bays (full shelf height,
     left→right) and carry all OBB labels through.

Example:
  .venv/bin/python tools/synth_hard_shelf_paste.py
"""

from __future__ import annotations

import argparse
import json
import math
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
class Bay:
    """Normalized shelf bay [0,1]. Cleared then filled."""

    x0: float
    y0: float
    x1: float
    y1: float
    # Prefer donor aspect: "wide" = horizontal piles; "any" = mixed
    prefer: str = "wide"


@dataclass
class Host:
    path: Path
    bays: list[Bay]
    note: str = ""


@dataclass
class Donor:
    rgb: np.ndarray  # HxWx3
    labels: list[list[tuple[float, float]]]  # quads in donor pixel space
    n_h: int
    n_v: int
    stem: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source-obb",
        type=Path,
        default=Path.home() / "ml/book-spines/raw/open-shelves/open-shelves.v9i.yolov8-obb",
    )
    p.add_argument("--out", type=Path, default=derived_dir("synth-hard-shelf"))
    p.add_argument("--per-host", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--preview", type=int, default=12)
    return p.parse_args()


def load_bgr(path: Path, max_side: int | None = None) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def parse_obb(line: str, w: int, h: int) -> dict | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    vals = list(map(float, parts[1:9]))
    pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
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
    abs_deg = abs(best_ang)
    aspect = max(bw, bh) / max(1e-6, min(bw, bh))
    return {
        "pts": pts,
        "xs": xs,
        "ys": ys,
        "horiz": abs_deg < 35 and aspect > 1.8,
        "vert": abs_deg > 55 and aspect > 1.8,
    }


def mine_donors(source: Path, rng: random.Random, limit: int = 40) -> list[Donor]:
    """Find images with horizontal piles / mixed stacks; crop the hard region."""
    label_files: list[Path] = []
    for split in ("train", "valid", "test"):
        d = source / split / "labels"
        if d.is_dir():
            label_files.extend(d.glob("*.txt"))
    rng.shuffle(label_files)

    donors: list[Donor] = []
    seen_bases: set[str] = set()
    for lp in label_files:
        if len(donors) >= limit:
            break
        split = lp.parent.parent.name
        img_path = next((source / split / "images").glob(lp.stem + ".*"), None)
        if img_path is None:
            continue
        base = img_path.name.split(".rf.")[0]
        if base in seen_bases:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for ln in lp.read_text().splitlines():
            if not ln.strip():
                continue
            b = parse_obb(ln, w, h)
            if b:
                boxes.append(b)
        if not boxes:
            continue
        n_h = sum(1 for b in boxes if b["horiz"])
        n_v = sum(1 for b in boxes if b["vert"])
        mixed = n_h >= 2 and n_v >= 3
        pile = n_h >= 6
        if not (mixed or pile):
            continue

        horiz = [b for b in boxes if b["horiz"]]
        if len(horiz) >= 3:
            hx0 = min(min(b["xs"]) for b in horiz)
            hx1 = max(max(b["xs"]) for b in horiz)
            hy0 = min(min(b["ys"]) for b in horiz)
            hy1 = max(max(b["ys"]) for b in horiz)
            pad = 0.03 * max(w, h)
            hx0 -= pad
            hy0 -= pad
            hx1 += pad
            hy1 += pad
            target = []
            for b in boxes:
                bx0, bx1 = min(b["xs"]), max(b["xs"])
                by0, by1 = min(b["ys"]), max(b["ys"])
                if bx1 < hx0 or bx0 > hx1 or by1 < hy0 or by0 > hy1:
                    continue
                target.append(b)
        else:
            target = boxes

        if len(target) < 4:
            continue
        x0 = max(0, int(min(min(b["xs"]) for b in target) - 2))
        y0 = max(0, int(min(min(b["ys"]) for b in target) - 2))
        x1 = min(w, int(max(max(b["xs"]) for b in target) + 2))
        y1 = min(h, int(max(max(b["ys"]) for b in target) + 2))
        rw, rh = x1 - x0, y1 - y0
        if rw < 60 or rh < 40:
            continue
        # Prefer wider-than-tall for shelf paste (horizontal piles)
        crop = img[y0:y1, x0:x1].copy()
        labels = [[(p[0] - x0, p[1] - y0) for p in b["pts"]] for b in target]
        seen_bases.add(base)
        donors.append(
            Donor(
                rgb=crop,
                labels=labels,
                n_h=sum(1 for b in target if b["horiz"]),
                n_v=sum(1 for b in target if b["vert"]),
                stem=f"{base}_h{n_h}_v{n_v}",
            )
        )
    return donors


def _sample_strip_color(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Median BGR of a strip; fall back to nearby pixels if strip is tiny."""
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([180, 180, 180], dtype=np.float32)
    return np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)


def clear_bay(img: np.ndarray, bay: Bay) -> np.ndarray:
    """Paint an empty shelf bay from border wood colors (no TELEA smear).

    Samples the thin rim left around the bay (shelf boards / side walls) and
    fills the interior with a vertical gradient: darker under the shelf above,
    lighter near the floor board. Keeps the outer 2–3 px of the bay so the
    physical shelf edges stay sharp.
    """
    h, w = img.shape[:2]
    x0, y0 = int(bay.x0 * w), int(bay.y0 * h)
    x1, y1 = int(bay.x1 * w), int(bay.y1 * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
        return img

    rim = max(3, int(0.012 * min(w, h)))
    # Sample shelf ceiling (top rim), floor (bottom rim), and side walls
    top_c = _sample_strip_color(img, x0, y0, x1, y0 + rim)
    bot_c = _sample_strip_color(img, x0, y1 - rim, x1, y1)
    left_c = _sample_strip_color(img, x0, y0, x0 + rim, y1)
    right_c = _sample_strip_color(img, x1 - rim, y0, x1, y1)
    back_c = 0.45 * top_c + 0.25 * bot_c + 0.15 * left_c + 0.15 * right_c

    out = img.copy()
    # Keep floor board + thin side walls; wipe almost to the upper shelf lip
    side = max(2, rim // 2)
    floor_keep = max(rim, int(0.05 * bh))
    ix0, iy0 = x0 + side, y0 + max(1, side // 2)
    ix1, iy1 = x1 - side, y1 - floor_keep
    if ix1 <= ix0 or iy1 <= iy0:
        return out

    fh, fw = iy1 - iy0, ix1 - ix0
    # Vertical gradient backboard → near-floor
    t = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None, None]
    # Darken slightly under the upper shelf (occlusion shadow)
    shade = 0.88 + 0.12 * t
    fill = (back_c * (1.0 - 0.35 * t) + bot_c * (0.35 * t)) * shade
    # Mild horizontal vignette toward side walls
    hx = np.linspace(0.0, 1.0, fw, dtype=np.float32)[None, :, None]
    side = 0.97 + 0.03 * np.minimum(hx, 1.0 - hx) * 2.0
    fill = fill * side
    # Tiny grain so it doesn't look flat plastic
    rng = np.random.default_rng((x0 * 131 + y0) & 0xFFFFFFFF)
    noise = rng.normal(0, 2.5, size=fill.shape).astype(np.float32)
    fill = np.clip(fill + noise, 0, 255).astype(np.uint8)
    out[iy0:iy1, ix0:ix1] = fill
    return out


def _match_lighting(donor_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Rough brightness match so paste isn't a gray cutout on a white shelf."""
    dm = float(np.mean(cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2GRAY))) + 1e-3
    rm = float(np.mean(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))) + 1e-3
    scale = float(np.clip(rm / dm, 0.65, 1.45))
    return np.clip(donor_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _paste_at(
    host: np.ndarray,
    bay: Bay,
    donor: Donor,
    left: int,
) -> tuple[np.ndarray, list[list[tuple[float, float]]], int]:
    """Scale donor to usable bay height; paste at `left`. Returns new left cursor.

    Always fills shelf height. If the scaled donor is wider than remaining bay
    width, crop from the left of the donor (keep height) — never squash down.
    """
    h, w = host.shape[:2]
    x0, y0 = int(bay.x0 * w), int(bay.y0 * h)
    x1, y1 = int(bay.x1 * w), int(bay.y1 * h)
    bay_h = max(1, y1 - y0)

    rim = max(3, int(0.012 * min(w, h)))
    side = max(2, rim // 2)
    floor_keep = max(rim, int(0.05 * bay_h))
    ix0, iy0 = x0 + side, y0 + max(1, side // 2)
    ix1, iy1 = x1 - side, y1 - floor_keep
    usable_h = max(1, iy1 - iy0)

    dh, dw = donor.rgb.shape[:2]
    scale = usable_h / dh
    nw, nh = max(8, int(dw * scale)), max(8, int(dh * scale))

    px = left
    remaining = ix1 - px
    if remaining < 16:
        return host, [], px

    # Full-height resize, then crop width if needed (never reduce height)
    resized = cv2.resize(donor.rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    label_scale_x = scale
    if nw > remaining:
        # Keep left portion of the hard region so spines stay shelf-height
        crop_w = remaining
        # Map crop back to donor label space fraction
        label_scale_x = scale
        resized = resized[:, :crop_w]
        nw = crop_w

    ref = host[iy0:iy1, max(ix0, px - 4) : min(ix1, px + 8)]
    if ref.size < 10:
        ref = host[iy0:iy1, ix0:ix1]
    if ref.size >= 10:
        resized = _match_lighting(resized, ref)

    py = iy1 - nh
    if py < iy0:
        py = iy0

    out = host.copy()
    sh = min(8, max(2, nh // 25))
    sy0 = max(iy0, py + nh - 1)
    sy1 = min(out.shape[0], sy0 + sh)
    if sy1 > sy0 and px + nw <= out.shape[1]:
        band = out[sy0:sy1, px : px + nw].astype(np.float32)
        for i in range(sy1 - sy0):
            band[i] *= 1.0 - 0.18 * (1.0 - i / max(1, sy1 - sy0))
        out[sy0:sy1, px : px + nw] = np.clip(band, 0, 255).astype(np.uint8)

    nh2 = min(nh, out.shape[0] - py)
    nw2 = min(nw, out.shape[1] - px)
    out[py : py + nh2, px : px + nw2] = resized[:nh2, :nw2]
    nh, nw = nh2, nw2

    # Labels: scale then drop any box mostly outside the cropped width
    quads = []
    for pts in donor.labels:
        q = [(p[0] * label_scale_x + px, p[1] * scale + py) for p in pts]
        xs = [p[0] for p in q]
        if max(xs) < px + 2 or min(xs) > px + nw - 2:
            continue
        # Clamp into paste rect
        q = [(min(px + nw - 1, max(px, x)), min(py + nh - 1, max(py, y))) for x, y in q]
        quads.append(q)
    return out, quads, px + nw


def fill_bay_with_donors(
    host: np.ndarray,
    bay: Bay,
    donors: list[Donor],
    prefer: str,
    rng: random.Random,
    used: set[str],
) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    """Tile hard regions left→right until the cleared bay is filled."""
    h, w = host.shape[:2]
    x0 = int(bay.x0 * w)
    x1 = int(bay.x1 * w)
    rim = max(3, int(0.012 * min(w, h)))
    side = max(2, rim // 2)
    cursor = x0 + side
    end = x1 - side
    all_quads: list[list[tuple[float, float]]] = []
    out = host
    guard = 0
    while cursor < end - 20 and guard < 10:
        guard += 1
        donor = pick_donor(donors, prefer if guard == 1 else "any", rng, used)
        out, quads, cursor2 = _paste_at(out, bay, donor, left=cursor)
        if cursor2 <= cursor:
            # Try another donor once; if still stuck, stop
            donor = pick_donor(donors, "any", rng, used)
            out, quads, cursor2 = _paste_at(out, bay, donor, left=cursor)
            if cursor2 <= cursor:
                break
        all_quads.extend(quads)
        cursor = cursor2 + max(1, int(0.003 * w))
    return out, all_quads


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


def draw_preview(img: np.ndarray, quads: list[list[tuple[float, float]]], path: Path) -> None:
    vis = img.copy()
    for q in quads:
        pts = np.array([(int(p[0]), int(p[1])) for p in q], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 220, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def build_hosts(scenes: Path, clutter_dir: Path) -> list[Host]:
    """Hand-tuned bays on cluttered hosts — clear NON-book clutter shelves only."""
    hosts: list[Host] = []

    office = scenes / "office1.jpeg"
    if office.exists():
        hosts.append(
            Host(
                path=office,
                note="office1: clear basket shelf + stethoscope/box shelf",
                bays=[
                    Bay(0.05, 0.66, 0.68, 0.80, prefer="wide"),  # basket + hole punch
                    Bay(0.05, 0.82, 0.55, 0.96, prefer="any"),  # box + cables / leaning
                ],
            )
        )

    bookcase = scenes / "bookcase.jpg"
    if bookcase.exists():
        hosts.append(
            Host(
                path=bookcase,
                note="bookcase: clear yoga shelf + bottom exercise clutter",
                bays=[
                    Bay(0.08, 0.64, 0.92, 0.78, prefer="wide"),  # yoga blocks / dumbbells
                    Bay(0.08, 0.79, 0.92, 0.95, prefer="wide"),  # bin / gym bag
                ],
            )
        )

    bedroom = scenes / "bedroom1.jpeg"
    if bedroom.exists():
        hosts.append(
            Host(
                path=bedroom,
                note="bedroom1: clear tablet/kindle end of shelf",
                bays=[Bay(0.78, 0.38, 0.96, 0.60, prefer="any")],
            )
        )

    # Frontal cluttered Commons — clear mid shelves that hold toys/boxes (not books-only)
    commons_bays = {
        "Liburu_apalak_01": [
            Bay(0.52, 0.48, 0.78, 0.58, prefer="wide"),  # brown box / CDs shelf
            Bay(0.28, 0.22, 0.48, 0.34, prefer="any"),  # toys / figurines shelf
        ],
        "Liburu_apalak_02": [
            Bay(0.52, 0.46, 0.78, 0.56, prefer="wide"),
            Bay(0.28, 0.20, 0.48, 0.32, prefer="any"),
        ],
        "biblioteca": [
            Bay(0.15, 0.42, 0.85, 0.52, prefer="wide"),
            Bay(0.15, 0.55, 0.85, 0.65, prefer="any"),
        ],
    }
    for p in sorted(clutter_dir.glob("work_clutter_*.jpg")):
        for key, bays in commons_bays.items():
            if key.lower() in p.name.lower():
                hosts.append(Host(path=p, note=f"commons:{p.stem}", bays=bays))
                break

    return hosts


def pick_donor(donors: list[Donor], prefer: str, rng: random.Random, used: set[str]) -> Donor:
    pool = donors
    if prefer == "wide":
        wide = [d for d in donors if d.n_h >= 5 and d.rgb.shape[1] >= d.rgb.shape[0] * 0.7]
        if wide:
            pool = wide
    unused = [d for d in pool if d.stem not in used] or pool
    d = rng.choice(unused)
    used.add(d.stem)
    return d


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    print("Mining hard donor regions from OBB labels ...")
    donors = mine_donors(args.source_obb, rng, limit=50)
    print(f"  {len(donors)} unique hard donors")
    if len(donors) < 5:
        print("Not enough hard donors.", file=sys.stderr)
        return 1
    for d in donors[:8]:
        print(f"    {d.stem}: {d.rgb.shape[1]}x{d.rgb.shape[0]} h={d.n_h} v={d.n_v} boxes={len(d.labels)}")

    scenes = Path("/Users/joebr/dev/book-train/scenes")
    clutter = Path.home() / "ml/book-spines/tmp/synth-cluttered-bgs"
    hosts = build_hosts(scenes, clutter)
    print(f"Hosts: {len(hosts)}")
    for h in hosts:
        print(f"  {h.path.name}: {len(h.bays)} bays — {h.note}")

    out = args.out
    img_dir = out / "images" / "train"
    lab_dir = out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "spines.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: spine\n"
    )

    # Save donor catalog for inspection
    donor_dir = out / "donors"
    donor_dir.mkdir(exist_ok=True)
    for i, d in enumerate(donors[:20]):
        cv2.imwrite(str(donor_dir / f"{i:02d}_{d.stem}.jpg"), d.rgb)

    n_out = 0
    previews = 0
    for host in hosts:
        if not host.path.exists():
            continue
        base = load_bgr(host.path, max_side=args.max_side)
        for i in range(args.per_host):
            canvas = base.copy()
            used: set[str] = set()
            # Clear selected bays first
            for bay in host.bays:
                canvas = clear_bay(canvas, bay)
            all_quads: list[list[tuple[float, float]]] = []
            # Fill cleared bays with tiled hard regions
            for bay in host.bays:
                canvas, quads = fill_bay_with_donors(canvas, bay, donors, bay.prefer, rng, used)
                all_quads.extend(quads)

            ch, cw = canvas.shape[:2]
            lines = [ln for q in all_quads if (ln := corners_to_yolo(q, cw, ch))]
            if len(lines) < 4:
                continue
            stem = f"hard_{host.path.stem}_{i:02d}"

            # Full-frame composite for visual review only (host books outside
            # cleared bays remain unlabeled — not for naive training).
            full_dir = out / "full_composites"
            full_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(full_dir / f"{stem}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if previews < args.preview:
                draw_preview(canvas, all_quads, out / "preview" / f"{stem}.jpg")
                previews += 1

            # Training chips: one crop per cleared bay (complete labels inside).
            # Re-fill per bay onto a cleared copy so each chip is self-contained.
            for bi, bay in enumerate(host.bays):
                chip_canvas = clear_bay(base.copy(), bay)
                chip_canvas, quads = fill_bay_with_donors(
                    chip_canvas, bay, donors, bay.prefer, rng, used
                )
                # Crop with small pad around bay
                hh, ww = chip_canvas.shape[:2]
                pad = 0.02
                cx0 = max(0, int((bay.x0 - pad) * ww))
                cy0 = max(0, int((bay.y0 - pad) * hh))
                cx1 = min(ww, int((bay.x1 + pad) * ww))
                cy1 = min(hh, int((bay.y1 + pad) * hh))
                chip = chip_canvas[cy0:cy1, cx0:cx1]
                chh, cww = chip.shape[:2]
                chip_lines = []
                for q in quads:
                    q2 = [(p[0] - cx0, p[1] - cy0) for p in q]
                    ln = corners_to_yolo(q2, cww, chh)
                    if ln:
                        chip_lines.append(ln)
                if len(chip_lines) < 3:
                    continue
                cstem = f"{stem}_bay{bi}"
                cv2.imwrite(str(img_dir / f"{cstem}.jpg"), chip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                (lab_dir / f"{cstem}.txt").write_text("\n".join(chip_lines) + "\n")
                n_out += 1
                draw_preview(
                    chip,
                    [[(p[0] - cx0, p[1] - cy0) for p in q] for q in quads],
                    out / "preview" / f"{cstem}.jpg",
                )
                print(f"  wrote {cstem} boxes={len(chip_lines)}")

            print(f"  full composite {stem} (review only)")

    # Move ~20% to val
    trains = sorted(img_dir.glob("*.jpg"))
    for p in trains[: max(1, len(trains) // 5)]:
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = out / kind / "train" / (p.stem + ext)
            dst = out / kind / "val" / (p.stem + ext)
            if src.exists():
                src.replace(dst)

    (out / "SOURCE.md").write_text(
        f"""# synth-hard-shelf

Hard-case synthesis: cluttered hosts → clear 1–2 shelf bays (inpaint) → paste
horizontal-pile / mixed-orientation *regions* mined from Open Shelves OBB labels.

| Field | Value |
|---|---|
| **id** | `synth-hard-shelf` |
| **script** | `tools/synth_hard_shelf_paste.py` |
| **spine_source** | `{args.source_obb}` |
| **images** | `{n_out}` |
| **note** | Train images are **bay chips** (cleared shelf + pasted hard region) so labels are complete. `full_composites/` are for visual review only — host books outside cleared bays are unlabeled there. |
"""
    )
    print(f"Done: {n_out} bay chips -> {out}")
    print("Train on images/ only (chips). full_composites/ is review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
