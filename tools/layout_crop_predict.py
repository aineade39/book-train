#!/usr/bin/env python3
"""Full-scene jigsaw crops guided by OBB shelves + orientation gaps.

Full-image detect → shelf bands + orientation-aware column blocks, each its
own crop by default ("crop each group of books on a shelf into its own
crop") → oversized groups (> ``--max-crop-dim-k`` × imgsz) are cut further
using pixel-texture-guided whitespace search rather than trusting OBB gaps
alone, so a seam prefers real low-texture whitespace over the OBB-only gap
midpoint (which can land on a book the detector missed) and a row can still
split even when the physical shelf-board line is slightly tilted in the
photo → quads that tile the image (including empty cells for missed
spines) → pad to rectangle (never upsize) → optional per-crop re-infer +
merge. A group that truly can't be split without cutting a book (at any
seam angle) is left as one larger crop -- the no-cut rule always wins over
granularity.

Plan is checked with hard rules in ``layout_crop_rules.py`` (no OBB cut, cover,
no overlap, finite quads).

Examples:
  .venv/bin/python tools/layout_crop_predict.py bookcase.jpg --overlay-plan
  .venv/bin/python tools/layout_crop_predict.py bookcase.jpg --write-crops /tmp/crops --infer-crops
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_crop_rules import hard_rules_ok, results_to_json, verify_plan  # noqa: E402
from paths import eval_dir, runs_dir  # noqa: E402
from tiled_predict_obb import (  # noqa: E402
    Det,
    ensure_ultralytics,
    latest_best_pt,
    nms_rotated,
    predict_crop,
    rotated_iou,
    run_single,
)


@dataclass
class CropPlan:
    shelf_id: int
    block_id: int
    angle_deg: float
    # Scene-space quad TL, TR, BR, BL (axis-aligned jigsaw cell)
    quad: list[tuple[float, float]]
    member_indices: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"shelf{self.shelf_id}_block{self.block_id}"

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """x0, y0, x1, y1."""
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return min(xs), min(ys), max(xs), max(ys)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="Shelf photo.")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--imgsz", type=int, default=1024, help="Model / pad canvas size.")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=500)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--angle-tol-deg",
        type=float,
        default=25.0,
        help="Max angle difference (deg) to stay in the same orientation block.",
    )
    p.add_argument(
        "--row-gap-k",
        type=float,
        default=0.2,
        help="Shelf bands merge when their y-extents are closer than this × median spine short-side.",
    )
    p.add_argument(
        "--col-gap-k",
        type=float,
        default=1.0,
        help="Column blocks split on an x-gap wider than this × median spine short-side "
        "(orientation change also splits, even with no gap).",
    )
    p.add_argument(
        "--min-block-members",
        type=int,
        default=2,
        help="Column blocks with fewer members are absorbed into the nearer neighbor block "
        "(also the minimum group size eligible for size-based re-splitting).",
    )
    p.add_argument(
        "--max-crop-dim-k",
        type=float,
        default=1.5,
        help="A shelf group (row band or column block) is re-split further when its height/width "
        "exceeds this × imgsz -- i.e. 'crop each shelf group on its own, further cropped if long'. "
        "Uses pixel-texture-guided whitespace search, not just OBB gaps.",
    )
    p.add_argument("--pad-value", type=int, default=114, help="Letterbox fill (Ultralytics gray).")
    p.add_argument("--plan-only", action="store_true", help="Plan + overlays only; no crop re-infer.")
    p.add_argument(
        "--infer-crops",
        action="store_true",
        help="Re-run detection on each padded crop and merge with first pass.",
    )
    p.add_argument("--write-crops", type=Path, default=None, help="Directory for padded crop PNGs.")
    p.add_argument(
        "--overlay-plan",
        action="store_true",
        help="Write scene overlay with crop-plan quads only (no OBBs, no crop files, no re-infer).",
    )
    p.add_argument(
        "--overlay-crops",
        action="store_true",
        help="Draw crop-boundary lines (quads) on the scene.",
    )
    p.add_argument(
        "--overlay-dets",
        action="store_true",
        help="Also draw OBB polygons (no labels) on the scene overlay.",
    )
    p.add_argument("--out", type=Path, default=None, help="Scene overlay image path.")
    p.add_argument("--json", type=Path, default=None, help="Summary JSON path.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Angle / geometry helpers
# ---------------------------------------------------------------------------


def norm_angle(a: float) -> float:
    a = a % math.pi
    if a < 0:
        a += math.pi
    return a


def angle_diff(a: float, b: float) -> float:
    d = abs(norm_angle(a) - norm_angle(b))
    return min(d, math.pi - d)


def circular_mean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(2 * norm_angle(a)) for a in angles)
    c = sum(math.cos(2 * norm_angle(a)) for a in angles)
    return norm_angle(0.5 * math.atan2(s, c))


def long_axis_angle(det: Det) -> float:
    """
    Absolute image-space direction (mod pi) of the OBB's *long* axis.

    Ultralytics OBB export doesn't guarantee ``w`` is the longer side: for
    this model, upright spines and lying-flat piles can both report
    ``angle`` near 0, with the true orientation encoded by which of w/h is
    numerically larger. Comparing raw ``det.angle`` therefore fails to tell
    upright and flat books apart -- this accounts for the w/h swap.
    """
    base = det.angle if det.w >= det.h else det.angle + math.pi / 2
    return norm_angle(base)


def _dist_from_vertical(theta: float) -> float:
    d = abs(norm_angle(theta) - math.pi / 2)
    return min(d, math.pi - d)


def all_edges(det: Det) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = det.corners()
    return [(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def side_edges(det: Det) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    The pair of OBB edges that face left/right -- the faces relevant for
    column seams. Edges come in two length classes: the "w-edges" run in
    direction ``det.angle``, the "h-edges" run in direction ``det.angle +
    90°``. Pick whichever class is closer to vertical in image space, which
    is robust regardless of which of w/h the model calls the long side.
    """
    pts = det.corners()
    h_edges = [(pts[0], pts[1]), (pts[2], pts[3])]
    w_edges = [(pts[1], pts[2]), (pts[3], pts[0])]
    if _dist_from_vertical(det.angle + math.pi / 2) <= _dist_from_vertical(det.angle):
        return h_edges
    return w_edges


def edge_mid_xy(edge: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    (x0, y0), (x1, y1) = edge
    return (x0 + x1) / 2, (y0 + y1) / 2


def det_y_span(det: Det) -> tuple[float, float]:
    ys = [c[1] for c in det.corners()]
    return min(ys), max(ys)


def det_x_span(det: Det) -> tuple[float, float]:
    xs = [c[0] for c in det.corners()]
    return min(xs), max(xs)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_from_points(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float, float]:
    a = p[1] - q[1]
    b = q[0] - p[0]
    c = -(a * p[0] + b * p[1])
    return a, b, c


def line_x_eq(x: float) -> tuple[float, float, float]:
    return 1.0, 0.0, -x


def line_y_eq(y: float) -> tuple[float, float, float]:
    return 0.0, 1.0, -y


def intersect_lines(
    l1: tuple[float, float, float], l2: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Intersect ax+by+c=0 lines. Returns None if parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    # a x + b y = -c
    x = (b2 * (-c1) - b1 * (-c2)) / det
    y = (a1 * (-c2) - a2 * (-c1)) / det
    return x, y


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares line ax+by+c=0 preferring near-horizontal (shelf) fits."""
    if len(points) < 2:
        y = points[0][1] if points else 0.0
        return line_y_eq(y)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    # y - m x - b = 0 → -m x + 1 y - b = 0
    return float(-m), 1.0, float(-b)


def side_edge_line(det: Det, prefer_min_x: bool) -> tuple[float, float, float]:
    """
    Most extreme side-face edge in x. This is normally just the OBB's true
    left/right face (``side_edges``, already orientation-correct); fall back
    to whichever of the 4 raw corner edges is most extreme in x if that
    candidate disagrees with the true extreme by more than a small margin
    (defensive only -- should rarely trigger).
    """
    all_e = all_edges(det)
    scored = sorted(((edge_mid_xy(e)[0], e) for e in all_e), key=lambda t: t[0])
    extreme = scored[0][1] if prefer_min_x else scored[-1][1]
    face_scored = sorted(((edge_mid_xy(e)[0], e) for e in side_edges(det)), key=lambda t: t[0])
    cand = face_scored[0][1] if prefer_min_x else face_scored[-1][1]
    short_dim = max(1.0, min(det.w, det.h))
    if abs(edge_mid_xy(cand)[0] - edge_mid_xy(extreme)[0]) <= 0.3 * short_dim:
        return line_from_points(*cand)
    return line_from_points(*extreme)


# ---------------------------------------------------------------------------
# Pixel-texture-guided whitespace search
#
# OBB-only gap-finding trusts the detector completely: a seam placed at the
# midpoint of a gap between two *detected* boxes can land squarely on a book
# the detector missed. Scanning the actual pixels for a low-texture strip
# (real whitespace/shelf-board is comparatively flat; ink and edges are not)
# lets a seam avoid likely-hidden content even when no OBB proves it's there.
# ---------------------------------------------------------------------------


def compute_edge_energy(image_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel |Laplacian| of grayscale image -- high where there's ink/edges."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(lap)


def _axis_profile(energy: np.ndarray, axis: str, lo: int, hi: int, fixed_lo: int, fixed_hi: int) -> np.ndarray:
    """Mean energy along ``axis`` ('x' or 'y') over [lo,hi), averaged across the
    perpendicular band [fixed_lo,fixed_hi). Out-of-image positions read as 0."""
    img_h, img_w = energy.shape
    length = max(0, hi - lo)
    out = np.zeros(length, dtype=np.float64)
    if axis == "x":
        f0, f1 = max(0, min(fixed_lo, img_h)), max(0, min(fixed_hi, img_h))
        p0, p1 = max(0, min(lo, img_w)), max(0, min(hi, img_w))
    else:
        f0, f1 = max(0, min(fixed_lo, img_w)), max(0, min(fixed_hi, img_w))
        p0, p1 = max(0, min(lo, img_h)), max(0, min(hi, img_h))
    if f1 <= f0 or p1 <= p0:
        return out
    patch = energy[f0:f1, p0:p1] if axis == "x" else energy[p0:p1, f0:f1]
    prof = patch.mean(axis=0) if axis == "x" else patch.mean(axis=1)
    off = p0 - lo
    out[off : off + prof.shape[0]] = prof
    return out


def find_pixel_seam(
    energy: np.ndarray,
    *,
    axis: str,
    fixed_lo: float,
    fixed_hi: float,
    search_lo: float,
    search_hi: float,
    forbidden: list[tuple[float, float]],
    smooth_px: int,
    margin_px: float,
) -> float | None:
    """
    Position (x if ``axis=='x'`` else y) of minimum texture energy within
    [search_lo,search_hi], averaged over the perpendicular
    [fixed_lo,fixed_hi) band, smoothed to resist single-pixel noise, and
    restricted to stay ``margin_px`` from the search-range ends and outside
    every ``forbidden`` sub-range (e.g. a detection's own span -- never pick
    a seam that grazes a known OBB). Returns ``None`` if every position is
    excluded.
    """
    lo_i, hi_i = int(math.floor(search_lo)), int(math.ceil(search_hi))
    if hi_i - lo_i < 3:
        return None
    prof = _axis_profile(energy, axis, lo_i, hi_i, int(round(fixed_lo)), int(round(fixed_hi)))
    k = max(1, int(smooth_px) | 1)
    if k > 1:
        kernel = np.ones(k, dtype=np.float64) / k
        prof = np.convolve(prof, kernel, mode="same")
    positions = np.arange(lo_i, hi_i, dtype=np.float64)
    allowed = (positions >= search_lo + margin_px) & (positions <= search_hi - margin_px)
    for f_lo, f_hi in forbidden:
        allowed &= ~((positions >= f_lo) & (positions <= f_hi))
    if not allowed.any():
        return None
    idx = np.where(allowed)[0]
    best = idx[int(np.argmin(prof[idx]))]
    return float(positions[best])


# ---------------------------------------------------------------------------
# Shelf / block clustering
# ---------------------------------------------------------------------------


def median_short_side(dets: list[Det]) -> float:
    """
    Median true short-side (spine thickness) length, i.e. ``min(w, h)`` per
    detection. Cannot just use ``d.h``: this model doesn't guarantee h is the
    short side (upright spines can report h as the *long* dimension), so
    blending h across mixed orientations would badly overestimate the scale.
    """
    return float(np.median([min(d.w, d.h) for d in dets])) if dets else 10.0


def band_extent(dets: list[Det], indices: list[int]) -> tuple[float, float]:
    """Strict (min, max) y-extent of a band's members."""
    tops = [det_y_span(dets[i])[0] for i in indices]
    bots = [det_y_span(dets[i])[1] for i in indices]
    return float(min(tops)), float(max(bots))


def build_shelf_bands(dets: list[Det], row_gap_k: float) -> list[list[int]]:
    """
    Row (shelf) bands via interval-union on OBB y-extents.

    Sweeping det extents sorted by y_top and unioning any that overlap or are
    closer than ``min_row_gap`` is correct by construction: a book can never
    straddle two bands, and two bands can never be separated when a book (or
    an outlier detection) bridges the gap between them. This replaces
    centroid clustering plus a later "merge if extents overlap" repair pass
    with a single pass.
    """
    if not dets:
        return []
    min_row_gap = max(6.0, row_gap_k * median_short_side(dets))

    order = sorted(range(len(dets)), key=lambda i: det_y_span(dets[i])[0])
    bands: list[list[int]] = [[order[0]]]
    cur_bot = det_y_span(dets[order[0]])[1]
    for idx in order[1:]:
        top, bot = det_y_span(dets[idx])
        if top <= cur_bot + min_row_gap:
            bands[-1].append(idx)
            cur_bot = max(cur_bot, bot)
        else:
            bands.append([idx])
            cur_bot = bot

    # Absorb tiny bands (likely single stray detections) into whichever
    # immediate neighbor band is closer, so noise doesn't spawn its own shelf.
    min_members = 2
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, band in enumerate(bands):
            if len(band) >= min_members:
                continue
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(bands)]
            if not neighbors:
                continue
            cy = float(np.mean([dets[j].cy for j in band]))
            best_j = min(
                neighbors,
                key=lambda j: abs(float(np.mean([dets[k].cy for k in bands[j]])) - cy),
            )
            bands[best_j] = sorted(bands[best_j] + band, key=lambda k: dets[k].cy)
            del bands[i]
            changed = True
            break

    for band in bands:
        band.sort(key=lambda i: dets[i].cy)
    return bands


def build_column_blocks(
    dets: list[Det],
    members: list[int],
    *,
    angle_tol_deg: float,
    col_gap_px: float,
    min_block_members: int,
) -> list[list[int]]:
    """
    Column blocks within one shelf band via a 1D chain-merge on members
    sorted by cx: consecutive detections stay in the same block only if both
    their x-extents are within ``col_gap_px`` of touching AND their
    orientation (long-axis angle) is within ``angle_tol_deg``. Either a real
    whitespace gap or an orientation change breaks the chain -- this is what
    lets touching piles of different orientation (e.g. upright spines flush
    against a horizontal stack) still get split, which pure x-gap grouping
    misses.
    """
    if not members:
        return []
    order = sorted(members, key=lambda i: dets[i].cx)
    blocks: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        prev_idx = blocks[-1][-1]
        prev_x1 = det_x_span(dets[prev_idx])[1]
        cur_x0 = det_x_span(dets[idx])[0]
        gap = cur_x0 - prev_x1
        same_orientation = (
            angle_diff(long_axis_angle(dets[prev_idx]), long_axis_angle(dets[idx]))
            <= math.radians(angle_tol_deg)
        )
        if gap <= col_gap_px and same_orientation:
            blocks[-1].append(idx)
        else:
            blocks.append([idx])

    # Absorb tiny blocks (likely a stray detection) into the nearer neighbor,
    # preferring a same-orientation neighbor when both are candidates.
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i, block in enumerate(blocks):
            if len(block) >= min_block_members:
                continue
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
            if not neighbors:
                continue
            block_angle = circular_mean([long_axis_angle(dets[k]) for k in block])
            block_cx = float(np.mean([dets[k].cx for k in block]))

            def neighbor_key(j: int) -> tuple[int, float]:
                n_angle = circular_mean([long_axis_angle(dets[k]) for k in blocks[j]])
                same = 0 if angle_diff(block_angle, n_angle) <= math.radians(angle_tol_deg) else 1
                n_cx = float(np.mean([dets[k].cx for k in blocks[j]]))
                return same, abs(n_cx - block_cx)

            best_j = min(neighbors, key=neighbor_key)
            blocks[best_j] = sorted(blocks[best_j] + block, key=lambda k: dets[k].cx)
            del blocks[i]
            changed = True
            break

    return blocks


def _rough_tilt_partition(
    dets: list[Det],
    idxs: list[int],
    energy: np.ndarray,
    *,
    lo: float,
    hi: float,
    x_lo: float,
    x_hi: float,
    n_samples: int = 24,
) -> tuple[list[int], list[int]] | None:
    """
    Rough guess at a tilted row boundary: sample local whitespace minima in
    narrow x-slices across the group's width (each slice sees only its own
    local content, so a shelf divider or an unrelated far-away column can't
    drown out the real local gap the way averaging over the *entire* width
    would), fit a line through those (x, y) points, and bipartition members
    by which side of that line their center falls on. This is only a seed
    for ``_sat_separating_line`` to refine into an exact separator -- it
    doesn't need to be perfect, just roughly right.
    """
    width = x_hi - x_lo
    if width <= 1.0 or len(idxs) < 4:
        return None
    step = width / n_samples
    slice_w = max(step, 40.0)
    margin = max(4.0, 0.08 * (hi - lo))
    mss = median_short_side([dets[i] for i in idxs])
    smooth_px = max(5, int(round(0.5 * mss)))
    points: list[tuple[float, float]] = []
    for k in range(n_samples):
        cx0 = x_lo + k * step
        cx1 = cx0 + slice_w
        y_local = find_pixel_seam(
            energy,
            axis="y",
            fixed_lo=cx0,
            fixed_hi=cx1,
            search_lo=lo,
            search_hi=hi,
            forbidden=[],
            smooth_px=smooth_px,
            margin_px=margin,
        )
        if y_local is not None:
            points.append(((cx0 + cx1) / 2.0, y_local))
    if len(points) < 2:
        return None
    line = fit_line(points)
    top = [i for i in idxs if _signed(line, (dets[i].cx, dets[i].cy)) < 0]
    bot = [i for i in idxs if _signed(line, (dets[i].cx, dets[i].cy)) >= 0]
    if not top or not bot:
        return None
    return top, bot


def _tilted_bipartition_seam(
    dets: list[Det],
    idxs: list[int],
    energy: np.ndarray,
    *,
    lo: float,
    hi: float,
    p_lo: float,
    p_hi: float,
    img_w: int,
    img_h: int,
) -> tuple[tuple[float, float, float], list[int], list[int]] | None:
    """
    Fallback for a row group whose OBB y-extents have genuinely zero gap
    anywhere (common with a photo taken at a slight tilt: the real
    shelf-board line isn't exactly ``y=const`` in image space, so *no*
    horizontal line can separate the two shelves without cutting some book
    at some x). Gets a rough bipartition from ``_rough_tilt_partition``,
    then asks the separating-axis theorem for an exact line -- allowed to be
    tilted -- that cleanly separates the two resulting groups. Returns
    ``None`` if the rough partition can't be found or the two groups aren't
    linearly separable at any angle (genuinely interleaved).
    """
    rough = _rough_tilt_partition(dets, idxs, energy, lo=lo, hi=hi, x_lo=p_lo, x_hi=p_hi)
    if rough is None:
        return None
    top, bot = rough
    corners_top = _all_corners(dets, top)
    corners_bot = _all_corners(dets, bot)
    line = _sat_separating_line(
        corners_top,
        corners_bot,
        top_l=line_x_eq(0.0),
        bot_l=line_x_eq(float(img_w)),
        img_w=img_w,
        img_h=img_h,
        prefer="horizontal",
    )
    if line is None or not _fully_separates(line, corners_top, corners_bot):
        return None
    return line, top, bot


def split_oversized_group(
    dets: list[Det],
    group: list[int],
    energy: np.ndarray,
    *,
    axis: str,
    max_dim: float,
    min_split_members: int,
    img_w: int,
    img_h: int,
) -> tuple[list[list[int]], list[tuple[float, float, float]]]:
    """
    Recursively split ``group`` (a row band if ``axis=='y'``, a column block
    if ``axis=='x'``) whenever its extent along ``axis`` exceeds ``max_dim``,
    implementing "crop each shelf group on its own, then further crop it if
    it's long": the group is the crop unit by default; size is what forces
    a further cut, not orientation or gaps.

    The split point normally comes from ``find_pixel_seam`` over the
    group's own extent, with every member's own span (plus a small pad)
    marked forbidden -- so even a size-forced cut can never land on a
    detected OBB. For ``axis=='y'`` only, if that axis-aligned search finds
    no safe gap at all (the whole point of the row-band merge in the first
    place), falls back to ``_tilted_bipartition_seam`` to look for a tilted
    separator before giving up. Returns ``(groups, seams)`` where
    ``seams[k]`` is the exact boundary line between ``groups[k]`` and
    ``groups[k+1]`` -- callers must use these lines directly rather than
    re-deriving a boundary from the groups' plain extents, since a tilted
    seam's two sides can have overlapping extents along ``axis``.
    """

    def own_span(i: int) -> tuple[float, float]:
        return det_y_span(dets[i]) if axis == "y" else det_x_span(dets[i])

    def extent(idxs: list[int]) -> tuple[float, float]:
        spans = [own_span(i) for i in idxs]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def perp_extent(idxs: list[int]) -> tuple[float, float]:
        spans = [det_x_span(dets[i]) if axis == "y" else det_y_span(dets[i]) for i in idxs]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def rec(idxs: list[int]) -> tuple[list[list[int]], list[tuple[float, float, float]]]:
        if len(idxs) < 2 * min_split_members:
            return [idxs], []
        lo, hi = extent(idxs)
        if hi - lo <= max_dim:
            return [idxs], []
        p_lo, p_hi = perp_extent(idxs)
        mss = median_short_side([dets[i] for i in idxs])
        pad = 0.15 * mss
        forbidden = [(own_span(i)[0] - pad, own_span(i)[1] + pad) for i in idxs]
        margin = max(4.0, 0.5 * mss)
        split = find_pixel_seam(
            energy,
            axis=axis,
            fixed_lo=p_lo,
            fixed_hi=p_hi,
            search_lo=lo,
            search_hi=hi,
            forbidden=forbidden,
            smooth_px=max(3, int(round(0.15 * mss))),
            margin_px=margin,
        )
        a: list[int] | None
        b: list[int] | None
        line: tuple[float, float, float] | None
        if split is not None:
            a = [i for i in idxs if own_span(i)[1] <= split]
            b = [i for i in idxs if own_span(i)[0] >= split]
            line = line_y_eq(split) if axis == "y" else line_x_eq(split)
            if not a or not b or len(a) + len(b) != len(idxs):
                a, b, line = None, None, None
            elif not _fully_separates(line, _all_corners(dets, a), _all_corners(dets, b)):
                a, b, line = None, None, None
        else:
            a, b, line = None, None, None
        if line is None and axis == "y":
            tilted = _tilted_bipartition_seam(
                dets, idxs, energy, lo=lo, hi=hi, p_lo=p_lo, p_hi=p_hi, img_w=img_w, img_h=img_h
            )
            if tilted is not None:
                line, a, b = tilted
        if line is None or a is None or b is None:
            return [idxs], []
        key = (lambda i: dets[i].cy) if axis == "y" else (lambda i: dets[i].cx)
        a_groups, a_seams = rec(sorted(a, key=key))
        b_groups, b_seams = rec(sorted(b, key=key))
        return a_groups + b_groups, a_seams + [line] + b_seams

    return rec(list(group))


def split_oversized_group_list(
    dets: list[Det],
    groups: list[list[int]],
    energy: np.ndarray,
    *,
    axis: str,
    max_dim: float,
    min_split_members: int,
    img_w: int,
    img_h: int,
) -> list[list[int]]:
    """Apply ``split_oversized_group`` to every group, preserving order along
    ``axis`` (discards the internal seam lines -- for use where a later step
    re-derives exact boundaries anyway, e.g. column-block seam resolution)."""
    key = (lambda i: dets[i].cy) if axis == "y" else (lambda i: dets[i].cx)
    out: list[list[int]] = []
    for group in groups:
        sub_groups, _ = split_oversized_group(
            dets,
            group,
            energy,
            axis=axis,
            max_dim=max_dim,
            min_split_members=min_split_members,
            img_w=img_w,
            img_h=img_h,
        )
        out.extend(sub_groups)
    out.sort(key=lambda g: float(np.mean([key(i) for i in g])))
    return out


# ---------------------------------------------------------------------------
# Full-scene jigsaw with OBB-edge seams (not AABB cuts through books)
# ---------------------------------------------------------------------------


def y_on_line(line: tuple[float, float, float], x: float) -> float:
    a, b, c = line
    if abs(b) < 1e-9:
        return 0.0
    return -(a * x + c) / b


def _signed(line: tuple[float, float, float], pt: tuple[float, float]) -> float:
    a, b, c = line
    return a * pt[0] + b * pt[1] + c


def _all_corners(dets: list[Det], indices: list[int]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in indices:
        pts.extend(dets[i].corners())
    return pts


def _fully_separates(
    line: tuple[float, float, float],
    corners_a: list[tuple[float, float]],
    corners_b: list[tuple[float, float]],
    eps: float = 1e-3,
) -> bool:
    """True iff every point in ``corners_a`` is strictly on one side of
    ``line`` and every point in ``corners_b`` is strictly on the other --
    i.e. the line is a valid cut that doesn't clip any OBB in either set."""
    sa = [_signed(line, p) for p in corners_a]
    sb = [_signed(line, p) for p in corners_b]
    if not sa or not sb:
        return False
    return (max(sa) < -eps and min(sb) > eps) or (min(sa) > eps and max(sb) < -eps)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone-chain convex hull, CCW, no repeated last point."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _seam_usable(
    line: tuple[float, float, float],
    top_l: tuple[float, float, float],
    bot_l: tuple[float, float, float],
    img_w: int,
    img_h: int,
) -> bool:
    """
    A seam line must cross both the row band's top and bottom boundary lines
    at finite, in-image points. A line nearly parallel to those (i.e. close
    to horizontal, since band boundaries are ``y = const``) can be a
    mathematically valid OBB separator yet useless as a column boundary --
    its intersection with the band lines shoots off far outside the image.
    """
    top_pt = intersect_lines(line, top_l)
    bot_pt = intersect_lines(line, bot_l)
    if top_pt is None or bot_pt is None:
        return False
    for x, y in (top_pt, bot_pt):
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if x < -2 or y < -2 or x > img_w + 2 or y > img_h + 2:
            return False
    return True


def _sat_separating_line(
    corners_a: list[tuple[float, float]],
    corners_b: list[tuple[float, float]],
    *,
    top_l: tuple[float, float, float],
    bot_l: tuple[float, float, float],
    img_w: int,
    img_h: int,
    prefer: str = "vertical",
) -> tuple[float, float, float] | None:
    """
    Exact linear-separability test via the separating axis theorem: two
    convex polygons are disjoint iff some hull-edge normal is a separating
    axis. Tries every edge normal of both hulls and, among the axes that do
    separate the two point sets *and* yield a seam usable given the
    ``top_l``/``bot_l`` boundary lines (``_seam_usable``), returns the line
    for whichever best matches ``prefer`` -- ``"vertical"`` for the usual
    column-split shape, ``"horizontal"`` for a row split whose true
    boundary is tilted (e.g. a shelf board that isn't exactly ``y=const``
    in image space due to camera perspective). Returns ``None`` iff no
    usable separating axis exists -- either the hulls actually overlap, or
    every valid separator is too close to parallel with the requested shape.
    """
    hull_a = _convex_hull(corners_a)
    hull_b = _convex_hull(corners_b)
    if len(hull_a) < 2 or len(hull_b) < 2:
        return None

    axes: list[tuple[float, float]] = []
    for hull in (hull_a, hull_b):
        n = len(hull)
        for i in range(n):
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            length = math.hypot(ex, ey)
            if length > 1e-9:
                axes.append((-ey / length, ex / length))

    best_line: tuple[float, float, float] | None = None
    best_score = math.inf
    for nx, ny in axes:
        proj_a = [nx * x + ny * y for x, y in corners_a]
        proj_b = [nx * x + ny * y for x, y in corners_b]
        max_a, min_a = max(proj_a), min(proj_a)
        max_b, min_b = max(proj_b), min(proj_b)
        if max_a < min_b:
            t = 0.5 * (max_a + min_b)
        elif max_b < min_a:
            t = 0.5 * (max_b + min_a)
        else:
            continue
        line = (nx, ny, -t)
        if not _seam_usable(line, top_l, bot_l, img_w, img_h):
            continue
        # vertical-ish seam: axis (normal) is near-horizontal, i.e. ny -> 0.
        # horizontal-ish seam: axis is near-vertical, i.e. nx -> 0.
        score = abs(ny) if prefer == "vertical" else abs(nx)
        if score < best_score:
            best_score, best_line = score, line
    return best_line


def find_separating_line(
    dets: list[Det],
    left_block: list[int],
    right_block: list[int],
    *,
    col_gap_px: float,
    top_l: tuple[float, float, float],
    bot_l: tuple[float, float, float],
    img_w: int,
    img_h: int,
    energy: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """
    Find a straight seam with every OBB corner of ``left_block`` strictly on
    one side and every corner of ``right_block`` on the other, so the crop
    boundary can never cut through a book on either side. The seam must also
    be ``_seam_usable`` within this row band (not near-parallel to the band's
    top/bottom boundaries), or it can't bound a finite quad.

    Tries, in priority order: the facing side-face of the two nearest-in-x
    end books (physically meaningful -- usually a real gap or a touching book
    face), the pixel-texture minimum within the gap (avoids landing on a book
    the detector missed even though the OBB-only gap looks clean), the gap
    midpoint as a fallback if pixel search is unavailable/inconclusive, then
    an exact separating-axis search (``_sat_separating_line``) over every
    member's OBB, which finds a valid separator whenever the two blocks'
    convex hulls don't actually overlap. Returns ``None`` if nothing usable
    separates every corner -- i.e. the two blocks are geometrically
    interleaved (e.g. a wide flat-lying pile behind an upright/leaning stack)
    and cannot be split by any straight line without cutting a book; the
    caller should merge them instead.
    """
    left_end = dets[max(left_block, key=lambda i: dets[i].cx)]
    right_end = dets[min(right_block, key=lambda i: dets[i].cx)]
    gap_lo = det_x_span(left_end)[1]
    gap_hi = det_x_span(right_end)[0]

    left_corners = _all_corners(dets, left_block)
    right_corners = _all_corners(dets, right_block)

    candidates: list[tuple[float, float, float]] = []
    # Any real (even small) whitespace gap: prefer the pixel-texture minimum
    # inside it over blindly trusting the OBB-only gap midpoint or a
    # touching-face candidate, since a gap this size is exactly what an
    # undetected book of typical width would hide inside.
    if gap_hi > gap_lo:
        if energy is not None:
            y_lo = min(y_on_line(top_l, gap_lo), y_on_line(top_l, gap_hi))
            y_hi = max(y_on_line(bot_l, gap_lo), y_on_line(bot_l, gap_hi))
            px_x = find_pixel_seam(
                energy,
                axis="x",
                fixed_lo=y_lo,
                fixed_hi=y_hi,
                search_lo=gap_lo,
                search_hi=gap_hi,
                forbidden=[],
                smooth_px=max(3, int(round(0.2 * max(col_gap_px, gap_hi - gap_lo)))),
                margin_px=max(2.0, 0.15 * (gap_hi - gap_lo)),
            )
            if px_x is not None:
                candidates.append(line_x_eq(px_x))
        candidates.append(line_x_eq(0.5 * (gap_lo + gap_hi)))
    candidates.append(side_edge_line(left_end, prefer_min_x=False))
    candidates.append(side_edge_line(right_end, prefer_min_x=True))

    for cand in candidates:
        if _seam_usable(cand, top_l, bot_l, img_w, img_h) and _fully_separates(
            cand, left_corners, right_corners
        ):
            return cand

    sat_line = _sat_separating_line(
        left_corners, right_corners, top_l=top_l, bot_l=bot_l, img_w=img_w, img_h=img_h
    )
    if sat_line is not None and _fully_separates(sat_line, left_corners, right_corners):
        return sat_line
    return None


def resolve_column_seams(
    dets: list[Det],
    blocks: list[list[int]],
    *,
    col_gap_px: float,
    top_l: tuple[float, float, float],
    bot_l: tuple[float, float, float],
    img_w: int,
    img_h: int,
    energy: np.ndarray | None = None,
) -> tuple[list[list[int]], list[tuple[float, float, float]]]:
    """
    Build vertical seams between consecutive column blocks, merging any
    adjacent pair for which ``find_separating_line`` fails (i.e. their OBBs
    are geometrically interleaved, like a wide flat-lying pile behind an
    upright/leaning stack, or no usable seam fits within this row band) so
    the result never cuts a book -- at the cost of fewer, larger blocks in
    those regions.
    """

    def sep(a: list[int], b: list[int]) -> tuple[float, float, float] | None:
        return find_separating_line(
            dets,
            a,
            b,
            col_gap_px=col_gap_px,
            top_l=top_l,
            bot_l=bot_l,
            img_w=img_w,
            img_h=img_h,
            energy=energy,
        )

    blocks = [list(b) for b in blocks]
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i in range(len(blocks) - 1):
            if sep(blocks[i], blocks[i + 1]) is None:
                blocks[i] = sorted(blocks[i] + blocks[i + 1], key=lambda k: dets[k].cx)
                del blocks[i + 1]
                changed = True
                break

    v_lines: list[tuple[float, float, float]] = [line_x_eq(0.0)]
    for bi in range(len(blocks) - 1):
        line = sep(blocks[bi], blocks[bi + 1])
        assert line is not None  # guaranteed separable by the merge loop above
        v_lines.append(line)
    v_lines.append(line_x_eq(float(img_w)))
    return blocks, v_lines


def quad_from_lines(
    top: tuple[float, float, float],
    bot: tuple[float, float, float],
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    img_w: int,
    img_h: int,
) -> list[tuple[float, float]] | None:
    tl = intersect_lines(top, left)
    tr = intersect_lines(top, right)
    br = intersect_lines(bot, right)
    bl = intersect_lines(bot, left)
    if None in (tl, tr, br, bl):
        return None
    assert tl and tr and br and bl
    pts = [tl, tr, br, bl]
    # Reject exploded geometry
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if x < -2 or y < -2 or x > img_w + 2 or y > img_h + 2:
            return None
    # Clamp tiny numeric drift onto image
    clamped = [
        (float(np.clip(x, 0, img_w)), float(np.clip(y, 0, img_h))) for x, y in pts
    ]
    # Degenerate?
    if dist(clamped[0], clamped[1]) < 2 or dist(clamped[0], clamped[3]) < 2:
        return None
    return clamped


def plan_crops(
    dets: list[Det],
    img_w: int,
    img_h: int,
    image_bgr: np.ndarray,
    *,
    angle_tol_deg: float,
    row_gap_k: float,
    col_gap_k: float,
    min_block_members: int,
    imgsz: int,
    max_crop_dim_k: float,
) -> list[CropPlan]:
    """
    Partition the image into quads bounded by shelf-band whitespace cuts and
    orientation-aware column seams. Row bands are built by ``build_shelf_bands``
    (interval-union, correct by construction — no repair pass needed). Column
    blocks are built by ``build_column_blocks``, which partitions each band's
    members directly, so block membership and quad ownership agree exactly.

    Each row band and each column block is, by default, its own crop unit
    ("crop each group of books on a shelf into its own crop"); a group is
    only cut further when its own extent exceeds ``max_crop_dim_k * imgsz``
    ("further cropped if it is long"), via ``split_oversized_group`` using
    pixel-texture-guided whitespace, not OBB gaps alone -- this is also what
    prevents a handful of mis-detected (too-tall) OBBs from silently fusing
    several real shelves into one oversized row band.
    """
    energy = compute_edge_energy(image_bgr)
    max_dim = max_crop_dim_k * imgsz

    original_bands = build_shelf_bands(dets, row_gap_k)
    col_gap_px = max(8.0, col_gap_k * median_short_side(dets))

    # Row bands are split (if oversized) with their exact internal seam
    # lines kept alongside the sub-bands, since a size-forced row split can
    # be a tilted line (``_tilted_bipartition_seam``) whose two sides have
    # overlapping y-extents -- re-deriving a boundary from plain min/max
    # extents (as for the simple gaps between *different* shelves below)
    # would be wrong in that case.
    row_spans: list[tuple[tuple[float, float, float], tuple[float, float, float], list[int]]] = []
    if not original_bands:
        row_spans.append((line_y_eq(0.0), line_y_eq(float(img_h)), []))
    else:
        extents = [band_extent(dets, b) for b in original_bands]
        expansions = [
            split_oversized_group(
                dets,
                band,
                energy,
                axis="y",
                max_dim=max_dim,
                min_split_members=min_block_members,
                img_w=img_w,
                img_h=img_h,
            )
            for band in original_bands
        ]
        prev = line_y_eq(0.0)
        first_top = extents[0][0]
        if first_top > 2.0:
            cut = line_y_eq(first_top)
            row_spans.append((prev, cut, []))
            prev = cut
        for i, band in enumerate(original_bands):
            sub_bands, internal_seams = expansions[i]
            if i + 1 < len(original_bands):
                mid = 0.5 * (extents[i][1] + extents[i + 1][0])
                end_cut = line_y_eq(float(np.clip(mid, 1.0, img_h - 1.0)))
            else:
                cut_y = extents[i][1] + 1.0
                end_cut = line_y_eq(float(np.clip(cut_y, 1.0, float(img_h))))
            boundaries = [prev] + internal_seams + [end_cut]
            for j, sub in enumerate(sub_bands):
                row_spans.append((boundaries[j], boundaries[j + 1], list(sub)))
            prev = end_cut
        if y_on_line(prev, img_w / 2) < img_h - 0.5:
            row_spans.append((prev, line_y_eq(float(img_h)), []))

    plans: list[CropPlan] = []
    for shelf_id, (top_l, bot_l, members) in enumerate(row_spans):
        if not members:
            quad = quad_from_lines(top_l, bot_l, line_x_eq(0.0), line_x_eq(float(img_w)), img_w, img_h)
            if quad:
                plans.append(
                    CropPlan(shelf_id=shelf_id, block_id=0, angle_deg=0.0, quad=quad, member_indices=[])
                )
            continue

        blocks = build_column_blocks(
            dets,
            members,
            angle_tol_deg=angle_tol_deg,
            col_gap_px=col_gap_px,
            min_block_members=min_block_members,
        )
        blocks = split_oversized_group_list(
            dets,
            blocks,
            energy,
            axis="x",
            max_dim=max_dim,
            min_split_members=min_block_members,
            img_w=img_w,
            img_h=img_h,
        )
        blocks, v_lines = resolve_column_seams(
            dets,
            blocks,
            col_gap_px=col_gap_px,
            top_l=top_l,
            bot_l=bot_l,
            img_w=img_w,
            img_h=img_h,
            energy=energy,
        )

        for block_id, block in enumerate(blocks):
            quad = quad_from_lines(top_l, bot_l, v_lines[block_id], v_lines[block_id + 1], img_w, img_h)
            if quad is None:
                continue
            ang = math.degrees(circular_mean([long_axis_angle(dets[i]) for i in block]))
            plans.append(
                CropPlan(
                    shelf_id=shelf_id,
                    block_id=block_id,
                    angle_deg=round(ang, 2),
                    quad=quad,
                    member_indices=list(block),
                )
            )

    return plans


# ---------------------------------------------------------------------------
# Warp + pad (no upsize)
# ---------------------------------------------------------------------------


def quad_output_size(quad: list[tuple[float, float]]) -> tuple[int, int]:
    tl, tr, br, bl = quad
    w = max(dist(tl, tr), dist(bl, br))
    h = max(dist(tl, bl), dist(tr, br))
    return max(1, int(round(w))), max(1, int(round(h)))


def warp_quad(
    image_bgr: np.ndarray,
    quad: list[tuple[float, float]],
    *,
    max_side: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    native_w, native_h = quad_output_size(quad)
    scale = 1.0
    if max_side is not None and max(native_w, native_h) > max_side:
        scale = max_side / max(native_w, native_h)
    out_w = max(1, int(round(native_w * scale)))
    out_h = max(1, int(round(native_h * scale)))
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image_bgr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    return warped, m, scale


def pad_no_upsize(
    crop_bgr: np.ndarray, canvas: int, pad_value: int
) -> tuple[np.ndarray, float, float, float]:
    h, w = crop_bgr.shape[:2]
    if w <= 0 or h <= 0:
        blank = np.full((canvas, canvas, 3), pad_value, dtype=np.uint8)
        return blank, 1.0, 0.0, 0.0

    scale = min(1.0, canvas / max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if scale < 1.0:
        resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = crop_bgr

    pad_x = (canvas - new_w) / 2.0
    pad_y = (canvas - new_h) / 2.0
    out = np.full((canvas, canvas, 3), pad_value, dtype=np.uint8)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out, scale, float(x0), float(y0)


def map_det_from_crop(
    det: Det,
    m_inv: np.ndarray,
    gain: float,
    pad_x: float,
    pad_y: float,
) -> Det | None:
    if gain <= 1e-9:
        return None
    cx_c = (det.cx - pad_x) / gain
    cy_c = (det.cy - pad_y) / gain
    w_c = det.w / gain
    h_c = det.h / gain

    c, s = math.cos(det.angle), math.sin(det.angle)
    v1x, v1y = c * w_c / 2, s * w_c / 2
    v2x, v2y = -s * h_c / 2, c * h_c / 2
    corners_w = np.array(
        [
            [cx_c + v1x + v2x, cy_c + v1y + v2y],
            [cx_c + v1x - v2x, cy_c + v1y - v2y],
            [cx_c - v1x - v2x, cy_c - v1y - v2y],
            [cx_c - v1x + v2x, cy_c - v1y + v2y],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    scene = cv2.perspectiveTransform(corners_w, m_inv).reshape(-1, 2)
    rect = cv2.minAreaRect(scene.astype(np.float32))
    (cx, cy), (rw, rh), angle_deg = rect
    if rw < 1 or rh < 1:
        return None
    if rw < rh:
        rw, rh = rh, rw
        angle_deg += 90.0
    return Det(float(cx), float(cy), float(rw), float(rh), math.radians(angle_deg), det.conf)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

COLOR_DET_FIRST = (255, 42, 4)  # orange — first-pass / full-image detections
COLOR_DET_NEW = (255, 0, 255)  # bright magenta — net-new after crop re-infer + merge


def partition_new_dets(merged: list[Det], first: list[Det], iou_thresh: float) -> tuple[list[Det], list[Det]]:
    """Split ``merged`` into detections that match a first-pass box vs net-new."""
    existing: list[Det] = []
    new: list[Det] = []
    for m in merged:
        if any(rotated_iou(m, f) >= iou_thresh for f in first):
            existing.append(m)
        else:
            new.append(m)
    return existing, new


def draw_overlay(
    image_bgr: np.ndarray,
    plans: list[CropPlan],
    dets: list[Det] | None,
    *,
    draw_dets: bool,
    new_dets: list[Det] | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    lw = max(2, int(round((w + h) / 2 * 0.002)))

    if draw_dets and dets:
        if new_dets is None:
            for det in dets:
                pts = np.array(det.corners(), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, COLOR_DET_FIRST, lw)
        else:
            new_set = set(id(d) for d in new_dets)
            for det in dets:
                if id(det) in new_set:
                    continue
                pts = np.array(det.corners(), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, COLOR_DET_FIRST, lw)
            new_lw = lw + 1
            for det in new_dets:
                pts = np.array(det.corners(), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, COLOR_DET_NEW, new_lw)

    palette = [
        (0, 220, 255),
        (0, 255, 128),
        (255, 180, 0),
        (255, 0, 200),
        (80, 80, 255),
        (0, 160, 255),
        (200, 255, 0),
        (255, 100, 100),
    ]
    for i, plan in enumerate(plans):
        color = palette[i % len(palette)]
        q = plan.quad
        for a, b in zip(q, q[1:] + q[:1]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, lw + 1)
        tl = plan.quad[0]
        label = plan.name if plan.member_indices else f"{plan.name}*"
        cv2.putText(
            out,
            label,
            (int(tl[0]) + 4, int(tl[1]) + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def default_json_path(image: Path) -> Path:
    eval_dir().mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return eval_dir(f"{image.stem}_layout_crops_{ts}.json")


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
    img_h, img_w = image_bgr.shape[:2]
    print(f"Image: {img_w}x{img_h}", flush=True)

    YOLO = ensure_ultralytics()
    print(f"Loading weights: {weights}", flush=True)
    model = YOLO(str(weights), task="obb")
    print("Running first-pass detection…", flush=True)

    first = run_single(
        model,
        image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        iou=args.iou,
        max_det=args.max_det,
    )
    # Drop detections whose center lies outside the image (common OBB edge noise).
    first = [
        d
        for d in first
        if 0.0 <= d.cx <= float(img_w) and 0.0 <= d.cy <= float(img_h)
    ]
    print(f"first-pass: {len(first)} spines (in-frame)", flush=True)

    plans = plan_crops(
        first,
        img_w,
        img_h,
        image_bgr,
        angle_tol_deg=args.angle_tol_deg,
        row_gap_k=args.row_gap_k,
        col_gap_k=args.col_gap_k,
        min_block_members=args.min_block_members,
        imgsz=args.imgsz,
        max_crop_dim_k=args.max_crop_dim_k,
    )
    rule_results = verify_plan(first, plans, img_w, img_h, angle_tol_deg=args.angle_tol_deg)
    ok = hard_rules_ok(rule_results)
    empty = sum(1 for p in plans if not p.member_indices)
    print(f"planned crops: {len(plans)} (rules_ok={ok}; empty cells={empty})", flush=True)
    for r in rule_results:
        flag = "OK" if r.ok else ("FAIL" if r.hard else "WARN")
        print(f"  [{flag}] {r.rule}: {r.detail}", flush=True)
    for p in plans:
        x0, y0, x1, y1 = p.rect
        print(
            f"  {p.name}: ~{x1 - x0:.0f}x{y1 - y0:.0f} @({x0:.0f},{y0:.0f}) "
            f"angle≈{p.angle_deg}° members={len(p.member_indices)}",
            flush=True,
        )

    overlay_plan = args.overlay_plan
    do_infer = args.infer_crops and not args.plan_only and not overlay_plan
    write_dir = None if overlay_plan else args.write_crops
    if write_dir:
        write_dir = write_dir.expanduser().resolve()
        write_dir.mkdir(parents=True, exist_ok=True)

    need_materialize = bool(write_dir) or do_infer
    crop_dets: list[Det] = []
    crop_meta: list[dict[str, Any]] = []

    if need_materialize:
        for plan in plans:
            warped, m, warp_scale = warp_quad(image_bgr, plan.quad, max_side=args.imgsz)
            padded, gain, pad_x, pad_y = pad_no_upsize(warped, args.imgsz, args.pad_value)
            m_inv = cv2.invert(m)[1]

            meta: dict[str, Any] = {
                "name": plan.name,
                "shelf_id": plan.shelf_id,
                "block_id": plan.block_id,
                "angle_deg": plan.angle_deg,
                "quad": [[round(x, 2), round(y, 2)] for x, y in plan.quad],
                "rect": [round(v, 2) for v in plan.rect],
                "members": len(plan.member_indices),
                "warped_size": [int(warped.shape[1]), int(warped.shape[0])],
                "warp_scale": round(warp_scale, 5),
                "gain": round(gain, 5),
                "pad": [round(pad_x, 2), round(pad_y, 2)],
            }

            if write_dir:
                out_crop = write_dir / f"{plan.name}.png"
                cv2.imwrite(str(out_crop), padded)
                meta["path"] = str(out_crop)
                print(
                    f"  wrote {out_crop.name} ({warped.shape[1]}x{warped.shape[0]} → {args.imgsz})",
                    flush=True,
                )

            if do_infer:
                raw = predict_crop(
                    model,
                    padded,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    offset_x=0,
                    offset_y=0,
                )
                mapped: list[Det] = []
                for d in raw:
                    md = map_det_from_crop(d, m_inv, gain, pad_x, pad_y)
                    if md is not None and md.w > 1 and md.h > 1:
                        mapped.append(md)
                meta["crop_detections"] = len(mapped)
                crop_dets.extend(mapped)

            crop_meta.append(meta)
    else:
        for plan in plans:
            crop_meta.append(
                {
                    "name": plan.name,
                    "shelf_id": plan.shelf_id,
                    "block_id": plan.block_id,
                    "angle_deg": plan.angle_deg,
                    "quad": [[round(x, 2), round(y, 2)] for x, y in plan.quad],
                    "rect": [round(v, 2) for v in plan.rect],
                    "members": len(plan.member_indices),
                }
            )

    merged = first
    new_dets: list[Det] = []
    if do_infer:
        merged = nms_rotated(first + crop_dets, args.iou, args.max_det)
        _, new_dets = partition_new_dets(merged, first, args.iou)
        print(
            f"crop-pass raw: {len(crop_dets)}  merged: {len(merged)}  "
            f"new: {len(new_dets)}  delta: {len(new_dets):+d}",
            flush=True,
        )

    draw_dets = args.overlay_dets and not overlay_plan
    want_overlay = overlay_plan or args.overlay_crops or args.overlay_dets or args.out is not None
    out_path = args.out
    if want_overlay:
        if out_path is None:
            if overlay_plan:
                suffix = "plan"
            elif args.overlay_crops:
                suffix = "crops"
            else:
                suffix = "layout"
            out_path = image_path.with_name(f"{image_path.stem}.{suffix}.png")
        overlay = draw_overlay(
            image_bgr,
            plans,
            merged if draw_dets else None,
            draw_dets=draw_dets,
            new_dets=new_dets if do_infer and draw_dets else None,
        )
        cv2.imwrite(str(out_path), overlay)
        print(f"Wrote overlay {out_path}", flush=True)

    payload: dict[str, Any] = {
        "image": str(image_path),
        "image_size": {"width": img_w, "height": img_h},
        "weights": str(weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "overlap": 0,
        "jigsaw": True,
        "rules_ok": ok,
        "rules": results_to_json(rule_results),
        "first_pass_count": len(first),
        "first_pass": [asdict(d) for d in first],
        "planned_crops": len(plans),
        "empty_cells": empty,
        "crops": crop_meta,
        "merged_count": len(merged),
        "new_after_merge_count": len(new_dets),
        "crop_pass_raw": len(crop_dets) if do_infer else 0,
    }
    if do_infer:
        payload["detections_merged"] = [asdict(d) for d in merged]

    json_path = args.json or default_json_path(image_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {json_path}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
