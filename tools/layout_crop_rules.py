#!/usr/bin/env python3
"""Verification rules for layout crop plans against first-pass OBB detections.

Hard rules
----------
R1 COVER_IMAGE
    Crop polygons tile the image: area ≈ W×H; every coarse grid sample is covered.

R2 NO_OVERLAP
    Pairwise interior intersection area of crop polygons is ≈ 0.

R3 NO_OBB_CUT
    No OBB has a corner *strictly inside* a crop other than the crop that owns
    the OBB center. Shared boundary touches are allowed.

R4 EVERY_OBB_OWNED
    Every OBB center lies in at least one crop polygon.

R5 QUAD_FINITE
    All crop corners are finite and inside the image (small pad).

Soft
----
S2 EMPTY_CELLS_OK
    Empty crops (0 members) are allowed for missed-spine recovery.

S3 BLOCK_ORIENTATION_PURITY
    Warn if a block's member angles spread beyond ``2 × angle_tol_deg`` from
    their circular mean — flags column segmentation that let two orientation
    classes leak into a single block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


class HasQuad(Protocol):
    name: str
    quad: list[tuple[float, float]]
    member_indices: list[int]


class HasCorners(Protocol):
    def corners(self) -> list[tuple[float, float]]: ...

    cx: float
    cy: float
    w: float
    h: float
    angle: float


def _norm_angle(a: float) -> float:
    a = a % math.pi
    if a < 0:
        a += math.pi
    return a


def _angle_diff(a: float, b: float) -> float:
    d = abs(_norm_angle(a) - _norm_angle(b))
    return min(d, math.pi - d)


def _circular_mean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(2 * _norm_angle(a)) for a in angles)
    c = sum(math.cos(2 * _norm_angle(a)) for a in angles)
    return _norm_angle(0.5 * math.atan2(s, c))


def _long_axis_angle(det: HasCorners) -> float:
    """Mirrors ``layout_crop_predict.long_axis_angle``: the model doesn't
    guarantee w is the long side, so orientation must account for w vs h."""
    base = det.angle if det.w >= det.h else det.angle + math.pi / 2
    return _norm_angle(base)


@dataclass
class RuleResult:
    rule: str
    hard: bool
    ok: bool
    detail: str


def _poly_area(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    a = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _point_on_segment(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float, eps: float = 1.0
) -> bool:
    cross = abs((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1))
    if cross > eps * max(1.0, math.hypot(x2 - x1, y2 - y1)):
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -eps:
        return False
    if dot > (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps:
        return False
    return True


def _point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if _point_on_segment(x, y, xi, yi, xj, yj, eps=1.0):
            return True
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _point_strictly_inside(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    if not _point_in_poly(x, y, poly):
        return False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if _point_on_segment(x, y, x1, y1, x2, y2, eps=1.5):
            return False
    return True


def _clip_poly(
    subject: list[tuple[float, float]], edge_a: tuple[float, float], edge_b: tuple[float, float]
) -> list[tuple[float, float]]:
    def side(p):
        return (edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0])

    def intersect(p, q):
        a1 = edge_b[1] - edge_a[1]
        b1 = edge_a[0] - edge_b[0]
        c1 = a1 * edge_a[0] + b1 * edge_a[1]
        a2 = q[1] - p[1]
        b2 = p[0] - q[0]
        c2 = a2 * p[0] + b2 * p[1]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return p
        return ((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)

    out: list[tuple[float, float]] = []
    if not subject:
        return out
    for i, cur in enumerate(subject):
        prev = subject[i - 1]
        cur_in = side(cur) >= 0
        prev_in = side(prev) >= 0
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
    return out


def _poly_intersection_area(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    inter = list(a)
    for i in range(len(b)):
        if not inter:
            break
        inter = _clip_poly(inter, b[i], b[(i + 1) % len(b)])
    return _poly_area(inter)


def _ensure_ccw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    signed = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    return list(reversed(pts)) if signed < 0 else list(pts)


def owning_crop(x: float, y: float, plans: list[HasQuad]) -> HasQuad | None:
    hits = [p for p in plans if _point_in_poly(x, y, p.quad)]
    if not hits:
        return None
    return min(hits, key=lambda p: _poly_area(p.quad))


def verify_plan(
    dets: list[HasCorners],
    plans: list[HasQuad],
    img_w: int,
    img_h: int,
    *,
    grid: int = 48,
    angle_tol_deg: float = 25.0,
) -> list[RuleResult]:
    results: list[RuleResult] = []
    if not plans:
        results.append(RuleResult("R5_QUAD_FINITE", True, False, "no crops"))
        return results

    pad = 3.0
    bad_finite = []
    for p in plans:
        for x, y in p.quad:
            if not (math.isfinite(x) and math.isfinite(y)):
                bad_finite.append(p.name)
                break
            if x < -pad or y < -pad or x > img_w + pad or y > img_h + pad:
                bad_finite.append(f"{p.name}@({x:.0f},{y:.0f})")
                break
    results.append(
        RuleResult(
            "R5_QUAD_FINITE",
            True,
            not bad_finite,
            "ok" if not bad_finite else f"bad corners: {bad_finite[:8]}",
        )
    )

    overlaps = []
    polys = [_ensure_ccw(list(p.quad)) for p in plans]
    for i, a in enumerate(polys):
        for j in range(i + 1, len(polys)):
            area = _poly_intersection_area(a, polys[j])
            if area > 4.0:
                overlaps.append(f"{plans[i].name}/{plans[j].name}:{area:.0f}")
    results.append(
        RuleResult(
            "R2_NO_OVERLAP",
            True,
            not overlaps,
            "ok" if not overlaps else f"overlaps {overlaps[:6]}",
        )
    )

    area_sum = sum(_poly_area(p) for p in polys)
    expected = float(img_w * img_h)
    area_ok = abs(area_sum - expected) <= 0.02 * expected
    miss = 0
    multi = 0
    step_x = img_w / grid
    step_y = img_h / grid
    for gy in range(grid):
        for gx in range(grid):
            x = (gx + 0.5) * step_x
            y = (gy + 0.5) * step_y
            hits = sum(1 for p in plans if _point_in_poly(x, y, p.quad))
            if hits == 0:
                miss += 1
            elif hits > 2:
                multi += 1
    results.append(
        RuleResult(
            "R1_COVER_IMAGE",
            True,
            area_ok and miss == 0,
            f"area={area_sum:.0f}/{expected:.0f} miss={miss} multi={multi}",
        )
    )

    orphan: list[int] = []
    cut_violations: list[int] = []
    split: list[str] = []
    for di, det in enumerate(dets):
        center_owner = owning_crop(det.cx, det.cy, plans)
        if center_owner is None:
            orphan.append(di)
            continue
        bad: list[str] = []
        for cx, cy in det.corners():
            for p in plans:
                if p.name == center_owner.name:
                    continue
                if _point_strictly_inside(cx, cy, p.quad):
                    bad.append(p.name)
        # Crop edge through OBB: any crop edge (except owner) with midpoint inside OBB.
        obb = det.corners()
        for p in plans:
            if p.name == center_owner.name:
                continue
            q = p.quad
            for i in range(4):
                a, b = q[i], q[(i + 1) % 4]
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                if _point_strictly_inside(mx, my, obb):
                    bad.append(p.name)
        if bad:
            cut_violations.append(di)
            split.append(f"det{di}:{center_owner.name}->{sorted(set(bad))}")

    results.append(
        RuleResult(
            "R4_EVERY_OBB_OWNED",
            True,
            not orphan,
            "ok" if not orphan else f"orphans n={len(orphan)} ids={orphan[:12]}",
        )
    )
    results.append(
        RuleResult(
            "R3_NO_OBB_CUT",
            True,
            not cut_violations,
            "ok" if not cut_violations else f"split={split[:8]}",
        )
    )
    results.append(
        RuleResult(
            "S2_EMPTY_CELLS_OK",
            False,
            True,
            f"empty_cells={sum(1 for p in plans if not p.member_indices)}",
        )
    )

    purity_bad: list[str] = []
    for p in plans:
        if len(p.member_indices) < 2:
            continue
        angles = [_long_axis_angle(dets[i]) for i in p.member_indices]
        mean_a = _circular_mean(angles)
        spread = max(_angle_diff(a, mean_a) for a in angles)
        if spread > math.radians(angle_tol_deg * 2):
            purity_bad.append(f"{p.name}:spread={math.degrees(spread):.1f}deg")
    results.append(
        RuleResult(
            "S3_BLOCK_ORIENTATION_PURITY",
            False,
            not purity_bad,
            "ok" if not purity_bad else f"impure blocks: {purity_bad[:8]}",
        )
    )
    return results


def hard_rules_ok(results: list[RuleResult]) -> bool:
    return all(r.ok for r in results if r.hard)


def results_to_json(results: list[RuleResult]) -> list[dict[str, Any]]:
    return [{"rule": r.rule, "hard": r.hard, "ok": r.ok, "detail": r.detail} for r in results]
