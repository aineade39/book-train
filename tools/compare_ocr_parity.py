#!/usr/bin/env python3
"""OCR Mac vs iOS parity scorer -- see book-id-ios's "OCR Mac vs iOS
compare" plan and its README.md "OCR Mac vs iOS parity" section.

Scores OCR-only exports from `SpineIdentificationPipeline.runOCROnly`
(detect -> warp -> Vision OCR, never catalog matching) against the
optimize-gemini oracles, and separately checks Mac-vs-iOS OCR agreement on
identical ("pinned") detections and Mac-vs-iOS detector drift.

Inputs are the directories `xcrun xcresulttool export attachments` writes,
each containing a `manifest.json` plus the exported JSON files themselves
(named by UUID -- `manifest.json` maps each back to the
"<sceneId>.<pass>.json" name `Tests/BookIDOCRTests/OCRParityTests.swift`
gave the `XCTAttachment`, ignoring the `_<index>_<uuid>` suffix XCTest
appends on export):

    xcodebuild test -scheme BookID-macOS -destination 'platform=macOS' \
        -resultBundlePath mac.xcresult
    xcrun xcresulttool export attachments --path mac.xcresult \
        --output-path mac-export --filter '*.json'

Usage:
  # After the Mac pass: stage its mac-native JSONs for the iOS test bundle
  # (see project.yml's BookIDOCRTests-iOS `mac-detections` resource).
  .venv/bin/python3 tools/compare_ocr_parity.py --mac-dir mac-export \
      --copy-mac-native-to ../book-id-ios/Tests/BookIDOCRTests/Fixtures/mac-detections

  # After both passes: score everything.
  .venv/bin/python3 tools/compare_ocr_parity.py \
      --mac-dir mac-export --ios-dir ios-export --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spine_matching_parity_fixture import normalize_for_search  # noqa: E402

# `suggestedHumanReadableName` in manifest.json is the XCTAttachment name
# ("<sceneId>.<pass>.json") plus an "_<index>_<uuid>" disambiguation suffix
# XCTest appends on export -- it is not the on-disk filename.
ATTACHMENT_NAME_RE = re.compile(
    r"^(?P<scene_id>.+)\.(?P<pass_name>mac-native|ios-native|ios-pinned)(?:_\d+_[0-9A-Fa-f-]{36})?\.json$"
)

HIT_THRESHOLD = 70.0
WEAK_THRESHOLD = 40.0
DETECTOR_IOU_THRESHOLD = 0.5


@dataclass
class OBB:
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians -- matches SpineCore.OBBDetection.angle verbatim

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        v1x, v1y = c * self.w / 2, s * self.w / 2
        v2x, v2y = -s * self.h / 2, c * self.h / 2
        return [
            (self.cx + v1x + v2x, self.cy + v1y + v2y),
            (self.cx + v1x - v2x, self.cy + v1y - v2y),
            (self.cx - v1x - v2x, self.cy - v1y - v2y),
            (self.cx - v1x + v2x, self.cy - v1y + v2y),
        ]

    @property
    def diag(self) -> float:
        return math.hypot(self.w, self.h)


def obb_from_spine(spine: dict) -> OBB:
    return OBB(cx=spine["cx"], cy=spine["cy"], w=spine["w"], h=spine["h"], angle=spine["angle"])


# MARK: - Rotated-box geometry (ported verbatim from
# SpineCore/SpineGeometry.swift / tools/tiled_predict_obb.py so oracle
# point-in-box pairing and detector-drift IoU use the exact same math
# already proven out in both other implementations)


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def ensure_ccw(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    area = sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )
    return list(reversed(pts)) if area < 0 else pts


def clip_polygon(
    subject: list[tuple[float, float]], edge_a: tuple[float, float], edge_b: tuple[float, float]
) -> list[tuple[float, float]]:
    if not subject:
        return []

    def side(p: tuple[float, float]) -> float:
        return (edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0])

    def intersect(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
        a1, b1 = edge_b[1] - edge_a[1], edge_a[0] - edge_b[0]
        c1 = a1 * edge_a[0] + b1 * edge_a[1]
        a2, b2 = q[1] - p[1], p[0] - q[0]
        c2 = a2 * p[0] + b2 * p[1]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return p
        return ((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)

    out: list[tuple[float, float]] = []
    for i in range(len(subject)):
        current, previous = subject[i], subject[i - 1]
        cur_in, prev_in = side(current) >= 0, side(previous) >= 0
        if cur_in:
            if not prev_in:
                out.append(intersect(previous, current))
            out.append(current)
        elif prev_in:
            out.append(intersect(previous, current))
    return out


def polygon_area(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    return (
        abs(
            sum(
                pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                for i in range(len(pts))
            )
        )
        / 2.0
    )


def rotated_iou(a: OBB, b: OBB) -> float:
    reach = (a.diag + b.diag) / 2
    if math.hypot(a.cx - b.cx, a.cy - b.cy) >= reach:
        return 0.0
    quad_a, quad_b = ensure_ccw(a.corners()), ensure_ccw(b.corners())
    inter = quad_a
    for i in range(len(quad_b)):
        if not inter:
            break
        inter = clip_polygon(inter, quad_b[i], quad_b[(i + 1) % len(quad_b)])
    inter_area = polygon_area(inter)
    if inter_area <= 0:
        return 0.0
    union = polygon_area(quad_a) + polygon_area(quad_b) - inter_area
    return inter_area / union if union > 0 else 0.0


# MARK: - Export discovery


def discover_exports(directory: Path) -> dict[str, dict[str, Path]]:
    """scene_id -> {pass_name: path}, read from an `xcresulttool export
    attachments` output directory's manifest.json."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest.json under {directory} -- pass a directory `xcrun xcresulttool "
            "export attachments --output-path ...` wrote to."
        )
    manifest = json.loads(manifest_path.read_text())
    out: dict[str, dict[str, Path]] = {}
    for test_entry in manifest:
        for att in test_entry.get("attachments", []):
            m = ATTACHMENT_NAME_RE.match(att.get("suggestedHumanReadableName", ""))
            if not m:
                continue
            out.setdefault(m["scene_id"], {})[m["pass_name"]] = directory / att["exportedFileName"]
    return out


def copy_mac_native_to(mac_exports: dict[str, dict[str, Path]], dest: Path) -> None:
    """Renames+copies each scene's mac-native export to
    "<sceneId>.mac-native.json" under `dest` -- the fixed filename
    `Tests/BookIDOCRTests/OCRParityTests.swift`'s pinned pass looks up by
    exact name in the bundled `mac-detections` resource folder."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for scene_id, passes in sorted(mac_exports.items()):
        src = passes.get("mac-native")
        if src is None:
            print(f"  skip {scene_id}: no mac-native export found", file=sys.stderr)
            continue
        dst = dest / f"{scene_id}.mac-native.json"
        shutil.copyfile(src, dst)
        print(f"  {src.name} -> {dst}")
        copied += 1
    print(f"Copied {copied} mac-native export(s) into {dest}")
    if copied:
        print("Run `xcodegen generate` in book-id-ios, then re-run the iOS test.")


# MARK: - Oracle scoring


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fuzzy_score(a: str, b: str) -> float:
    return fuzz.token_set_ratio(normalize_for_search(a), normalize_for_search(b))


def oracle_guard(payload: dict, oracle: dict) -> str | None:
    oimg = oracle["image"]
    if "sha256" not in oimg:
        return "oracle has no image.sha256 (incomplete oracle) -- skipping oracle scoring"
    if payload["imageWidth"] != oimg["w"] or payload["imageHeight"] != oimg["h"]:
        return (
            f"image dims mismatch: export {payload['imageWidth']}x{payload['imageHeight']} "
            f"vs oracle {oimg['w']}x{oimg['h']}"
        )
    if payload["imageSHA256"] != oimg["sha256"]:
        return (
            f"image sha256 mismatch: export {payload['imageSHA256'][:12]}... vs oracle "
            f"{oimg['sha256'][:12]}... (the oracle's source image file has changed on disk "
            "since the oracle was generated)"
        )
    return None


def pair_oracle(oracle_books: list[dict], oracle_w: float, oracle_h: float, spines: list[dict]) -> dict:
    """Point-in-OBB containment first, falling back to nearest-box distance
    (within one box-diagonal) -- resolved one-to-one by a greedy assignment
    that prefers containment matches, then ascending distance."""
    obbs = [obb_from_spine(s) for s in spines]
    avg_diag = sum(o.diag for o in obbs) / len(obbs) if obbs else 0.0
    fallback_radius = avg_diag if avg_diag > 0 else 200.0

    candidates: list[tuple[bool, float, int, int]] = []  # (not_contained, dist, book_idx, spine_idx)
    for bi, book in enumerate(oracle_books):
        y1000, x1000 = book["point_2d"]
        px, py = (x1000 / 1000.0) * oracle_w, (y1000 / 1000.0) * oracle_h
        for si, obb in enumerate(obbs):
            contained = point_in_polygon(px, py, obb.corners())
            dist = math.hypot(px - obb.cx, py - obb.cy)
            if contained or dist <= fallback_radius:
                candidates.append((not contained, dist, bi, si))

    candidates.sort()
    used_books: set[int] = set()
    used_spines: set[int] = set()
    assignment: dict[int, int] = {}
    for _, _, bi, si in candidates:
        if bi in used_books or si in used_spines:
            continue
        used_books.add(bi)
        used_spines.add(si)
        assignment[bi] = si

    buckets = {"hit": 0, "weak": 0, "fail": 0, "unpaired": 0}
    scores: list[float] = []
    books_out = []
    for bi, book in enumerate(oracle_books):
        if bi not in assignment:
            buckets["unpaired"] += 1
            books_out.append({"title": book.get("title"), "status": "unpaired"})
            continue
        text = spines[assignment[bi]].get("assembledText", "")
        score = fuzzy_score(text, book.get("title", "")) if text.strip() else 0.0
        scores.append(score)
        status = "hit" if score >= HIT_THRESHOLD else ("weak" if score >= WEAK_THRESHOLD else "fail")
        buckets[status] += 1
        books_out.append({
            "title": book.get("title"), "status": status, "score": round(score, 1), "assembledText": text,
        })

    return {
        "buckets": buckets,
        "meanScore": round(sum(scores) / len(scores), 1) if scores else None,
        "totalBooks": len(oracle_books),
        "books": books_out,
    }


# MARK: - Mac vs iOS


def compare_mac_ios(mac_payload: dict, ios_pinned_payload: dict) -> dict:
    if mac_payload["imageSHA256"] != ios_pinned_payload["imageSHA256"]:
        return {"error": "imageSHA256 mismatch between mac-native and ios-pinned exports; not comparable"}

    mac_by_id = {s["id"]: s for s in mac_payload["spines"]}
    ios_by_id = {s["id"]: s for s in ios_pinned_payload["spines"]}

    agree = 0
    ratios: list[float] = []
    disagreements = []
    missing_on_ios = 0
    for det_id, mac_spine in mac_by_id.items():
        ios_spine = ios_by_id.get(det_id)
        if ios_spine is None:
            missing_on_ios += 1
            continue
        mac_norm = normalize_for_search(mac_spine.get("assembledText", ""))
        ios_norm = normalize_for_search(ios_spine.get("assembledText", ""))
        ratio = fuzz.token_set_ratio(mac_norm, ios_norm) if (mac_norm or ios_norm) else 100.0
        ratios.append(ratio)
        if mac_norm == ios_norm:
            agree += 1
        else:
            disagreements.append({
                "id": det_id, "mac": mac_spine.get("assembledText", ""),
                "ios": ios_spine.get("assembledText", ""), "ratio": round(ratio, 1),
            })

    disagreements.sort(key=lambda d: d["ratio"])
    total = len(mac_by_id)
    return {
        "totalDetections": total,
        "exactAgreementRate": round(100.0 * agree / total, 1) if total else None,
        "meanTokenSetRatio": round(sum(ratios) / len(ratios), 1) if ratios else None,
        "missingOnIOS": missing_on_ios,
        "worstDisagreements": disagreements[:10],
    }


def compare_detector_drift(mac_native: dict, ios_native: dict) -> dict:
    mac_obbs = [obb_from_spine(s) for s in mac_native["spines"]]
    ios_obbs = [obb_from_spine(s) for s in ios_native["spines"]]

    candidates = []
    for i, a in enumerate(mac_obbs):
        for j, b in enumerate(ios_obbs):
            iou = rotated_iou(a, b)
            if iou >= DETECTOR_IOU_THRESHOLD:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    matched = 0
    for _, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matched += 1

    return {
        "macCount": len(mac_obbs), "iosCount": len(ios_obbs), "matchedIoU0.5": matched,
        "macOnly": len(mac_obbs) - matched, "iosOnly": len(ios_obbs) - matched,
    }


# MARK: - Reporting


def print_summary(report: dict) -> None:
    print(f"{'scene':<14}{'pass':<13}{'hit':>5}{'weak':>6}{'fail':>6}{'unpaired':>10}{'mean':>7}")
    print("-" * 61)
    for scene_id, scene in sorted(report["scenes"].items()):
        oracle = scene.get("oracle", {})
        if "error" in oracle:
            print(f"{scene_id:<14}(oracle)      -- oracle: {oracle['error']}")
            continue
        for pass_name, stats in sorted(oracle.items()):
            if "error" in stats:
                print(f"{scene_id:<14}{pass_name:<13}-- {stats['error']}")
                continue
            b = stats["buckets"]
            mean = stats["meanScore"] if stats["meanScore"] is not None else 0.0
            print(f"{scene_id:<14}{pass_name:<13}{b['hit']:>5}{b['weak']:>6}{b['fail']:>6}{b['unpaired']:>10}{mean:>7.1f}")

    mac_vs_ios_rows = [(sid, s["macVsIOS"]) for sid, s in sorted(report["scenes"].items()) if "macVsIOS" in s]
    if mac_vs_ios_rows:
        print()
        print(f"{'scene':<14}{'agree%':>8}{'meanRatio':>11}{'missing':>9}")
        for scene_id, mvi in mac_vs_ios_rows:
            if "error" in mvi:
                print(f"{scene_id:<14}-- {mvi['error']}")
                continue
            print(f"{scene_id:<14}{mvi['exactAgreementRate']:>7.1f}%{mvi['meanTokenSetRatio']:>11.1f}{mvi['missingOnIOS']:>9}")

    drift_rows = [(sid, s["detectorDrift"]) for sid, s in sorted(report["scenes"].items()) if "detectorDrift" in s]
    if drift_rows:
        print()
        print(f"{'scene':<14}{'mac#':>6}{'ios#':>6}{'matched@0.5':>13}")
        for scene_id, dd in drift_rows:
            print(f"{scene_id:<14}{dd['macCount']:>6}{dd['iosCount']:>6}{dd['matchedIoU0.5']:>13}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mac-dir", required=True, type=Path, help="xcresulttool export attachments output dir for the macOS pass")
    parser.add_argument("--ios-dir", type=Path, default=None, help="Same, for the iOS pass. Omit for a Mac-only oracle-scoring run.")
    parser.add_argument(
        "--oracles-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "optimize-gemini" / "fixtures" / "oracles",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the full report as JSON here")
    parser.add_argument(
        "--copy-mac-native-to", type=Path, default=None,
        help="Rename+copy each scene's mac-native export to <sceneId>.mac-native.json under this "
        "directory (the iOS test's bundled mac-detections resource folder), then exit.",
    )
    args = parser.parse_args()

    mac_exports = discover_exports(args.mac_dir)
    if args.copy_mac_native_to:
        copy_mac_native_to(mac_exports, args.copy_mac_native_to)
        return 0

    ios_exports = discover_exports(args.ios_dir) if args.ios_dir else {}
    scene_ids = sorted(set(mac_exports) | set(ios_exports))
    if not scene_ids:
        print(f"No OCR export JSON found under {args.mac_dir}" + (f" or {args.ios_dir}" if args.ios_dir else ""), file=sys.stderr)
        return 1

    report: dict = {"scenes": {}}
    for scene_id in scene_ids:
        scene_report: dict = {}
        oracle_path = args.oracles_dir / f"{scene_id}.json"
        oracle = load_json(oracle_path) if oracle_path.exists() else None

        payloads = {**mac_exports.get(scene_id, {}), **ios_exports.get(scene_id, {})}
        payloads = {pass_name: load_json(path) for pass_name, path in payloads.items()}

        if oracle is None:
            scene_report["oracle"] = {"error": f"no oracle at {oracle_path}"}
        else:
            oracle_scores = {}
            for pass_name, payload in payloads.items():
                guard_error = oracle_guard(payload, oracle)
                oracle_scores[pass_name] = (
                    {"error": guard_error} if guard_error
                    else pair_oracle(oracle["books"], oracle["image"]["w"], oracle["image"]["h"], payload["spines"])
                )
            scene_report["oracle"] = oracle_scores

        if "mac-native" in payloads and "ios-pinned" in payloads:
            scene_report["macVsIOS"] = compare_mac_ios(payloads["mac-native"], payloads["ios-pinned"])
        if "mac-native" in payloads and "ios-native" in payloads:
            scene_report["detectorDrift"] = compare_detector_drift(payloads["mac-native"], payloads["ios-native"])

        report["scenes"][scene_id] = scene_report

    print_summary(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
