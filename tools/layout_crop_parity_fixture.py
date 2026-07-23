#!/usr/bin/env python3
"""Fixed OBB-detection fixture for the opt-in Swift/Python layout-crop parity
test (`Tests/SpineCoreTests/ParityIntegrationTests.swift`).

Bypasses the YOLO model entirely: feeds a hand-authored `Det` list straight
into `layout_crop_predict.plan_crops` / `layout_crop_rules.verify_plan`
against a uniform-gray synthetic scene (so pixel-texture-guided seam search
is deterministic — zero energy everywhere), and prints the resulting plan
plus rule outcomes as JSON on stdout.

The Swift test constructs the identical `Det` list in Swift and diffs the
two outputs; keep both lists numerically in sync if you change this file.
Not part of the production CLI pipeline — invoked only by the Swift test.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_crop_predict import plan_crops  # noqa: E402
from layout_crop_rules import hard_rules_ok, results_to_json, verify_plan  # noqa: E402
from tiled_predict_obb import Det  # noqa: E402

IMG_W = 900
IMG_H = 500


def _deg(a: float) -> float:
    return math.radians(a)


# Two shelves: shelf 0 has two column blocks (touching triplets) separated by
# a wide gap; shelf 1 has one upright block and one block rotated ~80° (forces
# an orientation-based column split with no gap requirement).
DETS = [
    Det(80, 160, 50, 200, _deg(0), 0.90),
    Det(150, 160, 50, 200, _deg(0), 0.90),
    Det(220, 160, 50, 200, _deg(0), 0.90),
    Det(500, 160, 50, 200, _deg(0), 0.90),
    Det(570, 160, 50, 200, _deg(0), 0.90),
    Det(640, 160, 50, 200, _deg(0), 0.90),
    Det(80, 400, 50, 200, _deg(0), 0.85),
    Det(150, 400, 50, 200, _deg(0), 0.85),
    Det(220, 400, 50, 200, _deg(80), 0.85),
    Det(290, 400, 50, 200, _deg(80), 0.85),
]


def main() -> int:
    image_bgr = np.full((IMG_H, IMG_W, 3), 114, dtype=np.uint8)
    plans = plan_crops(
        DETS,
        IMG_W,
        IMG_H,
        image_bgr,
        angle_tol_deg=25.0,
        row_gap_k=0.2,
        col_gap_k=1.0,
        min_block_members=2,
        imgsz=1024,
        max_crop_dim_k=1.5,
    )
    rules = verify_plan(DETS, plans, IMG_W, IMG_H, angle_tol_deg=25.0)
    payload = {
        "img_w": IMG_W,
        "img_h": IMG_H,
        "rules_ok": hard_rules_ok(rules),
        "rules": results_to_json(rules),
        "plans": [
            {
                "shelf_id": p.shelf_id,
                "block_id": p.block_id,
                "angle_deg": p.angle_deg,
                "quad": [[round(x, 3), round(y, 3)] for x, y in p.quad],
                "member_indices": list(p.member_indices),
            }
            for p in plans
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
