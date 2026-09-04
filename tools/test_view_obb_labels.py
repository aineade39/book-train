#!/usr/bin/env python3
"""Unit tests for tools/view_obb_labels.py parsers / discovery."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from view_obb_labels import (  # noqa: E402
    discover_samples,
    parse_createml_boxes,
    parse_labelme_shapes,
    parse_yolo_obb_txt,
)


class TestViewObbLabels(unittest.TestCase):
    def test_parse_yolo_obb_txt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text(
                "0 0 0 1 0 1 1 0 1\n"
                "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n",
                encoding="utf-8",
            )
            boxes = parse_yolo_obb_txt(p, w=100, h=200)
            self.assertEqual(len(boxes), 2)
            np.testing.assert_allclose(boxes[0][0], [0.0, 0.0])
            np.testing.assert_allclose(boxes[0][2], [100.0, 200.0])
            np.testing.assert_allclose(boxes[1][0], [10.0, 40.0])

    def test_parse_labelme_rectangle(self) -> None:
        doc = {
            "shapes": [
                {
                    "label": "spine",
                    "shape_type": "rectangle",
                    "points": [[10, 20], [40, 80]],
                }
            ]
        }
        boxes = parse_labelme_shapes(doc)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].shape, (4, 2))
        np.testing.assert_allclose(boxes[0][0], [10, 20])
        np.testing.assert_allclose(boxes[0][2], [40, 80])

    def test_parse_createml_boxes(self) -> None:
        anns = [
            {"label": "spine", "coordinates": {"x": 50, "y": 100, "width": 20, "height": 40}}
        ]
        boxes = parse_createml_boxes(anns)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0][0], [40, 80])
        np.testing.assert_allclose(boxes[0][2], [60, 120])

    def test_discover_yolo_layout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            img_dir = root / "train" / "images"
            lbl_dir = root / "train" / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            img = img_dir / "foo.jpg"
            cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
            (lbl_dir / "foo.txt").write_text("0 0 0 1 0 1 1 0 1\n", encoding="utf-8")
            samples = discover_samples(root, labels=None, split="train")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].kind, "yolo")
            assert samples[0].label is not None
            self.assertEqual(samples[0].label.name, "foo.txt")

    def test_parse_manual_boxes_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(
                json.dumps(
                    {
                        "boxes": [
                            {"corners_px": [[1, 2], [3, 2], [3, 4], [1, 4]]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            from view_obb_labels import load_boxes_file

            boxes = load_boxes_file(p, w=10, h=10)
            self.assertEqual(len(boxes), 1)
            np.testing.assert_allclose(boxes[0][0], [1, 2])

    def test_discover_createml(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            img = root / "spine_0001.jpg"
            cv2.imwrite(str(img), np.zeros((16, 16, 3), dtype=np.uint8))
            (root / "annotations.json").write_text(
                json.dumps(
                    [
                        {
                            "imagefilename": "spine_0001.jpg",
                            "annotation": [
                                {
                                    "label": "spine",
                                    "coordinates": {
                                        "x": 8,
                                        "y": 8,
                                        "width": 4,
                                        "height": 8,
                                    },
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            samples = discover_samples(root, labels=None, split=None)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].kind, "createml")
            assert samples[0].boxes is not None
            self.assertEqual(len(samples[0].boxes), 1)


if __name__ == "__main__":
    unittest.main()
