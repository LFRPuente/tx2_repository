from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as vision


class PieceMeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (20, 10), (75, 95), (180, 180, 180), -1)
        cv2.rectangle(self.image, (125, 10), (190, 125), (210, 210, 210), -1)
        self.boxes = [
            {"x": 125, "y": 10, "w": 65, "h": 115, "conf": 0.80},
            {"x": 20, "y": 10, "w": 55, "h": 85, "conf": 0.90},
        ]
        self.calibration = {
            "reference_y": 150.0,
            "inch_per_px": 0.1,
            "reference_offset_in": 0.0,
        }

    def test_each_box_produces_an_individual_horizontal_measurement(self) -> None:
        pieces, summary = vision.analyze_piece_boxes(
            self.image,
            self.boxes,
            self.calibration,
            frame_idx=10,
            time_sec=1.0,
        )

        self.assertEqual([piece["piece_id"] for piece in pieces], [1, 2])
        self.assertEqual([piece["box"]["x"] for piece in pieces], [20, 125])
        self.assertEqual(summary["detected_count"], 2)
        self.assertEqual(summary["valid_count"], 2)

        first_line = pieces[0]["sobel"]["line"]
        second_line = pieces[1]["sobel"]["line"]
        self.assertTrue(first_line["horizontal"])
        self.assertTrue(second_line["horizontal"])
        self.assertEqual(first_line["y1"], first_line["y2"])
        self.assertEqual(second_line["y1"], second_line["y2"])
        self.assertAlmostEqual(first_line["x1"], 20.0)
        self.assertAlmostEqual(first_line["x2"], 75.0)
        self.assertAlmostEqual(second_line["x1"], 125.0)
        self.assertAlmostEqual(second_line["x2"], 190.0)
        self.assertAlmostEqual(first_line["y"], 93.0, delta=2.0)
        self.assertAlmostEqual(second_line["y"], 124.0, delta=2.0)

        first_measurement = pieces[0]["measurement"]
        second_measurement = pieces[1]["measurement"]
        self.assertAlmostEqual(first_measurement["x"], 47.5)
        self.assertAlmostEqual(second_measurement["x"], 157.5)
        self.assertLess(first_measurement["measurement_in"], second_measurement["measurement_in"])
        self.assertAlmostEqual(
            summary["minimum_in"],
            first_measurement["measurement_in"],
        )
        self.assertAlmostEqual(
            summary["maximum_in"],
            second_measurement["measurement_in"],
        )

    def test_measurement_uses_piece_line_midpoint(self) -> None:
        measurement = vision.measurement_from_sobel(
            {
                "line": {
                    "x1": 10.0,
                    "y1": 90.0,
                    "x2": 30.0,
                    "y2": 90.0,
                }
            },
            self.calibration,
            width=240,
        )

        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement["x"], 20.0)
        self.assertAlmostEqual(measurement["measurement_in"], -6.0)

    def test_piece_front_uses_falling_edge_instead_of_box_bottom(self) -> None:
        image = np.full((160, 180, 3), 65, dtype=np.uint8)
        cv2.rectangle(image, (40, 15), (135, 88), (205, 205, 205), -1)
        cv2.rectangle(image, (40, 89), (135, 115), (35, 35, 35), -1)
        box = {"x": 40, "y": 15, "w": 95, "h": 110, "conf": 0.9}

        sobel = vision.sobel_projection_for_piece(image, box)

        self.assertTrue(sobel["is_valid"])
        self.assertAlmostEqual(sobel["line"]["y"], 88.0, delta=3.0)
        self.assertLess(sobel["line"]["y"], box["y"] + box["h"] - 20.0)

    def test_original_overlay_keeps_one_front_per_piece(self) -> None:
        pieces, _summary = vision.analyze_piece_boxes(
            self.image,
            self.boxes,
            self.calibration,
        )
        overlay = vision.mvp_original_overlay_for_pieces(
            pieces,
            self.calibration,
            np.eye(3, dtype=np.float64),
            rect_width=self.image.shape[1],
        )

        self.assertEqual(len(overlay["piece_fronts"]), 2)
        self.assertEqual(
            [front["piece_id"] for front in overlay["piece_fronts"]],
            [1, 2],
        )
        self.assertIsNotNone(overlay["reference_line"])


if __name__ == "__main__":
    unittest.main()
