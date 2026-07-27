from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as vision


class PieceBoxRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 1000, 3), dtype=np.uint8)
        self.profile = {
            "source": "test",
            "sample_count": 40,
            "median_width_ratio": 0.10,
            "median_height_ratio": 0.20,
            "median_pitch_ratio": 0.11,
        }

    @staticmethod
    def box(center_x: float, confidence: float = 0.9) -> dict:
        return {
            "x": center_x - 50.0,
            "y": 100.0,
            "w": 100.0,
            "h": 60.0,
            "conf": confidence,
        }

    def test_overlapping_predictions_keep_the_highest_confidence_box(self) -> None:
        boxes = [
            self.box(100.0, confidence=0.95),
            {**self.box(103.0, confidence=0.40), "x": 53.0},
            self.box(210.0, confidence=0.90),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(len(processed), 2)
        self.assertEqual(diagnostics["removed_overlap_count"], 1)
        self.assertAlmostEqual(processed[0]["conf"], 0.95)

    def test_supported_large_gap_infers_one_missing_piece(self) -> None:
        boxes = [
            self.box(100.0),
            self.box(210.0),
            self.box(430.0),
            self.box(540.0),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        inferred = [box for box in processed if box.get("inferred")]
        self.assertEqual(len(inferred), 1)
        self.assertAlmostEqual(inferred[0]["x"] + inferred[0]["w"] / 2.0, 320.0)
        self.assertEqual(diagnostics["missing_candidate_count"], 1)
        self.assertEqual(diagnostics["inferred_count"], 1)

    def test_isolated_pair_does_not_create_an_unbounded_guess(self) -> None:
        boxes = [self.box(100.0), self.box(430.0)]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(len(processed), 2)
        self.assertEqual(diagnostics["missing_candidate_count"], 0)
        self.assertFalse(any(box.get("inferred") for box in processed))

    def test_severe_size_outlier_is_removed_from_piece_detections(self) -> None:
        boxes = [
            {"x": 0.0, "y": 0.0, "w": 82.0, "h": 26.0, "conf": 0.12},
            self.box(210.0),
            self.box(320.0),
            self.box(430.0),
            self.box(540.0),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(len(processed), 4)
        self.assertEqual(diagnostics["removed_size_count"], 1)
        self.assertTrue(all(box["h"] >= 50.0 for box in processed))

    def test_small_exclusion_zone_overlap_is_allowed(self) -> None:
        box = self.box(100.0)
        zone = {"x": 140.0, "y": 100.0, "w": 100.0, "h": 60.0}

        kept, removed = vision.filter_boxes_by_exclusion_zones(
            [box],
            [zone],
            image_shape=self.image.shape,
        )

        self.assertEqual(kept, [box])
        self.assertEqual(removed, [])

    def test_box_mostly_inside_exclusion_zone_is_removed(self) -> None:
        box = self.box(100.0)
        zone = {"x": 120.0, "y": 100.0, "w": 100.0, "h": 60.0}

        kept, removed = vision.filter_boxes_by_exclusion_zones(
            [box],
            [zone],
            image_shape=self.image.shape,
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(removed), 1)
        self.assertAlmostEqual(removed[0]["overlap_ratio"], 0.30)


if __name__ == "__main__":
    unittest.main()
