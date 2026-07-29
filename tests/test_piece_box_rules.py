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

    @staticmethod
    def sized_box(center_x: float, width: float, confidence: float = 0.9) -> dict:
        return {
            "x": center_x - width / 2.0,
            "y": 100.0,
            "w": width,
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

    def test_coherent_frame_width_overrides_a_different_dataset_profile(self) -> None:
        boxes = [
            self.sized_box(100.0, 72.0),
            self.sized_box(180.0, 72.0),
            self.sized_box(260.0, 72.0),
            self.sized_box(340.0, 72.0),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(len(processed), 4)
        self.assertTrue(diagnostics["frame_width_reliable"])
        self.assertEqual(diagnostics["profile_source"], "frame")
        self.assertAlmostEqual(diagnostics["typical_width_px"], 72.0)
        self.assertEqual(diagnostics["removed_size_count"], 0)

    def test_coherent_box_widths_are_normalized_without_moving_centers(self) -> None:
        boxes = [
            self.sized_box(100.0, 90.0),
            self.sized_box(210.0, 100.0),
            self.sized_box(320.0, 110.0),
            self.sized_box(430.0, 100.0),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(diagnostics["adjusted_width_count"], 2)
        self.assertTrue(all(abs(box["w"] - 100.0) < 1e-6 for box in processed))
        centers = [box["x"] + box["w"] / 2.0 for box in processed]
        self.assertEqual(centers, [100.0, 210.0, 320.0, 430.0])

    def test_frame_width_outlier_is_removed_from_a_coherent_group(self) -> None:
        boxes = [
            self.sized_box(100.0, 100.0),
            self.sized_box(210.0, 100.0),
            self.sized_box(320.0, 140.0),
            self.sized_box(430.0, 100.0),
            self.sized_box(540.0, 100.0),
        ]

        processed, diagnostics = vision.apply_piece_box_rules(
            self.image,
            boxes,
            verify_inferred=False,
            profile=self.profile,
        )

        self.assertEqual(diagnostics["removed_size_count"], 1)
        self.assertEqual(len([box for box in processed if not box.get("inferred")]), 4)
        self.assertTrue(all(box["w"] == 100.0 for box in processed))

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
