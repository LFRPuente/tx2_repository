from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as vision


class SpatialScaleMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            {
                "x1": 100.0,
                "y1": 100.0,
                "x2": 100.0,
                "y2": 200.0,
                "px": 100.0,
                "inches": 6.0,
            },
            {
                "x1": 900.0,
                "y1": 100.0,
                "x2": 900.0,
                "y2": 200.0,
                "px": 100.0,
                "inches": 5.0,
            },
            {
                "x1": 200.0,
                "y1": 300.0,
                "x2": 400.0,
                "y2": 300.0,
                "px": 200.0,
                "inches": 8.0,
            },
        ]
        self.scale_map = vision.build_spatial_scale_map(self.segments, 1000, 500)
        self.calibration = {
            "segments": self.segments,
            "inch_per_px": 0.05,
            "reference_y": 100.0,
            "reference_offset_in": vision.MEASUREMENT_REFERENCE_OFFSET_IN,
            "scale_map": self.scale_map,
        }

    def test_axis_references_build_independent_maps(self) -> None:
        vertical = self.scale_map["axes"]["y"]
        horizontal = self.scale_map["axes"]["x"]

        self.assertEqual(vertical["sample_count"], 2)
        self.assertEqual(vertical["mode"], "linear_x")
        self.assertEqual(horizontal["sample_count"], 1)
        self.assertAlmostEqual(vertical["minimum_inch_per_px"], 0.05)
        self.assertAlmostEqual(vertical["maximum_inch_per_px"], 0.06)

    def test_vertical_scale_interpolates_and_clamps_outside_coverage(self) -> None:
        center = vision.spatial_scale_at(self.calibration, 500.0, 150.0, "y")
        outside = vision.spatial_scale_at(self.calibration, 0.0, 150.0, "y")

        self.assertAlmostEqual(center["inch_per_px"], 0.055)
        self.assertFalse(center["extrapolated"])
        self.assertAlmostEqual(outside["inch_per_px"], 0.06)
        self.assertTrue(outside["extrapolated"])

    def test_piece_distance_integrates_the_local_vertical_scale(self) -> None:
        spatial = vision.integrate_vertical_scale(
            self.calibration,
            x=500.0,
            y_start=100.0,
            y_end=300.0,
        )

        self.assertAlmostEqual(spatial["distance_in"], 11.0)
        self.assertAlmostEqual(spatial["mean_inch_per_px"], 0.055)
        self.assertFalse(spatial["extrapolated"])

    def test_sobel_measurement_uses_map_instead_of_global_average(self) -> None:
        sobel = {
            "line": {
                "x1": 400.0,
                "y1": 300.0,
                "x2": 600.0,
                "y2": 300.0,
            }
        }

        measurement = vision.measurement_from_sobel(
            sobel,
            self.calibration,
            width=1000,
        )

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement["delta_in"], 11.0)
        self.assertAlmostEqual(
            measurement["measurement_in"],
            vision.MEASUREMENT_REFERENCE_OFFSET_IN + 11.0,
        )
        self.assertEqual(measurement["scale_source"], "linear_x")
        self.assertEqual(measurement["scale_sample_count"], 2)

    def test_global_scale_remains_the_fallback(self) -> None:
        calibration = {
            "inch_per_px": 0.0625,
            "reference_y": 100.0,
            "scale_map": vision.build_spatial_scale_map([], 1000, 500),
        }

        spatial = vision.integrate_vertical_scale(calibration, 500.0, 100.0, 260.0)

        self.assertAlmostEqual(spatial["distance_in"], 10.0)
        self.assertEqual(spatial["source"], "global")

    def test_two_dimensional_map_uses_nearest_sample_outside_hull(self) -> None:
        segments = []
        for x, y, inch_per_px in (
            (100.0, 100.0, 0.050),
            (900.0, 100.0, 0.052),
            (900.0, 400.0, 0.054),
            (100.0, 400.0, 0.056),
        ):
            segments.append(
                {
                    "x1": x,
                    "y1": y - 50.0,
                    "x2": x,
                    "y2": y + 50.0,
                    "px": 100.0,
                    "inches": 100.0 * inch_per_px,
                }
            )
        calibration = {
            "inch_per_px": 0.053,
            "scale_map": vision.build_spatial_scale_map(segments, 1000, 500),
        }

        inside = vision.spatial_scale_at(calibration, 500.0, 250.0, "y")
        outside = vision.spatial_scale_at(calibration, 0.0, 40.0, "y")

        self.assertEqual(calibration["scale_map"]["axes"]["y"]["mode"], "idw_2d")
        self.assertFalse(inside["extrapolated"])
        self.assertAlmostEqual(inside["inch_per_px"], 0.053, places=3)
        self.assertTrue(outside["extrapolated"])
        self.assertEqual(outside["source"], "nearest_2d")
        self.assertAlmostEqual(outside["inch_per_px"], 0.050)


if __name__ == "__main__":
    unittest.main()
