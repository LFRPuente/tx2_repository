from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_mvp_app import draw_rectified_overlay


class LiveExclusionZoneTests(unittest.TestCase):
    def test_rectified_overlay_draws_configured_exclusion_zone(self) -> None:
        rectified = np.zeros((30, 40, 3), dtype=np.uint8)
        calibration = {
            "reference_y": None,
            "exclusion_zones": [{"x": 2.0, "y": 2.0, "w": 14.0, "h": 20.0}],
        }

        overlay = draw_rectified_overlay(rectified, [], calibration)

        blue, green, red = overlay[10, 10]
        self.assertGreater(int(red), int(blue))
        self.assertGreater(int(red), int(green))
        self.assertTrue(np.array_equal(overlay[28, 38], rectified[28, 38]))


if __name__ == "__main__":
    unittest.main()
