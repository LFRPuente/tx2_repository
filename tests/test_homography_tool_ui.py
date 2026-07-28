from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homography_web_app import HTML


class HomographyToolUiTests(unittest.TestCase):
    def test_measurement_segments_can_move_by_endpoint_or_as_a_line(self) -> None:
        self.assertIn("selectedSegment: null, draggingSegment: null", HTML)
        self.assertIn("function nearestMeasureSegmentPart(cx, cy)", HTML)
        self.assertIn("function beginMeasureSegmentDrag(hit, imagePoint)", HTML)
        self.assertIn("function updateMeasureSegmentDrag(imagePoint)", HTML)
        self.assertIn("drag.part === 'start'", HTML)
        self.assertIn("drag.part === 'end'", HTML)
        self.assertIn("imagePoint.x - drag.anchor.x", HTML)

    def test_moving_a_segment_recalculates_its_scale(self) -> None:
        self.assertIn(
            "segment.px = Math.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);",
            HTML,
        )
        self.assertIn(
            "segment.inch_per_px = segment.px > 0 ? Number(segment.inches) / segment.px : null;",
            HTML,
        )
        self.assertIn("updateMeasureInfo();", HTML)


if __name__ == "__main__":
    unittest.main()
