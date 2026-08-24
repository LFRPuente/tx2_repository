from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOL_SCRIPT = (REPO_ROOT / "static" / "js" / "homography_tool.js").read_text(
    encoding="utf-8"
)
TOOL_TEMPLATE = (REPO_ROOT / "templates" / "homography_tool.html").read_text(
    encoding="utf-8"
)


class HomographyToolUiTests(unittest.TestCase):
    def test_measurement_segments_can_move_by_endpoint_or_as_a_line(self) -> None:
        self.assertIn("selectedSegment: null, draggingSegment: null", TOOL_SCRIPT)
        self.assertIn("function nearestMeasureSegmentPart(cx, cy)", TOOL_SCRIPT)
        self.assertIn("function beginMeasureSegmentDrag(hit, imagePoint)", TOOL_SCRIPT)
        self.assertIn("function updateMeasureSegmentDrag(imagePoint)", TOOL_SCRIPT)
        self.assertIn("drag.part === 'start'", TOOL_SCRIPT)
        self.assertIn("drag.part === 'end'", TOOL_SCRIPT)
        self.assertIn("imagePoint.x - drag.anchor.x", TOOL_SCRIPT)

    def test_moving_a_segment_recalculates_its_scale(self) -> None:
        self.assertIn(
            "segment.px = Math.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);",
            TOOL_SCRIPT,
        )
        self.assertIn(
            "segment.inch_per_px = segment.px > 0 ? Number(segment.inches) / segment.px : null;",
            TOOL_SCRIPT,
        )
        self.assertIn("updateMeasureInfo();", TOOL_SCRIPT)

    def test_tool_template_uses_external_assets(self) -> None:
        self.assertIn("filename='css/homography_tool.css'", TOOL_TEMPLATE)
        self.assertIn("filename='js/homography_tool.js'", TOOL_TEMPLATE)
        self.assertNotIn("<style>", TOOL_TEMPLATE)
        self.assertNotIn("<script>", TOOL_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
