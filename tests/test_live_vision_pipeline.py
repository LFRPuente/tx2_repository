from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import live_mvp_app as live


class LiveVisionPipelineTests(unittest.TestCase):
    def test_live_processor_applies_calibrated_zones_and_keeps_box_rules(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(
                process_fps=10.0,
                conf=0.10,
                imgsz=960,
                model=Path("individual-pieces.pt"),
            ),
            live.FrameBuffer(maxlen=8),
        )
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        item = {
            "index": 17,
            "utc": "frame-17",
            "monotonic": 101.25,
            "frame": frame,
        }
        zones = [{"x": 0.0, "y": 0.0, "w": 5.0, "h": 24.0}]
        calibration = {
            "reference_y": None,
            "exclusion_zones": zones,
            "exclusion_max_box_overlap": 0.20,
        }
        boxes = [{"x": 8.0, "y": 3.0, "w": 8.0, "h": 16.0, "conf": 0.91}]
        diagnostics = {
            "raw_count": 2,
            "removed_exclusion_count": 1,
            "final_count": 1,
            "exclusion_max_overlap": 0.20,
        }
        pieces = [
            {
                "piece_id": 1,
                "box": boxes[0],
                "confidence": 0.91,
                "sobel": {"line": None, "is_valid": False},
                "measurement": None,
                "valid": False,
            }
        ]
        summary = {"detected_count": 1, "valid_count": 0, "invalid_count": 1}

        with (
            patch.object(
                live.vision,
                "load_homography",
                return_value=(np.eye(3), (32, 24), {}),
            ),
            patch.object(
                live.vision,
                "load_measurement_calibration",
                return_value=calibration,
            ),
            patch.object(
                live.vision,
                "predict_yolo_boxes_with_rules",
                return_value=(boxes, diagnostics),
            ) as predict,
            patch.object(
                live.vision,
                "analyze_piece_boxes",
                return_value=(pieces, summary),
            ) as analyze,
            patch.object(
                live.vision,
                "mvp_original_overlay_for_pieces",
                return_value={"reference_line": None, "piece_fronts": []},
            ),
        ):
            result = processor._process(item)

        predict.assert_called_once()
        predict_kwargs = predict.call_args.kwargs
        self.assertEqual(predict_kwargs["exclusion_zones"], zones)
        self.assertEqual(predict_kwargs["exclusion_max_overlap"], 0.20)
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], boxes)
        self.assertEqual(result["box_rules"], diagnostics)
        self.assertEqual(result["pieces"], pieces)
        self.assertEqual(result["measurement_summary"], summary)


if __name__ == "__main__":
    unittest.main()
