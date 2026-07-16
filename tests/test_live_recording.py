from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from live_mvp_app import HTML, ClipRecorder, FrameBuffer
from tools.plc_triggered_video_recorder import edge_matches


class FrameBufferTests(unittest.TestCase):
    def test_capacity_is_bounded(self) -> None:
        buffer = FrameBuffer(maxlen=8)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        for index in range(20):
            buffer.append(
                {
                    "index": index,
                    "utc": f"frame-{index}",
                    "monotonic": float(index),
                    "frame": frame,
                }
            )

        self.assertEqual(
            buffer.stats(),
            {"count": 8, "first_index": 12, "last_index": 19},
        )


class PlcEdgeTests(unittest.TestCase):
    def test_changed_mode_accepts_boolean_edges(self) -> None:
        self.assertTrue(edge_matches("rising", "changed"))
        self.assertTrue(edge_matches("falling", "changed"))
        self.assertFalse(edge_matches("", "changed"))

    def test_live_ui_exposes_plc_signal_state(self) -> None:
        self.assertIn('id="plc-signal"', HTML)
        self.assertIn("plc.last_trigger", HTML)
        self.assertIn("PLC signal received", HTML)
        self.assertNotIn("TX2 Vision", HTML)
        self.assertNotIn("<h1>Live MVP</h1>", HTML)


class ClipRecorderTests(unittest.TestCase):
    def test_second_event_closes_first_clip_without_overlapping_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=0.4,
                record_fps=10.0,
                capture_fps=30.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=20)
            recorder = ClipRecorder(args, buffer)
            stop_feeder = threading.Event()

            def feed_frames() -> None:
                index = 0
                while not stop_feeder.is_set():
                    frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
                    buffer.append(
                        {
                            "index": index,
                            "utc": f"frame-{index}",
                            "monotonic": time.perf_counter(),
                            "frame": frame,
                        }
                    )
                    index += 1
                    time.sleep(0.01)

            feeder = threading.Thread(target=feed_frames, daemon=True)
            feeder.start()
            self.addCleanup(stop_feeder.set)
            self.addCleanup(feeder.join, 1.0)

            deadline = time.perf_counter() + 1.0
            while buffer.latest() is None and time.perf_counter() < deadline:
                time.sleep(0.01)

            first_event_time = time.perf_counter()
            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": first_event_time,
                }
            )
            time.sleep(0.16)
            second_event_time = time.perf_counter()
            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": second_event_time,
                }
            )

            self.assertEqual(recorder.snapshot()["active_recordings"], 1)
            deadline = time.perf_counter() + 3.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            self.assertFalse(recorder.snapshot()["recording"])
            self.assertEqual(recorder.snapshot()["error"], "")

            sidecars = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in Path(temp_dir).rglob("*.json")
            ]
            sidecars.sort(key=lambda item: item["clip_index"])
            self.assertEqual(len(sidecars), 2)
            first_clip, second_clip = sidecars

            self.assertEqual(first_clip["stop_reason"], "next_plc_signal")
            self.assertLess(first_clip["record_seconds"], 0.4)
            self.assertGreater(first_clip["record_seconds"], 0.1)
            self.assertEqual(first_clip["next_event"]["event_read_monotonic"], second_event_time)
            self.assertEqual(second_clip["stop_reason"], "max_duration")
            self.assertAlmostEqual(second_clip["record_seconds"], 0.4, places=3)
            self.assertEqual(second_clip["frames_written"], 4)
            self.assertLess(first_clip["last_frame_index"], second_clip["first_frame_index"])

            for data in sidecars:
                video = cv2.VideoCapture(data["video_path"])
                try:
                    self.assertTrue(video.isOpened())
                    self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), data["frames_written"])
                    self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 10.0, delta=0.2)
                finally:
                    video.release()

    def test_failed_replacement_clip_is_not_hidden_by_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=0.3,
                record_fps=10.0,
                capture_fps=30.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=20)
            recorder = ClipRecorder(args, buffer)
            stop_feeder = threading.Event()

            def feed_frames() -> None:
                index = 0
                while not stop_feeder.is_set():
                    buffer.append(
                        {
                            "index": index,
                            "utc": f"frame-{index}",
                            "monotonic": time.perf_counter(),
                            "frame": np.full((48, 64, 3), index % 255, dtype=np.uint8),
                        }
                    )
                    index += 1
                    time.sleep(0.01)

            feeder = threading.Thread(target=feed_frames, daemon=True)
            feeder.start()
            self.addCleanup(stop_feeder.set)
            self.addCleanup(feeder.join, 1.0)

            deadline = time.perf_counter() + 1.0
            while buffer.latest() is None and time.perf_counter() < deadline:
                time.sleep(0.01)

            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": time.perf_counter(),
                }
            )
            time.sleep(0.05)
            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": time.perf_counter() - 5.0,
                }
            )

            deadline = time.perf_counter() + 3.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            snapshot = recorder.snapshot()
            self.assertFalse(snapshot["recording"])
            self.assertEqual(snapshot["failed_recording_count"], 1)
            self.assertIn("No frames were available", snapshot["error"])
            self.assertEqual(len(list(Path(temp_dir).rglob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
