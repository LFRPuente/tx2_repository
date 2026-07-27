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

from live_mvp_app import HTML, ClipRecorder, FrameBuffer, LiveProcessor, representative_snapshots
from tools.plc_triggered_video_recorder import edge_matches


class FakeOverlayProcessor:
    def __init__(self, buffer: FrameBuffer, *, detects_piece: bool = True) -> None:
        self.buffer = buffer
        self.detects_piece = detects_piece

    def recording_frame(self) -> dict | None:
        item = self.buffer.latest()
        if item is None:
            return None
        processed = item.copy()
        processed["frame"] = np.full_like(item["frame"], (0, 0, 240))
        return processed

    def snapshot(self, include_images: bool = False) -> dict:
        item = self.buffer.latest()
        if item is None:
            return {"result": None}
        pieces = (
            [
                {
                    "piece_id": 1,
                    "valid": False,
                    "box": {"x": 1, "y": 1, "w": 4, "h": 4, "conf": 0.9},
                    "sobel": {},
                    "measurement": None,
                }
            ]
            if self.detects_piece
            else []
        )
        return {
            "result": {
                "frame_index": int(item["index"]),
                "frame_utc": item["utc"],
                "frame_monotonic": float(item["monotonic"]),
                "pieces": pieces,
                "measurement_summary": {
                    "detected_count": len(pieces),
                    "valid_count": 0,
                    "invalid_count": len(pieces),
                },
            }
        }


class FakeSnapshotProcessor:
    def __init__(self) -> None:
        self.result: dict | None = None

    def recording_frame(self) -> dict | None:
        return None

    def snapshot(self, include_images: bool = False) -> dict:
        return {"result": self.result}


class FakeDatabase:
    backend_name = "sqlite"

    def __init__(self) -> None:
        self.created_event_ids: list[str] = []
        self.deleted_event_ids: list[str] = []

    def create_measurement_event(self, **kwargs) -> str:
        event_id = str(kwargs["event_id"])
        self.created_event_ids.append(event_id)
        return event_id

    def delete_measurement_event(self, event_id: str) -> None:
        self.deleted_event_ids.append(event_id)


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


class LiveProcessorTests(unittest.TestCase):
    def test_recording_frame_is_available_without_entering_api_payloads(self) -> None:
        processor = LiveProcessor(SimpleNamespace(process_fps=10.0), FrameBuffer(maxlen=8))
        recording_frame = {
            "index": 7,
            "utc": "frame-7",
            "monotonic": 7.0,
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        processor._set_state(
            result={
                "frame_index": 7,
                "original_image": "encoded",
                "_recording_frame": recording_frame,
            }
        )

        self.assertEqual(processor.recording_frame()["index"], 7)
        self.assertNotIn("_recording_frame", processor.snapshot(include_images=True)["result"])


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


class HistoryTests(unittest.TestCase):
    def test_representative_snapshots_include_the_full_clip_range(self) -> None:
        snapshots = [{"index": index} for index in range(64)]

        selected = representative_snapshots(snapshots, 6)

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0]["index"], 0)
        self.assertEqual(selected[-1]["index"], 63)
        self.assertEqual(len({item["index"] for item in selected}), 6)


class ClipRecorderTests(unittest.TestCase):
    def test_processing_snapshots_are_strictly_inside_the_plc_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = FakeSnapshotProcessor()
            recorder = ClipRecorder(
                SimpleNamespace(record_seconds=8.0),
                FrameBuffer(maxlen=8),
                processor,
            )
            snapshots: list[dict] = []
            seen: set[int] = set()
            analysis_dir = Path(temp_dir)

            for frame_index, frame_monotonic in ((1, 9.9), (2, 10.0), (3, 17.9), (4, 18.0)):
                processor.result = {
                    "frame_index": frame_index,
                    "frame_utc": f"frame-{frame_index}",
                    "frame_monotonic": frame_monotonic,
                }
                recorder._capture_processing_snapshot(analysis_dir, snapshots, seen, 10.0, 18.0)

            self.assertEqual([item["frame_index"] for item in snapshots], [2, 3])

    def test_overlapping_events_keep_separate_fixed_duration_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=0.4,
                record_fps=10.0,
                capture_fps=30.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=20)
            recorder = ClipRecorder(args, buffer, FakeOverlayProcessor(buffer))
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
            time.sleep(0.06)
            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": time.perf_counter(),
                }
            )

            self.assertEqual(recorder.snapshot()["active_recordings"], 2)
            deadline = time.perf_counter() + 3.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            self.assertFalse(recorder.snapshot()["recording"])
            self.assertEqual(recorder.snapshot()["error"], "")

            sidecars = sorted(Path(temp_dir).rglob("*.json"))
            self.assertEqual(len(sidecars), 2)
            for sidecar_path in sidecars:
                data = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.assertEqual(data["frames_written"], 4)
                self.assertAlmostEqual(data["video_duration_seconds"], 0.4, places=3)
                self.assertEqual(data["video_content"], "yolo_processed_overlay")

                video = cv2.VideoCapture(data["video_path"])
                try:
                    self.assertTrue(video.isOpened())
                    self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                    self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 10.0, delta=0.2)
                    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
                    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
                    self.assertIn(codec.lower(), {"avc1", "h264"})
                    self.assertEqual(data["video_codec"], "h264")
                    ok, encoded_frame = video.read()
                    self.assertTrue(ok)
                    blue, green, red = encoded_frame.mean(axis=(0, 1))
                    self.assertGreater(red, blue + 100)
                    self.assertGreater(red, green + 100)
                finally:
                    video.release()

    def test_failed_overlapping_clip_is_not_hidden_by_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=0.3,
                record_fps=10.0,
                capture_fps=30.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=20)
            recorder = ClipRecorder(args, buffer, FakeOverlayProcessor(buffer))
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

    def test_clip_without_detected_pieces_is_removed_from_disk_and_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                source="video",
                video=Path("test.mp4"),
                plc_endpoint="opc.tcp://test",
                event_node="ns=2;s=MeasureLength",
                watchdog_node="ns=2;s=VisionWD",
                record_seconds=0.3,
                record_fps=10.0,
                capture_fps=30.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=20)
            database = FakeDatabase()
            recorder = ClipRecorder(
                args,
                buffer,
                FakeOverlayProcessor(buffer, detects_piece=False),
                database,
            )
            recorder.vision_configuration = {"configuration_hash": "test"}
            stop_feeder = threading.Event()

            def feed_frames() -> None:
                index = 0
                while not stop_feeder.is_set():
                    buffer.append(
                        {
                            "index": index,
                            "utc": f"frame-{index}",
                            "monotonic": time.perf_counter(),
                            "frame": np.zeros((48, 64, 3), dtype=np.uint8),
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

            deadline = time.perf_counter() + 3.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            snapshot = recorder.snapshot()
            self.assertFalse(snapshot["recording"])
            self.assertEqual(snapshot["discarded_clip_count"], 1)
            self.assertEqual(
                snapshot["last_discarded_clip"]["reason"],
                "no_piece_detected",
            )
            self.assertEqual(snapshot["failed_recording_count"], 0)
            self.assertEqual(snapshot["error"], "")
            self.assertEqual(database.deleted_event_ids, database.created_event_ids)
            self.assertEqual(list(Path(temp_dir).rglob("*.mp4")), [])
            self.assertEqual(list(Path(temp_dir).rglob("*.json")), [])
            self.assertEqual(list(Path(temp_dir).rglob("*_analysis")), [])


if __name__ == "__main__":
    unittest.main()
