from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from live_mvp_app import (
    HTML,
    HISTORY_HTML,
    ClipRecorder,
    DatabaseReconciler,
    FrameBuffer,
    LiveProcessor,
    build_raw_rtsp_url,
    direct_raw_capture_enabled,
    mark_measurement_evidence_snapshot,
    measurement_evidence_snapshots,
    measurement_marker_active,
    representative_snapshots,
)
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
        processed["raw_frame"] = item["frame"]
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


class DatabaseReconcilerTests(unittest.TestCase):
    def test_synced_sidecar_missing_from_database_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            clip_dir = output_dir / "live_plc_clips" / "2026-07-28"
            clip_dir.mkdir(parents=True)
            sidecar_path = clip_dir / "live_0001_raw.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "event_id": "retained-event",
                        "event_key": "retained-key",
                        "event": {},
                        "db_sync_status": "synced",
                        "db_sync_backend": "sqlite",
                    }
                ),
                encoding="utf-8",
            )

            class ReconcileDatabase:
                backend_name = "sqlite"

                def __init__(self) -> None:
                    self.event_ids: set[str] = set()
                    self.synced_paths: list[Path] = []

                def all_measurement_event_ids(self) -> set[str]:
                    return self.event_ids.copy()

                def sync_sidecar(self, path: Path, _output_dir: Path) -> str:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    event_id = str(data["event_id"])
                    self.event_ids.add(event_id)
                    self.synced_paths.append(path)
                    return event_id

                def delete_measurement_event(self, event_id: str) -> None:
                    self.event_ids.discard(event_id)

            args = SimpleNamespace(
                output_dir=output_dir,
                plc_endpoint="opc.tcp://test",
                event_node="ns=2;s=MeasureLength",
                watchdog_node="ns=2;s=VisionWD",
            )
            database = ReconcileDatabase()
            with patch(
                "live_mvp_app.build_vision_configuration",
                return_value={"configuration_hash": "test"},
            ):
                reconciler = DatabaseReconciler(args, database)

            reconciler.sync_once()

            self.assertEqual(database.synced_paths, [sidecar_path])
            self.assertEqual(database.event_ids, {"retained-event"})
            self.assertEqual(reconciler.snapshot()["synced"], 1)


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


class RawCaptureTests(unittest.TestCase):
    def test_raw_rtsp_requests_camera_native_resolution_and_thirty_fps(self) -> None:
        args = SimpleNamespace(
            camera_user="axis user",
            camera_password="p@ss word",
            camera_ip="10.14.115.241",
            codec="h264",
            raw_camera_resolution="2880x2160",
            raw_record_fps=30.0,
            raw_rtsp_url="",
        )

        url = build_raw_rtsp_url(args)

        self.assertIn("resolution=2880x2160", url)
        self.assertIn("fps=30", url)
        self.assertIn("videozfpsmode=fixed", url)
        self.assertIn("videokeyframeinterval=30", url)
        self.assertIn("axis%20user:p%40ss%20word@", url)

    def test_direct_raw_copy_is_only_used_for_camera_sources(self) -> None:
        self.assertTrue(
            direct_raw_capture_enabled(
                SimpleNamespace(save_raw_clips=True, source="rtsp")
            )
        )
        self.assertFalse(
            direct_raw_capture_enabled(
                SimpleNamespace(save_raw_clips=True, source="video")
            )
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
        self.assertIn("Last PLC Cut Signal", HTML)
        self.assertNotIn("Last PLC signal", HTML)
        self.assertIn("measurement-taken", HTML)
        self.assertIn("data.recorder?.measurement_marker_active", HTML)
        self.assertNotIn("TX2 Vision", HTML)
        self.assertNotIn("<h1>Live MVP</h1>", HTML)


class HistoryTests(unittest.TestCase):
    def test_history_only_renders_the_green_measurement_evidence(self) -> None:
        self.assertIn("PLC + 2 s measurement frame", HISTORY_HTML)
        self.assertIn('class="measurement-evidence"', HISTORY_HTML)
        self.assertIn("Raw clip", HISTORY_HTML)
        self.assertNotIn("Up to 6 representative captures", HISTORY_HTML)
        self.assertNotIn("No diagram evidence", HISTORY_HTML)
        self.assertNotIn("SQLite temporal is active", HISTORY_HTML)

    def test_measurement_evidence_uses_only_the_canonical_snapshot(self) -> None:
        snapshots = [
            {"frame_index": 1, "frame_monotonic": 10.1, "is_canonical": 0},
            {"frame_index": 2, "frame_monotonic": 12.0, "is_canonical": 1},
            {"frame_index": 3, "frame_monotonic": 12.1, "is_canonical": 0},
        ]

        selected = measurement_evidence_snapshots(snapshots)

        self.assertEqual([item["frame_index"] for item in selected], [2])

    def test_representative_snapshots_include_the_full_clip_range(self) -> None:
        snapshots = [{"index": index} for index in range(64)]

        selected = representative_snapshots(snapshots, 6)

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0]["index"], 0)
        self.assertEqual(selected[-1]["index"], 63)
        self.assertEqual(len({item["index"] for item in selected}), 6)


class ClipRecorderTests(unittest.TestCase):
    def test_configuration_snapshot_refreshes_after_calibration_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            homography_path = output_dir / "homography_selection.json"
            calibration_path = output_dir / "table_measurement_calibration.json"
            model_path = output_dir / "best.pt"
            homography_path.write_text("{}", encoding="utf-8")
            calibration_path.write_text('{"version": 1}', encoding="utf-8")
            model_path.write_bytes(b"model")
            recorder = ClipRecorder(
                SimpleNamespace(
                    output_dir=output_dir,
                    model=model_path,
                    conf=0.1,
                    imgsz=960,
                ),
                FrameBuffer(maxlen=8),
            )
            configurations = [
                {"calibration_sha256": "first"},
                {"calibration_sha256": "second"},
            ]

            with patch(
                "live_mvp_app.build_vision_configuration",
                side_effect=configurations,
            ) as build:
                first = recorder._configuration_snapshot()
                cached = recorder._configuration_snapshot()
                calibration_path.write_text('{"version": 22}', encoding="utf-8")
                refreshed = recorder._configuration_snapshot()

            self.assertIs(first, cached)
            self.assertEqual(first["calibration_sha256"], "first")
            self.assertEqual(refreshed["calibration_sha256"], "second")
            self.assertEqual(build.call_count, 2)

    def test_measurement_evidence_image_gets_a_green_perimeter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "evidence.jpg"
            cv2.imwrite(
                str(image_path),
                np.full((80, 120, 3), (0, 0, 240), dtype=np.uint8),
            )
            snapshot = {"original_overlay_path": str(image_path)}

            marked = mark_measurement_evidence_snapshot(snapshot)
            image = cv2.imread(str(image_path))
            perimeter = np.concatenate(
                (
                    image[:8].reshape(-1, 3),
                    image[-8:].reshape(-1, 3),
                    image[:, :8].reshape(-1, 3),
                    image[:, -8:].reshape(-1, 3),
                )
            )
            green_pixels = (
                (perimeter[:, 1] > perimeter[:, 0] + 30)
                & (perimeter[:, 1] > perimeter[:, 2] + 30)
            )

            self.assertTrue(marked)
            self.assertTrue(snapshot["measurement_evidence"])
            self.assertGreater(float(green_pixels.mean()), 0.10)

    def test_measurement_marker_starts_two_seconds_after_the_plc_event(self) -> None:
        self.assertFalse(measurement_marker_active(11.999, 10.0))
        self.assertTrue(measurement_marker_active(12.0, 10.0))
        self.assertTrue(measurement_marker_active(12.799, 10.0))
        self.assertFalse(measurement_marker_active(12.8, 10.0))

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
                save_raw_clips=True,
                measurement_delay_seconds=0.2,
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
                self.assertEqual(data["raw_video_content"], "axis_camera_raw")
                self.assertEqual(data["raw_video_capture_mode"], "processed_buffer")
                self.assertEqual(data["raw_video_fps"], 10.0)
                self.assertEqual(data["raw_video_frames"], 4)
                self.assertEqual(data["measurement_delay_seconds"], 0.2)
                self.assertEqual(data["measurement_marker_first_video_frame_index"], 2)
                self.assertEqual(data["measurement_marker_frames_written"], 2)

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
                    green_perimeter_found = False
                    while True:
                        ok, encoded_frame = video.read()
                        if not ok:
                            break
                        perimeter = np.concatenate(
                            (
                                encoded_frame[:8].reshape(-1, 3),
                                encoded_frame[-8:].reshape(-1, 3),
                                encoded_frame[:, :8].reshape(-1, 3),
                                encoded_frame[:, -8:].reshape(-1, 3),
                            )
                        )
                        green_pixels = (
                            (perimeter[:, 1] > perimeter[:, 0] + 30)
                            & (perimeter[:, 1] > perimeter[:, 2] + 30)
                        )
                        if float(green_pixels.mean()) > 0.10:
                            green_perimeter_found = True
                    self.assertTrue(green_perimeter_found)
                finally:
                    video.release()

                raw_video = cv2.VideoCapture(data["raw_video_path"])
                try:
                    self.assertTrue(raw_video.isOpened())
                    self.assertEqual(int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                    self.assertAlmostEqual(
                        raw_video.get(cv2.CAP_PROP_FPS),
                        10.0,
                        delta=0.2,
                    )
                    green_perimeter_found = False
                    red_overlay_found = False
                    while True:
                        ok, raw_frame = raw_video.read()
                        if not ok:
                            break
                        blue, green, red = raw_frame.mean(axis=(0, 1))
                        if red > blue + 100 and red > green + 100:
                            red_overlay_found = True
                        perimeter = np.concatenate(
                            (
                                raw_frame[:8].reshape(-1, 3),
                                raw_frame[-8:].reshape(-1, 3),
                                raw_frame[:, :8].reshape(-1, 3),
                                raw_frame[:, -8:].reshape(-1, 3),
                            )
                        )
                        green_pixels = (
                            (perimeter[:, 1] > perimeter[:, 0] + 30)
                            & (perimeter[:, 1] > perimeter[:, 2] + 30)
                        )
                        if float(green_pixels.mean()) > 0.10:
                            green_perimeter_found = True
                    self.assertFalse(red_overlay_found)
                    self.assertFalse(green_perimeter_found)
                finally:
                    raw_video.release()

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
