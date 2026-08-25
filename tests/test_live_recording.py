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
    ClipRecorder,
    DatabaseReconciler,
    FrameBuffer,
    LiveProcessor,
    app as flask_app,
    build_raw_rtsp_url,
    direct_raw_capture_enabled,
    mark_measurement_evidence_snapshot,
    measurement_evidence_snapshots,
    measurement_marker_active,
    plc_status_with_signal_state,
)
from tools.plc_triggered_video_recorder import edge_matches

LIVE_TEMPLATE = (REPO_ROOT / "templates" / "live.html").read_text(encoding="utf-8")
LIVE_SCRIPT = (REPO_ROOT / "static" / "js" / "live.js").read_text(encoding="utf-8")
LIVE_STYLE = (REPO_ROOT / "static" / "css" / "live.css").read_text(encoding="utf-8")
HISTORY_TEMPLATE = (REPO_ROOT / "templates" / "history.html").read_text(
    encoding="utf-8"
)
HISTORY_SCRIPT = (REPO_ROOT / "static" / "js" / "history.js").read_text(
    encoding="utf-8"
)


class FakeOverlayProcessor:
    def __init__(
        self,
        buffer: FrameBuffer,
        *,
        detects_piece: bool = True,
        processing_delay_seconds: float = 0.0,
    ) -> None:
        self.buffer = buffer
        self.detects_piece = detects_piece
        self.processing_delay_seconds = processing_delay_seconds
        self.processed_indices: list[int] = []
        self.evidence_indices: list[int] = []

    def _overlay_item(self, item: dict | None) -> dict | None:
        if item is None:
            return None
        processed = item.copy()
        color = (
            (240, 0, 0)
            if str(item.get("utc", "")).startswith("pre-")
            else (0, 0, 240)
        )
        processed["frame"] = np.full_like(item["frame"], color)
        processed["raw_frame"] = item["frame"]
        return processed

    def process_clip_frame(
        self,
        item: dict,
        *,
        include_evidence_images: bool = False,
    ) -> tuple[dict, float]:
        if self.processing_delay_seconds > 0:
            time.sleep(self.processing_delay_seconds)
        self.processed_indices.append(int(item["index"]))
        if include_evidence_images:
            self.evidence_indices.append(int(item["index"]))
        result = self.snapshot(include_images=True)["result"].copy()
        result["frame_index"] = int(item["index"])
        result["frame_utc"] = item["utc"]
        result["frame_monotonic"] = float(item["monotonic"])
        result["_recording_frame"] = self._overlay_item(item)
        return result, 1.0

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

    def test_frames_between_returns_the_pretrigger_window(self) -> None:
        buffer = FrameBuffer(maxlen=8)
        for index, frame_monotonic in enumerate((7.9, 8.0, 9.0, 10.0)):
            buffer.append(
                {
                    "index": index,
                    "utc": f"frame-{index}",
                    "monotonic": frame_monotonic,
                    "frame": np.zeros((4, 4, 3), dtype=np.uint8),
                }
            )

        selected = buffer.frames_between(8.0, 10.0)

        self.assertEqual([item["index"] for item in selected], [1, 2])

    def test_latest_at_or_before_selects_the_camera_frame_nearest_the_plc(self) -> None:
        buffer = FrameBuffer(maxlen=8)
        for index, frame_monotonic in enumerate((9.8, 9.95, 10.05)):
            buffer.append(
                {
                    "index": index,
                    "utc": f"frame-{index}",
                    "monotonic": frame_monotonic,
                    "frame": np.zeros((4, 4, 3), dtype=np.uint8),
                }
            )

        selected = buffer.latest_at_or_before(10.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 1)


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
    def test_internal_recording_frame_does_not_enter_api_payloads(self) -> None:
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

        self.assertNotIn("_recording_frame", processor.snapshot(include_images=True)["result"])

    def test_historical_clip_frames_do_not_rewind_the_latest_result(self) -> None:
        processor = LiveProcessor(SimpleNamespace(process_fps=10.0), FrameBuffer(maxlen=8))
        processor._set_state(
            last_frame_index=90,
            result={"frame_index": 90, "pieces": []},
        )
        historical_result = {
            "frame_index": 80,
            "frame_utc": "frame-80",
            "_original_jpeg": b"processed-frame-80",
        }

        with patch.object(processor, "_process", return_value=historical_result):
            processor.process_clip_frame(
                {
                    "index": 80,
                    "utc": "frame-80",
                    "monotonic": 80.0,
                    "frame": np.zeros((8, 8, 3), dtype=np.uint8),
                }
            )

        self.assertEqual(processor.snapshot()["result"]["frame_index"], 90)


class PlcEdgeTests(unittest.TestCase):
    def test_changed_mode_accepts_boolean_edges(self) -> None:
        self.assertTrue(edge_matches("rising", "changed"))
        self.assertTrue(edge_matches("falling", "changed"))
        self.assertFalse(edge_matches("", "changed"))

    def test_plc_signal_state_marks_only_recent_triggers(self) -> None:
        status = {
            "last_trigger": {"event_read_monotonic": 100.0},
        }
        with patch("live_mvp_app.time.perf_counter", return_value=103.5):
            recent = plc_status_with_signal_state(status)
        with patch("live_mvp_app.time.perf_counter", return_value=105.0):
            stale = plc_status_with_signal_state(status)

        self.assertTrue(recent["signal_recent"])
        self.assertEqual(recent["last_trigger_age_seconds"], 3.5)
        self.assertFalse(stale["signal_recent"])

    def test_live_ui_exposes_plc_signal_state(self) -> None:
        self.assertIn('id="plc-signal"', LIVE_TEMPLATE)
        self.assertIn("plc.last_trigger", LIVE_SCRIPT)
        self.assertIn("PLC signal received", LIVE_SCRIPT)
        self.assertIn("Last PLC Cut Signal", LIVE_SCRIPT)
        self.assertNotIn("Last PLC signal", LIVE_SCRIPT)
        self.assertIn("measurement-taken", LIVE_STYLE)
        self.assertIn("data.recorder?.measurement_marker_active", LIVE_SCRIPT)
        self.assertIn("updatePlcSignal(data.plc || {})", LIVE_SCRIPT)
        self.assertIn("keepLiveVideoNearEdge", LIVE_SCRIPT)
        self.assertIn("/api/live/frame?metadata=1", LIVE_SCRIPT)
        self.assertIn('src="/api/live/stream.mp4"', LIVE_TEMPLATE)
        self.assertNotIn("/api/live/image.jpg?frame=", LIVE_SCRIPT)
        self.assertIn("analysisRequestInFlight", LIVE_SCRIPT)
        self.assertNotIn("TX2 Vision", LIVE_TEMPLATE.split("<body>", 1)[-1])
        self.assertNotIn("<h1>Live MVP</h1>", LIVE_TEMPLATE)

    def test_frontend_assets_are_served_from_templates_and_static_files(self) -> None:
        with flask_app.test_client() as client:
            live_response = client.get("/")
            history_response = client.get("/history")
            script_response = client.get("/static/js/live.js")
            style_response = client.get("/static/css/live.css")
            try:
                self.assertEqual(live_response.status_code, 200)
                self.assertEqual(history_response.status_code, 200)
                self.assertEqual(script_response.status_code, 200)
                self.assertEqual(style_response.status_code, 200)
                self.assertIn(b"/static/js/live.js", live_response.data)
                self.assertIn(b"/static/css/history.css", history_response.data)
                self.assertNotIn(b"<style>", live_response.data)
                self.assertNotIn(b"<script>", live_response.data)
            finally:
                live_response.close()
                history_response.close()
                script_response.close()
                style_response.close()


class HistoryTests(unittest.TestCase):
    def test_history_only_renders_the_green_measurement_evidence(self) -> None:
        self.assertIn("PLC signal measurement frame", HISTORY_SCRIPT)
        self.assertIn('class="measurement-evidence"', HISTORY_SCRIPT)
        self.assertIn("Raw clip", HISTORY_SCRIPT)
        self.assertNotIn("Up to 6 representative captures", HISTORY_SCRIPT)
        self.assertNotIn("No diagram evidence", HISTORY_SCRIPT)
        self.assertNotIn("SQLite temporal is active", HISTORY_TEMPLATE)

    def test_measurement_evidence_uses_only_the_canonical_snapshot(self) -> None:
        snapshots = [
            {"frame_index": 1, "frame_monotonic": 10.1, "is_canonical": 0},
            {"frame_index": 2, "frame_monotonic": 12.0, "is_canonical": 1},
            {"frame_index": 3, "frame_monotonic": 12.1, "is_canonical": 0},
        ]

        selected = measurement_evidence_snapshots(snapshots)

        self.assertEqual([item["frame_index"] for item in selected], [2])

class ClipRecorderTests(unittest.TestCase):
    def test_shutdown_rejects_new_plc_recordings(self) -> None:
        args = SimpleNamespace(record_seconds=8.0)
        recorder = ClipRecorder(args, FrameBuffer(maxlen=8))

        recorder.stop(timeout_seconds=0.1)
        recorder.start_event_clip(
            {
                "event_edge": "rising",
                "event_read_monotonic": time.perf_counter(),
            }
        )

        self.assertEqual(recorder.snapshot()["clip_index"], 0)
        self.assertTrue(recorder.snapshot()["shutting_down"])

    def test_frame_collector_preserves_clip_frames_while_processing_lags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=1.0,
                pre_trigger_seconds=0.0,
                record_fps=10.0,
                capture_fps=100.0,
                save_raw_clips=False,
                measurement_delay_seconds=0.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=100)
            processor = FakeOverlayProcessor(
                buffer,
                processing_delay_seconds=0.03,
            )
            recorder = ClipRecorder(args, buffer, processor)
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
            deadline = time.perf_counter() + 6.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            sidecars = list(Path(temp_dir).rglob("*.json"))
            self.assertEqual(len(sidecars), 1)
            data = json.loads(sidecars[0].read_text(encoding="utf-8"))
            self.assertGreater(data["processed_source_frame_count"], 8)
            self.assertEqual(
                data["processed_source_frame_count"],
                len(processor.processed_indices),
            )
            self.assertEqual(
                processor.processed_indices,
                list(
                    range(
                        processor.processed_indices[0],
                        processor.processed_indices[-1] + 1,
                    )
                ),
            )

    def test_retention_removes_corrupt_sidecar_and_inferred_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            clip_dir = output_dir / "live_plc_clips" / "2026-07-29"
            clip_dir.mkdir(parents=True)
            newest_sidecar = clip_dir / "live_0002.json"
            newest_sidecar.write_text('{"event_id": "newest"}', encoding="utf-8")
            expired_sidecar = clip_dir / "live_0001.json"
            expired_sidecar.write_bytes(b"\x00" * 64)
            expired_video = clip_dir / "live_0001.mp4"
            expired_video.write_bytes(b"video")
            expired_raw_video = clip_dir / "live_0001_raw.mp4"
            expired_raw_video.write_bytes(b"raw")
            expired_analysis = clip_dir / "live_0001_analysis"
            expired_analysis.mkdir()
            (expired_analysis / "evidence.jpg").write_bytes(b"image")
            database = FakeDatabase()
            recorder = ClipRecorder(
                SimpleNamespace(output_dir=output_dir, max_clips=1),
                FrameBuffer(maxlen=8),
                database=database,
            )

            recorder._enforce_retention()

            self.assertTrue(newest_sidecar.exists())
            self.assertFalse(expired_sidecar.exists())
            self.assertFalse(expired_video.exists())
            self.assertFalse(expired_raw_video.exists())
            self.assertFalse(expired_analysis.exists())
            self.assertEqual(database.deleted_event_ids, [])
            self.assertEqual(recorder.error, "")

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

    def test_measurement_marker_starts_at_the_plc_event(self) -> None:
        self.assertFalse(measurement_marker_active(9.999, 10.0))
        self.assertTrue(measurement_marker_active(10.0, 10.0))
        self.assertTrue(measurement_marker_active(10.799, 10.0))
        self.assertFalse(measurement_marker_active(10.8, 10.0))

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

    def test_clip_starts_before_plc_and_marks_measurement_at_the_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=Path(temp_dir),
                record_seconds=3.0,
                pre_trigger_seconds=0.2,
                record_fps=10.0,
                capture_fps=30.0,
                save_raw_clips=False,
                measurement_delay_seconds=0.0,
                max_clips=10,
            )
            buffer = FrameBuffer(maxlen=30)
            processor = FakeOverlayProcessor(buffer)
            recorder = ClipRecorder(args, buffer, processor)
            event_monotonic = time.perf_counter()
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            for index, offset in enumerate((-0.25, -0.15, -0.05)):
                buffer.append(
                    {
                        "index": index,
                        "utc": f"pre-{index}",
                        "monotonic": event_monotonic + offset,
                        "frame": frame.copy(),
                    }
                )

            stop_feeder = threading.Event()

            def feed_frames() -> None:
                index = 3
                while not stop_feeder.is_set():
                    buffer.append(
                        {
                            "index": index,
                            "utc": f"post-{index}",
                            "monotonic": time.perf_counter(),
                            "frame": frame.copy(),
                        }
                    )
                    index += 1
                    time.sleep(0.01)

            feeder = threading.Thread(target=feed_frames, daemon=True)
            feeder.start()
            self.addCleanup(stop_feeder.set)
            self.addCleanup(feeder.join, 1.0)

            recorder.start_event_clip(
                {
                    "event_edge": "rising",
                    "event_read_monotonic": event_monotonic,
                }
            )
            deadline = time.perf_counter() + 5.0
            while recorder.snapshot()["recording"] and time.perf_counter() < deadline:
                time.sleep(0.02)
            stop_feeder.set()
            feeder.join(timeout=1.0)

            sidecars = list(Path(temp_dir).rglob("*.json"))
            self.assertEqual(len(sidecars), 1)
            data = json.loads(sidecars[0].read_text(encoding="utf-8"))
            self.assertEqual(data["frames_written"], 30)
            self.assertEqual(data["first_frame_index"], 1)
            self.assertEqual(data["pre_trigger_seconds"], 0.2)
            self.assertAlmostEqual(data["post_trigger_seconds"], 2.8, places=3)
            self.assertEqual(data["plc_event_video_offset_seconds"], 0.2)
            self.assertEqual(data["measurement_delay_seconds"], 0.0)
            self.assertEqual(data["measurement_actual_offset_seconds"], 0.0)
            self.assertAlmostEqual(
                data["measurement_source_frame_offset_seconds"],
                -0.05,
                places=3,
            )
            self.assertTrue(data["measurement_event_frame_processed"])
            self.assertEqual(data["event_frame_processing_error"], "")
            self.assertTrue(
                data["processing_snapshots"][0]["measurement_event_frame"]
            )
            self.assertEqual(data["measurement_marker_first_video_frame_index"], 2)
            self.assertEqual(data["measurement_marker_frames_written"], 8)
            self.assertEqual(data["processing_mode"], "plc_triggered_clip")
            self.assertEqual(
                data["processed_source_frame_count"],
                len(processor.processed_indices),
            )
            self.assertEqual(
                len(processor.processed_indices),
                len(set(processor.processed_indices)),
            )
            self.assertGreater(data["processed_source_frame_count"], 3)
            self.assertEqual(processor.evidence_indices, [2])
            self.assertEqual(data["processing_snapshot_count"], 1)
            video = cv2.VideoCapture(data["video_path"])
            try:
                frame_count = 0
                while True:
                    ok, processed_frame = video.read()
                    if not ok:
                        break
                    center = processed_frame[12:-12, 12:-12]
                    blue, _green, red = center.mean(axis=(0, 1))
                    self.assertGreater(max(blue, red), 100)
                    frame_count += 1
                self.assertEqual(frame_count, 30)
            finally:
                video.release()

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
                self.assertEqual(data["processing_mode"], "plc_triggered_clip")
                self.assertGreaterEqual(data["processed_source_frame_count"], 1)
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
                record_seconds=1.0,
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

            deadline = time.perf_counter() + 5.0
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
