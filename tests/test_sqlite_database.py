from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tx2_database import RevisionConflict, build_event_key, event_uuid
from tx2_sqlite_database import SQLiteDatabaseRepository


def write_sidecar(output_dir: Path) -> tuple[Path, str]:
    day_dir = output_dir / "live_plc_clips" / "2026-07-27"
    day_dir.mkdir(parents=True)
    video_path = day_dir / "clip.mp4"
    video_path.write_bytes(b"video")
    event = {
        "event_edge": "rising",
        "event_value": True,
        "previous_event_value": False,
        "event_source_timestamp": "2026-07-27T12:00:00+00:00",
        "read_utc": "2026-07-27T12:00:00.010+00:00",
        "event_read_monotonic": 100.0,
    }
    endpoint = "opc.tcp://10.14.6.48:49320"
    event_node = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
    key = build_event_key(event, endpoint, event_node)
    event_id = event_uuid(key)
    snapshots = [
        {
            "frame_index": 10,
            "frame_monotonic": 100.2,
            "frame_utc": "2026-07-27T12:00:00.200+00:00",
            "processed_utc": "2026-07-27T12:00:00.250+00:00",
            "original_width": 1920,
            "original_height": 1080,
            "measurement_summary": {
                "detected_count": 2,
                "valid_count": 1,
                "invalid_count": 1,
            },
            "pieces": [
                {
                    "piece_id": 1,
                    "valid": True,
                    "confidence": 0.91,
                    "box": {"conf": 0.91},
                    "sobel": {"edge_confidence": 0.82, "crm_px": 121.0},
                    "measurement": {
                        "measurement_in": 120.5,
                        "delta_in": 4.5,
                        "delta_px": 42.0,
                    },
                },
                {
                    "piece_id": 2,
                    "valid": False,
                    "confidence": 0.77,
                    "box": {"conf": 0.77},
                    "sobel": {"edge_confidence": 0.31},
                    "measurement": {},
                },
            ],
        }
    ]
    sidecar = {
        "event_id": event_id,
        "event_key": key,
        "event": event,
        "plc_endpoint": endpoint,
        "plc_event_node": event_node,
        "plc_watchdog_node": "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD",
        "camera_source": "test",
        "saved_at": "2026-07-27T12:00:08.100+00:00",
        "first_frame_utc": "2026-07-27T12:00:00.100+00:00",
        "last_frame_utc": "2026-07-27T12:00:08.100+00:00",
        "first_frame_index": 9,
        "last_frame_index": 89,
        "video_path": str(video_path),
        "processing_snapshots": snapshots,
        "vision_configuration": {
            "homography_sha256": "h" * 64,
            "calibration_sha256": "c" * 64,
            "model_sha256": "m" * 64,
            "homography_json": {"source_points": []},
            "calibration_json": {"scale": 1.0},
            "model_path": "model.pt",
            "model_name": "model.pt",
            "confidence_threshold": 0.5,
            "inference_size": 960,
            "app_commit_sha": "abc123",
        },
    }
    sidecar_path = day_dir / "clip.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path, event_id


class SQLiteDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "outputs"
        self.output_dir.mkdir()
        self.database_path = self.output_dir / "tx2_live_mvp.sqlite3"
        self.repository = SQLiteDatabaseRepository(self.database_path)
        self.repository.open()
        self.repository.validate_schema()

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def test_sync_persists_piece_counts_snapshots_and_assets(self) -> None:
        sidecar_path, event_id = write_sidecar(self.output_dir)

        actual_id = self.repository.sync_sidecar(sidecar_path, self.output_dir)
        events = self.repository.list_measurement_events()
        detail = self.repository.get_measurement_event(event_id)

        self.assertEqual(actual_id, event_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["detected_piece_count"], 2)
        self.assertEqual(events[0]["valid_piece_count"], 1)
        self.assertIsNotNone(detail)
        self.assertEqual(len(detail["pieces"]), 2)
        self.assertEqual(len(detail["snapshots"]), 1)
        self.assertTrue(
            any(asset["asset_type"] == "video" for asset in detail["assets"])
        )
        self.assertTrue(self.repository.health().ok)

    def test_operator_corrections_are_audited_and_survive_resync(self) -> None:
        sidecar_path, event_id = write_sidecar(self.output_dir)
        self.repository.sync_sidecar(sidecar_path, self.output_dir)
        detail = self.repository.get_measurement_event(event_id)
        piece_id = detail["pieces"][0]["id"]

        self.repository.set_operator_measurement(
            event_id=event_id,
            piece_id=piece_id,
            measurement_in=Decimal("121.5"),
            reason="Tape verification",
            operator_id="operator.1",
            operator_display_name="Operator One",
            expected_revision=0,
            source_ip="127.0.0.1",
        )
        with self.assertRaises(RevisionConflict):
            self.repository.set_operator_measurement(
                event_id=event_id,
                piece_id=piece_id,
                measurement_in=Decimal("122"),
                reason="Stale edit",
                operator_id="operator.1",
                operator_display_name=None,
                expected_revision=0,
                source_ip="127.0.0.1",
            )

        self.repository.sync_sidecar(sidecar_path, self.output_dir)
        corrected = self.repository.get_measurement_event(event_id)["pieces"][0]

        self.assertEqual(corrected["operator_measurement_in"], 121.5)
        self.assertEqual(corrected["effective_measurement_in"], 121.5)
        self.assertEqual(corrected["operator_revision"], 1)
        self.assertEqual(len(corrected["revisions"]), 1)
        self.assertEqual(len(self.repository.operator_histories()), 1)

        self.repository.clear_operator_measurement(
            event_id=event_id,
            piece_id=piece_id,
            reason="Return to automatic",
            operator_id="operator.1",
            operator_display_name="Operator One",
            expected_revision=1,
            source_ip="127.0.0.1",
        )
        cleared = self.repository.get_measurement_event(event_id)["pieces"][0]
        self.assertIsNone(cleared["operator_measurement_in"])
        self.assertEqual(cleared["effective_measurement_in"], 120.5)
        self.assertEqual(cleared["operator_revision"], 2)
        self.assertEqual(len(cleared["revisions"]), 2)

    def test_database_reopens_without_losing_history(self) -> None:
        sidecar_path, event_id = write_sidecar(self.output_dir)
        self.repository.sync_sidecar(sidecar_path, self.output_dir)
        self.repository.close()

        reopened = SQLiteDatabaseRepository(self.database_path)
        reopened.open()
        try:
            self.assertIsNotNone(reopened.get_measurement_event(event_id))
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
