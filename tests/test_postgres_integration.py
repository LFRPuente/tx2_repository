from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import uuid
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.apply_postgres_migrations import migration_statements
from tx2_database import DatabaseRepository, RevisionConflict


@unittest.skipUnless(
    os.environ.get("TX2_TEST_POSTGRES_DSN"),
    "TX2_TEST_POSTGRES_DSN is not configured",
)
class PostgreSQLRepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        cls.schema = f"tx2_test_{uuid.uuid4().hex}"
        cls.admin = psycopg.connect(
            os.environ["TX2_TEST_POSTGRES_DSN"],
            autocommit=True,
        )
        cls.admin.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema))
        )
        cls.admin.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(cls.schema))
        )
        migration_sql = (
            REPO_ROOT / "db" / "migrations" / "001_initial.sql"
        ).read_text(encoding="utf-8")
        for statement in migration_statements(migration_sql):
            cls.admin.execute(statement)
        cls.repository = DatabaseRepository(
            make_conninfo(
                os.environ["TX2_TEST_POSTGRES_DSN"],
                options=f"-csearch_path={cls.schema}",
            )
        )
        cls.repository.open()
        cls.repository.validate_schema()

    @classmethod
    def tearDownClass(cls) -> None:
        from psycopg import sql

        cls.repository.close()
        cls.admin.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema))
        )
        cls.admin.close()

    def test_sidecar_sync_is_idempotent_and_corrections_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            clip_dir = output_dir / "live_plc_clips" / "2026-07-27"
            analysis_dir = clip_dir / "clip_analysis"
            analysis_dir.mkdir(parents=True)
            video_path = clip_dir / "clip.mp4"
            raw_video_path = clip_dir / "clip_raw.mp4"
            original_path = analysis_dir / "original.jpg"
            rectified_path = analysis_dir / "rectified.jpg"
            video_path.write_bytes(b"test-video")
            raw_video_path.write_bytes(b"test-raw-video")
            original_path.write_bytes(b"test-original")
            rectified_path.write_bytes(b"test-rectified")
            sidecar_path = clip_dir / "clip.json"
            sidecar = {
                "saved_at": "2026-07-27T12:00:08+00:00",
                "event_id": str(uuid.uuid4()),
                "event_key": uuid.uuid4().hex,
                "event": {
                    "read_utc": "2026-07-27T12:00:00+00:00",
                    "event_source_timestamp": "2026-07-27T12:00:00+00:00",
                    "event_server_timestamp": None,
                    "event_read_monotonic": 10.0,
                    "event_edge": "rising",
                    "event_value": True,
                    "previous_event_value": False,
                    "watchdog_value": 10,
                },
                "plc_endpoint": "opc.tcp://127.0.0.1:49320",
                "plc_event_node": "event",
                "plc_watchdog_node": "watchdog",
                "camera_source": "test",
                "vision_configuration": {
                    "homography_sha256": "h" * 64,
                    "calibration_sha256": "c" * 64,
                    "model_sha256": "m" * 64,
                    "homography_json": {"matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                    "calibration_json": {"inch_per_px": 0.1},
                    "model_path": "model.pt",
                    "model_name": "model.pt",
                    "confidence_threshold": 0.5,
                    "inference_size": 960,
                    "app_commit_sha": "test",
                },
                "first_frame_index": 1,
                "last_frame_index": 80,
                "first_frame_utc": "2026-07-27T12:00:00+00:00",
                "last_frame_utc": "2026-07-27T12:00:08+00:00",
                "video_path": str(video_path),
                "raw_video_path": str(raw_video_path),
                "processing_snapshots": [
                    {
                        "frame_index": 1,
                        "frame_monotonic": 10.1,
                        "frame_utc": "2026-07-27T12:00:00.1+00:00",
                        "processed_utc": "2026-07-27T12:00:00.2+00:00",
                        "original_width": 1920,
                        "original_height": 1080,
                        "measurement_summary": {
                            "detected_count": 1,
                            "valid_count": 1,
                            "invalid_count": 0,
                        },
                        "pieces": [
                            {
                                "piece_id": 1,
                                "valid": True,
                                "confidence": 0.91,
                                "box": {"x": 10, "y": 20, "w": 30, "h": 40, "conf": 0.91},
                                "sobel": {
                                    "is_valid": True,
                                    "edge_confidence": 0.85,
                                    "crm_px": 1.0,
                                },
                                "measurement": {
                                    "measurement_in": 120.0,
                                    "delta_in": 2.0,
                                    "delta_px": 20.0,
                                },
                            }
                        ],
                        "original_overlay_path": str(original_path),
                        "rectified_overlay_path": str(rectified_path),
                    }
                ],
            }
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

            event_id = self.repository.sync_sidecar(sidecar_path, output_dir)
            repeated_event_id = self.repository.sync_sidecar(sidecar_path, output_dir)

            self.assertEqual(event_id, repeated_event_id)
            events = self.repository.list_measurement_events(limit=10)
            matching = [event for event in events if str(event["id"]) == event_id]
            self.assertEqual(len(matching), 1)
            detail = self.repository.get_measurement_event(event_id)
            self.assertEqual(len(detail["pieces"]), 1)
            self.assertTrue(
                any(
                    asset["asset_type"] == "raw_video"
                    for asset in detail["assets"]
                )
            )
            piece = detail["pieces"][0]
            self.repository.set_operator_measurement(
                event_id=event_id,
                piece_id=str(piece["id"]),
                measurement_in=Decimal("121.5"),
                reason="Manual verification",
                operator_id="operator.test",
                operator_display_name="Test Operator",
                expected_revision=0,
                source_ip="127.0.0.1",
            )
            with self.assertRaises(RevisionConflict):
                self.repository.set_operator_measurement(
                    event_id=event_id,
                    piece_id=str(piece["id"]),
                    measurement_in=Decimal("122"),
                    reason="Stale update",
                    operator_id="operator.test",
                    operator_display_name="Test Operator",
                    expected_revision=0,
                    source_ip="127.0.0.1",
                )
            self.repository.clear_operator_measurement(
                event_id=event_id,
                piece_id=str(piece["id"]),
                reason="Correction no longer required",
                operator_id="operator.test",
                operator_display_name="Test Operator",
                expected_revision=1,
                source_ip="127.0.0.1",
            )
            updated = self.repository.get_measurement_event(event_id)
            self.assertEqual(updated["pieces"][0]["operator_revision"], 2)
            self.assertEqual(len(updated["pieces"][0]["revisions"]), 2)


if __name__ == "__main__":
    unittest.main()
