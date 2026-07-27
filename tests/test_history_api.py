from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import live_mvp_app as live
from tx2_database import RevisionConflict


class FakeDatabase:
    def __init__(self) -> None:
        self.last_update = None
        self.conflict = False

    def list_measurement_events(self, *, limit: int, before: str | None):
        return [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "status": "complete",
                "created_at": "2026-07-27T12:00:00+00:00",
                "detected_piece_count": 1,
                "valid_piece_count": 1,
            }
        ]

    def get_measurement_event(self, event_id: str):
        return {
            "id": event_id,
            "status": "complete",
            "detected_piece_count": 1,
            "valid_piece_count": 1,
            "snapshots": [],
            "pieces": [
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "piece_number": 1,
                    "is_valid": True,
                    "review_required": False,
                    "automatic_measurement_in": 120,
                    "operator_measurement_in": None,
                    "effective_measurement_in": 120,
                    "operator_revision": 0,
                    "revisions": [],
                }
            ],
            "assets": [],
        }

    def set_operator_measurement(self, **values):
        if self.conflict:
            raise RevisionConflict("stale revision")
        self.last_update = values
        return values

    def clear_operator_measurement(self, **values):
        self.last_update = {**values, "clear": True}
        return values


class HistoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_database = live._database
        self.previous_args = getattr(live, "_args", None)
        self.temp_dir = tempfile.TemporaryDirectory()
        live._args = SimpleNamespace(output_dir=Path(self.temp_dir.name))
        live._database = FakeDatabase()
        self.client = live.app.test_client()

    def tearDown(self) -> None:
        live._database = self.previous_database
        if self.previous_args is not None:
            live._args = self.previous_args
        self.temp_dir.cleanup()

    def test_history_list_is_backed_by_postgresql_mode(self) -> None:
        response = self.client.get("/api/history/events?limit=20")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["database_mode"], "postgresql")
        self.assertEqual(payload["count"], 1)

    def test_operator_measurement_converts_units_and_records_identity(self) -> None:
        response = self.client.patch(
            "/api/history/events/11111111-1111-1111-1111-111111111111/"
            "pieces/22222222-2222-2222-2222-222222222222/operator-measurement",
            json={
                "feet": 10,
                "inches": 1,
                "sixteenths": 8,
                "reason": "Verified against tape",
                "operator_id": "operator.1",
                "operator_display_name": "Operator One",
                "expected_revision": 0,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(float(live._database.last_update["measurement_in"]), 121.5)
        self.assertEqual(live._database.last_update["operator_id"], "operator.1")

    def test_stale_operator_revision_returns_conflict(self) -> None:
        live._database.conflict = True

        response = self.client.patch(
            "/api/history/events/11111111-1111-1111-1111-111111111111/"
            "pieces/22222222-2222-2222-2222-222222222222/operator-measurement",
            json={
                "measurement_in": 121,
                "reason": "Stale test",
                "operator_id": "operator.1",
                "expected_revision": 0,
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.get_json()["conflict"])


if __name__ == "__main__":
    unittest.main()
