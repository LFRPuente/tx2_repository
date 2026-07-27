from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_mvp_app import clip_sidecars, operator_measurement_inches
from tools.apply_postgres_migrations import migration_statements
from tx2_database import (
    DatabaseRepository,
    build_event_key,
    relative_asset_path,
    resolve_asset_path,
    select_canonical_snapshot,
    snapshot_pieces,
)


def snapshot(
    frame_index: int,
    monotonic: float,
    valid_count: int,
    invalid_count: int,
    edge_confidence: float,
) -> dict:
    detected_count = valid_count + invalid_count
    return {
        "frame_index": frame_index,
        "frame_monotonic": monotonic,
        "measurement_summary": {
            "detected_count": detected_count,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
        },
        "pieces": [
            {
                "piece_id": index + 1,
                "valid": index < valid_count,
                "measurement": {"measurement_in": 120 + index},
                "sobel": {"edge_confidence": edge_confidence},
                "box": {"conf": 0.9},
            }
            for index in range(detected_count)
        ],
    }


class CanonicalSnapshotTests(unittest.TestCase):
    def test_prefers_valid_count_then_invalid_count_then_edge_confidence(self) -> None:
        snapshots = [
            snapshot(1, 100.1, 1, 0, 0.95),
            snapshot(2, 100.2, 2, 1, 0.90),
            snapshot(3, 100.3, 2, 0, 0.70),
            snapshot(4, 100.4, 2, 0, 0.88),
        ]

        selected = select_canonical_snapshot(snapshots, event_monotonic=100.0)

        self.assertEqual(selected["frame_index"], 4)

    def test_limits_primary_candidates_to_first_one_and_a_half_seconds(self) -> None:
        snapshots = [
            snapshot(1, 50.5, 1, 0, 0.80),
            snapshot(2, 52.0, 4, 0, 0.99),
        ]

        selected = select_canonical_snapshot(snapshots, event_monotonic=50.0)

        self.assertEqual(selected["frame_index"], 1)

    def test_uses_whole_clip_when_timing_is_unavailable(self) -> None:
        snapshots = [
            snapshot(1, 10.0, 1, 0, 0.70),
            snapshot(2, 11.0, 2, 0, 0.60),
        ]
        for item in snapshots:
            item.pop("frame_monotonic")

        selected = select_canonical_snapshot(snapshots, event_monotonic=10.0)

        self.assertEqual(selected["frame_index"], 2)

    def test_legacy_single_measurement_becomes_one_piece(self) -> None:
        pieces = snapshot_pieces(
            {
                "measurement": {"measurement_in": 125.5, "delta_in": 5.5},
                "sobel": {"is_valid": True, "edge_confidence": 0.8},
                "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "conf": 0.9}],
            }
        )

        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["piece_id"], 1)
        self.assertTrue(pieces[0]["valid"])


class DatabaseUtilityTests(unittest.TestCase):
    def test_event_key_is_deterministic_and_signal_specific(self) -> None:
        event = {
            "read_utc": "2026-07-27T12:00:00+00:00",
            "event_source_timestamp": "2026-07-27T12:00:00+00:00",
            "event_edge": "rising",
            "event_value": True,
        }
        first = build_event_key(event, "opc.tcp://localhost:49320", "node")
        second = build_event_key(dict(event), "opc.tcp://localhost:49320", "node")
        changed = build_event_key({**event, "read_utc": "2026-07-27T12:00:01+00:00"}, "opc.tcp://localhost:49320", "node")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_asset_paths_must_stay_inside_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset = root / "live_plc_clips" / "clip.mp4"
            asset.parent.mkdir()
            asset.write_bytes(b"video")

            relative = relative_asset_path(asset, root)

            self.assertEqual(relative, "live_plc_clips/clip.mp4")
            self.assertEqual(resolve_asset_path(relative, root), asset.resolve())
            with self.assertRaises(ValueError):
                relative_asset_path(root.parent / "outside.mp4", root)
            with self.assertRaises(ValueError):
                resolve_asset_path("../outside.mp4", root)

    def test_database_requires_an_explicit_dsn(self) -> None:
        with self.assertRaises(ValueError):
            DatabaseRepository("")

    def test_operator_units_convert_to_inches(self) -> None:
        value = operator_measurement_inches(
            {"feet": 10, "inches": 3, "sixteenths": 8}
        )

        self.assertEqual(float(value), 123.5)

    def test_operator_units_reject_invalid_inch_fields(self) -> None:
        with self.assertRaises(ValueError):
            operator_measurement_inches(
                {"feet": 10, "inches": 12, "sixteenths": 0}
            )

    def test_migration_runner_owns_transaction_boundaries(self) -> None:
        statements = migration_statements("BEGIN; CREATE TABLE example(id integer); COMMIT;")

        self.assertEqual(statements, ["CREATE TABLE example(id integer)"])

    def test_sidecar_listing_tolerates_retention_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            day_dir = output_dir / "live_plc_clips" / "2026-07-27"
            day_dir.mkdir(parents=True)
            stable = day_dir / "stable.json"
            vanishing = day_dir / "vanishing.json"
            stable.write_text("{}", encoding="utf-8")
            vanishing.write_text("{}", encoding="utf-8")
            original_stat = Path.stat

            def flaky_stat(path: Path, *args, **kwargs):
                if path == vanishing:
                    raise FileNotFoundError(path)
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", flaky_stat):
                result = clip_sidecars(output_dir)

            self.assertEqual(result, [stable])


if __name__ == "__main__":
    unittest.main()
